# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1-schedule jagged zerocheck: the height-aware multi-chip sumcheck round.

The statement, in one sum — the claim the transcript carries::

    sum_{x in {0,1}^nrv} eq(zeta, x) * sum_c lambda_c * (
        C_{alpha_c}(trace_c(x)) - C_{alpha_c}(0_row) * geq_c(x)
        + sum_j beta**(j+1) * col_{c,j}(x)
    )  =  sum_c lambda_c * v_c

Per chip the constraint part sums to ZERO (AIR constraints vanish on real
witness rows — the zerocheck statement) and the column part to v_c, the
chip's GKR opening claim at ``zeta`` — so one single-phase sumcheck over the
``nrv`` row variables proves every chip's constraint zero-sum and its
opening claim together; there is no separate opening pass. The three
challenges batch three dimensions (`JaggedZerocheckSummand`): alpha a
chip's constraints, beta its columns, lambda across chips.

Chips keep their own heights: each trace lives in a fixed-width buffer — its
real height padded to the next multiple of 4, SP1's round-0 alignment — with
the live prefix packed at the front, halving per round
(``ceil(num_real / 2)``, SP1's ``next_start_indices_and_column_heights``
schedule) while the buffer width stays put. The fixed width keeps every
per-chip op shape-invariant across rounds, so each chip's constraint circuit
compiles to ONE kernel reused by every round and t-point: the
``constraint_eval`` live-width bound masks rows past the live prefix to the
field's zero (a recognizing emitter skips their circuit work entirely), and
the zero tail makes the full-width sums byte-equal to sums truncated at the
live bound — field zero-adds are exact. Storage stays proportional to
``sum(num_cols * num_real)`` instead of ``num_chips * num_cols *
2**num_vars``. Rows past a chip's real height read as zero; their constant
constraint value ``C_alpha(0_row)`` is subtracted via a virtual
``x >= num_real`` indicator, making the effective summand
``eq * (C_alpha(trace) - C_alpha(0_row) * geq)`` — zero past the real prefix.
The GKR column batch (``beta`` + ``claims``) adds
``sum_j beta**(j+1) * col_j``, turning each chip's proved sum from zero into
its GKR opening at ``zeta``; padded rows are unaffected (a zero row's column
term vanishes with it). The protocol content is declared on
``JaggedZerocheckSummand``; ``prove_jagged_zerocheck`` is the SP1-schedule
driver around it.

Round loop is one ``jax.lax.scan``, so the traced graph is O(1) in
``num_vars`` (one round body, not ``num_vars`` copies): the fixed-width
buffers make every per-chip operand shape round-invariant, so the round carry
holds a fixed shape and each chip's constraint circuit compiles to ONE kernel
the scan reuses across all rounds. The carry is ``(buffers, virtual-geqs,
chip claims, eq adjustment, live heights, eq table, transcript)``; the live
heights ride as traced ``int32`` (the ``constraint_eval`` live-width bound and
the eq gather index), and the eq table folds in-carry (summing adjacent pairs
drops one variable) rather than being recomputed per round. The per-chip loop
stays Python-unrolled inside the body — chips differ in fixed width — so the
graph is O(num_chips), never O(num_chips * num_vars). The round-poly trick
below is SP1's rather than the product summand the ``zorch.sumcheck`` marker
carries.

Round polys mirror SP1's ``sum_as_poly_in_last_variable``: the degree-4 poly
is pinned by evaluations at t in {0, 2, 4}, the claim identity
``p(1) = claim - p(0)``, and an implicit zero at the root of the bound eq
factor scaling every term — the Gruen compression, assembled by
``zorch.sumcheck.gruen`` (this engine's extra points are {2, 4}; the {0, 2, 4}
choice is SP1's). Polys travel in COEFFICIENT form (lambda-RLC across chips)
through the transcript — SP1's encoding, intentionally different from the
evaluation-form messages of zorch's dense sumcheck driver.

Variable order: zorch's ``expand_eq_to_hypercube`` indexes the hypercube with
``zeta[n-1]`` at the LSB, so the even/odd row pairing fixes ``zeta[n-1-round]``
— SP1's back-to-front order (its ``jagged_point`` is the challenge list
reversed; the reversal is the consumer's).

References (pinned at the same SP1 commit as ``coeffs.rlc_coeffs``):
- round-poly trick — ``sum_as_poly_in_last_variable``,
  https://github.com/fractalyze/sp1/blob/640d8b80c/slop/crates/sumcheck/src/poly.rs
- column-height schedule — ``next_start_indices_and_column_heights``,
  https://github.com/fractalyze/sp1/blob/640d8b80c/sp1-gpu/crates/utils/src/jagged.rs
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
from jax import Array, lax
from zk_dtypes import efinfo

from zorch.constraint_eval import constraint_eval
from zorch.poly.eq import contract_hypercube_step, eq_factor, expand_eq_to_hypercube
from zorch.poly.geq import VirtualGeq
from zorch.sumcheck.gruen import (
    fold_round_scalars,
    interp_matrix,
    round_coeffs_from_matrix,
)
from zorch.sumcheck.prover import RoundMsg, split_pairs, zero_extend
from zorch.transcript import Transcript, sample_challenge

from sp1_zorch.zerocheck.coeffs import gkr_powers

if TYPE_CHECKING:
    from zorch.sumcheck.gruen import GruenSummand


def _challenge_limbs(dtype) -> int:
    """Transcript squeezes per challenge of ``dtype``: an extension field
    takes ``degree`` base squeezes reinterpreted (SP1's ``sample_ext_element``
    convention, fractalyze/sp1-zorch#88); the transcript's own field takes
    one."""
    try:
        return efinfo(dtype).degree
    except ValueError:
        return 1


def _constraint_and_column_term(
    eval_fn: Callable[[Array], Array],
    rows_t: Array,
    alpha: Array,
    eq: Array,
    gkr_powers: Array,
    live_width: Array,
) -> Array:
    """A constrained chip's inner sum at one t-point: the alpha-folded
    constraint evaluation plus the GKR column term, eq-weighted and summed
    over the pair axis with rows past ``live_width`` masked to zero.

    The per-row column term ``sum_c rows[r,c] * gkr_powers[c]`` rides
    ``constraint_eval``'s ``column_weights`` operand on every backend: a
    recognizing emitter adds the dot in-kernel while the row is already
    loaded (the cross-row reduce stays external, keeping the big round
    one-thread-per-row), and an unrecognizing backend inlines the
    decomposition's identical dot."""
    return jnp.sum(
        constraint_eval(
            eval_fn,
            rows_t,
            alpha,
            live_width=live_width,
            column_weights=gkr_powers,
        )
        * eq
    )


def _column_term(
    rows_t: Array, alpha: Array, eq: Array, gkr_powers: Array, live_width: Array
) -> Array:
    """A lookup-only chip's whole inner sum at one t-point: just the
    eq-weighted GKR column term as a standalone matmul. Such a chip has NO
    transition constraints — SP1's Byte / Program / Range ship K = 0 in every
    real shard, so this is a production strategy, not a defensive one
    (`constraint_eval` rejects an empty alpha)."""
    del alpha, live_width  # no constraint fold to mask or bound
    return jnp.sum((rows_t @ gkr_powers) * eq)


# A chip's term strategy — `_constraint_and_column_term` with its eval_fn
# bound, or `_column_term` — chosen once at summand init:
# (rows_t, alpha, eq, gkr_powers, live_width) -> the inner sum at one t-point.
_TermFn = Callable[[Array, Array, Array, Array, Array], Array]


@dataclass(frozen=True)
class JaggedZerocheckSummand:
    """SP1's jagged zerocheck summand — the protocol, separated from the
    scan driver below. Per chip ``c`` at row ``x`` the summand is

        eq(zeta, x) * ( C_{alpha_c}(trace_c(x))
                        - C_{alpha_c}(0_row) * geq_c(x)
                        + sum_j beta**(j+1) * col_{c,j}(x) )

    and the round polynomial is the ``lambdas``-RLC across chips — three
    batchings under three challenges: ``alphas[c]`` folds chip ``c``'s K
    constraints (descending powers, ``coeffs.rlc_coeffs``; empty for a
    lookup-only chip — SP1's Byte / Program / Range ship K = 0 in every
    real shard), ``beta`` batches its columns (the GKR opening term), and
    ``lambdas`` batches across chips. Chips are RLC'd anew every round,
    never folded — a chip index is a batch axis, not a sumcheck variable,
    so the reduction is single-phase.

    Evaluation points: ``{0, 2, 4}`` are computed (`chip_evals`); t = 1 and
    the bound eq factor's root come free (the claim identity and the Gruen
    zero, assembled by the driver through ``zorch.sumcheck.gruen`` — this
    class satisfies its ``GruenSummand`` seam: ``degree - 2`` extra points
    beyond s(0)). Round 0 drops the constraint term at t = 0 — that term IS
    the zerocheck statement (constraints vanish on real witness rows) — but
    never the column term, which does not vanish there. The
    ``C(0_row) * geq`` subtraction is this summand's padding correction
    (every jagged summand has one; LogUp's dual is the neutral-fraction
    virtual mass in ``zorch.logup_gkr.jagged_prover``): rows past a chip's
    real height read as the MLE's canonical zero-extension, whose constant
    constraint value would otherwise leak into the sum."""

    # SP1's zerocheck round-poly degree: constraint degree <= 3 plus the eq
    # factor; the constraint-degree bound is the caller's contract (not
    # probeable).
    DEGREE: ClassVar[int] = 4

    eval_fns: Sequence[Callable[[Array], Array]]
    alphas: Sequence[Array]
    lambdas: Array
    beta: Array
    # Each chip's term strategy — constraint+column, or column-only for a
    # lookup-only chip — chosen once here off its alpha's static length, so
    # `chip_evals` runs the same code for every chip.
    _term_fns: tuple[_TermFn, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_term_fns",
            tuple(
                partial(_constraint_and_column_term, fn)
                if alpha.shape[-1] > 0
                else _column_term
                for fn, alpha in zip(self.eval_fns, self.alphas)
            ),
        )

    @property
    def degree(self) -> int:
        return self.DEGREE

    def extra_ts(self, dtype) -> tuple[Array, ...]:
        """SP1's materialized extra points ``{2, 4}`` — degree 4 makes
        d - 2 = 2 extras beyond s(0) (the Gruen seam's invariant); the
        {0, 2, 4} choice is SP1's ``sum_as_poly_in_last_variable``."""
        return (jnp.array(2, dtype), jnp.array(4, dtype))

    def chip_evals(
        self,
        i: int,
        p0: Array,
        diff: Array,
        eq: Array,
        gkr_powers: Array,
        padded_row_adj: Array,
        last: Array,
        eq_adj: Array,
        nr_live: Array,
        vgeq: VirtualGeq,
        *,
        is_zero: bool,
        is_round0: Array,
    ) -> tuple[Array, Array, Array]:
        """Chip ``i``'s computed ``(y0, y2, y4)`` round-poly evaluations —
        SP1's ``sum_as_poly_in_last_variable``: inner sums over the chip's
        fixed pair width with rows past the live bound masked to zero
        (byte-equal to SP1's truncated sums), and the virtual-geq padded-row
        correction per t-point. ``eq`` is the round's eq table sliced to this
        chip's pair width; ``gkr_powers``/``padded_row_adj`` are the driver's
        per-chip expansions of ``beta`` and the ``C(0_row)`` probe.
        ``nr_live`` is the chip's live height this round (a traced ``int32``
        — the live-width bound and the eq gather index).

        ``is_zero`` is STATIC (``ceil(nr/2)`` fixes any ``nr >= 1``, so only
        a chip that starts empty is ever empty): its width-0 buffer and zero
        adjustment collapse every t-point to zero, leaving the claim identity
        alone — without running ``constraint_eval`` on an empty trace."""
        alpha = self.alphas[i]
        term = self._term_fns[i]
        ef = last.dtype
        zero = jnp.zeros((), ef)
        two = jnp.array(2, ef)
        four = jnp.array(4, ef)

        if is_zero:
            return zero, zero, zero

        num_non_padded = (nr_live + 1) // 2

        # The computed evaluation points {0, 2, 4} — SP1's choice of materialized
        # t-points; t = 1 and the eq root come free (claim identity / Gruen zero).
        t_traces = (p0, p0 + two * diff, p0 + four * diff)

        # The t-points run separately (not batched) so the t-point-specific
        # round-0 t=0 constraint drop can be applied by zeroing `alpha`
        # (`sum_k C_k*0 = 0`), leaving the alpha-independent column term
        # intact.
        def inner(rows: Array, *, mask_round0: bool) -> Array:
            a = (
                jnp.where(is_round0, jnp.zeros_like(alpha), alpha)
                if mask_round0
                else alpha
            )
            return term(rows.T, a, eq, gkr_powers, num_non_padded)

        inner_0 = inner(t_traces[0], mask_round0=True)
        inner_2 = inner(t_traces[1], mask_round0=False)
        inner_4 = inner(t_traces[2], mask_round0=False)

        # A live chip keeps nr_live >= 1, so threshold_half is in [0, eq.shape[0]):
        # an in-bounds dynamic gather, the same element a static index would read.
        threshold_half = num_non_padded - 1
        msb_lagrange = eq_adj * eq[threshold_half]

        def correct(y: Array, t_val: Array) -> Array:
            # The padding correction (the jagged summands' shared concept —
            # LogUp's dual is the neutral-fraction virtual mass in
            # zorch.logup_gkr.jagged_prover): scale by the bound eq factor and
            # subtract the zero-extension leak — rows past the live prefix
            # read as the MLE's canonical zero-extension, whose constant
            # constraint value (padded_row_adj = C_alpha(0_row)) enters the
            # inner sum; the virtual geq's closed form removes that mass
            # without materializing the 2**num_vars indicator.
            eq_last = eq_factor(t_val, last)
            vg = vgeq.fix_last_variable(t_val).eval_at(threshold_half)
            return eq_last * (y * eq_adj - padded_row_adj * vg * msb_lagrange)

        y_0 = correct(inner_0, zero)
        y_2 = correct(inner_2, two)
        y_4 = correct(inner_4, four)
        return y_0, y_2, y_4

    def combine_chips(self, polys: Array) -> Array:
        """The cross-chip lambda-RLC of the per-chip coefficient polys —
        re-applied every round; the wire carries only the RLC'd poly."""
        return jnp.dot(self.lambdas, polys)


# Wire-shape alias: the verifier and the byte-match harness size the round
# polys off the summand's degree.
DEGREE = JaggedZerocheckSummand.DEGREE

if TYPE_CHECKING:
    # Seam conformance pin (zorch docs/conventions.md "Seam conformance
    # pins"): any type-checker pass over this module fails if the summand
    # drifts off the GruenSummand vocabulary the Gruen assembly reads.
    _gruen_summand: type[GruenSummand] = JaggedZerocheckSummand


def prove_jagged_zerocheck(
    summand: JaggedZerocheckSummand,
    traces: Sequence[Array],
    num_reals: Sequence[int],
    zeta: Array,
    transcript: Transcript,
    *,
    claims: Sequence[Array],
) -> tuple[list[Array], Transcript, RoundMsg]:
    """Run the SP1-schedule jagged multi-chip zerocheck sumcheck.

    The protocol content — the summand equation, the three batching
    challenges, the computed evaluation points, the round-0 rule — is
    ``summand``'s (`JaggedZerocheckSummand`); this driver owns the
    SP1-schedule plumbing: the fixed-width buffers, the scan, the Gruen
    assembly, the Fiat-Shamir thread, and the per-round fold.

    ``traces[c]`` is chip ``c``'s column-major ``(num_cols_c, num_reals[c])``
    trace — exactly its real rows; the driver owns all padding (the zero-tail
    soundness contract is internal). ``zeta`` is the eq point and each of its
    coordinates gets exactly one round.

    ``claims[c]`` — chip ``c``'s GKR opening at ``zeta`` — seeds its
    ``p(1) = claim - p(0)`` identity; the matching column term rides
    ``summand.beta``.

    Returns the final per-chip ``(num_cols_c, h)`` folded traces
    (``h in {0, 2}``; position 0 holds the column's evaluation at the
    sumcheck point), the advanced transcript, and the stacked ``RoundMsg``
    whose ``round_poly`` is ``(num_vars, DEGREE+1)`` in COEFFICIENT form."""
    eval_fns = summand.eval_fns
    alphas = summand.alphas
    lambdas = summand.lambdas
    beta = summand.beta
    num_chips = len(traces)
    if num_chips == 0:
        raise ValueError("at least one chip is required")
    if lambdas.ndim != 1:
        raise ValueError(f"lambdas must be 1-D, got shape {lambdas.shape}")
    if not (
        num_chips == len(eval_fns) == len(num_reals) == len(alphas) == lambdas.shape[0]
    ):
        raise ValueError(
            "eval_fns, traces, num_reals, alphas, and lambdas must agree on the "
            f"chip count: {len(eval_fns)}, {num_chips}, {len(num_reals)}, "
            f"{len(alphas)}, {lambdas.shape[0]}"
        )
    if len(claims) != num_chips:
        raise ValueError(
            f"claims must give one GKR opening per chip: got {len(claims)} "
            f"for {num_chips} chips"
        )
    num_vars = int(zeta.shape[0])
    width = 1 << num_vars
    nrs = [int(nr) for nr in num_reals]
    for i, (trace, nr) in enumerate(zip(traces, nrs)):
        if trace.ndim != 2:
            raise ValueError(f"traces[{i}] must be 2-D, got shape {trace.shape}")
        if not 0 <= nr <= width:
            raise ValueError(f"num_reals[{i}] must be within [0, {width}], got {nr}")
        if trace.shape[1] != nr:
            raise ValueError(
                f"traces[{i}] has height {trace.shape[1]} but num_reals[{i}] is "
                f"{nr}; pass exactly the real rows — the driver owns the padding"
            )

    ef = zeta.dtype
    ef_limbs = _challenge_limbs(ef)
    one = jnp.ones((), ef)
    zero = jnp.zeros((), ef)

    # C_alpha(0_row), the constant every padded row contributes — probed once
    # per chip; num_real == 0 and constraint-less chips never trace their
    # constraint formula. A single zero row suffices: bounding the eval at
    # live_width=1 routes it through the compact loop-form GPU emitter instead
    # of the monolithic CSE unroll a non-bounded wide circuit triggers — which
    # on the koalabear Global chip is a 271k-instr kernel and a >660s
    # cold-compile cliff. The loop-form emitter engages on a single-row
    # ([1, nc]) trace as of fractalyze/zkx#704; the same circuit then lowers to
    # ~785 instrs and Global cold-compiles in a few seconds.
    adjs = [
        zero
        if nrs[i] == 0 or alphas[i].shape[-1] == 0
        else constraint_eval(
            eval_fns[i],
            jnp.zeros((1, traces[i].shape[0]), dtype=ef),
            alphas[i],
            live_width=1,
        )[0]
        for i in range(num_chips)
    ]

    # Fixed-width buffers: each chip's width is pinned once at its padded
    # round-0 height and never shrinks; folds re-extend into the same width
    # so every per-chip op (and its compiled kernel) is round-invariant. The
    # buffer rides the scan carry, so it is promoted to the round's field up
    # front: a base-field trace would otherwise change dtype the first time it
    # folds against an extension-field challenge, which a fixed-shape carry
    # forbids (the embedding is exact, so the round polys are unchanged).
    widths = [((nr + 3) // 4) * 4 for nr in nrs]
    bufs = [zero_extend(traces[i], widths[i]).astype(ef) for i in range(num_chips)]
    # int32 threshold from the start: as a scan carry it must keep the same
    # leaf type fix_last_variable produces (it folds to a traced int32).
    vgeqs = [VirtualGeq(jnp.asarray(nr, jnp.int32), one, zero) for nr in nrs]
    chip_claims = jnp.stack(list(claims))
    # The summand's beta expanded into per-chip column weights, each sliced to
    # its chip's width.
    max_cols = max(t.shape[0] for t in traces)
    gkr_all = gkr_powers(beta, max_cols) if max_cols else jnp.zeros(0, ef)
    chip_gkr = [gkr_all[: t.shape[0]] for t in traces]
    eq_adj = one
    # Only a chip that starts empty is ever empty (see `chip_evals`); the
    # rest carry their live height as a traced int32 round carry. Zero-height
    # chips are production, not defensive: an SP1 shard carries its shape's
    # FULL chip set, and chips the program never exercised ship zero rows
    # (six of the fibonacci fixture's 35 main chips).
    is_zero_chip = [nr == 0 for nr in nrs]
    nrs_live = [jnp.asarray(nr, jnp.int32) for nr in nrs]

    # Round-varying scan inputs, all O(1) to build (no per-round constraint
    # work): `last` is zeta back-to-front (the round binds zeta[n-1-rnd]); the
    # Gruen matrix maps each round's computed evaluations at the summand's
    # points plus the two free ones (t=1 from the claim identity, the implicit
    # zero at the bound eq factor's root) to coefficients, prebuilt per round
    # outside the scan as `zorch.sumcheck.gruen` prescribes for fixed-shape
    # drivers; `is_round0` flags the constraint skip at t=0.
    extra_ts = summand.extra_ts(ef)
    if len(extra_ts) != summand.degree - 2:
        raise ValueError(
            f"a degree-{summand.degree} Gruen round materializes s(0) plus "
            f"{summand.degree - 2} extra points, got {len(extra_ts)}"
        )
    last_xs = zeta[::-1]
    interp_xs = jax.vmap(lambda z: interp_matrix(extra_ts, z))(last_xs)
    # A traced flag rather than a peeled first round: unrolling round 0 out
    # of the scan would put a second copy of every chip circuit in the graph
    # — the compile cliff the O(1) scan exists to avoid.
    is_round0_xs = jnp.arange(num_vars) == 0

    # The eq table over the not-yet-bound variables, folded in-carry: summing
    # adjacent pairs drops the bound variable (expand_eq builds it LSB-last, so
    # eq_m[2j]+eq_m[2j+1] = eq_{m-1}[j] exactly), so it stays this round's table
    # without a per-round re-expansion. Width is the widest pair count; chips
    # slice their own prefix, the dead tail past a chip's live height is zero.
    eq_buf = expand_eq_to_hypercube(zeta[: num_vars - 1], one)
    eq_width = eq_buf.shape[0]

    def fold_eq(buf: Array) -> Array:
        if eq_width < 2:
            return buf  # single round: no next-round table is needed
        return zero_extend(contract_hypercube_step(buf), eq_width)

    def round_step(carry, xs):
        bufs, vgeqs, chip_claims, eq_adj, nrs_live, eq_buf, transcript = carry
        last, interp, is_round0 = xs

        # Only constraint_eval (inside `chip_evals`) is shape-divergent, so
        # the chip loop runs it unrolled and collects the (y0, y2, y4)
        # round-poly evaluations as scalars.
        y0s, y2s, y4s = [], [], []
        p0s = []
        diffs = []
        for i in range(num_chips):
            p0, p1 = split_pairs(bufs[i])
            diff = p1 - p0
            p0s.append(p0)
            diffs.append(diff)
            y_0, y_2, y_4 = summand.chip_evals(
                i,
                p0,
                diff,
                eq_buf[: widths[i] // 2],
                chip_gkr[i],
                adjs[i],
                last,
                eq_adj,
                nrs_live[i],
                vgeqs[i],
                is_zero=is_zero_chip[i],
                is_round0=is_round0,
            )
            y0s.append(y_0)
            y2s.append(y_2)
            y4s.append(y_4)

        # The Gruen assembly + RLC tail is uniform-shape across chips, so the
        # chip evaluations ride its batch axis — one matrix product for all
        # chips instead of num_chips tiny per-chip launches — transposed to
        # keep coefficients on the last axis for the post-round
        # `fold_round_scalars`; the summand takes the cross-chip RLC. Field ops
        # are exact, so the reassociation gives byte-identical round polys.
        y0 = jnp.stack(y0s)
        y2 = jnp.stack(y2s)
        y4 = jnp.stack(y4s)
        polys = round_coeffs_from_matrix(interp, y0, chip_claims, (y2, y4)).T
        rlc = summand.combine_chips(polys)

        # SP1 binds each variable with one extension element (its
        # ``sample_ext_element``) — the shared ``sample_challenge`` rule.
        transcript = transcript.observe(rlc)
        transcript, r = sample_challenge(transcript, ef, ef_limbs)

        bufs = [zero_extend(p0s[i] + r * diffs[i], widths[i]) for i in range(num_chips)]
        vgeqs = [vg.fix_last_variable(r) for vg in vgeqs]
        nrs_live = [(nr + 1) // 2 for nr in nrs_live]
        chip_claims, eq_adj = fold_round_scalars(polys, r, eq_adj, last)

        carry = (
            bufs,
            vgeqs,
            chip_claims,
            eq_adj,
            nrs_live,
            fold_eq(eq_buf),
            transcript,
        )
        return carry, RoundMsg(round_poly=rlc, challenge=r)

    init = (bufs, vgeqs, chip_claims, eq_adj, nrs_live, eq_buf, transcript)
    (bufs, *_, transcript), msgs = lax.scan(
        round_step, init, (last_xs, interp_xs, is_round0_xs)
    )

    # The first pair of each buffer is the whole fold result; the rest of the
    # fixed width is dead zeros.
    return [b[:, :2] for b in bufs], transcript, msgs
