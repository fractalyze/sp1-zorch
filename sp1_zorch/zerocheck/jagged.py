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

Round loop is one ``frx.lax.scan``, so the traced graph is O(1) in
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

import frx
import frx.numpy as jnp
from frx import Array, lax
from zk_dtypes import efinfo

from zorch.constraint_eval import constraint_eval
from zorch.poly.eq import contract_hypercube_step, eq_factor, expand_eq_to_hypercube
from zorch.poly.geq import VirtualGeq
from zorch.sumcheck.gruen import (
    fold_round_scalars,
    interp_matrix,
    round_coeffs_from_matrix,
)
from zorch.sumcheck.prover import RoundMsg
from zorch.transcript import Transcript, sample_challenge

from sp1_zorch.zerocheck._round_composite import zerocheck_round_poly
from sp1_zorch.zerocheck.coeffs import gkr_powers

if TYPE_CHECKING:
    from zorch.sumcheck.gruen import GruenSummand


def _zero_extend(arr: Array, width: int) -> Array:
    """Zero-extend the last axis to `width` -- the re-extension the fixed-width
    round carry pairs with its fold: the live prefix halves each round while the
    buffer width stays put, and the dead tail stays exactly zero so full-width
    reductions match live-prefix-truncated ones byte-for-byte (field zero-adds
    are exact). In-tree because zorch#412 dropped `sumcheck.prover.zero_extend`
    (internally dead there once its sumcheck family moved to the domain layer);
    the SP1-schedule fixed-width buffer here still needs the re-extension."""
    pad = width - arr.shape[-1]
    if pad < 0:
        raise ValueError(f"width {width} < last-axis size {arr.shape[-1]}")
    if pad == 0:
        return arr
    return jnp.concatenate([arr, jnp.zeros((*arr.shape[:-1], pad), arr.dtype)], axis=-1)


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
    eval_fn: Callable[[Array, Array], Array],
    public_values: Array,
    rows: Array,
    alpha: Array,
    gkr_powers: Array,
    live_width: Array,
) -> Array:
    """A constrained chip's PER-ROW summand value at one t-point: the
    alpha-folded constraint evaluation plus the GKR column term, eq-unweighted
    (the cross-row eq-weighted reduce is the caller's — `_reduce_and_assemble`
    — this is the round marker's future decomposition-body operand) with rows
    past ``live_width`` masked to zero.

    ``rows`` is the chip's row-major ``[rows, num_cols]`` trace read directly —
    the orientation `constraint_eval` already consumes, now the carry's native
    layout so there is no transpose (fractalyze/sp1-zorch#242). The
    `start_offset` window that lets a chip read its segment out of a shared
    class buffer is the follow-on slice — it needs the emitter recognizer that
    disambiguates a second scalar-s32 operand, which rides the wheel rebuild.

    ``public_values`` rides ``constraint_eval``'s ``aux_operands`` so the 2-ary
    ``eval_fn(rows, pv)`` reads the statement as a declared operand — not a
    closure, which would carry a tracer into the composite under the jitted
    stage body.

    The per-row column term ``sum_c rows[r,c] * gkr_powers[c]`` rides
    ``constraint_eval``'s ``column_weights`` operand on every backend: a
    recognizing emitter adds the dot in-kernel while the row is already
    loaded (the cross-row reduce stays external, keeping the big round
    one-thread-per-row), and an unrecognizing backend inlines the
    decomposition's identical dot."""
    return constraint_eval(
        eval_fn,
        rows,
        alpha,
        live_width=live_width,
        column_weights=gkr_powers,
        aux_operands=(public_values,),
    )


def _column_term(
    rows: Array, alpha: Array, gkr_powers: Array, live_width: Array
) -> Array:
    """A lookup-only chip's whole PER-ROW summand value at one t-point: just
    the GKR column term as a standalone matmul, eq-unweighted. Such a chip has
    NO transition constraints — SP1's Byte / Program / Range ship K = 0 in
    every real shard, so this is a production strategy, not a defensive one
    (`constraint_eval` rejects an empty alpha).

    ``rows`` is the chip's row-major ``[rows, num_cols]`` trace, so the dot is
    the direct ``rows @ gkr_powers`` (no transpose)."""
    del alpha, live_width  # no constraint fold to mask or bound
    return rows @ gkr_powers


# A chip's term strategy — `_constraint_and_column_term` with its eval_fn
# bound, or `_column_term` — chosen once at summand init:
# (rows, alpha, gkr_powers, live_width) -> the per-row summand value at one
# t-point, eq-unweighted (the cross-row reduce lives in `_summand_values`'s
# caller). ``rows`` is the chip's row-major ``[rows, num_cols]`` trace.
_TermFn = Callable[[Array, Array, Array, Array], Array]


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

    Evaluation points: ``{0, 2, 4}`` are computed (`chip_raw_evals`); t = 1 and
    the bound eq factor's root come free (the claim identity and the Gruen
    zero, assembled by the driver through ``zorch.sumcheck.gruen`` — this
    class satisfies its ``GruenSummand`` seam: ``degree - 2`` extra points
    beyond s(0)). Round 0 drops the constraint term at t = 0 — that term IS
    the zerocheck statement (constraints vanish on real witness rows) — but
    never the column term, which does not vanish there. The
    ``C(0_row) * geq`` subtraction is this summand's padding correction
    (every jagged summand has one; LogUp's dual is the neutral-fraction
    virtual mass, ``zorch.logup_gkr.prover.LogupSummand.correct``): rows past a chip's
    real height read as the MLE's canonical zero-extension, whose constant
    constraint value would otherwise leak into the sum. SP1's trace-internal
    padding is a different thing and needs no correction — those rows sit
    below ``num_real``, are built constraint-satisfying by trace-gen, and
    are summed as real rows; only the shared-hypercube zero-extension is
    corrected."""

    # SP1's zerocheck round-poly degree: constraint degree <= 3 plus the eq
    # factor; the constraint-degree bound is the caller's contract (not
    # probeable).
    DEGREE: ClassVar[int] = 4

    # 2-ary ``eval_fn(rows, public_values)``: the statement rides as a declared
    # ``constraint_eval`` operand (see ``_constraint_and_column_term``), not a
    # closure that would carry a tracer into the composite under the jitted
    # stage body.
    eval_fns: Sequence[Callable[[Array, Array], Array]]
    alphas: Sequence[Array]
    lambdas: Array
    beta: Array
    public_values: Array
    # Each chip's term strategy — constraint+column, or column-only for a
    # lookup-only chip — chosen once here off its alpha's static length, so
    # `chip_raw_evals` runs the same code for every chip.
    _term_fns: tuple[_TermFn, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_term_fns",
            tuple(
                partial(_constraint_and_column_term, fn, self.public_values)
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

    def chip_raw_evals(
        self,
        i: int,
        p0: Array,
        diff: Array,
        eq: Array,
        gkr_powers: Array,
        nr_live: Array,
        *,
        is_zero: bool,
        is_round0: Array,
    ) -> tuple[Array, Array, Array]:
        """Chip ``i``'s uncorrected inner sums at the computed points {0, 2, 4}.

        Off the driver path since round_step moved to the ``zorch.sumcheck.round``
        marker (fractalyze/zorch#394): retained as the pre-refactor byte oracle
        that ``jagged_test.test_reduce_and_assemble_matches_summand_slots`` checks
        the ``_summand_values`` + ``_reduce_and_assemble`` seam against. Delegates
        to ``_summand_values`` so the two cannot drift. Removal tracked in #242.

        The GruenSummand evals slot, SP1's
        ``sum_as_poly_in_last_variable``: sums over the chip's fixed pair
        width with rows past the live bound masked to zero (byte-equal to
        SP1's truncated sums). ``eq`` is the round's eq table sliced to this
        chip's pair width; ``gkr_powers`` is the driver's per-chip expansion
        of ``beta``; ``nr_live`` is the chip's live height this round (a
        traced ``int32`` — the live-width bound). The t-points run
        separately (not batched) exactly so round 0's t = 0 constraint drop
        fits: zeroing ``alpha`` (``sum_k C_k*0 = 0``) drops the constraint
        term while the alpha-independent column term stays.

        ``is_zero`` is STATIC (``ceil(nr/2)`` fixes any ``nr >= 1``, so only
        a chip that starts empty is ever empty): its width-0 buffer collapses
        every t-point to zero without running ``constraint_eval`` on an
        empty trace; `correct` short-circuits the same way.

        Delegates to `_summand_values` for the per-row values and reduces them
        here — keeping this method's summed-tuple return byte-identical while
        the per-row seam stays reusable outside the `GruenSummand` slot."""
        if is_zero:
            zero = jnp.zeros((), p0.dtype)
            return zero, zero, zero
        v0, v2, v4 = _summand_values(self._term_fns[i], self.alphas[i], p0, diff,
                                     gkr_powers, nr_live, is_round0)
        return jnp.sum(v0 * eq), jnp.sum(v2 * eq), jnp.sum(v4 * eq)

    def correct(
        self,
        raws: tuple[Array, Array, Array],
        eq: Array,
        last: Array,
        eq_adj: Array,
        padded_row_adj: Array,
        nr_live: Array,
        vgeq: VirtualGeq,
        *,
        is_zero: bool,
    ) -> tuple[Array, Array, Array]:
        """The padding correction — the GruenSummand correction slot.

        Off the driver path (round_step uses the marker); retained with
        `chip_raw_evals` as the pre-refactor byte oracle for the seam test.
        Removal tracked in #242.

        (LogUp's dual is `LogupSummand.correct`, the neutral-fraction
        virtual mass): scale each computed point by the bound eq factor and
        subtract the zero-extension leak — rows past the live prefix read
        as the MLE's canonical zero-extension, whose constant constraint
        value (``padded_row_adj = C_alpha(0_row)``, the driver's per-chip
        probe) enters the inner sum; the virtual geq's closed form removes
        that mass without materializing the 2**num_vars indicator.

        A live chip keeps ``nr_live >= 1``, so the straddle index is in
        ``[0, eq.shape[0])`` — an in-bounds dynamic gather, the same
        element a static index would read. An ``is_zero`` chip
        short-circuits with its `chip_raw_evals`: zero evals and zero
        adjustment leave the claim identity alone."""
        ef = last.dtype
        zero = jnp.zeros((), ef)
        two = jnp.array(2, ef)
        four = jnp.array(4, ef)

        if is_zero:
            return zero, zero, zero

        threshold_half = (nr_live + 1) // 2 - 1
        msb_lagrange = eq_adj * eq[threshold_half]

        def one_point(y: Array, t_val: Array) -> Array:
            eq_last = eq_factor(t_val, last)
            vg = vgeq.fix_last_variable(t_val).eval_at(threshold_half)
            return eq_last * (y * eq_adj - padded_row_adj * vg * msb_lagrange)

        return (
            one_point(raws[0], zero),
            one_point(raws[1], two),
            one_point(raws[2], four),
        )

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


def _summand_values(term, alpha, p0, diff, gkr_powers, nr_live, is_round0):
    """Per-t (t=0,2,4) per-row summand values (constraint+column via `term`;
    round-0 drops the constraint at t=0 by zeroing alpha). The marker's
    summand-value operand — `constraint_eval` stays external, eq-unweighted.

    ``p0`` / ``diff`` are the row-major ``[rows, num_cols]`` even/odd row halves
    of the carry, so each t-trace is handed to ``term`` with no transpose
    (fractalyze/sp1-zorch#242)."""
    ef = p0.dtype
    two, four = jnp.array(2, ef), jnp.array(4, ef)
    num_non_padded = (nr_live + 1) // 2
    t_traces = (p0, p0 + two * diff, p0 + four * diff)

    def one(rows, *, mask_round0):
        a = jnp.where(is_round0, jnp.zeros_like(alpha), alpha) if mask_round0 else alpha
        # per-row, eq-unweighted; the carry is row-major, so no transpose
        return term(rows, a, gkr_powers, num_non_padded)

    return (one(t_traces[0], mask_round0=True),
            one(t_traces[1], mask_round0=False),
            one(t_traces[2], mask_round0=False))


def _zero_chip_poly(interp, claim, ef):
    """A statically-empty chip's round poly: nothing sums, so every computed
    point is zero and only the claim identity survives — the trivial
    claim-identity Gruen poly, no reduce or correction."""
    z = jnp.zeros((), ef)
    return round_coeffs_from_matrix(interp, z, claim, (z, z))


def _reduce_and_assemble(vals, eq, interp, claim, last, eq_adj, padded_row_adj,
                         nr_live, vgeq):
    """A live chip's round poly: eq-weighted reduce of the per-t summand values
    + geq zero-extension correction + Gruen degree-4 assembly. Byte-identical to
    chip_raw_evals+correct+round_coeffs_from_matrix; also the marker's
    decomposition body (`_round_composite._decomp` mirrors it)."""
    ef = last.dtype
    y_raw = tuple(jnp.sum(v * eq) for v in vals)
    # A runtime-empty chip (nr_live == 0 — a cap-class chip the program never
    # exercised) has no live row: the last-live-row index (nr_live+1)//2 - 1 is
    # -1, which corrupts the padding correction and leaks a spurious term for a
    # constraint that is nonzero on an all-zero row (e.g. DivRem). Clamp the
    # index and vanish the reduce to the trivial claim identity — byte-identical
    # to the exact path's static `is_zero_chip` branch (its width-0 buffer
    # reaches _zero_chip_poly by another route), at no extra Gruen dot.
    empty = nr_live == 0
    threshold_half = jnp.maximum((nr_live + 1) // 2 - 1, 0)
    msb_lagrange = eq_adj * eq[threshold_half]

    def corr(y, t_val):
        eq_last = eq_factor(t_val, last)
        vg = vgeq.fix_last_variable(t_val).eval_at(threshold_half)
        val = eq_last * (y * eq_adj - padded_row_adj * vg * msb_lagrange)
        return jnp.where(empty, jnp.zeros((), ef), val)

    zero, two, four = jnp.zeros((), ef), jnp.array(2, ef), jnp.array(4, ef)
    y0, y2, y4 = corr(y_raw[0], zero), corr(y_raw[1], two), corr(y_raw[2], four)
    return round_coeffs_from_matrix(interp, y0, claim, (y2, y4))


@dataclass(frozen=True)
class TotalCapClass:
    """The static shard-invariance class of the total-cap zerocheck round
    (fractalyze/sp1-zorch#242): the two buffer bounds every shard of one chip
    set shares, so shards that differ only in per-chip row counts compile ONCE.

    - ``area_cap`` bounds the JAGGED-PACKED round buffer: chips sit in one flat
      column-major buffer, chip ``c`` occupying ``num_cols_c`` segments of its
      even-padded live height, so the live area is
      ``Σ_c num_cols_c * evenpad(num_real_c)`` — no chip pays another chip's
      column count (the dense-rectangle form priced every chip at the widest
      chip's width and blew past device memory on wide-chip sets). The
      ``+ 2*window`` tail keeps the last column's window read in bounds on the
      halved buffer.
    - ``window`` (``W``) bounds any chip's round-0 live pair count
      (``ceil(num_real / 2)``): the per-column window height every chip's
      constraint reads through, and (as ``2*window``) the per-chip block height
      each pre-padded trace arrives at.

    Neither derives from a specific shard's heights — those ride as a traced
    ``int32`` vector, and the segment offsets are their cumsum — so the round
    body's compile keys on this class plus the chip set alone. ``from_heights``
    builds the a-priori-tight class of ONE shard (the static per-shard-compile
    fallback); a whole block passes a class bounding every shard it contains."""

    area_cap: int
    window: int

    @classmethod
    def from_heights(
        cls, num_reals: Sequence[int], num_cols: Sequence[int]
    ) -> "TotalCapClass":
        window = max([1, *((nr + 1) // 2 for nr in num_reals)])
        area = sum(
            nc * (nr + nr % 2)
            for nr, nc in zip(num_reals, num_cols, strict=True)
        )
        return cls(area_cap=area + 2 * window, window=window)

    def shrink_schedule(
        self, sum_cols: int, rounds: int
    ) -> tuple[list[int], list[int]]:
        """Static per-round buffer bounds for the shrinking round prefix: at
        round ``r`` every chip's live height is ``ceil(h / 2^r)``, so the
        packed area halves per round while these bounds stay static.
        ``evenpad(ceil(h/2)) <= evenpad(h)/2 + 1`` per COLUMN gives the
        packed-area recurrence ``area' = area/2 + sum_cols`` (rounded up to
        even, so segment offsets stay pair-aligned), and the window bound
        halves exactly (``ceil(ceil(h/2^r)/2) = ceil(h/2^{r+1})``). Returns
        ``rounds + 1`` entries of ``(area_cap_r, window_r)``; entry 0 is the
        class itself. Every value derives from the class + the chip set's
        static total column count alone, never a shard's heights, so a compile
        keyed on these stays keyed on the class."""
        caps, wins = [self.area_cap], [self.window]
        area = self.area_cap - 2 * self.window
        for _ in range(rounds):
            # WORKAROUND (floor 2, byte-neutral — the extra row is masked dead):
            # a width-1 window makes eval_fn's scalar-constant lift a RESHAPE
            # (scalar -> [1]; any wider width is a broadcast), and XLA layout
            # normalization mis-folds reshape(bitcast-convert(s32[] constant))
            # into an s32[1] constant whose literal stays rank-0 — the verifier
            # then rejects the module. Real fix: the constant-literal handling
            # in XLA layout normalization; drop the floor once that lands.
            w = max((wins[-1] + 1) // 2, 2)
            area = area // 2 + sum_cols
            area += area % 2
            caps.append(area + 2 * w)
            wins.append(w)
        return caps, wins


def pack_flat_arrival(
    traces: Sequence[Array],
    num_reals: Sequence[int],
    cap_class: TotalCapClass,
) -> Array:
    """The class-shaped flat jagged arrival, packed EAGERLY with host-known
    heights: chip c's exact-height columns as ``evenpad(h_c)``-row segments at
    the same ``cols*evenpad(h)`` cumsum offsets ``_prove_total_cap`` derives
    from the traced heights (the two layouts MUST agree), zeros to
    ``area_cap``. Stays in the traces' own (base) field — the body lifts per
    element. No chip is padded to the class window, so the arrival is the live
    area, not ``2W`` per chip (a wide class made that uniform padding overflow
    int32 element indexing and dwarf the packed buffer itself).

    Heights are host ints here, so this ALSO closes the traced-path validation
    gap: an under-bounding class fails loud at pack time instead of silently
    corrupting in the body."""
    need = TotalCapClass.from_heights(
        [int(nr) for nr in num_reals], [int(t.shape[0]) for t in traces]
    )
    if cap_class.area_cap < need.area_cap or cap_class.window < need.window:
        raise ValueError(
            f"total_cap_class {cap_class} does not bound this shard "
            f"(needs area_cap >= {need.area_cap}, window >= {need.window})"
        )
    dtype = traces[0].dtype if traces else jnp.int32
    pieces = []
    used = 0
    for t, nr in zip(traces, num_reals, strict=True):
        h = int(nr)
        if h == 0 or t.shape[0] == 0:
            continue
        if t.shape[1] != h:
            raise ValueError(
                f"pack_flat_arrival wants exact-height traces: got "
                f"{t.shape[1]} rows for height {h}"
            )
        if h % 2:
            t = jnp.pad(t, ((0, 0), (0, 1)))
        pieces.append(t.reshape(-1))
        used += t.shape[0] * t.shape[1]
    pieces.append(jnp.zeros((cap_class.area_cap - used,), dtype))
    return jnp.concatenate(pieces)


# Rounds unrolled at statically halving buffer bounds before the fixed-shape
# tail scan (see the shrink-prefix comment in `_prove_total_cap`). Work decays
# ~2x per unrolled round, so past ~5 the tail is already <4% of round-0 and
# another round only buys compile time (each prefix round compiles its own
# body).
_SHRINK_ROUNDS = 5


def _prove_total_cap(
    summand: JaggedZerocheckSummand,
    traces: Sequence[Array],
    static_nrs: Sequence[int | None],
    num_reals: Sequence[int | Array],
    cap_class: TotalCapClass | None,
    zeta: Array,
    transcript: Transcript,
    claims: Sequence[Array],
    adjs: Sequence[Array],
    ef,
    ef_limbs: int,
    num_vars: int,
    flat_arrival: Array | None = None,
    num_cols: Sequence[int] | None = None,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """The total-cap round carry: ONE flat JAGGED-PACKED buffer
    (fractalyze/sp1-zorch#242), chip ``c`` occupying ``num_cols_c``
    column-major segments of its even-padded live height at a runtime element
    offset (cumsum of ``cols_j * evenpad(h_j)``). Every column segment is even,
    so a whole-buffer stride-2 fold never crosses a segment boundary, and no
    chip pays another chip's column count — the memory property the dense
    ``[ROW_CAP, MAX_COLS]`` rectangle form lacked (one wide chip priced every
    chip at its width and blew past device memory on wide-chip sets).

    Heights are TRACED: ``num_reals`` rides as an ``int32`` vector, segment
    offsets are cumsums of it, and every shape (``area_cap``, ``W``) is a
    static class constant, so shards of one ``TotalCapClass`` + chip set share
    one executable. When ``cap_class`` is ``None`` every height is a host int
    and the class is derived from them (the static per-shard-compile
    fallback); when a class is passed, each chip's trace must arrive
    column-major pre-padded to ``2*window`` rows (zeros past its live height).

    The fold, the ``constraint_eval`` per-row summand, the geq zero-extension
    correction, and the Gruen assembly are unchanged from the per-chip driver —
    only the buffer layout/windowing differs. Each live chip's constraint reads
    its ``[W, cols_c]`` window IN PLACE via ``constraint_eval``'s
    ``start_offset``/``col_stride`` (column ``c`` at
    ``flat[off + c*stride + row]``), masking past its live pairs; the pack and
    the per-round fold re-pack are whole-buffer gathers driven by the traced
    segment starts, so no per-chip slice/update materializes."""
    num_chips = len(num_cols) if num_cols is not None else len(traces)
    one = jnp.ones((), ef)
    zero = jnp.zeros((), ef)

    cols_arr = (
        [int(c) for c in num_cols]
        if num_cols is not None
        else [int(t.shape[0]) for t in traces]
    )
    sum_cols = sum(cols_arr)
    traced = any(nr is None for nr in static_nrs)
    if cap_class is None:
        # Static heights (the per-shard-compile fallback): derive this shard's
        # own a-priori-tight class. A traced height cannot size a static buffer,
        # so `prove_jagged_zerocheck` rejects it before reaching here.
        cap_class = TotalCapClass.from_heights(
            [int(nr) for nr in static_nrs], cols_arr  # type: ignore[arg-type]
        )
    # W ≥ every chip's round-0 live pair count (ceil(nr/2)); the per-column
    # window height every chip's constraint reads through. The +2W area tail
    # keeps the last column's W-window read in bounds on the halved buffer;
    # area_cap and every segment are even, so the whole-buffer stride-2 fold
    # stays pair-aligned.
    W = cap_class.window
    area_cap = cap_class.area_cap
    row_block = 2 * W  # each chip's even-padded UNFOLDED arrival height
    if not traced:
        # Static heights can check the class bound; traced heights cannot (a
        # runtime value cannot gate a compile), so an under-bounding class on
        # the traced path silently corrupts — the caller owns the bound, built
        # from the per-shard ZC_CLASS maxima.
        need = TotalCapClass.from_heights(
            [int(nr) for nr in static_nrs], cols_arr  # type: ignore[arg-type]
        )
        if area_cap < need.area_cap or W < need.window:
            raise ValueError(
                f"total_cap_class {cap_class} does not bound this shard "
                f"(needs area_cap >= {need.area_cap}, window >= {need.window})"
            )
    if traced and flat_arrival is None:
        for i, t in enumerate(traces):
            if t.shape[1] != row_block:
                raise ValueError(
                    f"traces[{i}] has a runtime height, so it must arrive "
                    f"pre-padded to 2*window = {row_block} rows: got {t.shape[1]}"
                )
    # Only a STATICALLY empty chip skips; a traced chip stays live (its runtime
    # emptiness rides the `nr_live == 0` clamp inside `zerocheck_round_poly`).
    is_zero_chip = [nr == 0 for nr in static_nrs]

    # Per-chip column weights, exactly the chip's own columns — jagged packing
    # has no padded columns to weight.
    widest = max(cols_arr) if cols_arr else 0
    gkr_all = gkr_powers(summand.beta, widest) if widest else jnp.zeros(0, ef)
    chip_gkr = [gkr_all[: cols_arr[i]] for i in range(num_chips)]

    # Static segment maps: the flat buffer is sum_cols column segments in chip
    # order; a whole-buffer gather resolves (chip, column) per element from
    # these compile-time tables plus the traced segment starts.
    cols_v = jnp.asarray(cols_arr, jnp.int32)
    chip_of_seg = jnp.asarray(
        [i for i, nc in enumerate(cols_arr) for _ in range(nc)], jnp.int32
    )
    col_of_seg = jnp.asarray(
        [c for nc in cols_arr for c in range(nc)], jnp.int32
    )

    def _evenpad(h: Array) -> Array:
        return h + (h % 2)

    def _area_offsets(h: Array) -> Array:
        """Each chip's runtime element offset in the flat buffer: cumsum of the
        prior chips' areas ``cols_j * evenpad(h_j)``. Every term is even, so
        offsets stay pair-aligned for the whole-buffer stride-2 fold."""
        per_chip = cols_v * _evenpad(h)
        return jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(per_chip)])[
            :num_chips
        ]

    def _gather_halves(
        area_out: int,
        live_rows: Array,
        src_off: Array,
        src_stride: Array,
        fetch,
    ) -> tuple[Array, Array]:
        """Build the packed buffer of static UNFOLDED size ``area_out``
        directly as its stride-2 halves ``(p0h, diffh)`` — the unfolded buffer
        never materializes, which is what keeps the round-0 arena at ~1.5x the
        live area instead of 2x (the unfolded+both-halves peak OOMs the big
        classes). Half element ``j`` lands in segment ``s`` (traced starts,
        ``evenpad(live_rows[chip])/2`` rows each) at half-row ``rh``; its
        unfolded pair rows are ``2*rh`` / ``2*rh + 1``, each reading
        ``fetch(src_off[chip] + col*src_stride[chip] + row)`` when
        ``row < live_rows[chip]`` and exact field zero otherwise (the even-pad
        partner, the class tail, and every dead element) — so
        ``p0h[j] = v0`` and ``diffh[j] = v1 - v0`` match the unfolded pack's
        stride-2 slices byte for byte."""
        seg_lens = (_evenpad(live_rows) // 2)[chip_of_seg]
        seg_starts = jnp.concatenate(
            [jnp.zeros((1,), jnp.int32), jnp.cumsum(seg_lens)]
        )
        j = jnp.arange(area_out // 2, dtype=jnp.int32)
        seg = jnp.clip(
            jnp.searchsorted(seg_starts, j, side="right") - 1, 0, sum_cols - 1
        )
        rh = j - seg_starts[seg]
        chip = chip_of_seg[seg]
        base = src_off[chip] + col_of_seg[seg] * src_stride[chip]
        live = live_rows[chip]

        def at(row):
            src = jnp.clip(base + row, 0, None)
            return jnp.where(row < live, fetch(src), zero)

        v0 = at(2 * rh)
        v1 = at(2 * rh + 1)
        return v0, v1 - v0

    # Initial pack: one gather from the concatenated column-major arrival
    # blocks (chip i's trace [cols_i, arrival_rows_i] flattens to cols_i
    # arrival-height segments; arrival offsets/strides are static). Rows past
    # a chip's live height read the arrival zero-pad or are masked to zero, so
    # the packed buffer is live rows + exact field zeros.
    heights = jnp.stack([jnp.asarray(nr, jnp.int32) for nr in num_reals])
    if flat_arrival is not None:
        # The prologue already packed the class layout eagerly
        # (`pack_flat_arrival`, same cumsum offsets the traced heights derive),
        # so the initial halves are two strided slices. The base-field
        # subtract embeds exactly into the extension field (the embedding is a
        # ring homomorphism on the base subfield), byte-matching the
        # convert-then-subtract order.
        if flat_arrival.shape != (area_cap,):
            raise ValueError(
                f"flat_arrival must be the class-shaped [{area_cap}] buffer, "
                f"got {flat_arrival.shape}"
            )
        p0h0 = flat_arrival[0::2].astype(ef)
        diffh0 = (flat_arrival[1::2] - flat_arrival[0::2]).astype(ef)
    else:
        arr_rows = [int(t.shape[1]) for t in traces]
        arr_off_list: list[int] = []
        acc = 0
        for i in range(num_chips):
            arr_off_list.append(acc)
            acc += cols_arr[i] * arr_rows[i]
        # The arrival stays in ITS OWN field (base for real traces); the lift
        # to the round's extension field happens per gathered element, fused
        # into the pack — an eager `.astype(ef)` would materialize a 4x-wider
        # copy of the whole arrival.
        arr_flat = (
            jnp.concatenate([t.reshape(-1) for t in traces])
            if num_chips
            else jnp.zeros((0,), ef)
        )
        p0h0, diffh0 = _gather_halves(
            area_cap,
            live_rows=heights,
            src_off=jnp.asarray(arr_off_list, jnp.int32),
            src_stride=jnp.asarray(arr_rows, jnp.int32),
            fetch=lambda src: jnp.take(arr_flat, src, mode="clip").astype(ef),
        )

    vgeqs = [VirtualGeq(jnp.asarray(nr, jnp.int32), one, zero) for nr in num_reals]
    chip_claims = jnp.stack(list(claims))
    eq_adj = one

    extra_ts = summand.extra_ts(ef)
    if len(extra_ts) != summand.degree - 2:
        raise ValueError(
            f"a degree-{summand.degree} Gruen round materializes s(0) plus "
            f"{summand.degree - 2} extra points, got {len(extra_ts)}"
        )
    last_xs = zeta[::-1]
    interp_xs = frx.vmap(lambda z: interp_matrix(extra_ts, z))(last_xs)
    is_round0_xs = jnp.arange(num_vars) == 0

    eq_buf = expand_eq_to_hypercube(zeta[: num_vars - 1], one)
    eq_width = eq_buf.shape[0]
    if W > eq_width:
        raise ValueError(
            f"total_cap_class.window {W} exceeds the round-0 half-width "
            f"{eq_width} (2**(num_vars-1)); a pair count cannot exceed it"
        )

    two, four = jnp.array(2, ef), jnp.array(4, ef)

    def _jagged_window(flat: Array, off: Array, stride: Array, n_rows: int,
                       n_cols: int) -> Array:
        """A chip's ``[n_rows, n_cols]`` window out of the flat halved buffer:
        column ``c`` is the rank-1 slice at ``off + c*stride`` (unmasked — the
        caller owns the dead-row zeroing)."""
        return jnp.stack(
            [
                lax.dynamic_slice(flat, (off + c * stride,), (n_rows,))
                for c in range(n_cols)
            ],
            axis=1,
        )

    def make_round_step(w_in: int, cap_out: int, shrink_eq: bool):
        """One round at static window height ``w_in``, re-packing the fold into
        a flat ``[cap_out]`` buffer. ``shrink_eq`` lets the eq table halve
        (the unrolled shrink prefix); the scan tail keeps every carry shape
        fixed instead (``cap_out`` = the input cap, eq zero-extended back)."""
        rows_w = jnp.arange(w_in, dtype=jnp.int32)

        def fold_eq(buf: Array) -> Array:
            if buf.shape[0] < 2:
                return buf
            contracted = contract_hypercube_step(buf)
            if not shrink_eq:
                return _zero_extend(contracted, buf.shape[0])
            # Keep at least the floored window's rows (see `shrink_schedule`):
            # the zero rows pair with masked dead rows, so the eq-weighted
            # reduce is untouched.
            return _zero_extend(contracted, max(contracted.shape[0], 2))

        def round_step(carry, xs):
            p0h, diffh, vgeqs, chip_claims, eq_adj, heights, eq_buf, transcript = carry
            last, interp, is_round0 = xs
            # Each chip's element offset / per-column stride on the HALVED flat
            # buffer: segments are even, so halving the offsets and strides
            # lands exactly on the folded segments. The carry IS the halves —
            # the unfolded buffer never exists (see `_gather_halves`), so
            # there is no per-round slice/subtract either.
            folded_off = _area_offsets(heights) // 2
            half_stride = _evenpad(heights) // 2
            eq_slice = eq_buf[:w_in]
            # The 3 sumcheck t-points are p0h + t*diffh for t in {0,2,4}. Pass p0h
            # (base) and diffh (delta) as the two shared flat halves + t as a RUNTIME
            # fold coefficient, so constraint_eval folds `base+t*delta` per live row
            # INSIDE the kernel (no full-buffer `p0h+t*diffh` materialization) and one
            # compiled kernel serves every t-point. t=0 still carries delta (coeff 0)
            # so all three markers share one kernel shape. diffh stays materialized:
            # it is SHARED across the 3 t-points, so one full-buffer subtract is cheaper
            # than folding `p1h-p0h` in-kernel 3x (measured: fold-in was +68ms).
            t_coeffs = (zero, two, four)

            polys_list = []
            for i in range(num_chips):
                if is_zero_chip[i]:
                    polys_list.append(_zero_chip_poly(interp, chip_claims[i], ef))
                    continue
                live_pair = (heights[i] + 1) // 2
                off_i = folded_off[i]
                stride_i = half_stride[i]
                if summand.alphas[i].shape[-1] > 0:
                    alpha = summand.alphas[i]
                    # Round 0 drops the constraint at t=0 (that term IS the zerocheck
                    # statement) by zeroing alpha; the column term stays.
                    a0 = jnp.where(is_round0, jnp.zeros_like(alpha), alpha)
                    vals = tuple(
                        constraint_eval(
                            summand.eval_fns[i],
                            p0h,
                            a0 if k == 0 else alpha,
                            live_width=live_pair,
                            start_offset=off_i,
                            window_rows=w_in,
                            col_stride=stride_i,
                            num_cols=cols_arr[i],
                            delta=diffh,
                            fold_coeff=t_coeffs[k],
                            column_weights=chip_gkr[i],
                            aux_operands=(summand.public_values,),
                        )
                        for k in range(3)
                    )
                else:
                    # Lookup-only chip (no transition constraints): just the GKR
                    # column term, on explicitly masked windows (dead rows
                    # straddle the next segment, so they are NOT zero in place).
                    mask = (rows_w < live_pair)[:, None]  # [w_in, 1]
                    win_p0 = jnp.where(
                        mask,
                        _jagged_window(p0h, off_i, stride_i, w_in, cols_arr[i]),
                        zero,
                    )
                    win_diff = jnp.where(
                        mask,
                        _jagged_window(diffh, off_i, stride_i, w_in, cols_arr[i]),
                        zero,
                    )
                    t_traces = (
                        win_p0, win_p0 + two * win_diff, win_p0 + four * win_diff
                    )
                    vals = tuple(t_traces[k] @ chip_gkr[i] for k in range(3))
                polys_list.append(
                    zerocheck_round_poly(
                        vals, eq_slice, interp, chip_claims[i], last, eq_adj, adjs[i],
                        heights[i], vgeqs[i],
                    )
                )

            polys = jnp.stack(polys_list)
            rlc = summand.combine_chips(polys)
            transcript = transcript.observe(rlc)
            transcript, r = sample_challenge(transcript, ef, ef_limbs)

            # Fold every chip's live pairs (`p0 + r·diff`) and re-compact
            # straight into the next round's halves at the halved even-padded
            # segment offsets — one whole-buffer gather (no per-chip
            # window/update chain, no unfolded intermediate). A folded row's
            # source row index equals its index in the old half column
            # (evenpad(h)/2 == ceil(h/2) exactly), so `src_stride` is the old
            # half stride.
            new_heights = (heights + 1) // 2
            new_p0h, new_diffh = _gather_halves(
                cap_out,
                live_rows=new_heights,
                src_off=folded_off,
                src_stride=half_stride,
                fetch=lambda src: (
                    jnp.take(p0h, src, mode="clip")
                    + r * jnp.take(diffh, src, mode="clip")
                ),
            )

            vgeqs = [vg.fix_last_variable(r) for vg in vgeqs]
            chip_claims, eq_adj = fold_round_scalars(polys, r, eq_adj, last)
            carry = (
                new_p0h, new_diffh, vgeqs, chip_claims, eq_adj, new_heights,
                fold_eq(eq_buf), transcript,
            )
            return carry, RoundMsg(round_poly=rlc, challenge=r)

        return round_step

    # Shrinking round prefix + fixed-shape tail scan: every chip's live height
    # halves per round, so a single fixed-shape scan does full class-shaped
    # work `num_vars` times over rows that are ~all dead after the first few
    # folds. Unrolling the first `_SHRINK_ROUNDS` rounds at the statically
    # halving class bounds (`shrink_schedule`) makes the round work decay
    # geometrically — Σ 2^-r ≈ 2 round-0 units — and the tail scan then runs at
    # 2^-k scale. Byte-identical: only dead-zero buffer tails shrink, and field
    # zero-adds are exact. Each prefix round compiles its own body (shapes
    # differ), still keyed on the class alone; kernel sharing across chips and
    # t-points within a round is untouched.
    caps_r, wins_r = cap_class.shrink_schedule(sum_cols, _SHRINK_ROUNDS)
    unroll = min(_SHRINK_ROUNDS, num_vars)
    carry = (p0h0, diffh0, vgeqs, chip_claims, eq_adj, heights, eq_buf, transcript)
    prefix_msgs = []
    for rnd in range(unroll):
        step = make_round_step(wins_r[rnd], caps_r[rnd + 1], shrink_eq=True)
        carry, msg = step(carry, (last_xs[rnd], interp_xs[rnd], is_round0_xs[rnd]))
        prefix_msgs.append(msg)
    if prefix_msgs:
        msgs = frx.tree.map(lambda *ls: jnp.stack(ls), *prefix_msgs)
    if unroll < num_vars or not prefix_msgs:
        step = make_round_step(wins_r[unroll], caps_r[unroll], shrink_eq=False)
        carry, tail_msgs = lax.scan(
            step, carry, (last_xs[unroll:], interp_xs[unroll:], is_round0_xs[unroll:])
        )
        msgs = (
            frx.tree.map(lambda a, b: jnp.concatenate([a, b]), msgs, tail_msgs)
            if prefix_msgs
            else tail_msgs
        )
    (p0h, diffh, _vg, _cc, _ea, heights, _eqb, transcript) = carry

    # The first two rows of each chip's final segments hold the fold result
    # (rows past the live height are the invariant even-pad zeros); transpose
    # back to the column-major `(num_cols, 2)` folded-trace contract. Rows past
    # the final live height are masked to zero — a no-op for a live chip, and
    # the guard a traced runtime-empty chip needs (its zero-length segments
    # would otherwise read the next chip's rows). A statically empty chip owns
    # no elements at all.
    off = _area_offsets(heights) // 2
    stride = _evenpad(heights) // 2
    row2 = jnp.arange(2, dtype=jnp.int32)[:, None]
    finals = []
    for i in range(num_chips):
        if is_zero_chip[i]:
            finals.append(jnp.zeros((cols_arr[i], 2), ef))
            continue
        # Row 0 of a chip's final column is p0, row 1 is p0 + diff — the
        # halves hold the final fold pair directly.
        p0row = _jagged_window(p0h, off[i], stride[i], 1, cols_arr[i])
        drow = _jagged_window(diffh, off[i], stride[i], 1, cols_arr[i])
        win = jnp.concatenate([p0row, p0row + drow], axis=0)
        win = jnp.where(row2 < heights[i], win, zero)
        finals.append(win.T)
    return finals, transcript, msgs


def prove_jagged_zerocheck(
    summand: JaggedZerocheckSummand,
    traces: Sequence[Array],
    num_reals: Sequence[int | Array],
    zeta: Array,
    transcript: Transcript,
    *,
    claims: Sequence[Array],
    total_cap_class: TotalCapClass | None = None,
    flat_arrival: Array | None = None,
    num_cols: Sequence[int] | None = None,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """Run the SP1-schedule jagged multi-chip zerocheck sumcheck.

    The protocol content — the summand equation, the three batching
    challenges, the computed evaluation points, the round-0 rule — is
    ``summand``'s (`JaggedZerocheckSummand`); this driver owns the
    SP1-schedule plumbing: the fixed-width buffers, the scan, the Gruen
    assembly, the Fiat-Shamir thread, and the per-round fold.

    ``traces[c]`` is chip ``c``'s column-major trace. A chip's
    ``num_reals[c]`` entry is either a host int — the trace holds exactly its
    real rows and the driver owns all padding (the zero-tail soundness
    contract is internal) — or a traced int32 scalar, in which case the trace
    must arrive pre-padded to ``2*total_cap_class.window`` rows and the CALLER
    owns the zero tail past the live rows. Which entries are traced is a
    compile-time
    property: a host ``0`` statically empties the chip, while a traced height
    keeps the chip's circuit live and only bounds its rows at run time.
    ``zeta`` is the eq point and each of its coordinates gets exactly one
    round.

    ``claims[c]`` — chip ``c``'s GKR opening at ``zeta`` — seeds its
    ``p(1) = claim - p(0)`` identity; the matching column term rides
    ``summand.beta``.

    ``total_cap_class`` (optional) routes the single shared-buffer
    total-Σ-heights-cap round (`_prove_total_cap`) with TRACED heights: the
    ``TotalCapClass``'s ``area_cap`` / ``window`` size the one flat jagged
    buffer, the shard's real heights ride as the traced ``num_reals`` vector,
    and each chip's trace must arrive column-major pre-padded to ``2*window``
    rows (zeros past its live height). The compile keys on the class + chip
    set alone, so two shards of one class share one executable.
    ``total_cap_class`` is required whenever any ``num_reals`` entry is
    traced — a buffer bound cannot derive from a runtime height; without it,
    all heights must be host ints and the total-cap round derives its class
    per shard (per-shard compile).

    Returns the final per-chip ``(num_cols_c, h)`` folded traces
    (``h in {0, 2}``; position 0 holds the column's evaluation at the
    sumcheck point), the advanced transcript, and the stacked ``RoundMsg``
    whose ``round_poly`` is ``(num_vars, DEGREE+1)`` in COEFFICIENT form."""
    eval_fns = summand.eval_fns
    alphas = summand.alphas
    lambdas = summand.lambdas
    public_values = summand.public_values
    if (flat_arrival is None) != (num_cols is None):
        raise ValueError("flat_arrival and num_cols must be given together")
    if flat_arrival is not None and total_cap_class is None:
        raise ValueError("flat_arrival is the total_cap_class arrival form")
    num_chips = len(num_cols) if num_cols is not None else len(traces)
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
    # A host-int height keeps the exact-rows contract; a traced entry (int32
    # scalar) switches the chip to runtime height. The split is a
    # compile-time property, so the zero-chip skips below stay static.
    static_nrs = [None if isinstance(nr, Array) else int(nr) for nr in num_reals]
    if total_cap_class is None and any(nr is None for nr in static_nrs):
        raise ValueError(
            "a traced num_reals entry requires total_cap_class: a buffer "
            "bound cannot derive from a runtime height"
        )
    for i, (trace, nr) in enumerate(zip(traces, static_nrs)):
        if trace.ndim != 2:
            raise ValueError(f"traces[{i}] must be 2-D, got shape {trace.shape}")
        if nr is None:
            continue  # runtime height: the shape is pinned to its cap below
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
    # ([1, nc]) trace as of fractalyze/xla#704; the same circuit then lowers to
    # ~785 instrs and Global cold-compiles in a few seconds.
    probe_cols = (
        list(num_cols) if num_cols is not None
        else [t.shape[0] for t in traces]
    )
    adjs = [
        zero
        if static_nrs[i] == 0 or alphas[i].shape[-1] == 0
        else constraint_eval(
            eval_fns[i],
            jnp.zeros((1, probe_cols[i]), dtype=ef),
            alphas[i],
            live_width=1,
            aux_operands=(public_values,),
        )[0]
        for i in range(num_chips)
    ]

    return _prove_total_cap(
        summand, traces, static_nrs, list(num_reals), total_cap_class,
        zeta, transcript, list(claims), adjs, ef, ef_limbs, num_vars,
        flat_arrival=flat_arrival, num_cols=num_cols,
    )
