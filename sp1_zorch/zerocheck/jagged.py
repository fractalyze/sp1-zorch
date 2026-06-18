# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1-schedule jagged zerocheck: the height-aware multi-chip sumcheck round.

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
With GKR column batching (``beta`` + ``claims``) the summand gains
``sum_j beta**(j+1) * col_j``, turning each chip's proved sum from zero into
its GKR opening at ``zeta``; padded rows are unaffected (a zero row's column
term vanishes with it).

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
carries. The ``num_real``-bounded GPU fusion for equal-height chips stays in
``round.py``.

Round polys mirror SP1's ``sum_as_poly_in_last_variable``: the degree-4 poly
is pinned by evaluations at t in {0, 2, 4}, the claim identity
``p(1) = claim - p(0)``, and an implicit zero at ``b = (1-last)/(1-2*last)``
— the root of the bound eq factor ``(1-t)*(1-last) + t*last`` scaling every
term. Polys travel in COEFFICIENT form (lambda-RLC across chips) through the
transcript — SP1's encoding, intentionally different from the equal-height
path's evaluation-form messages.

Variable order: zorch's ``expand_eq_to_hypercube`` indexes the hypercube with
``zeta[n-1]`` at the LSB, so the even/odd row pairing fixes ``zeta[n-1-round]``
— SP1's back-to-front order (its ``jagged_point`` is the challenge list
reversed; the reversal is the consumer's).

References (pinned at the same SP1 commit as ``prover.rlc_coeffs``):
- round-poly trick — ``sum_as_poly_in_last_variable``,
  https://github.com/fractalyze/sp1/blob/640d8b80c/slop/crates/sumcheck/src/poly.rs
- column-height schedule — ``next_start_indices_and_column_heights``,
  https://github.com/fractalyze/sp1/blob/640d8b80c/sp1-gpu/crates/utils/src/jagged.rs
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array, lax
from zk_dtypes import efinfo

from zorch.constraint_eval import constraint_eval
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.geq import VirtualGeq
from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
    eval_coeffs,
)
from zorch.sumcheck.prover import RoundMsg
from zorch.transcript import Transcript, sample_challenge

from sp1_zorch.zerocheck.prover import gkr_powers

# SP1's zerocheck round-poly degree: constraint degree <= 3 plus the eq
# factor. The 4-point evaluation trick below is specific to this degree;
# the constraint-degree bound is the caller's contract (not probeable).
DEGREE = 4


def _challenge_limbs(dtype) -> int:
    """Transcript squeezes per challenge of ``dtype``: an extension field
    takes ``degree`` base squeezes reinterpreted (SP1's ``sample_ext_element``
    convention, fractalyze/sp1-zorch#88); the transcript's own field takes
    one."""
    try:
        return efinfo(dtype).degree
    except ValueError:
        return 1


def _zero_extend_cols(trace: Array, width: int) -> Array:
    """Zero-extend axis 1 to ``width``. The zero tail is load-bearing: it
    keeps the buffer's dead suffix exactly zero, so the GKR column term
    vanishes there and full-width sums match truncated sums byte-for-byte."""
    pad = width - trace.shape[1]
    if pad == 0:
        return trace
    return jnp.concatenate(
        [trace, jnp.zeros((trace.shape[0], pad), dtype=trace.dtype)], axis=1
    )


def _chip_round_poly(
    p0: Array,
    diff: Array,
    eq: Array,
    eval_fn: Callable[[Array], Array],
    alpha: Array,
    gkr_powers: Array | None,
    padded_row_adj: Array,
    claim: Array,
    last: Array,
    eq_adj: Array,
    nr_live: Array,
    vgeq: VirtualGeq,
    *,
    is_zero: bool,
    is_round0: Array,
) -> tuple[Array, Array, Array, Array]:
    """One chip's ``(y0, y1, y2, y4)`` round-poly evaluations — SP1's
    ``sum_as_poly_in_last_variable``: inner sums over the chip's fixed pair
    width with rows past the live bound masked to zero (byte-equal to SP1's
    truncated sums), the claim identity for t=1, and the virtual-geq
    padded-row correction per t-point. ``eq`` is the round's eq table sliced to
    this chip's pair width. ``nr_live`` is the chip's live height this round (a
    traced ``int32`` — the live-width bound and the eq gather index), and
    ``is_round0`` flags the first round, which drops the constraint term at t=0
    (the zerocheck statement: constraints vanish on real witness rows) — but
    never the ``gkr_powers`` column term, which does not vanish there.

    ``is_zero`` is STATIC (``ceil(nr/2)`` fixes any ``nr >= 1``, so only a chip
    that starts empty is ever empty): its width-0 buffer and zero adjustment
    collapse every t-point to zero, leaving the claim identity alone — without
    running ``constraint_eval`` on an empty trace."""
    ef = last.dtype
    one = jnp.ones((), ef)
    zero = jnp.zeros((), ef)
    two = jnp.array(2, ef)
    four = jnp.array(4, ef)

    if is_zero:
        y_0 = zero
        return y_0, claim - y_0, zero, zero

    # `constraint_eval` rejects an empty alpha — a lookup-only chip (e.g. SP1's
    # Byte) has no transition constraints to evaluate.
    has_constraints = alpha.shape[-1] > 0
    num_non_padded = (nr_live + 1) // 2

    def inner(rows: Array, *, mask_round0: bool) -> Array:
        rows_t = rows.T
        # Sum the constraint and GKR-column contributions as separate scalar
        # reductions (field add is associative, so this equals summing their
        # element-wise sum). Keeping the two vectors out of one element-wise add
        # before the reduction sidesteps a while-loop lowering miscompile when a
        # live-width-masked vector and the column matmul share a scan body.
        total = zero
        if has_constraints:
            c = constraint_eval(eval_fn, rows_t, alpha, live_width=num_non_padded)
            # Round 0 drops the constraint term at t=0; the kernel still emits
            # once and is masked, so the per-chip marker count stays flat across
            # rounds (one compiled kernel reused, never one per round).
            c = jnp.where(is_round0, zero, c) if mask_round0 else c
            total = total + jnp.sum(c * eq)
        if gkr_powers is not None:
            # Unmasked, but zero past the live prefix anyway: the buffer's dead
            # tail is exactly zero, and a zero row's column term vanishes with it.
            total = total + jnp.sum((rows_t @ gkr_powers) * eq)
        return total

    inner_0 = inner(p0, mask_round0=True)
    inner_2 = inner(p0 + two * diff, mask_round0=False)
    inner_4 = inner(p0 + four * diff, mask_round0=False)

    # A live chip keeps nr_live >= 1, so threshold_half is in [0, eq.shape[0]):
    # an in-bounds dynamic gather, the same element a static index would read.
    threshold_half = num_non_padded - 1
    msb_lagrange = eq_adj * eq[threshold_half]

    def correct(y: Array, t_val: Array) -> Array:
        # The bound eq factor (1-t)*(1-last) + t*last scales every term.
        eq_last = (one - t_val) * (one - last) + t_val * last
        vg = vgeq.fix_last_variable(t_val).eval_at(threshold_half)
        return eq_last * (y * eq_adj - padded_row_adj * vg * msb_lagrange)

    y_0 = correct(inner_0, zero)
    y_1 = claim - y_0
    y_2 = correct(inner_2, two)
    y_4 = correct(inner_4, four)
    return y_0, y_1, y_2, y_4


def prove_jagged_zerocheck(
    eval_fns: Sequence[Callable[[Array], Array]],
    traces: Sequence[Array],
    num_reals: Sequence[int],
    alphas: Sequence[Array],
    lambdas: Array,
    zeta: Array,
    transcript: Transcript,
    *,
    beta: Array | None = None,
    claims: Sequence[Array] | None = None,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """Run the SP1-schedule jagged multi-chip zerocheck sumcheck.

    ``traces[c]`` is chip ``c``'s column-major ``(num_cols_c, num_reals[c])``
    trace — exactly its real rows; the driver owns all padding (the zero-tail
    soundness contract is internal). ``alphas[c]`` is its constraint-RLC
    vector (``prover.rlc_coeffs``; empty for a lookup-only chip with no
    transition constraints) and ``lambdas[c]`` its cross-chip coefficient;
    ``zeta`` is the eq point, one round per coordinate.

    ``beta`` and ``claims`` switch on GKR column batching: chip ``c``'s
    summand gains ``sum_j beta**(j+1) * col_j`` and ``claims[c]`` — its GKR
    opening at ``zeta`` — seeds the ``p(1) = claim - p(0)`` identity. They
    come together (either alone breaks round 0's identity); omitted, the
    round proves the pure zero sum.

    Returns the final per-chip ``(num_cols_c, h)`` folded traces
    (``h in {0, 2}``; position 0 holds the column's evaluation at the
    sumcheck point), the advanced transcript, and the stacked ``RoundMsg``
    whose ``round_poly`` is ``(num_vars, DEGREE+1)`` in COEFFICIENT form."""
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
    if (beta is None) != (claims is None):
        raise ValueError(
            "beta and claims come together: the GKR column term reshapes every "
            "round poly and claims seed its p(1) = claim - p(0) identity"
        )
    if claims is not None and len(claims) != num_chips:
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
    inv_vand = compute_inv_vandermonde(DEGREE, ef)

    # C_alpha(0_row), the constant every padded row contributes — probed once
    # per chip; num_real == 0 and constraint-less chips never trace their
    # constraint formula. The zero-row constant is identical for any number of
    # rows, so the probe height is free: it is padded to a multi-row block and
    # bounded at live_width=1 (only row 0 is live, the rest mask to zero, and we
    # keep [0]) purely to engage the compact loop-form GPU emitter. That emitter
    # only fires on a multi-row trace — a size-1 leading dim is simplified away
    # and falls back to the monolithic CSE unroll, the koalabear Global compile
    # cliff (one 271k-instr kernel, >660s). With the multi-row block it lowers
    # to ~785 instrs and Global cold-compiles in ~5s. See fractalyze/zkx#702.
    probe_rows = 8
    adjs = [
        zero
        if nrs[i] == 0 or alphas[i].shape[-1] == 0
        else constraint_eval(
            eval_fns[i],
            jnp.zeros((probe_rows, traces[i].shape[0]), dtype=ef),
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
    bufs = [
        _zero_extend_cols(traces[i], widths[i]).astype(ef) for i in range(num_chips)
    ]
    # int32 threshold from the start: as a scan carry it must keep the same
    # leaf type fix_last_variable produces (it folds to a traced int32).
    vgeqs = [VirtualGeq(jnp.asarray(nr, jnp.int32), one, zero) for nr in nrs]
    if claims is None:
        # Pure zerocheck: sum eq * (C - C(0)*geq) = 0.
        chip_claims: list[Array] = [zero] * num_chips
        chip_gkr: list[Array | None] = [None] * num_chips
    else:
        chip_claims = list(claims)
        max_cols = max(t.shape[0] for t in traces)
        gkr_all = gkr_powers(beta, max_cols) if max_cols else jnp.zeros(0, ef)
        chip_gkr = [gkr_all[: t.shape[0]] for t in traces]
    eq_adj = one
    # Only a chip that starts empty is ever empty (see `_chip_round_poly`); the
    # rest carry their live height as a traced int32 round carry.
    is_zero_chip = [nr == 0 for nr in nrs]
    nrs_live = [jnp.asarray(nr, jnp.int32) for nr in nrs]

    # Round-varying scan inputs, all O(1) to build (no per-round constraint
    # work): `last` is zeta back-to-front (the round binds zeta[n-1-rnd]); the
    # degree-3 Lagrange basis pins p(3) over {0,1,2,4} plus the implicit zero at
    # b = (1-last)/(1-2*last), the root of the eq-last factor that scales every
    # term; `is_round0` flags the constraint skip at t=0.
    last_xs = zeta[::-1]
    b_xs = (one - last_xs) / (one - jnp.array(2, ef) * last_xs)
    domain_xs = jnp.concatenate(
        [jnp.broadcast_to(jnp.array([0, 1, 2, 4], ef), (num_vars, 4)), b_xs[:, None]],
        axis=1,
    )
    three = jnp.array(3, ef)
    basis_3_xs = jax.vmap(compute_lagrange_basis, in_axes=(None, 0))(three, domain_xs)
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
        half = eq_width // 2
        return jnp.concatenate(
            [buf.reshape(half, 2).sum(axis=1), jnp.zeros((half,), ef)]
        )

    def round_step(carry, xs):
        bufs, vgeqs, chip_claims, eq_adj, nrs_live, eq_buf, transcript = carry
        last, basis_3, is_round0 = xs

        polys = []
        p0s = []
        diffs = []
        for i in range(num_chips):
            p0 = bufs[i][:, 0::2]
            diff = bufs[i][:, 1::2] - p0
            p0s.append(p0)
            diffs.append(diff)
            y_0, y_1, y_2, y_4 = _chip_round_poly(
                p0,
                diff,
                eq_buf[: widths[i] // 2],
                eval_fns[i],
                alphas[i],
                chip_gkr[i],
                adjs[i],
                chip_claims[i],
                last,
                eq_adj,
                nrs_live[i],
                vgeqs[i],
                is_zero=is_zero_chip[i],
                is_round0=is_round0,
            )
            y_3 = jnp.dot(jnp.stack([y_0, y_1, y_2, y_4, zero]), basis_3)
            polys.append(jnp.dot(inv_vand, jnp.stack([y_0, y_1, y_2, y_3, y_4])))

        rlc = jnp.dot(lambdas, jnp.stack(polys))

        # SP1 binds each variable with one extension element (its
        # ``sample_ext_element``) — the shared ``sample_challenge`` rule.
        transcript = transcript.observe(rlc)
        transcript, alpha_r = sample_challenge(transcript, ef, ef_limbs)

        bufs = [
            _zero_extend_cols(p0s[i] + alpha_r * diffs[i], widths[i])
            for i in range(num_chips)
        ]
        vgeqs = [vg.fix_last_variable(alpha_r) for vg in vgeqs]
        nrs_live = [(nr + 1) // 2 for nr in nrs_live]
        chip_claims = [eval_coeffs(polys[i], alpha_r) for i in range(num_chips)]
        eq_adj = eq_adj * (alpha_r * last + (one - alpha_r) * (one - last))

        carry = (
            bufs,
            vgeqs,
            chip_claims,
            eq_adj,
            nrs_live,
            fold_eq(eq_buf),
            transcript,
        )
        return carry, RoundMsg(round_poly=rlc, challenge=alpha_r)

    init = (bufs, vgeqs, chip_claims, eq_adj, nrs_live, eq_buf, transcript)
    (bufs, *_, transcript), msgs = lax.scan(
        round_step, init, (last_xs, basis_3_xs, is_round0_xs)
    )

    # The first pair of each buffer is the whole fold result; the rest of the
    # fixed width is dead zeros.
    return [b[:, :2] for b in bufs], transcript, msgs
