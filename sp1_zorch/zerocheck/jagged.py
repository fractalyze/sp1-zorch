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

Own per-round Python loop, NOT zorch's scan driver: per-chip widths change
every round (and differ across chips), which a fixed-shape scan carry cannot
hold, and the round-poly trick below is SP1's rather than the product summand
the ``zorch.sumcheck`` marker carries. The ``num_real``-bounded GPU fusion for
equal-height chips stays in ``round.py``.

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

import jax.numpy as jnp
from jax import Array

from zorch.constraint_eval import constraint_eval
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.geq import VirtualGeq
from zorch.poly.univariate import (
    compute_inv_vandermonde,
    compute_lagrange_basis,
    eval_coeffs,
)
from zorch.sumcheck.prover import RoundMsg
from zorch.transcript import Transcript

from sp1_zorch.zerocheck.prover import gkr_powers

# SP1's zerocheck round-poly degree: constraint degree <= 3 plus the eq
# factor. The 4-point evaluation trick below is specific to this degree;
# the constraint-degree bound is the caller's contract (not probeable).
DEGREE = 4


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


def _fit_width(v: Array, width: int) -> Array:
    """Truncate or zero-extend a 1-D vector to ``width``. The eq table starts
    wider than any chip's pair width and ends narrower; either way the live
    prefix is preserved and the appended tail multiplies masked zeros only."""
    if v.shape[0] >= width:
        return v[:width]
    return jnp.concatenate([v, jnp.zeros((width - v.shape[0],), dtype=v.dtype)])


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
    num_real: int,
    vgeq: VirtualGeq,
    *,
    skip_constraint_at_zero: bool,
) -> tuple[Array, Array, Array, Array]:
    """One chip's ``(y0, y1, y2, y4)`` round-poly evaluations — SP1's
    ``sum_as_poly_in_last_variable``: inner sums over the chip's fixed pair
    width with rows past the live bound masked to zero (byte-equal to SP1's
    truncated sums), the claim identity for t=1, and the virtual-geq
    padded-row correction per t-point. ``eq`` arrives pre-fitted to the same
    pair width. Round 0 skips the constraint term at t=0 (the zerocheck
    statement: constraints vanish on real witness rows) — but never the
    ``gkr_powers`` column term, which does not vanish there."""
    ef = last.dtype
    one = jnp.ones((), ef)
    zero = jnp.zeros((), ef)
    two = jnp.array(2, ef)
    four = jnp.array(4, ef)
    num_non_padded = (num_real + 1) // 2

    if num_real == 0:
        inner_0 = inner_2 = inner_4 = zero
    else:
        # `constraint_eval` rejects an empty alpha — a lookup-only chip
        # (e.g. SP1's Byte) has no transition constraints to evaluate.
        has_constraints = alpha.shape[-1] > 0

        def inner(rows: Array, *, skip_constraint: bool = False) -> Array:
            rows_t = rows.T
            skip = skip_constraint or not has_constraints
            v = (
                zero
                if skip
                else constraint_eval(
                    eval_fn, rows_t, alpha, live_width=num_non_padded
                )
            )
            if gkr_powers is not None:
                # Unmasked, but zero past the live prefix anyway: the buffer's
                # dead tail is exactly zero, and a zero row's column term
                # vanishes with it.
                v = v + rows_t @ gkr_powers
            return jnp.sum(v * eq)

        inner_0 = inner(p0, skip_constraint=skip_constraint_at_zero)
        inner_2 = inner(p0 + two * diff)
        inner_4 = inner(p0 + four * diff)

    # num_real == 0 puts the threshold pair index at -1; everything below
    # zeroes out through the guard (the inner sums are already zero).
    threshold_half = num_non_padded - 1
    msb_lagrange = eq_adj * (eq[threshold_half] if threshold_half >= 0 else zero)

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
    one = jnp.ones((), ef)
    zero = jnp.zeros((), ef)
    inv_vand = compute_inv_vandermonde(DEGREE, ef)

    # C_alpha(0_row), the constant every padded row contributes — probed once
    # per chip; num_real == 0 and constraint-less chips never trace their
    # constraint formula.
    adjs = [
        zero
        if nrs[i] == 0 or alphas[i].shape[-1] == 0
        else constraint_eval(
            eval_fns[i], jnp.zeros((1, traces[i].shape[0]), dtype=ef), alphas[i]
        )[0]
        for i in range(num_chips)
    ]

    # Fixed-width buffers: each chip's width is pinned once at its padded
    # round-0 height and never shrinks; folds re-extend into the same width
    # so every per-chip op (and its compiled kernel) is round-invariant.
    widths = [((nr + 3) // 4) * 4 for nr in nrs]
    bufs = [_zero_extend_cols(traces[i], widths[i]) for i in range(num_chips)]
    pair_max = max(w // 2 for w in widths)
    vgeqs = [VirtualGeq(nr, one, zero) for nr in nrs]
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

    round_polys = []
    challenges = []
    for rnd in range(num_vars):
        last = zeta[num_vars - 1 - rnd]
        rest_z = zeta[: num_vars - 1 - rnd]
        eq_rest = (
            expand_eq_to_hypercube(rest_z, one)
            if rest_z.shape[0] > 0
            else jnp.ones((1,), ef)
        )
        # One round-varying reshape, shared by all chips; the per-chip slice
        # of it below keeps round-invariant shapes.
        eq_pairs = _fit_width(eq_rest, pair_max)
        # Domain pinning p(3): evals at {0, 1, 2, 4} plus the implicit zero at
        # b — the root of the eq-last factor that scales every term.
        b = (one - last) / (one - jnp.array(2, ef) * last)
        domain = jnp.concatenate([jnp.array([0, 1, 2, 4], dtype=ef), b[None]])
        basis_3 = compute_lagrange_basis(jnp.array(3, ef), domain)

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
                eq_pairs[: widths[i] // 2],
                eval_fns[i],
                alphas[i],
                chip_gkr[i],
                adjs[i],
                chip_claims[i],
                last,
                eq_adj,
                nrs[i],
                vgeqs[i],
                skip_constraint_at_zero=(rnd == 0),
            )
            y_3 = jnp.dot(jnp.stack([y_0, y_1, y_2, y_4, zero]), basis_3)
            polys.append(jnp.dot(inv_vand, jnp.stack([y_0, y_1, y_2, y_3, y_4])))

        rlc = jnp.dot(lambdas, jnp.stack(polys))

        transcript, r = transcript.observe_and_sample(rlc, 1)
        alpha_r = r[0]

        bufs = [
            _zero_extend_cols(p0s[i] + alpha_r * diffs[i], widths[i])
            for i in range(num_chips)
        ]
        vgeqs = [vg.fix_last_variable(alpha_r) for vg in vgeqs]
        nrs = [(nr + 1) // 2 for nr in nrs]
        chip_claims = [eval_coeffs(polys[i], alpha_r) for i in range(num_chips)]
        eq_adj = eq_adj * (alpha_r * last + (one - alpha_r) * (one - last))

        round_polys.append(rlc)
        challenges.append(alpha_r)

    # The first pair of each buffer is the whole fold result; the rest of the
    # fixed width is dead zeros.
    return (
        [b[:, :2] for b in bufs],
        transcript,
        RoundMsg(round_poly=jnp.stack(round_polys), challenge=jnp.stack(challenges)),
    )
