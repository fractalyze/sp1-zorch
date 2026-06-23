# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1-schedule jagged evaluation-proof sumcheck as a composable ``Round``.

``JaggedEvalRound`` is the stage-5 evaluation proof in zorch's IOP-``Round`` form,
so it sequences with the other stages —
``ProveChain([Commit(), LogUpGkr(), ZeroCheck(), JaggedEval()])``. It reproves
SP1's jagged PCS opening sumchecks byte-identically: the OUTER Hadamard sumcheck
``Σ_i D(i)·J̃(i)`` over the committed dense buffer (round polys + ``dense_eval``)
whose folded point feeds the INNER branching-program sumcheck reproving
``J̃(z_row, z_col, z_final)``. The stacked BaseFold open of ``D`` at ``z_final``
is the remaining half of stage 5.

SP1 folds **LSB-first** (even/odd pairing ``[0::2]``/``[1::2]``), round polys
travel in **coefficient** form ``[c0, c1, c2]``, and the proof point is the
challenge list reversed (insert-at-front). zorch's ``SumcheckRound`` / ``prove``
fold MSB-first over a fixed dense shape — they can't byte-match SP1's LSB-first
jagged schedule, so this ``Round`` runs its own loop over zorch's order-free leaf
blocks (``build_jagged_layout`` / ``bp_eval_core`` / ``eval_coeffs``), same as
``zerocheck/jagged.py``. (Consequently it does not emit the ``zorch.sumcheck``
composite — the dense-only SVO / register-resident codegen does not apply; the
jagged equivalent is separate GPU-codegen work.)

The inner challenges are sampled from the threaded transcript; ``z_col`` /
``z_trace`` arrive on the carry (fixed upstream — ``z_col`` at commitment,
``z_trace`` by the outer sumcheck).

References (same SP1 commit as ``zerocheck/jagged.py``):
- coefficient-form deg-2 round poly — ``process_univariate_polynomial``.
- LSB-first elimination — ``fix_last_variable_kernel`` (``dim-1-round``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from zk_dtypes import efinfo

from zorch.fusion import fused_region
from zorch.pcs.jagged.poly import (
    _TRANSITION_ROWS,
    _offset_bit_tensor,
    bp_eval_core,
    build_jagged_layout,
    build_prefix_sums,
    msb_first_bits,
    partial_eval,
)
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import eval_coeffs
from zorch.round import Round
from zorch.sumcheck.prover import (
    SUMCHECK_MARKER,
    SUMCHECK_MARKER_VERSION,
    _state_leaves,
)
from zorch.transcript import (
    DuplexState,
    DuplexTranscript,
    Transcript,
    sample_challenge,
)
from zorch.utils.bits import log2_ceil_usize

# Degree of the inner branching-program sumcheck round poly (eq-weighted BP
# summand: a product of the degree-1 eq factor and the degree-1 BP difference).
_INNER_DEGREE = 2

# `composite.attributes["poly_form"]` value routing the inner jagged-eval round
# body to its dedicated zkx emitter. Distinct from the LogUp-GKR jagged path's
# "coefficient" (same coefficient-form round poly + LSB fold, but a different
# summand: branching-program × column-eq, not the LogUp rational), so it needs
# its own `ParseRoundPolyForm` value + emitter (the BP DP is not an inlinable
# `zorch.sumcheck.combine` region — it carries loops, matmuls, and gathers).
_JAGGED_EVAL_POLY_FORM = "jagged_eval"


@dataclass(frozen=True)
class JaggedEvalInputs:
    """Carry into ``JaggedEvalRound``: the committed columns' jagged layout plus
    the points the upstream rounds fixed.

    ``col_heights`` is the per-unit-column height list and ``all_claims`` the
    matching ``(L,)`` per-column GKR openings (see ``assemble_columns``).
    ``dense`` is the combined committed dense buffer ``D`` (both rounds' raw
    packed columns concatenated, padded to ``2^n``) over which the outer
    Hadamard sumcheck runs; the outer point ``z_final`` it produces feeds the
    inner sumcheck, so it is no longer carried in."""

    col_heights: tuple[int, ...]
    all_claims: Array
    z_row: Array
    z_col: Array
    dense: Array


@dataclass(frozen=True)
class JaggedEvalMsg:
    """Proof message: the outer Hadamard sumcheck (initial column claim, its
    coefficient-form round polys, the folded point ``z_final``, and
    ``dense_eval = D(z_final)``) and the inner branching-program sumcheck
    transcript (coefficient-form round polys, the folded point, the reproved
    claim)."""

    outer_sumcheck_claim: Array
    outer_sumcheck_polys: Array
    outer_sumcheck_point: Array
    dense_eval: Array
    inner_sumcheck_polys: Array
    inner_point: Array
    inner_claimed_sum: Array


def assemble_columns(
    row_counts_rounds: Sequence[Sequence[int]],
    column_counts_rounds: Sequence[Sequence[int]],
    column_claims_rounds: Sequence[Array],
    *,
    dtype,
) -> tuple[list[int], Array]:
    """Flatten the per-round (row_counts, column_counts, real claims) into the
    per-unit-column height list and the full column-claim buffer.

    Each chip contributes ``column_count`` unit columns of height ``row_count``;
    the last two ``column_counts`` per round are SP1's stacking dummies, so the
    claim buffer appends ``cc[-2]+cc[-1]`` zero claims after each round's real
    ones (matching SP1's ``prove_trusted_evaluations`` layout)."""
    col_heights: list[int] = []
    claim_blocks: list[Array] = []
    for rcs, ccs, claims_r in zip(
        row_counts_rounds, column_counts_rounds, column_claims_rounds, strict=True
    ):
        for rc, cc in zip(rcs, ccs, strict=True):
            col_heights.extend([int(rc)] * int(cc))
        n_pad = int(ccs[-2]) + int(ccs[-1])
        claim_blocks.append(jnp.asarray(claims_r, dtype=dtype))
        if n_pad:
            claim_blocks.append(jnp.zeros((n_pad,), dtype=dtype))
    return col_heights, jnp.concatenate(claim_blocks, axis=0)


def sample_z_col(
    transcript: Transcript, num_columns: int, dtype
) -> tuple[Transcript, Array]:
    """One extension challenge per column variable — SP1 samples ``z_col`` as
    extension elements, not stacked base squeezes. One definition driven by
    the prover stage and its verifier dual."""
    limbs = efinfo(dtype).degree
    parts: list[Array] = []
    for _ in range(log2_ceil_usize(num_columns)):
        transcript, challenge = sample_challenge(transcript, dtype, limbs)
        parts.append(challenge)
    z_col = jnp.stack(parts) if parts else jnp.zeros((0,), dtype)
    return transcript, z_col


def merged_prefix_bits(col_heights: Sequence[int], num_bits: int, *, dtype) -> Array:
    """The ``(L, 2·num_bits)`` merged prefix-bit buffer ``bits(t_c) ‖
    bits(t_{c+1})`` — the branching-program input both the inner sumcheck and
    its verifier leaf check read."""
    prefix_int = build_prefix_sums(list(col_heights))
    bits = msb_first_bits(prefix_int, num_bits)
    return jnp.asarray(np.concatenate([bits[:-1], bits[1:]], axis=1), dtype=dtype)


def outer_sumcheck_claim(all_claims: Array, z_col: Array) -> Array:
    """``Σ_c eq(z_col, c)·claim[c]`` over the padded 2^⌈log L⌉ column hypercube."""
    dtype = z_col.dtype
    col_eq = expand_eq_to_hypercube(z_col, jnp.ones((), dtype))  # (2^n_c,)
    width = col_eq.shape[0]
    padded = jnp.concatenate(
        [all_claims, jnp.zeros((width - all_claims.shape[0],), dtype=dtype)], axis=0
    )
    return jnp.sum(col_eq * padded)


def build_outer_indicator(
    col_heights: Sequence[int],
    z_row: Array,
    z_col: Array,
    target_size: int,
    *,
    dtype,
) -> Array:
    """Materialize ``J̃(z_row, z_col, ·)`` over the dense area ``[0, target_size)``.

    ``partial_eval`` allocates ``2^n_d`` (``n_d`` = prefix-bit width, which can
    exceed the dense area's address width when the total area is itself a power
    of two), reading its scatter offsets from the canonical-limb tensor; every
    nonzero entry lands below ``total_area <= target_size``, so the tail is zero
    and slicing to ``target_size`` recovers the indicator over the dense area."""
    heights = list(col_heights)
    l_max = len(heights)
    _, cfg = build_jagged_layout(heights, l_max, z_row.shape[0], dtype)
    offsets = _offset_bit_tensor(heights, l_max, cfg)
    return partial_eval(offsets, z_row, z_col, cfg=cfg)[:target_size]


def outer_sumcheck(
    dense: Array,
    indicator: Array,
    claim: Array,
    transcript: Transcript,
) -> tuple[Array, Array, Array, Transcript]:
    """Outer Hadamard sumcheck ``Σ_i D(i)·J̃(i) = claim``, LSB-first.

    Returns ``(round_polys (n,3), z_final (n,), dense_eval, transcript)`` where
    ``n = log2(len(dense))``. Folds even/odd pairs (``[0::2]``/``[1::2]``) one
    variable per round, observing each coefficient-form degree-2 round poly
    ``[s(0), claim-2·s(0)-s(∞), s(∞)]`` and sampling the next challenge; the
    point is the challenge list reversed (SP1's insert-at-front). ``dense_eval``
    is ``D(z_final)`` — the indicator factor is reproved by the inner sumcheck,
    not folded into the eval. Mirrors ``inner_sumcheck``'s LSB-first idiom over a
    flat Hadamard product (no branching program)."""
    state_a = dense
    state_b = indicator
    n_rounds = (state_a.shape[0] - 1).bit_length()
    ef = claim.dtype
    ef_limbs = efinfo(ef).degree
    two = jnp.array(2, ef)

    cur = claim
    polys: list[Array] = []
    challenges: list[Array] = []
    for _ in range(n_rounds):
        p0a, p1a = state_a[0::2], state_a[1::2]
        p0b, p1b = state_b[0::2], state_b[1::2]
        s0 = jnp.sum(p0a * p0b)
        s_inf = jnp.sum((p1a - p0a) * (p1b - p0b))
        coef = jnp.stack([s0, cur - two * s0 - s_inf, s_inf])

        # SP1 binds each variable with one extension element (its
        # ``sample_ext_element``) — the shared ``sample_challenge`` rule.
        transcript = transcript.observe(coef)
        transcript, alpha = sample_challenge(transcript, ef, ef_limbs)
        state_a = p0a + alpha * (p1a - p0a)
        state_b = p0b + alpha * (p1b - p0b)
        cur = eval_coeffs(coef, alpha)
        polys.append(coef)
        challenges.append(alpha)

    dense_eval = state_a[0]
    z_final = jnp.stack(challenges)[::-1]
    return jnp.stack(polys), z_final, dense_eval, transcript


def _bp_all(
    buf: Array,
    z_row: Array,
    z_trace: Array,
    t_matrix: Array,
    num_bits: int,
    bp_num_vars: int,
) -> Array:
    """Branching program evaluated over all ``L`` delta columns of a merged
    prefix-bit buffer: ``vmap`` of ``bp_eval_core`` over ``bits(t_c) ‖
    bits(t_{c+1})``."""
    return jax.vmap(
        lambda left, right: bp_eval_core(
            z_row, z_trace, left, right, t_matrix, bp_num_vars
        )
    )(buf[:, :num_bits], buf[:, num_bits:])


def _run_inner_rounds(
    merged: Array,
    weights: Array,
    claimed_sum: Array,
    transcript: Transcript,
    *,
    z_row: Array,
    z_trace: Array,
    t_matrix: Array,
    num_bits: int,
    bp_num_vars: int,
    dtype,
) -> tuple[Array, Array, Transcript]:
    """The per-variable jagged-eval inner sumcheck loop, shared by the plain and
    marked paths so the ``zorch.sumcheck`` marker decomposes byte-identically.

    Returns ``(round_polys (n_vars,3), challenges (n_vars,) in round order,
    transcript)``; ``n_vars = 2·num_bits`` eliminated LSB-first (buffer column
    ``n_vars-1`` down to ``0``). The round loop is one ``lax.scan`` so the rounds
    trace to a single reused body — an unrolled Python loop builds an O(rounds)
    XLA graph whose compile dominates the cold jagged-eval stage, while the scan's
    trace is O(1) in the round count (carry shapes are fixed; only column
    ``round_idx`` changes). Byte-equal to the unrolled loop: same descending round
    order, same arithmetic, same observe/sample sequence threaded through the
    carry."""
    n_vars = 2 * num_bits
    one = jnp.ones((), dtype)
    two = jnp.array(2, dtype)
    ef_limbs = efinfo(dtype).degree

    def _round(
        carry: tuple[Array, Array, Array, Transcript], round_idx: Array
    ) -> tuple[tuple[Array, Array, Array, Transcript], tuple[Array, Array]]:
        buf, weights, claim, transcript = carry
        bits_i = merged[:, round_idx]
        eq0 = one - bits_i
        bp0 = _bp_all(
            buf.at[:, round_idx].set(0), z_row, z_trace, t_matrix, num_bits, bp_num_vars
        )
        bp1 = _bp_all(
            buf.at[:, round_idx].set(1), z_row, z_trace, t_matrix, num_bits, bp_num_vars
        )
        p0 = jnp.sum(weights * eq0 * bp0)
        p_inf = jnp.sum(weights * (bits_i - eq0) * (bp1 - bp0))
        coef = jnp.stack([p0, claim - two * p0 - p_inf, p_inf])

        # One extension element per variable, as in ``outer_sumcheck``.
        transcript = transcript.observe(coef)
        transcript, alpha = sample_challenge(transcript, dtype, ef_limbs)
        buf = buf.at[:, round_idx].set(alpha)
        weights = weights * (alpha * bits_i + (one - alpha) * eq0)
        claim = eval_coeffs(coef, alpha)
        return (buf, weights, claim, transcript), (coef, alpha)

    # SP1 absorbs the claimed J̃ value before the rounds (sp1-zorch#90). Observing
    # it HERE (not in the caller) keeps the marked composite's first sponge entry
    # at the (0, 0) block boundary the register-resident jagged-eval kernel
    # assumes — the kernel observes the same claim operand inside, so the marker
    # decomposes byte-identically.
    transcript = transcript.observe(claimed_sum)
    init = (merged, weights, claimed_sum, transcript)
    (_, _, _, transcript), (polys, challenges) = lax.scan(
        _round, init, jnp.arange(n_vars - 1, -1, -1)
    )
    return polys, challenges, transcript


def _inner_sumcheck_marked(
    merged: Array,
    weights: Array,
    claimed_sum: Array,
    transcript: DuplexTranscript,
    *,
    z_row: Array,
    z_trace: Array,
    t_matrix: Array,
    num_bits: int,
    bp_num_vars: int,
    dtype,
) -> tuple[Array, Array, Transcript]:
    """Wrap ``_run_inner_rounds`` in the ``zorch.sumcheck`` composite, Fiat-Shamir
    INSIDE, so the body is the *same* scan and the result is bit-identical to the
    plain path; an unrecognized marker decomposes straight to that scan.

    Operand ABI the zkx ``jagged_eval`` recognizer/emitter consumes by position
    (the BP DP summand is hardcoded in the emitter, so there is no
    ``zorch.sumcheck.combine`` region): ``[merged, weights, claim, z_row, z_trace,
    t_matrix][5 DuplexState leaves][auto-lifted poseidon2 round constants]``. The
    five sponge leaves thread the duplex state; the Fiat-Shamir permutation rides
    as the nested ``zorch.poseidon2`` marker inside ``sample_challenge``.
    ``fold_order="lsb"`` / ``poly_form="jagged_eval"`` declare the schedule.
    Results: ``[folded][5 leaves][round polys][challenges]`` — the leading folded
    slot (final reduced claim) matches the kernel's flat output."""
    perm, rate = transcript.permutation, transcript.rate
    leaves = _state_leaves(transcript.state)
    n_vars = 2 * num_bits
    t_matrix_shape = t_matrix.shape

    def body(*operands: Array, **_attrs: object) -> tuple[Array, ...]:
        bmerged, bweights, bclaim, bz_row, bz_trace, bt = operands[:6]
        # The kernel reads merged / t_matrix as flat buffers (the emitter's
        # operand ABI), so they ride flattened; restore the 2D shapes the scan
        # body indexes.
        bmerged = bmerged.reshape(-1, n_vars)
        bt = bt.reshape(t_matrix_shape)
        lv = operands[6 : 6 + len(leaves)]
        polys, challenges, t = _run_inner_rounds(
            bmerged,
            bweights,
            bclaim,
            DuplexTranscript(perm, rate, DuplexState(*lv)),
            z_row=bz_row,
            z_trace=bz_trace,
            t_matrix=bt,
            num_bits=num_bits,
            bp_num_vars=bp_num_vars,
            dtype=dtype,
        )
        leaves_out = _state_leaves(cast(DuplexTranscript, t).state)
        # The recognized kernel emits a leading folded slot (the final reduced
        # claim) in its result tuple; the host body matches it so the marker
        # decomposes byte-identically. Both compute it the same way.
        folded = eval_coeffs(polys[-1], challenges[-1])
        return (folded, *leaves_out, polys, challenges)

    out = fused_region(
        body,
        merged.reshape(-1),
        weights,
        claimed_sum,
        z_row,
        z_trace,
        t_matrix.reshape(-1),
        *leaves,
        name=SUMCHECK_MARKER,
        version=SUMCHECK_MARKER_VERSION,
        degree=_INNER_DEGREE,
        num_vars=n_vars,
        # The merged prefix-bit buffer is the single folding factor; z_row /
        # z_trace / t_matrix / weights ride as operands, not factors.
        num_factors=1,
        fold_order="lsb",
        poly_form=_JAGGED_EVAL_POLY_FORM,
    )
    _folded, *out_leaves, polys, challenges = out
    t = DuplexTranscript(perm, rate, DuplexState(*out_leaves))
    return polys, challenges, t


def inner_sumcheck(
    col_heights: Sequence[int],
    z_row: Array,
    z_col: Array,
    z_trace: Array,
    transcript: Transcript,
    *,
    dtype,
) -> tuple[Array, Array, Array, Transcript]:
    """Reprove ``J̃(z_row, z_col, z_trace)`` via the branching-program sumcheck.

    Returns ``(round_polys (n_vars,3), inner_point (n_vars,), claimed_sum,
    transcript)``. ``n_vars = 2·n_d`` over the merged prefix-bit buffer,
    eliminated LSB-first (buffer column ``n_vars-1`` down to ``0``); each round's
    coefficient-form poly is observed and the next challenge sampled.

    When the transcript carries a dedicated-fusion permutation (real poseidon2),
    the round loop is wrapped in a ``zorch.sumcheck`` composite a vendor codegens
    register-resident; otherwise it runs as the plain scan. The marker decomposes
    to the same scan, so both paths are byte-identical."""
    heights = list(col_heights)
    l_max = len(heights)
    _, cfg = build_jagged_layout(heights, l_max, z_row.shape[0], dtype)

    # num_bits = cfg.n_d holds prefix sums up to the total area; the merged
    # buffer carries bits(t_c) ‖ bits(t_{c+1}) -> n_vars = 2·n_d.
    num_bits = cfg.n_d
    # The BP indexes over n_d prefix bits, not z_trace's length — the layer loop
    # must cover all prefix bits or it drops the MSB (matches eval_jagged_mle's
    # num_vars = max(n_r, n_d)).
    bp_num_vars = max(cfg.n_r, cfg.n_d)
    t_matrix = jnp.asarray(_TRANSITION_ROWS, dtype=dtype)

    merged = merged_prefix_bits(heights, num_bits, dtype=dtype)

    col_eq = expand_eq_to_hypercube(z_col, jnp.ones((), dtype))
    weights = col_eq[:l_max]

    # claimed_sum = J̃(z_row, z_col, z_trace) = Σ_c eq(z_col,c)·bp_c. Computed via
    # jnp.sum (CPU EF reduce works) rather than eval_jagged_mle's trace-time
    # 1726-deep unroll, which compiles abysmally.
    claimed_sum = jnp.sum(
        weights * _bp_all(merged, z_row, z_trace, t_matrix, num_bits, bp_num_vars)
    )

    # The claimed J̃ value is absorbed inside _run_inner_rounds now (so the marked
    # composite enters the sponge at the (0, 0) boundary the register-resident
    # kernel observes from). Both the plain scan and the marked body absorb it
    # identically, and the verifier re-absorbs it the same way (sp1-zorch#90/#144).
    if isinstance(transcript, DuplexTranscript) and transcript.has_dedicated_fusion:
        polys, challenges, transcript = _inner_sumcheck_marked(
            merged,
            weights,
            claimed_sum,
            transcript,
            z_row=z_row,
            z_trace=z_trace,
            t_matrix=t_matrix,
            num_bits=num_bits,
            bp_num_vars=bp_num_vars,
            dtype=dtype,
        )
    else:
        polys, challenges, transcript = _run_inner_rounds(
            merged,
            weights,
            claimed_sum,
            transcript,
            z_row=z_row,
            z_trace=z_trace,
            t_matrix=t_matrix,
            num_bits=num_bits,
            bp_num_vars=bp_num_vars,
            dtype=dtype,
        )
    return polys, challenges[::-1], claimed_sum, transcript


class JaggedEvalRound(Round):
    """The jagged PCS evaluation-proof sumcheck as a composable IOP ``Round``.

    ``__call__`` maps ``(JaggedEvalInputs, transcript) -> (inputs, transcript,
    JaggedEvalMsg)`` so it sequences in ``ProveChain``. Runs the full sumcheck
    half: the outer Hadamard sumcheck ``Σ D·J̃`` over the committed dense buffer
    (round polys + ``dense_eval``), whose folded point ``z_final`` then feeds the
    inner branching-program sumcheck reproving ``J̃(z_row, z_col, z_final)``. See
    the module docstring for why both are bespoke loops, not ``SumcheckRound``s."""

    def __init__(self, *, dtype) -> None:
        self._dtype = dtype

    def __call__(
        self, carry: JaggedEvalInputs, transcript: Transcript
    ) -> tuple[JaggedEvalInputs, Transcript, JaggedEvalMsg]:
        claim = outer_sumcheck_claim(carry.all_claims, carry.z_col)
        indicator = build_outer_indicator(
            carry.col_heights,
            carry.z_row,
            carry.z_col,
            carry.dense.shape[0],
            dtype=self._dtype,
        )
        outer_polys, z_final, dense_eval, transcript = outer_sumcheck(
            carry.dense, indicator, claim, transcript
        )
        inner_polys, inner_point, inner_claimed_sum, transcript = inner_sumcheck(
            carry.col_heights,
            carry.z_row,
            carry.z_col,
            z_final,
            transcript,
            dtype=self._dtype,
        )
        msg = JaggedEvalMsg(
            outer_sumcheck_claim=claim,
            outer_sumcheck_polys=outer_polys,
            outer_sumcheck_point=z_final,
            dense_eval=dense_eval,
            inner_sumcheck_polys=inner_polys,
            inner_point=inner_point,
            inner_claimed_sum=inner_claimed_sum,
        )
        return carry, transcript, msg


__all__ = [
    "JaggedEvalInputs",
    "JaggedEvalMsg",
    "JaggedEvalRound",
    "assemble_columns",
    "merged_prefix_bits",
    "outer_sumcheck_claim",
    "build_outer_indicator",
    "outer_sumcheck",
    "inner_sumcheck",
    "sample_z_col",
]
