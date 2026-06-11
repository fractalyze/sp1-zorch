# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 LogUp-GKR verifier: the zorch-native dual of ``prove_logup_gkr``.

Mirrors SP1's reference verifier, pinned for diffing:
https://github.com/fractalyze/sp1/blob/e2c02f376/crates/hypercube/src/logup_gkr/verifier.rs
The Fiat-Shamir legs run the same glue Rounds the prover drives (grind via
the transcript's own ``check_witness``, head challenges, output bind, the
chip-openings absorb), the per-layer replay is zorch's jagged GKR verifier
chain, and the consumer-side leaf check closes the reduction: the first
layer's MLE evaluations recomputed from the opened column values must equal
the chain's reduced claim. Per-chip trace heights are statement inputs here
(the same source as the preamble's chip metadata); SP1 reads them off the
proof's opened-values degrees instead, which in our chain layering arrive
only with the zerocheck stage's message.

The output-layer acceptance leg closes the bus: ``sum(num/den)`` over the
circuit output must equal ``-digest``, where ``digest`` is SP1's
public-values interaction digest from ``eval_public_values``
(``sp1_zorch.logup_gkr.public_values``), folded under the head's
public-values challenge. With it, a from-scratch re-prove of an unbalanced
witness is rejected here — the shard-local soundness of the bus — alongside
the public-values constraint accumulator (well-formed public values fold to
zero). The leg needs the statement's public-values vector; a mechanics-only
caller (the layer-replay unit test) passes ``None`` to skip it.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from sp1_zorch.logup_gkr.circuit import GkrChip, generate_interaction_vals_batch
from sp1_zorch.logup_gkr.head import EF_LIMBS, HeadChallengesRound, OutputBindRound
from sp1_zorch.logup_gkr.prover import (
    ChipEvaluation,
    ChipOpeningsRound,
    LogupGkrProof,
)
from sp1_zorch.logup_gkr.public_values import eval_public_values
from zorch.logup_gkr.jagged_verifier import JaggedGkrLayerRound
from zorch.poly.geq import VirtualGeq
from zorch.poly.multilinear import eval_mle
from zorch.round import VerifyChain
from zorch.transcript import GrindingTranscript
from zorch.utils.bits import log2_ceil_usize


def virtual_padding_geq(threshold: Array | int, point: Array) -> Array:
    """Evaluate the ``index >= threshold`` indicator MLE at ``point``
    (MSB-first coordinates) — SP1's ``full_geq``, the mass of the virtual
    padding region above a chip's real rows.

    Runs zorch's ``VirtualGeq`` with unit coefficients through one
    partial-eval bind per coordinate; binding consumes the LSB variable, so
    the MSB-first point folds in reverse.
    """
    dtype = point.dtype
    geq = VirtualGeq(
        threshold=jnp.asarray(threshold, jnp.int32),
        geq_coefficient=jnp.ones((), dtype),
        eq_coefficient=jnp.zeros((), dtype),
    )
    # Indexed iteration: direct iteration over an extension-field array
    # dispatches `lax.sign` (the gotcha noted in zorch's `expand_eq`).
    for j in range(point.shape[0] - 1, -1, -1):
        geq = geq.fix_last_variable(point[j])
    return geq.eval_at(0)


def padding_geqs(
    heights: Iterable[int], trace_point: Array
) -> dict[int, Array]:
    """The ``index >= height`` indicator masses at ``trace_point``, one per
    distinct height — the padded-row corrections of the GKR leaf check and
    the zerocheck oracle check.

    The threshold domain is one variable wider than the trace point so a
    full-height chip (``height == 2^|trace_point|``) stays representable;
    the prepended zero pins the evaluation to the real half (SP1's
    ``point_extended``).
    """
    point_extended = jnp.pad(trace_point, (1, 0))
    return {h: virtual_padding_geq(h, point_extended) for h in set(heights)}


def _leaf_evaluations(
    gkr_chips: Sequence[GkrChip],
    openings: Mapping[str, ChipEvaluation],
    chip_heights: Mapping[str, int],
    alpha: Array,
    betas: Array,
    trace_point: Array,
) -> Array:
    """The first layer's per-interaction MLE values at ``trace_point`` as a
    ``[numerator, denominator]`` stack, recomputed from the opened column
    values — the verifier-side analogue of ``generate_first_layer``.

    Each opened value is the zero-extended column MLE at the point, and the
    interaction is affine in the columns, so evaluating it on the openings
    equals the point-eval of its per-row values — up to the virtual padding
    rows, whose fold-neutral ``(num 0, den 1)`` differs from the
    interaction's value on zero columns. The geq mass corrects exactly that:
    ``real - pad * geq`` zeroes the padding numerators and
    ``real + (1 - pad) * geq`` re-pins the padding denominators to one
    (SP1 verifier.rs, the trace-openings consistency block).
    """
    ef = alpha.dtype
    geq_by_height = padding_geqs(
        (chip_heights[c.name] for c in gkr_chips if c.interactions), trace_point
    )
    numerator_values: list[Array] = []
    denominator_values: list[Array] = []
    for chip in gkr_chips:
        if not chip.interactions:
            continue
        opening = openings[chip.name]
        # Row 0 is the opening, row 1 all-zero columns: one batched call
        # yields the real evaluation and the interaction's value on zero
        # columns (the padding row's worth) together.
        main = jnp.stack([opening.main, jnp.zeros_like(opening.main)])
        prep = (
            jnp.stack(
                [opening.preprocessed, jnp.zeros_like(opening.preprocessed)]
            )
            if opening.preprocessed is not None
            else None
        )
        geq = geq_by_height[chip_heights[chip.name]]
        for interaction in chip.interactions:
            nums, dens = generate_interaction_vals_batch(
                interaction, prep, main, alpha, betas
            )
            numerator_values.append(nums[0] - nums[1] * geq)
            denominator_values.append(dens[0] + (jnp.ones((), ef) - dens[1]) * geq)

    pad_count = (1 << log2_ceil_usize(len(numerator_values))) - len(numerator_values)
    numerator = jnp.pad(jnp.stack(numerator_values), (0, pad_count))
    denominator = jnp.pad(
        jnp.stack(denominator_values), (0, pad_count), constant_values=1
    )
    return jnp.stack([numerator, denominator])


def verify_logup_gkr(
    gkr_chips: Sequence[GkrChip],
    chip_names: Sequence[str],
    chip_heights: Mapping[str, int],
    proof: LogupGkrProof,
    transcript: GrindingTranscript,
    public_values: Array | None,
    *,
    num_betas: int,
    num_row_variables: int,
    pow_bits: int = 0,
) -> tuple[GrindingTranscript, Array, Array]:
    """Verify a LogUp-GKR stage proof on a transcript positioned after the
    shard preamble; returns ``(transcript, eval_point, ok)``.

    ``eval_point`` is the verifier's own derivation (the sampled challenges,
    not the wire copy); the caller threads it to the zerocheck dual. ``ok``
    is a traced scalar AND of every acceptance leg, so the whole dual can
    sit inside one ``@jit``. Malformed proof *structure* (wrong round or
    chip count, wrong output size) raises instead — shapes are host
    decisions, the same split SP1 makes between its shape errors and field
    checks.

    ``public_values`` is the shard's statement public-values vector; the
    output-layer leg checks ``sum(num/den)`` over the circuit output against
    ``-eval_public_values`` digest and folds the public-values constraint
    accumulator to zero. Pass ``None`` to skip the leg in a mechanics-only
    test — production callers (the shard dual) always provide it.
    """
    total_interactions = sum(len(c.interactions) for c in gkr_chips)
    num_interaction_variables = log2_ceil_usize(total_interactions)
    if len(proof.round_proofs) != num_row_variables:
        raise ValueError(
            f"need one layer proof per row variable ({num_row_variables}), "
            f"got {len(proof.round_proofs)}"
        )
    expected_output = 1 << (num_interaction_variables + 1)
    if proof.circuit_output.numerator.shape[0] != expected_output:
        raise ValueError(
            f"circuit output must have {expected_output} entries, got "
            f"{proof.circuit_output.numerator.shape[0]}"
        )
    # Grind gate. The prover's GrindRound judges host-side and raises; the
    # dual needs the verdict as a traced leg of ok, so it calls the same
    # one-definition predicate directly.
    transcript, ok_pow = transcript.check_witness(pow_bits, proof.witness)

    _, transcript, head = HeadChallengesRound(num_betas)(None, transcript)
    carry, transcript, _ = OutputBindRound(proof.circuit_output)(None, transcript)

    # SP1 rejects a zero output denominator before walking the layers: the
    # bus-balance fractions divide by it, and a zero would let an adversary
    # park unbalanced mass in an undefined term.
    denominator = proof.circuit_output.denominator
    ok_denominator = jnp.all(denominator != jnp.zeros((), denominator.dtype))

    chain = VerifyChain([JaggedGkrLayerRound(EF_LIMBS) for _ in proof.round_proofs])
    (num_eval, den_eval, eval_point), transcript, ok_layers = chain(
        carry, proof.round_proofs, transcript
    )
    # The wire carries the point for non-verifier consumers; pin the copy so
    # a stale serialization cannot drift past the dual. Shape-strict: a
    # broadcastable wrong-length copy must reject, not broadcast.
    ok_point = jnp.array_equal(eval_point, proof.eval_point)

    _, transcript, _ = ChipOpeningsRound(proof.chip_openings, chip_names)(
        None, transcript
    )

    leaf = _leaf_evaluations(
        gkr_chips,
        proof.chip_openings,
        chip_heights,
        head.alpha,
        head.betas,
        eval_point[num_interaction_variables:],
    )
    interaction_point = eval_point[:num_interaction_variables]
    # One batched eval contracts the interaction axis for the numerator and
    # denominator rows together — the eq hypercube expands once, not twice.
    expected = eval_mle(leaf, interaction_point, axis=1)
    ok_leaf = jnp.array_equal(jnp.stack([num_eval, den_eval]), expected)

    ok = ok_pow & ok_denominator & ok_layers & ok_point & ok_leaf

    if public_values is not None:
        # Output-layer bus balance: the circuit's cumulative sum cancels the
        # public-values interaction digest, and the public-values constraints
        # fold to zero. SP1 ``verify_logup_gkr``: ``output_cumulative_sum ==
        # -verify_public_values(...)`` and ``accumulator == 0``.
        accumulator, digest = eval_public_values(
            public_values, head.pv_challenge, head.alpha, head.betas
        )
        output = proof.circuit_output
        output_cumulative_sum = jnp.sum(output.numerator / output.denominator)
        ok_bus = output_cumulative_sum == -digest
        ok_accumulator = accumulator == jnp.zeros((), accumulator.dtype)
        ok = ok & ok_bus & ok_accumulator

    return transcript, eval_point, ok
