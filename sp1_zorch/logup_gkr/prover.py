# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 LogUp-GKR prover: the layered prove on zorch's jagged GKR blocks.

Challenger trajectory matches SP1's reference prover, pinned for diffing:
https://github.com/fractalyze/sp1/blob/e2c02f376/sp1-gpu/crates/sys/lib/logup_gkr/round.cu
Grind, then per shard: sample alpha (EF) -> beta seeds -> one discarded
public-values challenge -> build the circuit -> observe the output MLEs with
their length prefixes -> sample z1 -> per layer (output to input): sample
lambda (EF), run the materialized sumcheck, observe the four pair openings,
sample r (EF). Every EF challenge is four base squeezes, zorch's
``sample_challenge`` with four limbs.

Grinding searches for the witness; proving from a reference dump replays the
recorded one. The search loop arrives with the end-to-end shard prover --
until then ``witness`` is required when ``pow_bits > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax.numpy as jnp
from jax import Array, lax
from rw_constraints import Chip
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import (
    GkrChip,
    _chip_view,
    generate_circuit_layers,
    generate_first_layer,
)
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    LogUpGkrOutput,
    extract_jagged_outputs,
    jagged_layer_transition,
)
from zorch.logup_gkr.jagged_prover import JaggedGkrLayerRound, JaggedLayerProof
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.round import ProveChain
from zorch.transcript import Transcript, sample_challenge
from zorch.utils.bits import log2_ceil_usize, log2_strict_usize

# An SP1 extension-field challenge is four base-field squeezes.
_EF_LIMBS = 4


@dataclass(frozen=True)
class ChipEvaluation:
    """One chip's trace openings at the final GKR point."""

    main: Array  # (width,) EF, one eval per main column
    preprocessed: Array | None  # (prep width,) EF, when the chip has prep


@dataclass(frozen=True)
class LogupGkrProof:
    """The LogUp-GKR stage's proof: grind witness, circuit output, one round
    proof per layer (output to input), the final evaluation point, and the
    per-chip trace openings at it."""

    witness: Array
    circuit_output: LogUpGkrOutput
    round_proofs: list[JaggedLayerProof]
    eval_point: Array
    chip_openings: dict[str, ChipEvaluation]


def num_beta_values(chips: Mapping[str, Chip]) -> int:
    """SP1's beta count: ``max(interaction tuple width) + 1`` over the shard.

    Must match the reference prover or the challenger diverges at the beta
    seeds; mirrors ``max_tuple_width + 1`` in SP1's shard prover.
    """
    widths = [
        info.tuple_width
        for chip in chips.values()
        for info in (*chip.get_sends(), *chip.get_receives())
    ]
    return max(widths, default=0) + 1


def _bind_rows(mles: Array, r: Array) -> Array:
    """Bind one row variable of a ``[width, rows]`` stack, LSB-first."""
    return mles[:, 0::2] + r * (mles[:, 1::2] - mles[:, 0::2])


def _open_chip(trace: Array, rev_point: Array, real_height: int) -> Array:
    """Every column's MLE eval at the (reversed) row point, with the
    zero-extension to ``2^len(rev_point)`` factored out as a scalar.

    A chip folds at its own log-height: every row index below ``2^d`` has
    zero bits at coordinates ``k >= d``, so the implicit zero rows above
    contribute the product of ``(1 - rev_point[k])`` there -- no
    full-height pad buffer (SP1 evaluates the same factorization).
    """
    if real_height == 0:
        return jnp.zeros((trace.shape[1],), dtype=rev_point.dtype)
    log_h = max((real_height - 1).bit_length(), 0)
    pad = (1 << log_h) - real_height
    if pad > 0:
        trace = jnp.pad(trace, ((0, pad), (0, 0)))
    mles = trace.T
    for i in range(log_h):
        mles = _bind_rows(mles, rev_point[i])
    one = jnp.ones((), dtype=rev_point.dtype)
    correction = jnp.prod(one - rev_point[log_h:])
    return mles[:, 0] * correction


def open_traces(
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    eval_point: Array,
    transcript: Transcript,
    *,
    trace_dimension: int,
) -> tuple[Transcript, dict[str, ChipEvaluation]]:
    """Open every shard chip's traces at the final GKR point and absorb them.

    SP1 opens ALL shard chips (not just the GKR ones), preprocessed before
    main per chip, each eval length-prefixed. The absorb is one flat array
    in that exact element order -- the sponge eats elements one at a time
    either way, and per-eval transcript calls would re-trace the absorb
    scan per chip. Preprocessed traces open at their keygen height.
    """
    bf_dtype = main_region.dense.dtype
    rev_point = eval_point[-trace_dimension:][::-1]
    prep_name_to_idx = (
        {name: i for i, name in enumerate(prep_region.chip_names)}
        if prep_region is not None
        else {}
    )

    openings: dict[str, ChipEvaluation] = {}
    flat_parts: list[Array] = [jnp.array([len(main_region.chip_names)], bf_dtype)]
    for idx, name in enumerate(main_region.chip_names):
        main_eval = _open_chip(
            _chip_view(main_region, idx), rev_point, main_region.chip_heights[idx]
        )
        prep_eval = None
        if name in prep_name_to_idx:
            prep_idx = prep_name_to_idx[name]
            prep_eval = _open_chip(
                _chip_view(prep_region, prep_idx),
                rev_point,
                prep_region.chip_heights[prep_idx],
            )
            flat_parts.append(jnp.array([prep_eval.shape[0]], bf_dtype))
            flat_parts.append(lax.bitcast_convert_type(prep_eval, bf_dtype).reshape(-1))
        flat_parts.append(jnp.array([main_eval.shape[0]], bf_dtype))
        flat_parts.append(lax.bitcast_convert_type(main_eval, bf_dtype).reshape(-1))
        openings[name] = ChipEvaluation(main=main_eval, preprocessed=prep_eval)

    transcript = transcript.observe(jnp.concatenate(flat_parts))
    return transcript, openings


def extract_sp1_outputs(floor: JaggedGkrLayer) -> LogUpGkrOutput:
    """Output MLEs at SP1's fixed-depth floor.

    SP1's schedule saturates every interaction at two slots, one fold short
    of zorch's all-ones floor; its extractOutput kernel folds that last step
    inline. Run the missing transition, then interleave.
    """
    if all(rc == 2 for rc in floor.row_counts):
        floor = jagged_layer_transition(floor, (1,) * floor.num_interactions)
    return extract_jagged_outputs(floor)


def prove_logup_gkr(
    gkr_chips: Sequence[GkrChip],
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    transcript: Transcript,
    *,
    num_betas: int,
    num_row_variables: int,
    pow_bits: int = 0,
    witness: Array | None = None,
) -> tuple[Transcript, LogupGkrProof]:
    """Run the LogUp-GKR stage on a transcript positioned after the shard
    preamble (vk, public values, main commitment, chip metadata).

    Returns the advanced transcript and the proof; the caller opens the
    traces at ``proof.eval_point``.
    """
    bf_dtype = main_region.dense.dtype

    if pow_bits > 0:
        if witness is None:
            raise ValueError("pow_bits > 0 needs a witness (grinding not built)")
    else:
        witness = jnp.zeros((), dtype=bf_dtype)
    transcript = transcript.observe(witness)
    transcript, pow_sample = transcript.sample(1)
    if int(pow_sample[0]) & ((1 << pow_bits) - 1):
        raise ValueError(f"witness fails the {pow_bits}-bit proof of work")

    transcript, alpha = sample_challenge(transcript, EF, _EF_LIMBS)
    seeds = []
    for _ in range(log2_ceil_usize(num_betas)):
        transcript, seed = sample_challenge(transcript, EF, _EF_LIMBS)
        seeds.append(seed)
    # SP1 samples one extra public-values challenge here and discards it.
    transcript, _ = sample_challenge(transcript, EF, _EF_LIMBS)
    one = jnp.ones((), dtype=EF)
    betas = one[None] if not seeds else expand_eq_to_hypercube(jnp.stack(seeds), one)

    # No standalone binding for the first layer -- it is the largest, and a
    # stray reference would pin it past its round's release below.
    layers = generate_circuit_layers(
        generate_first_layer(gkr_chips, main_region, prep_region, alpha, betas),
        num_row_variables,
    )
    output = extract_sp1_outputs(layers[-1])

    # SP1 serializes the output MLEs with length prefixes.
    transcript = transcript.observe(jnp.array(output.numerator.shape[0], bf_dtype))
    transcript = transcript.observe(output.numerator)
    transcript = transcript.observe(jnp.array(output.denominator.shape[0], bf_dtype))
    transcript = transcript.observe(output.denominator)

    coords = []
    for _ in range(log2_strict_usize(output.numerator.shape[0])):
        transcript, c = sample_challenge(transcript, EF, _EF_LIMBS)
        coords.append(c)
    z1 = jnp.stack(coords)
    num_eval = eval_mle(output.numerator, z1)
    den_eval = eval_mle(output.denominator, z1)

    # layers.pop() walks output to input; the lazily consumed chain builds
    # each round on demand and releases it once proved, so at most one layer
    # of the pyramid stays live -- the planes sum to gigabytes at shard scale.
    chain = ProveChain(
        JaggedGkrLayerRound(layers.pop(), _EF_LIMBS) for _ in range(len(layers))
    )
    (_, _, eval_point), transcript, round_proofs = chain(
        (num_eval, den_eval, z1), transcript
    )

    transcript, chip_openings = open_traces(
        main_region,
        prep_region,
        eval_point,
        transcript,
        trace_dimension=num_row_variables + 1,
    )
    proof = LogupGkrProof(
        witness=witness,
        circuit_output=output,
        round_proofs=round_proofs,
        eval_point=eval_point,
        chip_openings=chip_openings,
    )
    return transcript, proof
