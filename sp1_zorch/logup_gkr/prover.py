# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 LogUp-GKR prover: the layered prove on zorch's jagged GKR blocks.

Challenger trajectory matches SP1's reference prover, pinned for diffing:
https://github.com/fractalyze/sp1/blob/e2c02f376/sp1-gpu/crates/sys/lib/logup_gkr/round.cu
Grind, then per shard: sample alpha (EF) -> beta seeds -> one discarded
public-values challenge -> build the circuit -> observe the output MLEs with
their length prefixes -> sample z1 -> per layer (output to input): sample
lambda (EF), run the materialized sumcheck, observe the four pair openings,
sample r (EF). Every EF challenge is four base squeezes, zorch's
``sample_challenge`` with four limbs. The head legs (through z1) live as the
shared glue Rounds in ``sp1_zorch.logup_gkr.head``.

Grinding searches for the witness; proving from a reference dump replays the
recorded one. The search loop arrives with the end-to-end shard prover --
until then ``witness`` is required when ``pow_bits > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array, lax
from rw_constraints import Chip
from zk_dtypes import efinfo

from zorch.pcs.jagged.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import (
    GkrChip,
    _chip_view,
    generate_first_layer,
    sp1_schedules,
)
from sp1_zorch.logup_gkr.head import (
    EF_LIMBS,
    GrindRound,
    HeadChallengesRound,
    OutputBindRound,
)
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    LogUpGkrOutput,
    extract_jagged_outputs,
    jagged_layer_transition,
    build_jagged_pyramid,
)
from zorch.logup_gkr.jagged_prover import (
    JaggedGkrLayerRound,
    JaggedLayerProof,
    RoundWidthCaps,
)
from zorch.round import ProveChain, Round
from zorch.transcript import GrindingTranscript, Transcript


# Pytree: both evals are array leaves (preprocessed is None for prep-less
# chips), so a carry holding these openings stays an arrays-only pytree.
@partial(
    jax.tree_util.register_dataclass,
    data_fields=["main", "preprocessed"],
    meta_fields=[],
)
@dataclass(frozen=True)
class ChipEvaluation:
    """One chip's trace openings at the final GKR point."""

    main: Array  # (width,) EF, one eval per main column
    preprocessed: Array | None  # (prep width,) EF, when the chip has prep

    def all_evals(self) -> Array:
        """The ``[main | prep]`` evaluation vector — the column order of the
        beta-power batching shared by the GKR opening claims and the
        zerocheck column batch."""
        if self.preprocessed is not None:
            return jnp.concatenate([self.main, self.preprocessed])
        return self.main


@dataclass(frozen=True)
class LogupGkrProof:
    """The LogUp-GKR stage's proof: grind witness, circuit output, one round
    proof per layer (output to input), the final evaluation point, and the
    per-chip trace openings at it.

    Each layer's sumcheck point rides on its ``JaggedLayerProof.point``
    (zorch retains it at prove time); the shard wire serializes it per layer
    (``point_and_eval``).
    """

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


def select_openings(
    openings: Mapping[str, ChipEvaluation], chip_names: Sequence[str]
) -> list[ChipEvaluation]:
    """Order a per-chip openings mapping by the caller's statement chips,
    rejecting a mapping that does not cover them exactly. The guard lives
    with the absorb Rounds consuming the selection because the mapping is
    proof-controlled once a verifier dual drives them: a missing chip would
    KeyError anyway, but an extra one would ride along silently."""
    if set(openings) != set(chip_names):
        raise ValueError("openings must cover exactly the statement chips")
    return [openings[name] for name in chip_names]


def flat_openings_absorb(
    evaluations: Sequence[ChipEvaluation], *, empty_prep_absorbs_zero: bool
) -> Array:
    """SP1's length-prefixed openings absorb as one flat base-field array:
    the chip count, then per chip preprocessed before main, each eval
    length-prefixed. One flat absorb because the sponge eats elements one at
    a time either way, and per-eval transcript calls would re-trace the
    absorb scan per chip.

    A chip with no preprocessed eval absorbs a bare zero length when
    ``empty_prep_absorbs_zero`` (SP1's empty-Vec framing on the zerocheck
    opened values) and nothing at all otherwise (SP1's GKR chip-openings
    framing). The two wire schedules share everything else; keeping them in
    one builder is what stops them drifting apart.
    """
    bf_dtype = efinfo(evaluations[0].main.dtype).base_field_dtype
    flat_parts: list[Array] = [jnp.array([len(evaluations)], bf_dtype)]
    for ev in evaluations:
        if ev.preprocessed is not None:
            flat_parts.append(jnp.array([ev.preprocessed.shape[0]], bf_dtype))
            flat_parts.append(
                lax.bitcast_convert_type(ev.preprocessed, bf_dtype).reshape(-1)
            )
        elif empty_prep_absorbs_zero:
            flat_parts.append(jnp.array([0], bf_dtype))
        flat_parts.append(jnp.array([ev.main.shape[0]], bf_dtype))
        flat_parts.append(lax.bitcast_convert_type(ev.main, bf_dtype).reshape(-1))
    return jnp.concatenate(flat_parts)


class ChipOpeningsRound(Round):
    """SP1's GKR chip-openings absorb schedule, single-sourced the same way
    as the preamble and the GKR head glue: the prover (``open_traces``)
    drives it with the openings it just computed, the verifier dual with the
    proof's recorded ones, so the two Fiat-Shamir streams cannot drift.
    ``chip_names`` fixes the absorb order -- the caller's statement, never
    the mapping's own iteration order. The message is the openings, the
    values this round binds."""

    def __init__(
        self, openings: Mapping[str, ChipEvaluation], chip_names: Sequence[str]
    ) -> None:
        self._openings = openings
        self._chip_names = chip_names

    def __call__(
        self, carry: Any, transcript: Transcript
    ) -> tuple[Any, Transcript, Mapping[str, ChipEvaluation]]:
        flat = flat_openings_absorb(
            select_openings(self._openings, self._chip_names),
            empty_prep_absorbs_zero=False,
        )
        return carry, transcript.observe(flat), self._openings


@partial(jax.jit, static_argnames=("trace_dimension",))
def open_traces(
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    eval_point: Array,
    transcript: Transcript,
    *,
    trace_dimension: int,
) -> tuple[Transcript, dict[str, ChipEvaluation]]:
    """Open every shard chip's traces at the final GKR point and absorb them
    via ``ChipOpeningsRound``.

    SP1 opens ALL shard chips (not just the GKR ones). Preprocessed traces
    open at their keygen height.

    ``@jit`` so the per-chip ``_chip_view`` + ``_open_chip`` + Fiat-Shamir absorb
    fuse into one program: under the eager LogUp-GKR fold this loop would
    otherwise dispatch op-by-op per chip and dominate the warm GKR time. It runs
    after the fold, so it pins no layer -- jitting it costs no memory."""
    rev_point = eval_point[-trace_dimension:][::-1]
    prep_name_to_idx = (
        {name: i for i, name in enumerate(prep_region.chip_names)}
        if prep_region is not None
        else {}
    )

    openings: dict[str, ChipEvaluation] = {}
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
        openings[name] = ChipEvaluation(main=main_eval, preprocessed=prep_eval)

    _, transcript, _ = ChipOpeningsRound(openings, main_region.chip_names)(
        None, transcript
    )
    return transcript, openings


def extract_sp1_outputs(floor: JaggedGkrLayer) -> LogUpGkrOutput:
    """Output MLEs at SP1's fixed-depth floor.

    SP1's schedule saturates every interaction at two slots, one fold short
    of zorch's all-ones floor; its extractOutput kernel folds that last step
    inline. Run the missing transition, then interleave.
    """
    if all(rc == 2 for rc in floor.row_counts):
        floor = jagged_layer_transition(floor, (1,) * floor.num_batches)
    return extract_jagged_outputs(floor)


def resolve_witness_and_grind(
    transcript: GrindingTranscript,
    *,
    pow_bits: int,
    witness: Array | None,
    bf_dtype: Any,
) -> tuple[Transcript, Array]:
    """Apply the witness-default policy and run the grind, returning the
    post-grind transcript and the resolved witness.

    With no witness and ``pow_bits > 0`` this now **searches** for one (the grind
    the docstring at the top of this module names); a supplied witness is
    **replayed** unchanged, byte-identical to the reference-dump path.

    Split out from ``prove_logup_gkr`` because ``GrindRound``'s ``pow_bits > 0``
    PoW verdict is a host-side ``bool(ok)`` that cannot run inside a traced
    region, so the grind stays eager while the body's inner zones self-jit.
    """
    if pow_bits < 0:
        # Fail closed at the stage boundary: a negative bit count is nonsense,
        # and the branch below would otherwise treat it as the zero-bit replay.
        raise ValueError("pow_bits must be non-negative")
    if pow_bits > 0 and witness is None:
        # Grind for the witness -- the "search" half this function's name
        # promises, now built. zorch's windowed grinder enumerates canonical
        # witnesses 0, 1, 2, ... for the lowest whose ``check_witness`` gate has
        # ``pow_bits`` zero low bits, host-validating it (``GrindError`` if none
        # is found in range) -- the exact gate ``GrindRound`` re-judges below, so
        # the found witness passes. Transcripts are immutable: grinding on
        # ``transcript`` only READS it to find the witness; the ``GrindRound``
        # line then advances the ORIGINAL transcript with that witness, a stream
        # byte-identical to the recorded-witness replay path (and to the FRI
        # open-phase grind already built the same way in ``jagged/open.py``:
        # ``t.grind(pow_bits)``).
        _, witness = transcript.grind(pow_bits)
    elif witness is None:
        # pow_bits == 0 with no witness: a dummy zero just advances the stream.
        # A *passed* witness at pow_bits == 0 is a recorded-witness replay -- the
        # zero-bit GrindRound gate observes it (the transcript's `message`)
        # without host-reading the verdict, so the stage stays jit-traceable AND
        # the transcript matches the judged pow_bits > 0 path. Zeroing it here
        # would diverge that transcript, so keep the caller's witness.
        witness = jnp.zeros((), dtype=bf_dtype)
    # The head schedule (grind, challenges, output binding) runs as the
    # shared glue Rounds -- the byte-match harness and the phase benchmark
    # thread the same definitions, so the three cannot drift.
    _, transcript, _ = GrindRound(witness, pow_bits=pow_bits)(None, transcript)
    return transcript, witness


# Right-size the fixed-width round buffer (xla#179) to a shared compile class.
# The cap pins ONE round-buffer width so the FS-less round kernels compile once
# per class -- the per-layer lay-in pads each layer to it. It must span the FULL
# shard range: every shard the driver hands us, from a tiny shard 0 (or a CPU
# test fixture) up to an arbitrarily wide one, gets a cap that tracks its own
# floor, never a fixed machine constant that would over-allocate a small shard
# (a 4M floor OOMs a 32-row layout) or cap out a big one.
#
# The class is the smallest multiple of the stacking height (2^21) that holds the
# shard's even-padded round-0 layout. 2^21 is SP1's CORE_LOG_STACKING_HEIGHT --
# the granularity its jagged commit stacks the trace at -- so the classes line up
# with SP1's own sizing (shard17 ~3M -> 4M, shard18 ~16.9M -> 18M, both exact
# multiples) and over-allocation stays below one stacking height (~2M), avoiding
# the 2^25 = 32M power of two that doubled the EF plane buffers to ~2.1 GB and
# OOM'd the widest shard. Below one stacking height, snapping up would grossly
# over-allocate a tiny shard, so there the class is the next power of two instead
# (still a multiple of 4, as the boundary handoff's two stride-2 halvings need
# row % 4 == 0).
_LOG_STACKING_HEIGHT = 21
_STACKING_HEIGHT = 1 << _LOG_STACKING_HEIGHT


def _row_cap(floor_padded: int) -> int:
    """The shard's round-buffer class: the smallest multiple of the stacking
    height (2^21) holding its even-padded round-0 layout, or -- below one
    stacking height -- the next power of two. Shards in the same class share the
    round-kernel compile; a bigger shard lands in a higher class and proves at
    its own size, so there is no fixed ceiling to OOM against."""
    if floor_padded < _STACKING_HEIGHT:
        cap = 4
        while cap < floor_padded:
            cap <<= 1
        return cap
    return -(-floor_padded // _STACKING_HEIGHT) * _STACKING_HEIGHT


def prove_logup_gkr_body(
    gkr_chips: Sequence[GkrChip],
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    transcript: Transcript,
    witness: Array,
    *,
    num_betas: int,
    num_row_variables: int,
) -> tuple[Transcript, LogupGkrProof]:
    """The grind-free LogUp-GKR body: head challenges, circuit build, the rolled
    pyramid sumcheck, and the trace openings, on a post-grind transcript.

    Pure traceable array work; each heavy inner zone (first-layer build,
    per-transition pyramid build, whole-layer sumcheck) is ``@jit``-ed on its
    own. ``witness`` is threaded only onto the returned proof.
    """
    _, transcript, head = HeadChallengesRound(num_betas)(None, transcript)

    # Build the pyramid as one fused traced region over the transitions (zorch's
    # build_jagged_pyramid, natural-width per zorch#370) rather than the eager
    # per-transition dispatch loop -- it collapses the ~20 heterogeneous
    # transition layers into one traced region, O(1) in the depth (sp1-zorch#143).
    # build_jagged_pyramid reads the first layer's row_counts to derive SP1's fold
    # schedule, so build it first.
    first = generate_first_layer(
        gkr_chips, main_region, prep_region, head.alpha, head.betas
    )
    layers = build_jagged_pyramid(
        first, sp1_schedules(first.row_counts, num_row_variables)
    )
    output = extract_sp1_outputs(layers[-1])
    carry, transcript, _ = OutputBindRound(output)(None, transcript)

    # Prove the floor-outward layer chain as an unrolled ProveChain of per-layer
    # JaggedGkrLayerRound. zorch retired the device-FS rolled `prove_jagged_pyramid`
    # (Fiat-Shamir now runs on the host between kernel launches); the unrolled
    # chain is byte-identical and the production path. Each layer traces once per
    # shape. Pop the layers into their rounds through a lazy generator (floor
    # first via `layers.pop()` per yield, NOT a materialized `proved` list): only
    # then does ProveChain's lazy consume release each proved layer before
    # building the next, so at most one big-witness layer stays live. A resident
    # list would pin the whole pyramid and defeat that invariant -- the runtime
    # host-RAM half of zorch#362 (the builder is the other half). Byte-match is
    # gated by the SP1 reference (verify_gkr_prove) and a captured CPU golden.
    #
    # Fixed-width round buffers (fractalyze/xla#179): the caps pin ONE operand
    # shape per round phase across every round and layer, so the FS-less round
    # kernels compile once per {row-cap class (see `_row_cap`), 2^niv interaction
    # class, dtype} instead of once per width. 2^niv is an SP1 protocol value that
    # fixes the round count, so it cannot be padded away.
    floor_padded = sum(rc + rc % 2 for rc in first.row_counts)
    caps = RoundWidthCaps(
        elements=_row_cap(floor_padded),
        eq_row=1 << num_row_variables,
        interaction=max(4, len(first.row_counts)),
    )
    # Each layer proves through the whole-layer jit zone (one executable per
    # layer): the caps pre-lay in zorch's `_jagged_round_via_zone` keys the
    # compile per nrv class, so shards share every layer program and XLA fuses
    # the inter-round glue instead of the host dispatching per round.
    chain = ProveChain(
        JaggedGkrLayerRound(layers.pop(), EF_LIMBS, caps=caps)
        for _ in range(len(layers))
    )
    (_, _, eval_point), transcript, round_proofs = chain(carry, transcript)

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
    traces at ``proof.eval_point``. The single source for the stage --
    ``LogupGkrRound`` calls it directly (host-side grind, then the body).
    """
    transcript, witness = resolve_witness_and_grind(
        transcript,
        pow_bits=pow_bits,
        witness=witness,
        bf_dtype=main_region.dense.dtype,
    )
    return prove_logup_gkr_body(
        gkr_chips,
        main_region,
        prep_region,
        transcript,
        witness,
        num_betas=num_betas,
        num_row_variables=num_row_variables,
    )
