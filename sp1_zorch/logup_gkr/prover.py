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

import frx
import frx.numpy as fnp
from frx import Array, lax
from rw_constraints import Chip
from zk_dtypes import efinfo

from zorch.pcs.jagged.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import (
    GkrCapClass,
    GkrChip,
    _arrival_offsets,
    capped_pyramid_widths,
    generate_first_layer_capped,
    interaction_chip_indices,
    pack_gkr_arrival,
    region_statics,
    repack_first_layer,
    sp1_next_row_counts,
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
    build_jagged_pyramid,
    extract_jagged_outputs,
    jagged_layer_transition,
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
    frx.tree_util.register_dataclass,
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
            return fnp.concatenate([self.main, self.preprocessed])
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
        return fnp.zeros((trace.shape[1],), dtype=rev_point.dtype)
    log_h = max((real_height - 1).bit_length(), 0)
    pad = (1 << log_h) - real_height
    if pad > 0:
        trace = fnp.pad(trace, ((0, pad), (0, 0)))
    mles = trace.T
    for i in range(log_h):
        mles = _bind_rows(mles, rev_point[i])
    one = fnp.ones((), dtype=rev_point.dtype)
    correction = fnp.prod(one - rev_point[log_h:])
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
    flat_parts: list[Array] = [fnp.array([len(evaluations)], bf_dtype)]
    for ev in evaluations:
        if ev.preprocessed is not None:
            flat_parts.append(fnp.array([ev.preprocessed.shape[0]], bf_dtype))
            flat_parts.append(
                lax.bitcast_convert_type(ev.preprocessed, bf_dtype).reshape(-1)
            )
        elif empty_prep_absorbs_zero:
            flat_parts.append(fnp.array([0], bf_dtype))
        flat_parts.append(fnp.array([ev.main.shape[0]], bf_dtype))
        flat_parts.append(lax.bitcast_convert_type(ev.main, bf_dtype).reshape(-1))
    return fnp.concatenate(flat_parts)


class ChipOpeningsRound(Round):
    """SP1's GKR chip-openings absorb schedule, single-sourced the same way
    as the preamble and the GKR head glue: the prover (``open_traces_capped``)
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


@partial(
    frx.jit,
    static_argnames=(
        "trace_dimension", "cap_class", "chip_names", "main_widths",
        "prep_names", "prep_widths", "prep_heights",
    ),
)
def open_traces_capped(
    main_flat: Array,
    prep_flat: Array | None,
    eval_point: Array,
    transcript: Transcript,
    *,
    trace_dimension: int,
    cap_class: GkrCapClass,
    chip_names: tuple[str, ...],
    main_widths: tuple[int, ...],
    prep_names: tuple[str, ...],
    prep_widths: tuple[int, ...],
    prep_heights: tuple[int, ...],
) -> tuple[Transcript, dict[str, ChipEvaluation]]:
    """Open every shard chip's trace at the final GKR point and absorb via
    ``ChipOpeningsRound``, on the class-shaped flat arrival — static slices
    at the class bounds, so the compile keys on the chip set + class alone.
    SP1 opens ALL shard chips; prep opens at its keygen height.
    Byte-identical at any admitted class: the arrival's zero rows fold into
    exactly the ``(1 - rev_point[k])`` factors ``_open_chip``'s
    zero-extension correction applies, and field mul is exact."""
    rev_point = eval_point[-trace_dimension:][::-1]
    main_offsets = _arrival_offsets(main_widths, cap_class.chip_heights)
    prep_offsets = _arrival_offsets(prep_widths, prep_heights)
    prep_name_to_idx = {name: i for i, name in enumerate(prep_names)}

    openings: dict[str, ChipEvaluation] = {}
    for idx, name in enumerate(chip_names):
        cap, width = cap_class.chip_heights[idx], main_widths[idx]
        start = main_offsets[idx]
        view = main_flat[start : start + width * cap].reshape(width, cap).T
        main_eval = _open_chip(view, rev_point, cap)
        prep_eval = None
        if name in prep_name_to_idx and prep_flat is not None:
            p_idx = prep_name_to_idx[name]
            p_h, p_w = prep_heights[p_idx], prep_widths[p_idx]
            p_start = prep_offsets[p_idx]
            p_view = prep_flat[p_start : p_start + p_w * p_h].reshape(p_w, p_h).T
            prep_eval = _open_chip(p_view, rev_point, p_h)
        openings[name] = ChipEvaluation(main=main_eval, preprocessed=prep_eval)

    _, transcript, _ = ChipOpeningsRound(openings, chip_names)(None, transcript)
    return transcript, openings


def extract_sp1_outputs(floor: JaggedGkrLayer) -> LogUpGkrOutput:
    """Output MLEs at SP1's fixed-depth floor.

    SP1's schedule saturates every interaction at two slots, one fold short
    of zorch's all-ones floor; its extractOutput kernel folds that last
    step inline. Slice the capacity to the saturated floor's exact
    ``2 * num_batches`` live rows, run the missing transition, interleave.
    All-2s saturation is the caller's obligation (counts are traced); the
    width gate rejects a capacity too small to hold it — the all-ones or
    mixed floors this contract does not cover.
    """
    floor_width = 2 * floor.num_batches
    if floor.width < floor_width:
        raise ValueError(
            f"extract_sp1_outputs expects the saturated all-2s floor "
            f"({floor_width} live rows); capacity width {floor.width} "
            f"cannot hold it"
        )
    floor = JaggedGkrLayer(
        numerator_0=floor.numerator_0[:floor_width],
        numerator_1=floor.numerator_1[:floor_width],
        denominator_0=floor.denominator_0[:floor_width],
        denominator_1=floor.denominator_1[:floor_width],
        row_counts=floor.row_counts,
    )
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
        witness = fnp.zeros((), dtype=bf_dtype)
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


def _prove_from_first_layer(
    first,
    class_counts: tuple[int, ...],
    heights: Array,
    seg_chip_idx: tuple[int, ...],
    slot_cap: int,
    transcript: Transcript,
    witness: Array,
    *,
    num_row_variables: int,
    open_fn,
) -> tuple[Transcript, LogupGkrProof]:
    """First-layer-onward prove: repack to the shard's tight traced counts
    inside the ``slot_cap`` capacity, fold the pyramid, bind the output,
    prove the layer chain, open via ``open_fn(eval_point, transcript)``.
    Pyramid buffers and round workspace follow ``slot_cap`` (the class
    total), not the sum of per-chip class maxima. The traced-geometry
    guards zorch cannot run host-side — the row-space fit and floor
    saturation — are discharged below against the class counts, which
    dominate every admitted shard.

    The chain MUST consume layers through the lazy ``layers.pop()``
    generator, not a materialized list — only then does ProveChain release
    each proved layer before building the next, keeping at most one
    big-witness layer live (the host-RAM half of zorch#362).

    The caps (fractalyze/xla#179) pin ONE operand shape per round phase, so
    the round kernels compile once per {row-cap class, 2^niv, dtype}; 2^niv
    is an SP1 protocol value and cannot be padded away.
    """
    num_segments = len(class_counts)
    if max(class_counts) > 1 << num_row_variables:
        raise ValueError(
            f"class slot count {max(class_counts)} exceeds the virtual "
            f"row space 2^{num_row_variables}"
        )
    # The recurrence is monotone and saturates any count >= 1 at 2, so a
    # class floor of all-2s pins every admitted shard's traced floor there.
    class_floor = class_counts
    for _ in range(num_row_variables - 1):
        class_floor = sp1_next_row_counts(class_floor)
    if any(rc != 2 for rc in class_floor):
        raise ValueError(
            f"class schedule does not saturate the floor in "
            f"{num_row_variables - 1} transitions: {class_floor}"
        )

    capacity = slot_cap + slot_cap % 2
    capped_first, out_counts = repack_first_layer(
        first,
        class_counts,
        heights,
        seg_chip_idx,
        capacity,
        num_row_variables - 1,
    )
    schedules = list(
        zip(
            out_counts,
            capped_pyramid_widths(
                slot_cap, num_segments, num_row_variables - 1
            ),
            strict=True,
        )
    )
    layers = build_jagged_pyramid(capped_first, schedules)
    output = extract_sp1_outputs(layers[-1])
    carry, transcript, _ = OutputBindRound(output)(None, transcript)

    caps = RoundWidthCaps(
        elements=_row_cap(capacity),
        eq_row=1 << num_row_variables,
        interaction=max(4, num_segments),
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

    transcript, chip_openings = open_fn(eval_point, transcript)
    proof = LogupGkrProof(
        witness=witness,
        circuit_output=output,
        round_proofs=round_proofs,
        eval_point=eval_point,
        chip_openings=chip_openings,
    )
    return transcript, proof


@partial(frx.jit, static_argnames=("num_betas",))
def _head_zone(
    transcript: Transcript, *, num_betas: int
) -> tuple[Transcript, Array, Array]:
    """``HeadChallengesRound`` as one compiled dispatch — eagerly its EF
    samples cost ~14 ms of warm host gaps between tiny permutes. Only the
    head fuses: swallowing the first-layer build too hands XLA every chip's
    intermediates at once and blows the wide-shard memory budget."""
    _, transcript, head = HeadChallengesRound(num_betas)(None, transcript)
    return transcript, head.alpha, head.betas


def prove_logup_gkr(
    gkr_chips: Sequence[GkrChip],
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    transcript: Transcript,
    *,
    num_betas: int,
    num_row_variables: int,
    cap_class: GkrCapClass | None = None,
    pow_bits: int = 0,
    witness: Array | None = None,
) -> tuple[Transcript, LogupGkrProof]:
    """Run the LogUp-GKR stage on a transcript positioned after the shard
    preamble — the single source for the stage (``LogupGkrStage``:
    host-side grind, then class-keyed inner zones).

    The one prove path is the shard-invariant class contract:
    class-shaped flat arrivals + one traced int32 heights
    vector, so every zone keys its compile on (chip set, class) — shards
    differing only in row counts share every executable. ``cap_class=None``
    derives the shard's own tight class (per-shard compile, same body,
    layout == the exact SP1 layout).

    Byte-identical across admitted classes: a wider class only adds
    fold-neutral (n=0, d=1) slots — fixed points of the layer fold, summand
    no-ops in the sumcheck (the virtual-mass correction subtracts exactly
    their eq weight), and zeros the open folds into its correction factors.
    """
    transcript, witness = resolve_witness_and_grind(
        transcript,
        pow_bits=pow_bits,
        witness=witness,
        bf_dtype=main_region.dense.dtype,
    )
    if cap_class is None:
        cap_class = GkrCapClass.from_heights(
            [int(h) for h in main_region.chip_heights]
        )
    main_flat, prep_flat, heights = pack_gkr_arrival(
        main_region, prep_region, cap_class
    )
    chip_names, main_widths, _ = region_statics(main_region)
    prep_names, prep_widths, prep_heights = region_statics(prep_region)

    cap_class.check_slot_cap(
        [int(h) for h in main_region.chip_heights], gkr_chips, chip_names
    )

    transcript, alpha, betas = _head_zone(transcript, num_betas=num_betas)
    first = generate_first_layer_capped(
        tuple(gkr_chips),
        main_flat,
        prep_flat,
        heights,
        alpha,
        betas,
        cap_class=cap_class,
        chip_names=chip_names,
        main_widths=main_widths,
        prep_names=prep_names,
        prep_widths=prep_widths,
        prep_heights=prep_heights,
    )
    return _prove_from_first_layer(
        first,
        cap_class.slot_counts(gkr_chips, chip_names),
        heights,
        interaction_chip_indices(tuple(gkr_chips), chip_names),
        cap_class.resolved_slot_cap(gkr_chips, chip_names),
        transcript,
        witness,
        num_row_variables=num_row_variables,
        open_fn=lambda eval_point, t: open_traces_capped(
            main_flat,
            prep_flat,
            eval_point,
            t,
            trace_dimension=num_row_variables + 1,
            cap_class=cap_class,
            chip_names=chip_names,
            main_widths=main_widths,
            prep_names=prep_names,
            prep_widths=prep_widths,
            prep_heights=prep_heights,
        ),
    )
