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

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
from frx import Array, lax
from rw_constraints import Chip
from zk_dtypes import efinfo
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    LogUpGkrOutput,
    build_jagged_pyramid,
    extract_jagged_outputs,
    jagged_layer_transition,
)
from zorch.logup_gkr.jagged_prover import (
    JaggedGkrLayerRound,
    RoundWidthCaps,
)
from zorch.round import ProverRound, prove_rounds
from zorch.stage import ProveResult, ProverStage
from zorch.transcript import Transcript

from sp1_zorch.logup_gkr.circuit import (
    GkrCapClass,
    GkrChip,
    _arrival_offsets,
    capped_pyramid_widths,
    generate_first_layer_capped,
    pack_gkr_arrival,
    region_statics,
    row_cap as _row_cap,
    sp1_next_row_counts,
    sp1_schedule_counts,
)
from sp1_zorch.logup_gkr.head import (
    EF_CHALLENGES,
    absorb_grind,
    absorb_long_message,
    bind_circuit_output,
    sample_head_challenges,
)
from sp1_zorch.types import (
    ChipEvaluation,
    GkrOutputClaim,
    LogupGkrProof,
    ShardClaim,
    ShardWitness,
)


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


# One fold chunk's arrival cells (rows x width). The open's live fold
# temporaries are one chunk's generations (~cells x the EF itemsize, 256 MiB
# here); the monolithic fold instead materializes a whole class arrival in
# the extension field at once -- 8 x slot_cap x 16 B, ~12 GiB on the
# 33-chip registry class, the largest single allocation of a staged prove.
# The budget must sit BELOW the per-chip cell counts it is meant to split:
# the registry 33-chip class tops out near 2^27 cells per chip (padded
# height <= 2^21, width <= 241), so a 2^27 budget planned every chip
# monolithic and left the arrival-sized arena intact. The chunked default
# is DISABLED: on GPU the lax.scan formulation materializes its sliced
# operands and pushes the open zone PAST the monolithic arena (measured
# 31.0 GiB device peak vs 24.3 GiB monolithic on the tight 33-chip class)
# before dying CUDA_ERROR_ILLEGAL_ADDRESS; the chunk plan needs a
# rework before it can be the default.
_OPEN_FOLD_CHUNK_CELLS = None


def _fold_chunk_rows(
    width: int, padded_height: int, chunk_cells: int | None
) -> int | None:
    """Rows per fold chunk: the largest power of two keeping one chunk
    within ``chunk_cells`` cells (>= 4 rows, so a chunk always folds), or
    ``None`` for the monolithic fold -- no budget, or the whole padded
    height fits one chunk (the single-chunk degenerate IS the monolithic
    fold)."""
    if chunk_cells is None:
        return None
    rows = max(chunk_cells // max(width, 1), 4)
    rows = 1 << (rows.bit_length() - 1)
    if rows >= padded_height:
        return None
    return rows


def _open_chip(
    trace: Array,
    rev_point: Array,
    real_height: int,
    *,
    chunk_cells: int | None = None,
) -> Array:
    """Every column's MLE eval at the (reversed) row point, with the
    zero-extension to ``2^len(rev_point)`` factored out as a scalar.

    A chip folds at its own log-height: every row index below ``2^d`` has
    zero bits at coordinates ``k >= d``, so the implicit zero rows above
    contribute the product of ``(1 - rev_point[k])`` there -- no
    full-height pad buffer (SP1 evaluates the same factorization).

    ``chunk_cells`` bounds the fold's live temporaries: the first bind
    promotes the base-field rows to the extension field, so the monolithic
    fold holds the whole trace in EF at once. Chunked, the first
    ``log2(chunk_rows)`` variables fold inside contiguous power-of-two row
    windows (a serial ``lax.scan``), then the remaining variables fold the
    per-chunk results. An LSB-first bind never pairs rows across an aligned
    power-of-two window, so each chunk-local generation is the monolithic
    fold's own restriction to its window; field ops are exact and
    elementwise, so the chunked fold is byte-identical, not approximate.
    """
    if real_height == 0:
        return fnp.zeros((trace.shape[1],), dtype=rev_point.dtype)
    log_h = max((real_height - 1).bit_length(), 0)
    pad = (1 << log_h) - real_height
    if pad > 0:
        trace = fnp.pad(trace, ((0, pad), (0, 0)))
    chunk_rows = _fold_chunk_rows(trace.shape[1], 1 << log_h, chunk_cells)
    if chunk_rows is None:
        mles = trace.T
        folded = 0
    else:
        folds = chunk_rows.bit_length() - 1

        def fold_chunk(carry: None, chunk: Array) -> tuple[None, Array]:
            mles = chunk.T
            for i in range(folds):
                mles = _bind_rows(mles, rev_point[i])
            return carry, mles[:, 0]

        _, reduced = lax.scan(
            fold_chunk, None, trace.reshape(-1, chunk_rows, trace.shape[1])
        )
        mles = reduced.T
        folded = folds
    for i in range(folded, log_h):
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


class ChipOpeningsRound:
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


@partial(frx.jit, static_argnames=("width", "cap", "chunk_cells"))
def _open_chip_zone(
    chip_flat: Array,
    rev_point: Array,
    *,
    width: int,
    cap: int,
    chunk_cells: int | None = None,
) -> Array:
    """ONE chip's capped open as its own executable. Per-chip zones keep the
    open's temp arena at one chip's fold generations (the EF promote of the
    largest chip, ~1 GiB at registry caps); a single whole-class zone instead
    holds every chip's promote in one module arena — 16 B x the class
    arrival, 12.2 GiB on the 33-chip registry class, which does not fit the
    32 GiB card next to the first-layer arena. The compile keys on
    ``(width, cap)`` alone, so quantized classes share entries across shards
    and blocks."""
    view = chip_flat.reshape(width, cap).T
    return _open_chip(view, rev_point, cap, chunk_cells=chunk_cells)


def open_traces_capped(
    main_flat: Array,
    prep_flat: Array | None,
    eval_point: Array,
    *,
    trace_dimension: int,
    cap_class: GkrCapClass,
    chip_names: tuple[str, ...],
    main_widths: tuple[int, ...],
    prep_names: tuple[str, ...],
    prep_widths: tuple[int, ...],
    prep_heights: tuple[int, ...],
    fold_chunk_cells: int | None = _OPEN_FOLD_CHUNK_CELLS,
) -> tuple[dict[str, ChipEvaluation], Array]:
    """Open every shard chip's trace at the final GKR point and build its
    absorb message, on the class-shaped flat arrival — static slices
    at the class bounds, so each per-chip zone keys on the chip set + class
    alone. SP1 opens ALL shard chips; prep opens at its keygen height.
    Byte-identical at any admitted class: the arrival's zero rows fold into
    exactly the ``(1 - rev_point[k])`` factors ``_open_chip``'s
    zero-extension correction applies, and field mul is exact.

    Not itself a jit zone: each chip folds in its own ``_open_chip_zone``
    executable (see its docstring for the memory contract), and the flat
    message assembles from the per-chip openings eagerly — small arrays,
    class-stable shapes.

    ``fold_chunk_cells`` bounds each chip zone's live fold temporaries
    (``_open_chip``'s chunked fold; ``None`` folds monolithically). The
    openings and the absorb message are byte-identical at any value — the
    knob moves memory, never the Fiat-Shamir stream."""
    rev_point = eval_point[-trace_dimension:][::-1]
    main_offsets = _arrival_offsets(main_widths, cap_class.chip_heights)
    prep_offsets = _arrival_offsets(prep_widths, prep_heights)
    prep_name_to_idx = {name: i for i, name in enumerate(prep_names)}

    openings: dict[str, ChipEvaluation] = {}
    for idx, name in enumerate(chip_names):
        cap, width = cap_class.chip_heights[idx], main_widths[idx]
        start = main_offsets[idx]
        chip_flat = main_flat[start : start + width * cap]
        main_eval = _open_chip_zone(
            chip_flat, rev_point, width=width, cap=cap, chunk_cells=fold_chunk_cells
        )
        prep_eval = None
        if name in prep_name_to_idx and prep_flat is not None:
            p_idx = prep_name_to_idx[name]
            p_h, p_w = prep_heights[p_idx], prep_widths[p_idx]
            p_start = prep_offsets[p_idx]
            prep_eval = _open_chip_zone(
                prep_flat[p_start : p_start + p_w * p_h],
                rev_point,
                width=p_w,
                cap=p_h,
                chunk_cells=fold_chunk_cells,
            )
        openings[name] = ChipEvaluation(main=main_eval, preprocessed=prep_eval)

    # The absorb itself stays outside the zones: on an interaction-heavy shard
    # this message is thousands of rate-blocks of serial sponge, which the
    # caller relocates to the host. `flat_openings_absorb` is still the one
    # definition of the message, shared with `ChipOpeningsRound` (the verifier
    # dual's path), so the two Fiat-Shamir streams cannot drift.
    flat = flat_openings_absorb(
        select_openings(openings, chip_names), empty_prep_absorbs_zero=False
    )
    return openings, flat


def absorb_chip_openings(
    transcript: Transcript, opened: tuple[dict[str, ChipEvaluation], Array]
) -> tuple[Transcript, dict[str, ChipEvaluation]]:
    """Absorb what ``open_traces_capped`` opened, outside its zone -- a traced
    region cannot reach the host sponge."""
    openings, flat = opened
    return absorb_long_message(transcript, flat), openings


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


@partial(frx.jit, static_argnames=("out_widths",))
def _pyramid_zone(
    first: JaggedGkrLayer, *, out_widths: tuple[int, ...]
) -> tuple[list[JaggedGkrLayer], tuple[Array, Array]]:
    """Fold the pyramid and extract SP1's floor outputs.

    Unrolls rather than scans: the per-level widths differ. Do not extend the
    zone up into the first-layer build -- that is a fan-in, not a chain, and
    blows the wide-shard memory budget (``_head_zone``).

    The bind stays outside: its absorbs may run on the host sponge, which a
    traced region cannot do.

    Outputs ride as a bare pair; ``LogUpGkrOutput`` is not a registered pytree.
    """
    schedules = list(
        zip(
            sp1_schedule_counts(first.row_counts, len(out_widths)),
            out_widths,
            strict=True,
        )
    )
    layers = build_jagged_pyramid(first, schedules)
    out = extract_sp1_outputs(layers[-1])
    return layers, (out.numerator, out.denominator)


def resolve_witness_and_grind(
    transcript: Transcript,
    *,
    pow_bits: int,
    pow_witness: Array | None,
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
    if pow_bits > 0 and pow_witness is None:
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
        _, pow_witness = transcript.grind(pow_bits)
    elif pow_witness is None:
        # pow_bits == 0 with no witness: a dummy zero just advances the stream.
        # A *passed* witness at pow_bits == 0 is a recorded-witness replay -- the
        # zero-bit GrindRound gate observes it (the transcript's `message`)
        # without host-reading the verdict, so the stage stays jit-traceable AND
        # the transcript matches the judged pow_bits > 0 path. Zeroing it here
        # would diverge that transcript, so keep the caller's witness.
        pow_witness = fnp.zeros((), dtype=bf_dtype)
    # The head schedule (grind, challenges, output binding) runs as the
    # shared glue Rounds -- the byte-match harness and the shard benchmark
    # thread the same definitions, so the three cannot drift.
    transcript = absorb_grind(transcript, pow_witness, pow_bits=pow_bits)
    return transcript, pow_witness


def _prove_from_first_layer(
    first: JaggedGkrLayer,
    class_counts: tuple[int, ...],
    slot_cap: int,
    transcript: Transcript,
    pow_witness: Array,
    *,
    num_row_variables: int,
    open_fn: Callable[
        [Array, Transcript], tuple[Transcript, dict[str, ChipEvaluation]]
    ],
) -> tuple[Transcript, LogupGkrProof]:
    """First-layer-onward prove over the tight-layout ``first`` layer: fold
    the pyramid, bind the output, prove the layer chain, open via
    ``open_fn(eval_point, transcript)``.
    Pyramid buffers and round workspace follow ``slot_cap`` (the class
    total), not the sum of per-chip class maxima. The traced-geometry
    guards zorch cannot run host-side — the row-space fit and floor
    saturation — are discharged below against the class counts, which
    dominate every admitted shard.

    The chain MUST consume layers through the lazy ``layers.pop()``
    generator, not a materialized list — only then does `prove_rounds` release
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
    # Class widths, so the zone's lay-in pad no-ops instead of copying planes.
    transition_widths = [
        _row_cap(w)
        for w in capped_pyramid_widths(slot_cap, num_segments, num_row_variables - 1)
    ]
    layers, (out_num, out_den) = _pyramid_zone(
        first, out_widths=tuple(transition_widths)
    )
    output = LogUpGkrOutput(numerator=out_num, denominator=out_den)
    transcript, carry = bind_circuit_output(transcript, output)

    layer_widths = [capacity, *transition_widths]
    layer_caps = [
        RoundWidthCaps(
            elements=_row_cap(w),
            eq_row=1 << num_row_variables,
            interaction=max(4, num_segments),
        )
        for w in layer_widths[: len(layers)]
    ]
    # Popped in lockstep with `layers`, floor first.
    (_, _, eval_point), transcript, round_proofs = prove_rounds(
        (
            JaggedGkrLayerRound(layers.pop(), EF_CHALLENGES, caps=layer_caps.pop())
            for _ in range(len(layers))
        ),
        carry,
        transcript,
    )

    transcript, chip_openings = open_fn(eval_point, transcript)
    proof = LogupGkrProof(
        pow_witness=pow_witness,
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
    transcript, head = sample_head_challenges(transcript, num_betas)
    return transcript, head.alpha, head.betas


def prove_logup_gkr(
    gkr_chips: Sequence[GkrChip],
    witness: ShardWitness,
    transcript: Transcript,
    *,
    num_betas: int,
    num_row_variables: int,
    cap_class: GkrCapClass | None = None,
    pow_bits: int = 0,
    pow_witness: Array | None = None,
) -> tuple[Transcript, LogupGkrProof]:
    """Run the LogUp-GKR stage on a transcript positioned after the shard
    preamble — the single source for the stage (``LogupGkrProver``:
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
    transcript, pow_witness = resolve_witness_and_grind(
        transcript,
        pow_bits=pow_bits,
        pow_witness=pow_witness,
        bf_dtype=witness.main_region.dense.dtype,
    )
    if cap_class is None:
        cap_class = GkrCapClass.from_heights(
            [int(h) for h in witness.main_region.chip_heights]
        )
    main_flat, prep_flat, heights = pack_gkr_arrival(
        witness.main_region, witness.prep_region, cap_class
    )
    chip_names, main_widths, _ = region_statics(witness.main_region)
    prep_names, prep_widths, prep_heights = region_statics(witness.prep_region)

    cap_class.check_slot_cap(
        [int(h) for h in witness.main_region.chip_heights], gkr_chips, chip_names
    )

    transcript, alpha, betas = _head_zone(transcript, num_betas=num_betas)
    slot_cap = cap_class.resolved_slot_cap(gkr_chips, chip_names)
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
        # Class width, so the zone's lay-in pad no-ops instead of copying.
        out_width=_row_cap(slot_cap + slot_cap % 2),
    )
    return _prove_from_first_layer(
        first,
        cap_class.slot_counts(gkr_chips, chip_names),
        cap_class.resolved_slot_cap(gkr_chips, chip_names),
        transcript,
        pow_witness,
        num_row_variables=num_row_variables,
        open_fn=lambda eval_point, t: absorb_chip_openings(
            t,
            open_traces_capped(
                main_flat,
                prep_flat,
                eval_point,
                trace_dimension=num_row_variables + 1,
                cap_class=cap_class,
                chip_names=chip_names,
                main_widths=main_widths,
                prep_names=prep_names,
                prep_widths=prep_widths,
                prep_heights=prep_heights,
            ),
        ),
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance -- driven by `prove_rounds`.
    _: type[ProverRound] = ChipOpeningsRound


class LogupGkrProver(
    ProverStage[ShardClaim, ShardWitness, GkrOutputClaim, LogupGkrProof]
):
    """LogUp-GKR stage over ``prove_logup_gkr``; writes the final
    evaluation point and per-chip openings as the claim zerocheck consumes.

    Eager orchestration, not one ``@jit`` body: a whole-body jit keeps
    every pyramid layer live at once (OOM on wide shards) and the grind's
    host-side ``pow_bits`` verdict cannot be traced. Every traced zone
    underneath keys its compile on (chip set, ``GkrCapClass``) — shards of
    one class share every executable; no pinned class means
    the shard's own tight class (per-shard compile, same body)."""

    def __init__(
        self,
        gkr_chips: Sequence[GkrChip],
        *,
        num_betas: int,
        num_row_variables: int,
        pow_bits: int = 0,
        pow_witness: Array | None = None,
        gkr_cap_class: GkrCapClass | None = None,
    ) -> None:
        self._gkr_chips = tuple(gkr_chips)
        self._num_betas = num_betas
        self._num_row_variables = num_row_variables
        self._pow_bits = pow_bits
        self._pow_witness = pow_witness
        self._gkr_cap_class = gkr_cap_class

    def prove(
        self,
        claim: ShardClaim,
        witness: ShardWitness,
        transcript: Transcript,
    ) -> ProveResult[GkrOutputClaim, LogupGkrProof]:
        # The verifier dual reads the row counts off the claim while the
        # prover has them in the witness's regions; a claim that disagrees
        # would otherwise only surface later, as a transcript divergence.
        assert claim.chip_metadata.by_chip() == {
            n: int(h)
            for n, h in zip(
                witness.main_region.chip_names,
                witness.main_region.chip_heights,
                strict=True,
            )
        }, "claim's chip metadata does not match the witness's main region"
        transcript, proof = prove_logup_gkr(
            self._gkr_chips,
            witness,
            transcript,
            num_betas=self._num_betas,
            num_row_variables=self._num_row_variables,
            pow_bits=self._pow_bits,
            pow_witness=self._pow_witness,
            cap_class=self._gkr_cap_class,
        )
        return ProveResult(
            GkrOutputClaim(proof.eval_point, proof.chip_openings),
            proof,
            transcript,
        )
