# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1-aligned LogUp-GKR circuit construction.

Mirrors SP1's reference CUDA first-layer tracegen, pinned for diffing:
https://github.com/fractalyze/sp1/blob/e2c02f376/sp1-gpu/crates/sys/lib/logup_gkr/tracegen.cu
(``populateLastCircuitLayer``), and emits zorch's ``JaggedGkrLayer`` so the
transitions run on zorch's scheme-agnostic jagged fold.

The SP1-specific pieces live here: the interaction fingerprint over
rw-constraints ``VirtualPairCol`` decompositions, the per-interaction slot
schedule (``sp1_col_h``) with fold-neutral trailing padding, the power-of-two
interaction padding, and the fixed-depth transition schedule
(``sp1_next_row_counts``) driving zorch's jagged fold. Numerator slots stay
in the main trace's BF dtype; denominators are EF.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Mapping, Sequence

import frx
import frx.numpy as jnp
from frx import Array, lax
from rw_constraints import Chip, Interaction

from zorch.pcs.jagged.region import JaggedRegion
from zorch.logup_gkr.circuit import JaggedGkrLayer
from zorch.utils.bits import log2_ceil_usize


@dataclass(frozen=True)
class GkrChip:
    """A chip's LogUp view: its name and SP1-ordered interactions."""

    name: str
    interactions: tuple[Interaction, ...]


def _by_sp1_index(info) -> int:
    return info.sp1_index if info.sp1_index is not None else 0


def build_gkr_chips(
    chips: Mapping[str, Chip], chip_names: Sequence[str]
) -> tuple[GkrChip, ...]:
    """Read typed interactions off each chip's manifest info, in SP1 order.

    Per chip: sends sorted by ``sp1_index``, then receives sorted by
    ``sp1_index`` — SP1's interaction enumeration order. Zero-interaction
    chips stay in the list (multi-block AIRs put their GKR work on a sibling
    control chip); stub chips (no typed interactions because the rw manifest
    width disagrees with the trace) are skipped rather than emitting garbage.
    """
    gkr_chips: list[GkrChip] = []
    for name in chip_names:
        sends, recvs = chips[name].get_sends(), chips[name].get_receives()
        if any(i.interaction is None for i in (*sends, *recvs)):
            continue
        sends = sorted(sends, key=_by_sp1_index)
        recvs = sorted(recvs, key=_by_sp1_index)
        gkr_chips.append(
            GkrChip(
                name=name,
                interactions=tuple(i.interaction for i in (*sends, *recvs)),
            )
        )
    return tuple(gkr_chips)


def generate_interaction_vals_batch(
    interaction: Interaction,
    preprocessed_trace: Array | None,
    main_trace: Array,
    alpha: Array,
    betas: Array,
) -> tuple[Array, Array]:
    """``(multiplicity, fingerprint)`` per row; multiplicity negated for
    receives. Fingerprint = ``alpha + betas[0]*kind + sum betas[i+1]*value_i``.
    """
    mult = interaction.multiplicity.apply_batch(preprocessed_trace, main_trace)
    if not interaction.is_send:
        mult = -mult

    height = main_trace.shape[0]
    fingerprint = jnp.broadcast_to(alpha + betas[0] * interaction.kind, (height,))
    for i, val_col in enumerate(interaction.values):
        value = val_col.apply_batch(preprocessed_trace, main_trace)
        fingerprint = fingerprint + betas[i + 1] * value
    return mult, fingerprint


def sp1_col_h(real_h: int) -> int:
    """Per-interaction column height ``ceil(max(real_h, 8) / 4)``
    (``populateLastCircuitLayer``); the layer reserves ``2 * col_h`` slots."""
    return (max(real_h, 8) + 3) // 4


@partial(frx.jit, static_argnames=("h", "w"))
def _chip_view_jit(dense: Array, start, *, h: int, w: int) -> Array:
    # Static-arg key is just (h, w); start stays traced so cross-shard offset
    # variation doesn't blow the pjit cache.
    return lax.dynamic_slice(dense, (start,), (h * w,)).reshape(w, h).T


def _chip_view(region: JaggedRegion, idx: int) -> Array:
    """The ``[height, width]`` row-major view of chip ``idx`` in ``region``
    (the dense buffer packs column-major), as one compiled dispatch."""
    h = region.chip_heights[idx]
    w = region.chip_widths[idx]
    return _chip_view_jit(region.dense, region.chip_starts[idx], h=h, w=w)


def _chip_mult_fingerprint(
    chip: GkrChip,
    main_trace: Array,
    prep_trace: Array | None,
    alpha: Array,
    betas: Array,
) -> tuple[Array, Array]:
    """Every interaction's per-row ``(multiplicity, fingerprint)`` as two
    ``[num_inter, height]`` stacks, computed as batched field ops rather than a
    per-interaction Python loop.

    Every multiplicity and value column is an affine form over the trace
    (``VirtualPairCol``: ``const + Σ wᵢ·colᵢ``). Stacking all forms into one
    ``[forms, width]`` weight matrix turns the whole per-row eval into a single
    field matmul ``W @ traceᵀ``; the fingerprint's ``Σ betas[i+1]·valueᵢ`` is an
    unrolled fold over the chip's (small) widest fingerprint, value forms padded
    with a zero form so short interactions add a field zero. Field sums are
    exact, so the result is byte-identical to the per-interaction affine formula
    (``generate_interaction_vals_batch``) at a fraction of the dispatch count.
    Matmul + gather + element-wise field mul/add are the only ops -- a field
    ``segment_sum`` (scatter-add) does not lower on the GPU emitter.
    """
    interactions = chip.interactions
    num_inter = len(interactions)
    height, main_w = main_trace.shape
    bf_dtype = main_trace.dtype  # base field; betas + fingerprint carry the EF

    # Fold the preprocessed trace into the column space only when a form actually
    # reads a prep column. A chip can sit in the prep region without using it,
    # and its prep view may be a different height (recursion shards keep prep
    # above main, and it's trimmed per chip), so an unconditional concat would
    # shape-mismatch. A prep weight with no prep data contributes zero, matching
    # ``VirtualPairCol.apply_batch``.
    uses_prep = prep_trace is not None and any(
        is_prep
        for it in interactions
        for vpc in (it.multiplicity, *it.values)
        for (_col, is_prep, _w) in vpc.column_weights
    )
    total_w = main_w + (prep_trace.shape[1] if uses_prep else 0)

    # Static tables over canonical ints. Duplicate columns within a form are
    # summed into one weight (field add is exact, so this matches the per-term
    # sum). Prep weights index into the main|prep column concat below.
    weight_rows: list[list[int]] = []
    constants: list[int] = []
    mult_form: list[int] = []  # [num_inter] form row of each multiplicity
    mult_sign: list[int] = []  # [num_inter] +1 send / -1 receive
    kinds: list[int] = []  # [num_inter]

    def _add_form(vpc) -> int:
        row = [0] * total_w
        for col_idx, is_prep, weight in vpc.column_weights:
            if is_prep:
                if not uses_prep:
                    continue  # prep weight with no prep trace -> zero contribution
                row[main_w + col_idx] += weight
            else:
                row[col_idx] += weight
        weight_rows.append(row)
        constants.append(vpc.constant)
        return len(weight_rows) - 1

    for interaction in interactions:
        mult_form.append(_add_form(interaction.multiplicity))
        mult_sign.append(1 if interaction.is_send else -1)
        kinds.append(interaction.kind)

    # Value forms padded to the chip's widest fingerprint; unused slots point at
    # a zero form (weight row of zeros, constant 0 -> evaluates to 0), so the
    # betas fold below adds a field zero there instead of needing a scatter.
    max_values = max((len(it.values) for it in interactions), default=0)
    weight_rows.append([0] * total_w)
    constants.append(0)
    zero_form = len(weight_rows) - 1
    value_idx: list[list[int]] = [[zero_form] * max_values for _ in range(num_inter)]
    for k, interaction in enumerate(interactions):
        for i, val_col in enumerate(interaction.values):
            value_idx[k][i] = _add_form(val_col)

    # forms[f] = const[f] + Σ_w W[f, w] · trace_cat[:, w]. Without a used prep
    # column total_w == main_w, so this is just the main trace.
    trace_cat = (
        jnp.concatenate([main_trace, prep_trace], axis=1) if uses_prep else main_trace
    )
    w_all = jnp.asarray(weight_rows, dtype=bf_dtype)  # [forms, total_w]
    c_all = jnp.asarray(constants, dtype=bf_dtype)  # [forms]
    forms = c_all[:, None] + w_all @ trace_cat.T  # [forms, height], BF

    # Multiplicity per interaction; receives negate (× field -1 == negation).
    mult = forms[jnp.asarray(mult_form)] * jnp.asarray(mult_sign, dtype=bf_dtype)[:, None]

    # Fingerprint = alpha + betas[0]·kind + Σ_i betas[i+1]·valueᵢ, the value sum
    # unrolled over the (small) widest fingerprint. Padded slots gather the zero
    # form, so betas[v+1]·0 adds a field zero (identity) -- byte-identical to the
    # per-interaction sum, all element-wise field ops the GPU emitter lowers.
    base = (alpha + betas[0] * jnp.asarray(kinds, dtype=bf_dtype))[:, None]
    fingerprint = jnp.broadcast_to(base, (num_inter, height))
    value_idx_arr = jnp.asarray(value_idx)  # [num_inter, max_values]
    for v in range(max_values):
        fingerprint = fingerprint + betas[v + 1] * forms[value_idx_arr[:, v]]
    return mult, fingerprint


@partial(
    frx.jit,
    static_argnames=(
        "chip", "main_start", "main_width", "cap", "prep_start",
        "prep_width", "prep_height",
    ),
)
def _chip_first_layer_capped(
    chip: GkrChip,
    main_flat: Array,
    prep_flat: Array | None,
    live_height: Array,
    alpha: Array,
    betas: Array,
    *,
    main_start: int,
    main_width: int,
    cap: int,
    prep_start: int = 0,
    prep_width: int = 0,
    prep_height: int = 0,
) -> tuple[Array, Array, Array, Array]:
    """One chip's four first-layer planes at a class height bound: static
    slices of the flat arrival + the traced ``live_height``, so the compile
    keys on the chip and class constants alone. Slots below
    ``live_height // 2`` hold the real pairs; the rest force the
    fold-neutral ``(n=0, d=1)``, byte-identical to the tight-class build's
    ``jnp.pad`` values. One ``@jit`` per chip — fusing all chips hands XLA
    every build's intermediates at once and blows the wide-shard budget."""
    height = cap
    main_trace = (
        main_flat[main_start : main_start + main_width * cap]
        .reshape(main_width, cap)
        .T
    )
    prep_trace = None
    if prep_width:
        prep_trace = (
            prep_flat[prep_start : prep_start + prep_width * prep_height]
            .reshape(prep_width, prep_height)
            .T
        )
        # Rows in [live_height, cap) only feed masked slots, so trimming or
        # zero-extending prep to the class bound is byte-safe.
        if prep_height >= cap:
            prep_trace = prep_trace[:cap]
        else:
            prep_trace = jnp.pad(prep_trace, ((0, cap - prep_height), (0, 0)))
    mult, fingerprint = _chip_mult_fingerprint(
        chip, main_trace, prep_trace, alpha, betas
    )

    live = jnp.arange(height // 2) < live_height // 2
    zero = jnp.zeros((), dtype=mult.dtype)
    one = jnp.ones((), dtype=fingerprint.dtype)
    slot_count = 2 * sp1_col_h(height)
    pad = ((0, 0), (0, slot_count - height // 2))
    return (
        jnp.pad(jnp.where(live, mult[:, 0::2], zero), pad).reshape(-1),
        jnp.pad(jnp.where(live, mult[:, 1::2], zero), pad).reshape(-1),
        jnp.pad(
            jnp.where(live, fingerprint[:, 0::2], one), pad, constant_values=1
        ).reshape(-1),
        jnp.pad(
            jnp.where(live, fingerprint[:, 1::2], one), pad, constant_values=1
        ).reshape(-1),
    )


def region_statics(
    region: JaggedRegion | None,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """A region's ``(chip_names, widths, heights)`` as the static tuples the
    class-shaped consumers take — hashable jit keys, no region pytree."""
    if region is None:
        return (), (), ()
    return (
        tuple(region.chip_names),
        tuple(int(w) for w in region.chip_widths),
        tuple(int(h) for h in region.chip_heights),
    )


def generate_first_layer(
    chips: Sequence[GkrChip],
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    alpha: Array,
    betas: Array,
) -> JaggedGkrLayer:
    """Build the first GKR layer with SP1-aligned per-interaction storage.

    Each interaction reserves ``2 * sp1_col_h(real_h)`` paired slots; the
    first ``real_h // 2`` hold real ``(mult, fingerprint)`` pairs split by
    even/odd row index, the rest the fold-neutral ``(n=0, d=1)`` —
    byte-equivalent to SP1's ``paddingValues``. Interactions missing past the
    power-of-two total get a full padding slot of ``2 * sp1_col_h(0) == 4``.

    Real heights must be even: an odd height would leave the odd-row side one
    slot short of the even-row side (the SP1 reference never produces one).

    One build path: the class-shaped build at the shard's own tight class,
    where the layout equals the exact SP1 layout byte-for-byte.
    """
    cap_class = GkrCapClass.from_heights(
        [int(h) for h in main_region.chip_heights]
    )
    main_flat, prep_flat, heights = pack_gkr_arrival(
        main_region, prep_region, cap_class
    )
    chip_names, main_widths, _ = region_statics(main_region)
    prep_names, prep_widths, prep_heights = region_statics(prep_region)
    return generate_first_layer_capped(
        chips,
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


def sp1_next_row_counts(row_counts: tuple[int, ...]) -> tuple[int, ...]:
    """One step of SP1's transition schedule: ``ceil(rc / 4) * 2`` per segment.

    Mirrors ``JaggedMle::next_start_indices_and_column_heights``:
    https://github.com/fractalyze/sp1/blob/e2c02f376/sp1-gpu/crates/utils/src/jagged.rs
    Each step folds a segment to half its (evened) height and rounds the
    result up to even, saturating at 2 — a saturated segment keeps its real
    fraction in the 0-child and re-pads the 1-child with the fold-neutral
    (n=0, d=1) row every remaining step.

    SP1 bookkeeps in col_h units; like whir-zorch's byte-matched prover, we
    apply the same step to the materialized per-plane slot counts (2x col_h).
    Past saturation the two layouts diverge in how much neutral padding they
    materialize, which never changes the underlying dense MLE.
    """
    return tuple(((rc + 3) // 4) * 2 for rc in row_counts)


def sp1_schedules(
    row_counts: tuple[int, ...], num_row_variables: int
) -> list[tuple[int, ...]]:
    """The per-transition ``out_row_counts`` SP1's fixed-depth circuit folds
    through: ``sp1_next_row_counts`` applied ``1 .. num_row_variables - 1`` times
    to the first layer's ``row_counts``. Threading it into zorch's
    ``build_jagged_pyramid`` builds the pyramid as one fused scan instead of
    an eager per-transition dispatch loop (sp1-zorch#143); the fused build is
    byte-matched against that eager reference loop in the tests."""
    # Fail closed on an invalid depth: num_row_variables < 1 has no first layer
    # to fold, so an empty schedule would silently build a degenerate pyramid
    # instead of erroring.
    if num_row_variables < 1:
        raise ValueError(f"num_row_variables must be >= 1, got {num_row_variables}")
    schedules: list[tuple[int, ...]] = []
    counts = row_counts
    for _ in range(num_row_variables - 1):
        counts = sp1_next_row_counts(counts)
        schedules.append(counts)
    return schedules


@dataclass(frozen=True)
class GkrCapClass:
    """Per-chip height bounds shared by every shard of one chip set — the
    LogUp-GKR analogue of zerocheck's ``TotalCapClass``.

    ``chip_heights[i]`` bounds main chip ``i``'s real height (main
    ``chip_names`` order), each even so the even/odd slot split never
    straddles a pair. Every first-layer slot count, transition schedule, and
    arrival offset derives from these bounds, so shards that differ only in
    row counts share every stage executable; the real heights ride as one
    traced int32 vector.
    """

    chip_heights: tuple[int, ...]

    @classmethod
    def from_heights(cls, heights: Sequence[int]) -> "GkrCapClass":
        """The a-priori-tight class of one shard (per-shard-compile
        fallback); real heights are even, so this is the identity bound."""
        return cls(tuple(int(h) + int(h) % 2 for h in heights))

    @classmethod
    def union(cls, *classes: "GkrCapClass") -> "GkrCapClass":
        """The smallest class admitting every input class — per-chip max.
        Assembles the cross-shard class from per-shard ``from_heights``."""
        if not classes:
            raise ValueError("union needs at least one class")
        counts = {len(c.chip_heights) for c in classes}
        if len(counts) != 1:
            raise ValueError(f"classes disagree on the chip count: {counts}")
        return cls(tuple(max(hs) for hs in zip(*(c.chip_heights for c in classes))))

    def check_bounds(self, heights: Sequence[int]) -> None:
        """Reject a shard the class does not admit — loudly, at pack time
        (the traced body cannot gate a compile on a runtime value)."""
        if len(heights) != len(self.chip_heights):
            raise ValueError(
                f"class covers {len(self.chip_heights)} chips, shard has "
                f"{len(heights)}"
            )
        for i, (h, cap) in enumerate(zip(heights, self.chip_heights)):
            if h % 2 != 0:
                raise ValueError(
                    f"chip {i} has odd real height {h}; the even/odd row "
                    f"split needs an even height"
                )
            if h > cap:
                raise ValueError(
                    f"chip {i} height {h} exceeds its class bound {cap}"
                )

    def slot_counts(
        self, gkr_chips: Sequence[GkrChip], chip_names: Sequence[str]
    ) -> tuple[int, ...]:
        """The class first-layer ``row_counts``: per interaction
        ``2 * sp1_col_h(bound)``, then the power-of-two interaction padding —
        ``generate_first_layer``'s accounting at the class bounds."""
        name_to_idx = {name: i for i, name in enumerate(chip_names)}
        counts: list[int] = []
        for chip in gkr_chips:
            slot_count = 2 * sp1_col_h(self.chip_heights[name_to_idx[chip.name]])
            counts.extend([slot_count] * len(chip.interactions))
        total = len(counts)
        n_pad = (1 << log2_ceil_usize(total)) - total
        counts.extend([2 * sp1_col_h(0)] * n_pad)
        return tuple(counts)

    def schedules(
        self,
        gkr_chips: Sequence[GkrChip],
        chip_names: Sequence[str],
        num_row_variables: int,
    ) -> list[tuple[int, ...]]:
        """The class transition schedule — ``sp1_schedules`` over the class
        slot counts; pointwise dominates every admitted shard's exact
        schedule (``sp1_next_row_counts`` is monotone)."""
        return sp1_schedules(
            self.slot_counts(gkr_chips, chip_names), num_row_variables
        )


def _arrival_offsets(
    widths: Sequence[int], heights: Sequence[int]
) -> tuple[int, ...]:
    """Chip offsets into a flat chip-major column-major arrival:
    ``cumsum(width * height)``. ``pack_gkr_arrival`` and the class-shaped
    consumers MUST derive the same offsets or views read across chips."""
    offsets = [0]
    for w, h in zip(widths, heights):
        offsets.append(offsets[-1] + int(w) * int(h))
    return tuple(offsets)


@partial(frx.jit, static_argnames=("cap_class", "main_widths"))
def _pack_main_zone(
    dense: Array,
    starts: Array,
    heights: Array,
    *,
    cap_class: GkrCapClass,
    main_widths: tuple[int, ...],
) -> Array:
    """The main arrival as ONE fused program: per chip, a vmapped column
    slice reads each column at its traced start/stride and the live mask
    zeroes rows past the traced height. Eagerly this pack was ~130
    dispatches copying the trace ~4x (slice, pad, transpose, concat) and
    dominated the warm serve (~1.7 s of a ~2 s unseen-shard serve); fused
    it is output-sized reads + writes. ``dense`` arrives padded to the
    class constant, so the compile keys on (class, widths) alone."""
    parts: list[Array] = []
    for i, (w, cap) in enumerate(zip(main_widths, cap_class.chip_heights)):
        if w * cap == 0:
            continue
        h = heights[i]
        start = starts[i]
        block = frx.vmap(
            lambda c: lax.dynamic_slice(dense, (start + c * h,), (cap,))
        )(jnp.arange(w, dtype=jnp.int32))  # [w, cap]
        live = jnp.arange(cap, dtype=jnp.int32)[None, :] < h
        parts.append(
            jnp.where(live, block, jnp.zeros((), dense.dtype)).reshape(-1)
        )
    return jnp.concatenate(parts)


def pack_gkr_arrival(
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    cap_class: GkrCapClass,
) -> tuple[Array, Array | None, Array]:
    """Pack the regions into class-shaped flat arrivals at the eager stage
    prologue: main chips as column-major ``[width, class height]`` blocks
    (live rows + zeros) at ``_arrival_offsets`` via the fused
    ``_pack_main_zone``, prep at its shard-invariant keygen heights, plus
    the real main heights as the one traced int32 vector. No
    ``JaggedRegion`` crosses into a jit body — its per-shard host-tuple
    metadata would key every zone compile."""
    heights_host = [int(h) for h in main_region.chip_heights]
    cap_class.check_bounds(heights_host)
    main_widths = tuple(int(w) for w in main_region.chip_widths)

    # Pad the dense to the CLASS constant (arrival area + one class height
    # of slack for the last column's full-cap read) so the pack zone's
    # operand shape is class-derived — any admitted shard, including a
    # synthetic warmup shard, hits the same executable. Out-of-chip reads
    # the slices make below a live height land in masked rows only.
    dense = main_region.dense
    arrival_len = _arrival_offsets(main_widths, cap_class.chip_heights)[-1]
    # >= any admitted dense (live area + its stacking pad) + full-cap slack
    # for the last column's slice.
    dense_cap = (
        arrival_len
        + (1 << main_region.log_stacking_height)
        + max(cap_class.chip_heights, default=0)
    )
    if dense.shape[0] > dense_cap:
        raise ValueError(
            f"dense length {dense.shape[0]} exceeds the class bound "
            f"{dense_cap}; the class does not admit this shard"
        )
    dense = jnp.pad(dense, (0, dense_cap - dense.shape[0]))
    heights = jnp.asarray(heights_host, jnp.int32)
    main_flat = _pack_main_zone(
        dense,
        jnp.asarray([int(s) for s in main_region.chip_starts], jnp.int32),
        heights,
        cap_class=cap_class,
        main_widths=main_widths,
    )

    prep_flat = None
    if prep_region is not None:
        prep_parts = [
            _chip_view(prep_region, idx).T.reshape(-1)
            for idx in range(len(prep_region.chip_names))
        ]
        prep_flat = jnp.concatenate(prep_parts)

    return main_flat, prep_flat, heights


def generate_first_layer_capped(
    gkr_chips: Sequence[GkrChip],
    main_flat: Array,
    prep_flat: Array | None,
    heights: Array,
    alpha: Array,
    betas: Array,
    *,
    cap_class: GkrCapClass,
    chip_names: tuple[str, ...],
    main_widths: tuple[int, ...],
    prep_names: tuple[str, ...],
    prep_widths: tuple[int, ...],
    prep_heights: tuple[int, ...],
) -> JaggedGkrLayer:
    """``generate_first_layer`` on the class-shaped flat arrival: static
    slices at the class bounds + the traced ``heights`` vector, so the
    build — and everything downstream reading ``row_counts`` — compiles
    once per (chip set, class). Byte-identical on every admitted shard
    (``_chip_first_layer_capped``)."""
    name_to_idx = {name: i for i, name in enumerate(chip_names)}
    prep_name_to_idx = {name: i for i, name in enumerate(prep_names)}
    main_offsets = _arrival_offsets(main_widths, cap_class.chip_heights)
    prep_offsets = _arrival_offsets(prep_widths, prep_heights)

    total_interactions = sum(len(c.interactions) for c in gkr_chips)
    padded_interactions = 1 << log2_ceil_usize(total_interactions)

    bf_dtype = main_flat.dtype
    ef_dtype = alpha.dtype

    n0_parts: list[Array] = []
    n1_parts: list[Array] = []
    d0_parts: list[Array] = []
    d1_parts: list[Array] = []
    row_counts: list[int] = []

    for chip in gkr_chips:
        if not chip.interactions:
            continue
        idx = name_to_idx[chip.name]
        cap = cap_class.chip_heights[idx]

        has_prep = chip.name in prep_name_to_idx and prep_flat is not None
        p_idx = prep_name_to_idx[chip.name] if has_prep else 0
        chip_n0, chip_n1, chip_d0, chip_d1 = _chip_first_layer_capped(
            chip,
            main_flat,
            prep_flat if has_prep else None,
            heights[idx],
            alpha,
            betas,
            main_start=main_offsets[idx],
            main_width=main_widths[idx],
            cap=cap,
            prep_start=prep_offsets[p_idx] if has_prep else 0,
            prep_width=prep_widths[p_idx] if has_prep else 0,
            prep_height=prep_heights[p_idx] if has_prep else 0,
        )
        n0_parts.append(chip_n0)
        n1_parts.append(chip_n1)
        d0_parts.append(chip_d0)
        d1_parts.append(chip_d1)
        row_counts.extend([2 * sp1_col_h(cap)] * len(chip.interactions))

    pad_slot_count = 2 * sp1_col_h(0)
    n_pad = padded_interactions - total_interactions
    if n_pad > 0:
        pad_total = n_pad * pad_slot_count
        n0_parts.append(jnp.zeros(pad_total, dtype=bf_dtype))
        n1_parts.append(jnp.zeros(pad_total, dtype=bf_dtype))
        d0_parts.append(jnp.ones(pad_total, dtype=ef_dtype))
        d1_parts.append(jnp.ones(pad_total, dtype=ef_dtype))
        row_counts.extend([pad_slot_count] * n_pad)

    return JaggedGkrLayer(
        numerator_0=jnp.concatenate(n0_parts),
        numerator_1=jnp.concatenate(n1_parts),
        denominator_0=jnp.concatenate(d0_parts),
        denominator_1=jnp.concatenate(d1_parts),
        row_counts=tuple(row_counts),
    )
