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

import jax
import jax.numpy as jnp
from jax import Array, lax
from rw_constraints import Chip, Interaction

from sp1_zorch.commit.region import JaggedRegion
from zorch.logup_gkr.circuit import JaggedGkrLayer, jagged_layer_transition
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


@partial(jax.jit, static_argnames=("h", "w"))
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


@partial(jax.jit, static_argnames=("chip",))
def _chip_first_layer_batched(
    chip: GkrChip,
    main_trace: Array,
    prep_trace: Array | None,
    alpha: Array,
    betas: Array,
) -> tuple[Array, Array, Array, Array]:
    """One chip's four first-layer planes, interaction-major, as batched field
    ops rather than a per-interaction Python loop.

    Every multiplicity and value column is an affine form over the trace
    (``VirtualPairCol``: ``const + Σ wᵢ·colᵢ``). Stacking all forms into one
    ``[forms, width]`` weight matrix turns the whole per-row eval into a single
    field matmul ``W @ traceᵀ``; the fingerprint's ``Σ betas[i+1]·valueᵢ`` is an
    unrolled fold over the chip's (small) widest fingerprint, value forms padded
    with a zero form so short interactions add a field zero. Field sums are
    exact, so the result is byte-identical to the per-interaction affine formula
    (``generate_interaction_vals_batch``) at a fraction of the dispatch count.
    Matmul + gather + element-wise field mul/add are the only ops -- a field
    ``segment_sum`` (scatter-add) does not lower on the GPU emitter. The even/odd
    slot split and fold-neutral padding match ``populateLastCircuitLayer``. The
    whole build rides one ``@jit`` boundary, so each chip is a single fused
    dispatch (``generate_first_layer`` runs eagerly, outside any prove jit).
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

    # Even/odd slot split + fold-neutral pad, flattened interaction-major. The
    # numerator pads with field 0 (Mont zero == jnp.pad's raw-zero default); the
    # denominator with the fold-neutral 1, matching the verifier's pad idiom.
    slot_count = 2 * sp1_col_h(height)
    pad_count = slot_count - height // 2
    pad = ((0, 0), (0, pad_count))
    return (
        jnp.pad(mult[:, 0::2], pad).reshape(-1),
        jnp.pad(mult[:, 1::2], pad).reshape(-1),
        jnp.pad(fingerprint[:, 0::2], pad, constant_values=1).reshape(-1),
        jnp.pad(fingerprint[:, 1::2], pad, constant_values=1).reshape(-1),
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
    """
    total_interactions = sum(len(c.interactions) for c in chips)
    padded_interactions = 1 << log2_ceil_usize(total_interactions)

    main_name_to_idx = {name: i for i, name in enumerate(main_region.chip_names)}
    prep_name_to_idx = (
        {name: i for i, name in enumerate(prep_region.chip_names)}
        if prep_region is not None
        else {}
    )

    bf_dtype = main_region.dense.dtype
    ef_dtype = alpha.dtype

    n0_parts: list[Array] = []
    n1_parts: list[Array] = []
    d0_parts: list[Array] = []
    d1_parts: list[Array] = []
    row_counts: list[int] = []

    for chip in chips:
        if not chip.interactions:
            continue
        main_trace = _chip_view(main_region, main_name_to_idx[chip.name])
        real_height = main_trace.shape[0]
        if real_height % 2 != 0:
            raise ValueError(
                f"chip {chip.name} has odd real height {real_height}; the "
                f"even/odd row split needs an even height"
            )
        if chip.name in prep_name_to_idx:
            # Trim prep to main's height for the per-row eval; recursion
            # shards keep prep at keygen height above main's.
            prep_trace = _chip_view(prep_region, prep_name_to_idx[chip.name])[
                :real_height
            ]
        else:
            prep_trace = None
        chip_n0, chip_n1, chip_d0, chip_d1 = _chip_first_layer_batched(
            chip, main_trace, prep_trace, alpha, betas
        )

        n0_parts.append(chip_n0)
        n1_parts.append(chip_n1)
        d0_parts.append(chip_d0)
        d1_parts.append(chip_d1)
        slot_count = 2 * sp1_col_h(real_height)
        row_counts.extend([slot_count] * len(chip.interactions))

    pad_slot_count = 2 * sp1_col_h(0)
    n_pad = padded_interactions - total_interactions
    if n_pad > 0:
        # One allocation per side instead of n_pad tiny ones.
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


def _sp1_schedules(
    row_counts: tuple[int, ...], num_row_variables: int
) -> list[tuple[int, ...]]:
    """The per-transition ``out_row_counts`` SP1's fixed-depth circuit folds
    through: ``sp1_next_row_counts`` applied ``1 .. num_row_variables - 1`` times
    to the first layer's ``row_counts``. This is exactly the schedule
    ``generate_circuit_layers`` walks, so threading it into zorch's
    ``scan_build_jagged_pyramid`` builds the same pyramid as one fused scan
    instead of the eager per-transition dispatch loop (sp1-zorch#143)."""
    # Fail closed on an invalid depth, matching generate_circuit_layers (the
    # eager path this replaces in the prover): num_row_variables < 1 has no
    # first layer to fold, so an empty schedule would silently build a
    # degenerate pyramid instead of erroring.
    if num_row_variables < 1:
        raise ValueError(f"num_row_variables must be >= 1, got {num_row_variables}")
    schedules: list[tuple[int, ...]] = []
    counts = row_counts
    for _ in range(num_row_variables - 1):
        counts = sp1_next_row_counts(counts)
        schedules.append(counts)
    return schedules


def generate_circuit_layers(
    first_layer: JaggedGkrLayer, num_row_variables: int
) -> list[JaggedGkrLayer]:
    """Every layer of the SP1-aligned circuit, first layer to the floor.

    SP1's circuit depth is fixed: ``num_row_variables - 1`` transitions
    (``max_log_row_count - 1`` row variables on a core shard) regardless of
    where individual segments saturate.

    A host-orchestrated Python loop, per the ``@jit`` convention
    (``fractalyze/zorch:docs/conventions.md``): the pyramid is heterogeneous
    step to step, so the fold/gather bodies inside ``jagged_layer_transition``
    are the fusion target, not this driver — and one fused pyramid graph
    would recompile per shard shape.
    """
    if num_row_variables < 1:
        raise ValueError(f"num_row_variables must be >= 1, got {num_row_variables}")
    layers = [first_layer]
    counts = first_layer.row_counts
    for _ in range(num_row_variables - 1):
        counts = sp1_next_row_counts(counts)
        layers.append(jagged_layer_transition(layers[-1], counts))
    return layers
