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


# Per-chip first-layer @jit closure cache. Eagerly, ~10 ops per interaction x
# tens of interactions x tens of chips is thousands of dispatches per shard;
# one closure per chip collapses that to one dispatch per chip chunk. Keyed by
# the GkrChip itself (name + interactions, both frozen) so a same-named chip
# with a different interaction set can never hit a stale closure; the
# structural hash runs once per chip per build, not per @jit call.
_CHIP_FIRST_LAYER_CACHE: dict = {}

# XLA caps kernel buffer args at 1024, which whole-chip closures blow past on
# curve chips (~1126 interactions); chunks also gate a kernel-size CUDA race
# observed on KeccakPermuteControl at 256 — 128 byte-matches the SP1 reference
# on the large-Keccak shards.
_INTERACTION_CHUNK = 128


def _get_chip_first_layer_jit(chip: GkrChip, has_prep: bool, bf_dtype, ef_dtype):
    """A cached ``fn(main_trace, [prep_trace,] alpha, betas)`` returning the
    chip's four first-layer parts, chunked per ``_INTERACTION_CHUNK``."""
    key = (chip, has_prep, bf_dtype, ef_dtype)
    if key in _CHIP_FIRST_LAYER_CACHE:
        return _CHIP_FIRST_LAYER_CACHE[key]

    interactions = chip.interactions
    chunks = [
        interactions[i : i + _INTERACTION_CHUNK]
        for i in range(0, len(interactions), _INTERACTION_CHUNK)
    ]

    def _make_chunk_body(chunk_inters):
        def _body(main_trace, prep_trace, alpha, betas):
            real_height = main_trace.shape[0]
            slot_count = 2 * sp1_col_h(real_height)
            # One shared pad per chunk — the slot shortfall is per-chip, not
            # per-interaction (even heights enforced by the caller). A
            # zero-length pad concat folds away under jit, so the full-slot
            # case needs no separate branch.
            pad_count = slot_count - real_height // 2
            pad_n = jnp.zeros(pad_count, dtype=bf_dtype)
            pad_d = jnp.ones(pad_count, dtype=ef_dtype)
            n0_parts, n1_parts, d0_parts, d1_parts = [], [], [], []
            for interaction in chunk_inters:
                nums, dens = generate_interaction_vals_batch(
                    interaction, prep_trace, main_trace, alpha, betas
                )
                n0_parts.append(jnp.concatenate([nums[0::2], pad_n]))
                n1_parts.append(jnp.concatenate([nums[1::2], pad_n]))
                d0_parts.append(jnp.concatenate([dens[0::2], pad_d]))
                d1_parts.append(jnp.concatenate([dens[1::2], pad_d]))
            return (
                jnp.concatenate(n0_parts),
                jnp.concatenate(n1_parts),
                jnp.concatenate(d0_parts),
                jnp.concatenate(d1_parts),
            )

        if has_prep:
            return jax.jit(_body)

        @jax.jit
        def _no_prep(main_trace, alpha, betas):
            return _body(main_trace, None, alpha, betas)

        return _no_prep

    chunk_fns = [_make_chunk_body(c) for c in chunks]

    def fn(*args):
        outs = [cf(*args) for cf in chunk_fns]
        return (
            jnp.concatenate([o[0] for o in outs]),
            jnp.concatenate([o[1] for o in outs]),
            jnp.concatenate([o[2] for o in outs]),
            jnp.concatenate([o[3] for o in outs]),
        )

    _CHIP_FIRST_LAYER_CACHE[key] = fn
    return fn


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
        has_prep = chip.name in prep_name_to_idx

        fn = _get_chip_first_layer_jit(chip, has_prep, bf_dtype, ef_dtype)
        if has_prep:
            # Trim prep to main's height for the per-row eval; recursion
            # shards keep prep at keygen height above main's.
            prep_trace = _chip_view(prep_region, prep_name_to_idx[chip.name])[
                :real_height
            ]
            chip_n0, chip_n1, chip_d0, chip_d1 = fn(
                main_trace, prep_trace, alpha, betas
            )
        else:
            chip_n0, chip_n1, chip_d0, chip_d1 = fn(main_trace, alpha, betas)

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
