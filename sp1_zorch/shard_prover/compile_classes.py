# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shard compile-class derivation and pin resolution.

The prove chain's heavy zones compile keyed on ``(chip set, class)`` — the
shard's runtime heights ride as traced values — so every shard of one class
shares one executable. This module is the single definition of that class
math and of the manifest/flag pin resolution, shared by every consumer (the
staged harness ``//tools:staged_prove_shard`` and the ``warm_shard_cache``
cache filler): the classes a warm fills are the classes a prove requests by
construction, not by two mirrored code paths staying in sync.

Class shapes on the wire:

- zerocheck pin spec (``--zc_class_json``): ``{"area_cap": N}``.
- GKR pin spec (``--gkr_class_json``): ``{"chip_heights": {name: bound}}``
  plus an optional ``"slot_cap"``.
- group-manifest entry (one shard's value in ``--group_manifest_json``):
  ``{"area_cap": N, "gkr": {name: bound}, "gkr_slot_cap": M}`` — every field
  optional, resolved field-by-field (:func:`resolve_classes`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sp1_zorch.logup_gkr.circuit import GkrCapClass
from sp1_zorch.zerocheck.jagged import TotalCapClass


def tight_classes(
    main_region: Any,
    prep_region: Any,
    order: Sequence[str],
    num_reals: Sequence[int],
    gkr_chips: Sequence[Any],
) -> tuple[TotalCapClass, GkrCapClass, int]:
    """The shard's a-priori-tight cap classes.

    Zerocheck: the prep-width join (by chip NAME) → per-chip total columns →
    :meth:`TotalCapClass.from_heights` over the real row counts. LogUp-GKR:
    :meth:`GkrCapClass.from_heights` over the REGION heights (what the stage
    packs — they agree with ``num_reals`` on real rows, but the pack's bound
    check runs on the region), plus its resolved first-layer slot cap.

    Returns ``(tc, gkr, slot_cap)``.
    """
    prep_widths = (
        {
            n: int(prep_region.chip_widths[k])
            for k, n in enumerate(prep_region.chip_names)
        }
        if prep_region is not None
        else {}
    )
    chip_cols = [
        int(main_region.chip_widths[i]) + prep_widths.get(name, 0)
        for i, name in enumerate(order)
    ]
    tc = TotalCapClass.from_heights([int(r) for r in num_reals], chip_cols)
    gkr = GkrCapClass.from_heights([int(h) for h in main_region.chip_heights])
    return tc, gkr, gkr.resolved_slot_cap(gkr_chips, order)


def jagged_class(main_region: Any, prep_region: Any) -> dict[str, Any]:
    """The fully derived jagged class — no pin flag exists for it.

    Same ``(L, n_d)`` ⇒ eval-zone cache hit; same ``K`` ⇒ open
    prologue/query hit; the fold zone is K-independent and always shared
    (sp1-zorch#274).
    """
    regions = [r for r in (prep_region, main_region) if r is not None]
    l_total = sum(sum(int(c) for c in r.column_counts) for r in regions)
    ks = [int(r.dense.shape[0]) >> int(r.log_stacking_height) for r in regions]
    total_area = sum(int(r.dense.shape[0]) for r in regions)
    return {
        "L": l_total,
        "n_d": (total_area - 1).bit_length() + 1,
        "K": ks,
        "rlc_bits": max(sum(ks) - 1, 0).bit_length(),
    }


def resolve_classes(
    order: Sequence[str],
    own_tc: TotalCapClass,
    own_gkr: GkrCapClass,
    *,
    manifest_entry: Mapping[str, Any] | None = None,
    zc_spec: Mapping[str, Any] | None = None,
    gkr_spec: Mapping[str, Any] | None = None,
) -> tuple[TotalCapClass, GkrCapClass]:
    """Resolve the shard's pinned classes, highest precedence first: the
    shard's group-manifest entry, then the global ``--zc_class_json`` /
    ``--gkr_class_json`` specs, then the shard's own tight class.

    A manifest entry overrides field-by-field: ``"area_cap"`` pins the
    zerocheck class; ``"gkr"`` (with an optional ``"gkr_slot_cap"``) pins the
    GKR class. A field absent from the entry falls through to the next
    precedence level, so a partial hand-written entry pins only what it
    names. A ``None`` slot cap leaves the first-layer slot total to be
    resolved from the chips at prove time.

    Returns ``(tc_class, gkr_class)``.
    """
    tc = own_tc
    if zc_spec is not None:
        tc = TotalCapClass(area_cap=int(zc_spec["area_cap"]))
    gkr = own_gkr
    if gkr_spec is not None:
        gkr = GkrCapClass(
            tuple(int(gkr_spec["chip_heights"][n]) for n in order),
            gkr_spec.get("slot_cap"),
        )
    if manifest_entry is not None:
        if "area_cap" in manifest_entry:
            tc = TotalCapClass(area_cap=int(manifest_entry["area_cap"]))
        if "gkr" in manifest_entry:
            gkr = GkrCapClass(
                tuple(int(manifest_entry["gkr"][n]) for n in order),
                (
                    int(manifest_entry["gkr_slot_cap"])
                    if "gkr_slot_cap" in manifest_entry
                    else None
                ),
            )
    return tc, gkr
