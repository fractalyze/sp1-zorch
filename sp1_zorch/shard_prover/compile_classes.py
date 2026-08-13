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

import hashlib
from collections.abc import Iterable, Mapping, Sequence
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


# --- class-keyed group manifest (zkvm-prover#176) ---------------------------
# The group manifest is a per-PROGRAM artifact keyed by chip-set class, not
# by per-block "shardN" position names. zkvm-prover's ``zkvm_sp1.manifest``
# owns the schema; sp1-zorch cannot import it, so the class naming, the cap
# quantization constants, and the read-side match rule are mirrored here by
# documented spec — golden-vector tests on both sides
# (``compile_classes_test`` / zkvm-prover ``test_manifest``) pin them to the
# same numbers.

_AREA_HEADROOM = (1, 2000)  # +0.05%
_AREA_QUANTUM = 1 << 15  # cells
_HEIGHT_REL_HEADROOM = (7, 50)  # +14% — the mid-chip (Mul-scale) rel drift
_HEIGHT_MIN_SLACK = 2048  # rows — the tiny-chip absolute-jitter floor
_HEIGHT_MAX_SLACK = 40960  # rows — the big-chip absolute drift bound
_HEIGHT_QUANTUM = 1 << 10  # rows
_SLOT_HEADROOM = (1, 25)  # +4%
_SLOT_QUANTUM = 1 << 15  # slots


def _quantize(x: int, headroom: tuple[int, int], quantum: int) -> int:
    """``ceil(x * (1 + num/den) / quantum) * quantum`` in exact integer
    arithmetic — a float boundary must never decide a compile key."""
    num, den = headroom
    padded = (int(x) * (den + num) + den - 1) // den
    return -(-padded // quantum) * quantum


def quantize_area(area: int) -> int:
    """The stored zerocheck ``area_cap`` for a tight area."""
    return _quantize(area, _AREA_HEADROOM, _AREA_QUANTUM)


def quantize_height(height: int) -> int:
    """The stored per-chip GKR height bound for a tight height.

    Per-chip slack = ``clamp(+14%, 2048, 40960)`` rows: cross-block height
    drift is absolute-bounded on big chips, relative on mid chips, and
    sub-percentage-floor on tiny chips — a flat percentage inflates the
    big chips (and the class's summed heights) far past the drift it
    covers.
    """
    num, den = _HEIGHT_REL_HEADROOM
    slack = (int(height) * num + den - 1) // den
    slack = min(max(slack, _HEIGHT_MIN_SLACK), _HEIGHT_MAX_SLACK)
    padded = int(height) + slack
    return -(-padded // _HEIGHT_QUANTUM) * _HEIGHT_QUANTUM


def quantize_slot(slot: int) -> int:
    """The stored ``gkr_slot_cap`` for a tight resolved slot bound."""
    return _quantize(slot, _SLOT_HEADROOM, _SLOT_QUANTUM)


def class_name(chips: Iterable[str]) -> str:
    """The class slug of a chip set: ``"{n}ch-{sig8}"``, sig8 = first 8 hex
    of sha256 over the newline-joined SORTED chip names — order-independent,
    stable across blocks/processes, greppable in a prove log."""
    names = sorted(str(n) for n in chips)
    sig = hashlib.sha256("\n".join(names).encode()).hexdigest()[:8]
    return f"{len(names)}ch-{sig}"


def manifest_entry_for(
    manifest: Mapping[str, Mapping[str, Any]],
    order: Sequence[str],
    name: str | None = None,
) -> Mapping[str, Any] | None:
    """The manifest entry allowed to pin a shard whose chip set is
    ``set(order)`` — the read-side match rule, mirroring
    ``zkvm_sp1.manifest.entry_for_shard``:

    - name lookup first (legacy per-block ``shardN`` manifests), but only
      when the named entry's ``gkr`` key set equals the live chips — a
      name collision across blocks must not serve a foreign class;
    - else any entry with EXACT chip-set equality — never a superset — the
      largest ``area_cap`` deterministically among several (legacy area
      clusters; a class-keyed manifest has exactly one per set);
    - else an area-only named entry (no ``gkr`` to match on) is trusted;
    - else ``None`` — the caller keeps its tight classes.
    """
    chips = frozenset(order)
    named = manifest.get(name) if name is not None else None
    if named is not None and "gkr" in named and frozenset(named["gkr"]) == chips:
        return named
    candidates = [
        (int(e.get("area_cap", -1)), key)
        for key, e in manifest.items()
        if "gkr" in e and frozenset(e["gkr"]) == chips
    ]
    if candidates:
        return manifest[max(candidates)[1]]
    if named is not None and "gkr" not in named:
        return named
    return None
