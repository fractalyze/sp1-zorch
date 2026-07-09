# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 chip loading over ``rw_constraints``.

``rw_constraints.Chip`` already carries the AIR constraints and LogUp
interactions a chip needs downstream; this module binds the SP1 field dtype
and owns the SP1↔rw naming seam. Dump files use SP1's PascalCase chip names,
the rw manifest uses snake_case; the pairing is genuinely irregular for many
chips (``byte_lookup`` ↔ ``Byte``), so it is read from each chip's manifest
``sp1_name`` field — emitted by the exporter from the schema — rather than a
table maintained here.
"""

from __future__ import annotations

import atexit
import functools
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from rw_constraints import Chip, ConstraintRegistry, bundled_constraints_dir
from zk_dtypes import koalabear_mont


@functools.cache
def _sp1_rw_name_maps() -> tuple[dict[str, str], dict[str, str]]:
    """``(sp1_name -> rw_name, rw_name -> sp1_name)`` from the sp1/v1 manifest.

    riscv-witness emits each chip's SP1 name into ``manifest.json`` (from the
    schema's ``sp1_name`` attribute), so the irregular SP1↔rw pairing is read
    from the shipped data rather than a table copied here — it cannot silently
    drift out of sync with the producer. Chips whose manifest omits
    ``sp1_name`` are skipped (they fall back to the ``.lower()`` /
    ``.capitalize()`` defaults below).
    """
    manifest = _registry().get_manifest("sp1", "v1")
    if manifest is None:
        raise FileNotFoundError(
            "rw-constraints ships no sp1/v1 manifest; cannot resolve chip names"
        )
    sp1_to_rw: dict[str, str] = {}
    rw_to_sp1: dict[str, str] = {}
    for chip in manifest.get("chips", []):
        sp1_name = chip.get("sp1_name")
        if not sp1_name:
            continue
        rw_to_sp1[chip["name"]] = sp1_name
        sp1_to_rw[sp1_name] = chip["name"]
    return sp1_to_rw, rw_to_sp1


def sp1_name_to_rw(sp1_name: str) -> str:
    sp1_to_rw, _ = _sp1_rw_name_maps()
    return sp1_to_rw.get(sp1_name, sp1_name.lower())


def rw_name_to_sp1(rw_name: str) -> str:
    """Inverse of :func:`sp1_name_to_rw`, for producer-named trace inputs
    (a co-located riscv-witness trace-gen hands rw-named chip dicts, while
    every downstream stage — chip_order, the preamble's Fiat-Shamir chip
    metadata — speaks SP1 names).

    Live bundles name chips with riscv-witness's per-zkVM registry prefix
    (``"sp1_add"``); the prefix is namespacing, not part of the chip name,
    so it is stripped before mapping.

    Chips the manifest doesn't map fall back to ``.capitalize()``, the inverse
    of ``sp1_name_to_rw``'s ``.lower()`` default, which is well-defined only
    for single-token names (``"add"`` -> ``"Add"``). A multi-token snake_case
    name with no manifest ``sp1_name`` is uninvertible — fail loudly rather
    than fabricate an SP1 name the transcript preamble would absorb.
    """
    rw_name = rw_name.removeprefix("sp1_")
    _, rw_to_sp1 = _sp1_rw_name_maps()
    sp1 = rw_to_sp1.get(rw_name)
    if sp1 is not None:
        return sp1
    if "_" in rw_name or not rw_name.islower():
        raise ValueError(
            f"unknown rw chip name {rw_name!r}: absent from the rw-constraints "
            "manifest's sp1_name map and not a single-token lowercase name, so "
            "its SP1 spelling cannot be derived"
        )
    return rw_name.capitalize()


def rw_names_for_chips(sp1_names) -> list[str]:
    """Bulk :func:`sp1_name_to_rw`, order-preserving and de-duplicated."""
    return list(dict.fromkeys(sp1_name_to_rw(n) for n in sp1_names))


@functools.cache
def _constraints_root() -> Path:
    """``rw_constraints``' bundled data as a plain directory tree.

    Bazel runfiles expose the wheel as a per-file symlink farm, so the
    registry's containment check — each chip file must ``resolve()`` inside
    the version dir — escapes into the store and raises ``Chip file escapes
    version dir``. Materializing one symlink-following copy per process
    restores a tree the registry accepts; plain pip installs skip the copy.
    The manifest probe is a proxy for the whole tree: bazel symlinks every
    file uniformly, so one resolved-in-place file means none are farmed.
    Drop once fractalyze/riscv-witness#1580 makes the check runfiles-safe.
    """
    src = bundled_constraints_dir()
    if src is None:
        raise FileNotFoundError(
            "rw-constraints is installed without its bundled constraint data"
        )
    probe = next(src.rglob("manifest.json"), None)
    if probe is not None and probe.resolve().is_relative_to(src.resolve()):
        return src
    dst = Path(
        tempfile.mkdtemp(prefix="rw-constraints-", dir=os.environ.get("TEST_TMPDIR"))
    )
    atexit.register(shutil.rmtree, dst, ignore_errors=True)
    shutil.copytree(src, dst / "constraints", symlinks=False)
    return dst / "constraints"


@functools.cache
def _registry() -> ConstraintRegistry:
    """One registry per process so its internal per-(target, version, dtype)
    chip cache survives across :func:`load_sp1_chips` calls — a full registry
    load execs every bundled chip module twice (~seconds)."""
    return ConstraintRegistry(_constraints_root())


def load_sp1_chips(
    target: str = "sp1",
    version: str = "v1",
    chip_names: Optional[list[str]] = None,
) -> dict[str, Chip]:
    """Load rw chip definitions with SP1's field dtype bound.

    Constraint / boundary / filler functions get ``koalabear_mont``;
    interactions keep the registry's ``jnp.uint32`` default (bitwise ops).
    """
    chips = _registry().load(target, version, constraint_field_dtype=koalabear_mont)
    if chip_names is not None:
        chips = {k: v for k, v in chips.items() if k in chip_names}
    return chips


def make_chip_stub(name: str, num_cols: int) -> Chip:
    """Constraint- and interaction-free placeholder ``Chip``.

    Keeps a chip in the shard for indexing when the rw manifest's column
    layout disagrees with the dumped trace (or the chip has no rw
    counterpart); downstream stages see an empty constraint set.

    Bypasses ``Chip.__init__`` (it unconditionally execs a source file) and
    mirrors its attribute set; a public empty-constructor upstream would
    replace this.
    """
    obj = Chip.__new__(Chip)
    obj.name = name
    obj.sp1_name = ""
    obj.num_cols = num_cols
    obj.num_blocks = 1
    obj._funcs = {}
    obj._constraints = {}
    obj._boundaries = {}
    obj._fillers = {}
    obj._interactions = {}
    obj._interaction_info = {}
    obj._pv_args = {}
    obj._constraint_blocks = {}
    return obj
