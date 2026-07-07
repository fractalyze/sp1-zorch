# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 chip loading over ``rw_constraints``.

``rw_constraints.Chip`` already carries the AIR constraints and LogUp
interactions a chip needs downstream; this module binds the SP1 field dtype
and owns the SP1↔rw naming seam. Dump files use SP1's PascalCase chip names,
the rw manifest uses snake_case — :data:`SP1_NAME_TO_RW` keeps the handful of
genuinely-irregular pairs in one place and everything else is ``.lower()``.
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

# SP1 PascalCase -> rw snake_case, ONLY where ``name.lower()`` is wrong.
# Mirrors riscv-witness tools/sp1/sp1_shard_prover trace_loader NAME_MAP.
# Interim: the rw manifest itself should carry each chip's SP1 name so
# consumers stop hand-maintaining copies of this table; drop it once
# fractalyze/riscv-witness#1580 ships the mapping in the wheel.
SP1_NAME_TO_RW = {
    "Byte": "byte_lookup",
    "DivRem": "divrem",
    "KeccakPermute": "keccak_permute",
    "KeccakPermuteControl": "keccak_permute_control",
    "LoadByte": "load_byte",
    "LoadDouble": "load_double",
    "LoadHalf": "load_half",
    "LoadWord": "load_word",
    "LoadX0": "load_x0",
    "MemoryBump": "memory_bump",
    "MemoryGlobalFinalize": "memory_global_final",
    "MemoryGlobalInit": "memory_global_init",
    "MemoryLocal": "memory_local",
    "Program": "program_rom",
    "ShiftLeft": "shift_left",
    "ShiftRight": "shift_right",
    "StateBump": "state_bump",
    "StoreByte": "store_byte",
    "StoreDouble": "store_double",
    "StoreHalf": "store_half",
    "StoreWord": "store_word",
    "SyscallCore": "syscall_core",
    "SyscallInstrs": "syscall_instrs",
    "SyscallPrecompile": "syscall_precompile",
    "UType": "utype",
    "Secp256k1AddAssign": "secp256k1_add",
    "Secp256k1DoubleAssign": "secp256k1_double",
    "Secp256r1AddAssign": "secp256r1_add",
    "Secp256r1DoubleAssign": "secp256r1_double",
    "Bn254AddAssign": "bn254_add",
    "Bn254DoubleAssign": "bn254_double",
    "Bls12381AddAssign": "bls12381_add",
    "Bls12381DoubleAssign": "bls12381_double",
    "Bn254FpOpAssign": "bn254_fp_op",
    "Bls12381FpOpAssign": "bls12381_fp_op",
    "Bn254Fp2AddSubAssign": "bn254_fp2_addsub",
    "Bls12381Fp2AddSubAssign": "bls12381_fp2_addsub",
    "Bn254Fp2MulAssign": "bn254_fp2_mul",
    "Bls12381Fp2MulAssign": "bls12381_fp2_mul",
    "EdAddAssign": "ed_add",
    "ShaExtend": "sha256_extend",
    "ShaExtendControl": "sha256_extend_control",
    "ShaCompress": "sha256_compress",
    "ShaCompressControl": "sha256_compress_control",
    "Uint256MulMod": "uint256_mul",
    "U256XU2048Mul": "u256x2048_mul",
}


def sp1_name_to_rw(sp1_name: str) -> str:
    return SP1_NAME_TO_RW.get(sp1_name, sp1_name.lower())


_RW_NAME_TO_SP1 = {rw: sp1 for sp1, rw in SP1_NAME_TO_RW.items()}


def rw_name_to_sp1(rw_name: str) -> str:
    """Inverse of :func:`sp1_name_to_rw`, for producer-named trace inputs
    (a co-located riscv-witness trace-gen hands rw-named chip dicts, while
    every downstream stage — chip_order, the preamble's Fiat-Shamir chip
    metadata — speaks SP1 names).

    Live bundles name chips with riscv-witness's per-zkVM registry prefix
    (``"sp1_add"``); the prefix is namespacing, not part of the chip name,
    so it is stripped before mapping.

    Outside the irregular table the forward map is ``.lower()``, whose
    inverse is well-defined only for single-token PascalCase names
    (``"add"`` -> ``"Add"``). A snake_case name missing from the table is
    therefore uninvertible — fail loudly rather than fabricate an SP1 name
    the transcript preamble would absorb.
    """
    rw_name = rw_name.removeprefix("sp1_")
    sp1 = _RW_NAME_TO_SP1.get(rw_name)
    if sp1 is not None:
        return sp1
    if "_" in rw_name or not rw_name.islower():
        raise ValueError(
            f"unknown rw chip name {rw_name!r}: not in SP1_NAME_TO_RW and not "
            "a single-token lowercase name, so its SP1 spelling cannot be "
            "derived; add the pair to chip_loader.SP1_NAME_TO_RW"
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
