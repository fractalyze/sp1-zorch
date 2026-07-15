# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shard loading from SP1 GPU-trace dumps.

A dump directory holds per-chip ``<Name>.meta`` (``num_real=``/``width=``
lines) + ``<Name>.bin`` (row-major raw Montgomery u32), an optional
``preprocessed/`` subdir in the same format, ``public_values.bin``, and a
canonical-integer ``gpu_vk.txt`` at the root. rsp captures put traces under
``gpu_traces/``, the gpu_fibonacci fixture under ``traces/`` — hence the
subdir autodetect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import frx.numpy as jnp
import numpy as np
import zk_dtypes
from frx import Array
from zk_dtypes import koalabear_mont

from sp1_zorch.shard_prover.chip_loader import (
    load_sp1_chips,
    make_chip_stub,
    rw_names_for_chips,
    sp1_name_to_rw,
)
from sp1_zorch.shard_prover.types import (
    MachineVerifyingKey,
    MainTraceData,
    ShardData,
    Traces,
)

_TRACE_SUBDIRS = ("gpu_traces", "traces")


@dataclass(frozen=True)
class DumpData:
    """Raw dump contents, before chip definitions attach."""

    traces: dict[str, Array]
    num_reals: dict[str, int]
    preprocessed: dict[str, Array]
    public_values: Array
    vk: MachineVerifyingKey


def _parse_kv_lines(text: str, *, skip_unkeyed: bool = False) -> dict[str, str]:
    """``key=value`` lines. Blank lines and stray whitespace are tolerated;
    any other malformed line fails loudly — these files are machine-generated,
    so silent skipping would mask dump-format drift. ``skip_unkeyed``
    tolerates exactly the ``--- section ---`` separator shape some diag files
    carry (``gpu_gkr_state.txt``'s ``--- round N ---``); anything else still
    fails loudly."""
    out = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=" not in line:
            if skip_unkeyed and stripped.startswith("---") and stripped.endswith("---"):
                continue
            raise ValueError(f"malformed dump line: {line!r}")
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_int_list(value: str) -> list[int]:
    """A dump's ``[a, b, c]`` canonical-int list."""
    return [int(t) for t in value.strip("[]").split(",")]


_EF_RE = re.compile(r"value: \[([0-9, ]+)\]")


def _parse_ef_list(value: str) -> Array:
    """All ``BinomialExtensionField { value: [...] }`` blobs in a dump value,
    as one EF array (canonical limbs Mont-encoded)."""
    rows = [[int(t) for t in m.group(1).split(",")] for m in _EF_RE.finditer(value)]
    return (
        jnp.array(rows, dtype=koalabear_mont)
        .reshape(-1)
        .view(zk_dtypes.koalabearx4_mont)
    )


def check_match(label: str, got, want) -> bool:
    """The byte-match runnables' OK/MISMATCH line: compare a value against
    its dump reference and print the verdict (got/want on mismatch)."""
    if isinstance(got, (int, list, tuple)):
        ok = got == want
    else:
        # array_equal, not all(got == want): shape divergence must read as
        # a mismatch, and broadcasting would equate e.g. (1,) with (1, 1).
        ok = bool(jnp.array_equal(got, want))
    print(f"{'OK ' if ok else 'MISMATCH'} {label}")
    if not ok:
        print(f"  got:  {got}")
        print(f"  want: {want}")
    return ok


def _read_meta(meta_path: Path) -> tuple[int, int]:
    meta = _parse_kv_lines(meta_path.read_text())
    return int(meta["num_real"]), int(meta["width"])


def _load_trace_dir(trace_dir: Path) -> tuple[dict[str, Array], dict[str, int]]:
    """Load every ``*.meta``-described matrix; zero-height chips keep their
    width as an empty ``(0, width)`` so layout stays meta-complete."""
    traces, num_reals = {}, {}
    for meta_path in sorted(trace_dir.glob("*.meta")):
        name = meta_path.stem
        height, width = _read_meta(meta_path)
        num_reals[name] = height
        bin_path = trace_dir / f"{name}.bin"
        if height == 0 or not bin_path.exists():
            traces[name] = jnp.zeros((0, width), dtype=koalabear_mont)
        else:
            raw = np.fromfile(bin_path, dtype=np.uint32)
            traces[name] = jnp.array(raw.reshape(height, width)).view(koalabear_mont)
    return traces, num_reals


def _load_vk(vk_path: Path) -> MachineVerifyingKey:
    """Parse the canonical-integer ``key=[v, ...]`` dump into Montgomery
    arrays (``astype`` encodes; the dump is canonical, not raw bytes)."""
    vals = {
        key: [int(x.strip()) for x in value.strip("[]").split(",")]
        for key, value in _parse_kv_lines(vk_path.read_text()).items()
    }

    def _arr(key: str) -> Array:
        return jnp.array(vals[key], dtype=jnp.uint32).astype(koalabear_mont)

    return MachineVerifyingKey(
        preprocessed_commit=_arr("preprocessed_commit"),
        pc_start=_arr("pc_start"),
        cum_sum_x=_arr("cum_sum_x"),
        cum_sum_y=_arr("cum_sum_y"),
        enable_untrusted=vals.get("enable_untrusted", [0])[0],
    )


def read_dump(fixture_dir: Path, trace_subdir: Optional[str] = None) -> DumpData:
    """Read a dump directory into arrays; no rw chip definitions involved."""
    fixture_dir = Path(fixture_dir)
    if trace_subdir is None:
        trace_subdir = next(
            (s for s in _TRACE_SUBDIRS if (fixture_dir / s).is_dir()), None
        )
        if trace_subdir is None:
            raise FileNotFoundError(
                f"{fixture_dir} has none of {_TRACE_SUBDIRS} trace subdirs"
            )
    trace_dir = fixture_dir / trace_subdir

    traces, num_reals = _load_trace_dir(trace_dir)
    prep_dir = trace_dir / "preprocessed"
    if prep_dir.is_dir():
        preprocessed, prep_reals = _load_trace_dir(prep_dir)
        # Empty prep matrices carry no commitment content — drop them.
        preprocessed = {n: t for n, t in preprocessed.items() if prep_reals[n] > 0}
    else:
        preprocessed = {}

    pv_raw = np.fromfile(trace_dir / "public_values.bin", dtype=np.uint32)
    public_values = jnp.array(pv_raw).view(koalabear_mont)

    return DumpData(
        traces=traces,
        num_reals=num_reals,
        preprocessed=preprocessed,
        public_values=public_values,
        vk=_load_vk(fixture_dir / "gpu_vk.txt"),
    )


def resolve_chips(traces: dict[str, Array], preprocessed: dict[str, Array]):
    """rw chip definitions for SP1-named trace matrices.

    A chip gets its rw constraints only when the manifest width agrees with
    the trace (manifest ``num_cols`` counts prep + main columns); otherwise
    it stays a constraint-less stub so chip indexing and trace layout survive
    the mismatch. One definition for both trace sources — the dump loader and
    the producer ingest — so the two paths cannot diverge on which chips
    carry constraints.
    """
    rw_chips = load_sp1_chips(chip_names=rw_names_for_chips(traces))
    chips = {}
    for name, trace in traces.items():
        main_width = trace.shape[1]
        prep_width = preprocessed[name].shape[1] if name in preprocessed else 0
        rw_chip = rw_chips.get(sp1_name_to_rw(name))
        if rw_chip is not None and rw_chip.num_cols == main_width + prep_width:
            chips[name] = rw_chip
        else:
            chips[name] = make_chip_stub(name, main_width)
    return chips


def load_fixture_shard(
    fixture_dir: Path, trace_subdir: Optional[str] = None
) -> ShardData:
    """Build a :class:`ShardData` from a GPU-dump directory; chips attach
    per :func:`resolve_chips` (constraints on a manifest width match, stubs
    otherwise)."""
    dump = read_dump(fixture_dir, trace_subdir)

    # dump.traces already iterates name-sorted (the reader walks sorted
    # .meta files), which fixes chip_order for every downstream stage.
    return ShardData(
        vk=dump.vk,
        preprocessed_traces=dump.preprocessed,
        main_trace_data=MainTraceData(
            traces=Traces.from_arrays(dump.traces, dump.num_reals),
            public_values=dump.public_values,
            chips=resolve_chips(dump.traces, dump.preprocessed),
        ),
    )
