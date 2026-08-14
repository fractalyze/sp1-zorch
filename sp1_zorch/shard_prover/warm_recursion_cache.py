# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fill the persistent compile cache with the RECURSION machine classes.

``warm_shard_cache``'s recursion sibling. Its own entrypoint, not a
``--warm_recursion`` flag there, because the two share no phase:
``warm_shard_cache`` is dump analysis — per-shard class derivation, manifest
folding, cover selection, peak-aware GPU packing over ~25 heavy workers —
while the recursion warm has no dump at all: a handful of fixed-shape entries
from a shape spec (the recursion circuits are block-independent), one worker
each. Only the worker-side ``frx.jit`` interception is shared, and both reach
it through ``warm_worker``.

Usage (same caller env contract as ``warm_shard_cache`` — the run's exact
``XLA_FLAGS``, ``JAX_PLATFORMS=cpu`` for this parent process, seed and prove
from the same venv/pin)::

    python -m sp1_zorch.shard_prover.warm_recursion_cache \\
        --spec_json=<program>/recursion_shapes.json --compile_cache_dir=<dir>

Workers run sequentially, one subprocess per entry, each writing the shared
``--compile_cache_dir`` under both cache-dir env spellings. Exit is nonzero when any
worker fails or the cache gained no entries — a warm that filled nothing is a
broken seed, not a success.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from absl import app, flags

from sp1_zorch.shard_prover.warm_recursion import StageShapes, load_spec

_SPEC_JSON = flags.DEFINE_string(
    "spec_json",
    None,
    "Recursion shape spec JSON (see warm_recursion.load_spec) — the fixed "
    "per-program stage shapes, captured once from any block's run.",
    required=True,
)
# Not ``--cache_dir``: absl flag names are one global registry, and the
# sibling ``warm_shard_cache`` already owns that spelling — importing both in
# one process (the test suite does) must not DuplicateFlagError.
_CACHE_DIR = flags.DEFINE_string(
    "compile_cache_dir",
    None,
    "Persistent compile cache dir the workers fill (and a real recursion "
    "prove later hits).",
    required=True,
)
_ENTRIES = flags.DEFINE_string(
    "entries",
    "",
    "Comma-separated spec entry names to warm (default: every entry).",
)


def select_entries(spec: Sequence[StageShapes], names_csv: str) -> list[StageShapes]:
    """The spec entries to warm: all, or the named subset (unknown names
    raise — a typo'd filter silently warming nothing would read as success)."""
    names = [n.strip() for n in names_csv.split(",") if n.strip()]
    if not names:
        return list(spec)
    by_name = {e.name: e for e in spec}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"unknown spec entries {unknown} (spec has {sorted(by_name)})")
    return [by_name[n] for n in names]


def worker_cmd(spec_json: str, entry_name: str) -> list[str]:
    """The per-entry worker invocation."""
    return [
        sys.executable,
        "-m",
        "sp1_zorch.shard_prover.warm_recursion_worker",
        spec_json,
        entry_name,
    ]


def worker_env(base_env: dict[str, str], cache_dir: str) -> dict[str, str]:
    """The worker's environment: the cache dir under BOTH env spellings (frx
    >= 0811 reads only ``FRX_``), the pool-release cap so autotune scratch is
    released between zone compiles, and no ``JAX_PLATFORMS`` override — the
    parent runs CPU-only, the worker needs the real backend."""
    env = dict(
        base_env,
        FRX_COMPILATION_CACHE_DIR=cache_dir,
        JAX_COMPILATION_CACHE_DIR=cache_dir,
        XLA_PYTHON_CLIENT_ALLOCATOR="cuda_async",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.5",
    )
    env.pop("JAX_PLATFORMS", None)
    return env


def main(argv: Sequence[str]) -> None:
    del argv
    spec_path = _SPEC_JSON.value
    entries = select_entries(load_spec(spec_path), _ENTRIES.value)
    cache = _CACHE_DIR.value
    Path(cache).mkdir(parents=True, exist_ok=True)
    env = worker_env(dict(os.environ), cache)
    ok = fail = 0
    for e in entries:
        print(
            f"=== warming recursion entry {e.name} (stage={e.stage}, "
            f"{len(e.chips)} chips) ===",
            flush=True,
        )
        rc = subprocess.run(worker_cmd(spec_path, e.name), env=env).returncode
        if rc == 0:
            ok += 1
        else:
            fail += 1
            print(f"  recursion warm worker for {e.name} exited {rc}", flush=True)
    n_entries = sum(1 for p in Path(cache).rglob("*") if p.is_file())
    print(
        f"=== recursion warm done: {ok}/{ok + fail} entries ok; "
        f"cache entries: {n_entries} ===",
        flush=True,
    )
    if fail or not ok or n_entries == 0:
        sys.exit(
            f"recursion warm FAILED: {fail} worker(s) failed, "
            f"{n_entries} cache entries"
        )


if __name__ == "__main__":
    app.run(main)
