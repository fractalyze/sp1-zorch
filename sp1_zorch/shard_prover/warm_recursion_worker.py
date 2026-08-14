# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Compile-only cache-fill worker for one recursion warm entry.

Reuses ``warm_worker``'s interception wholesale: importing it (FIRST, before
any chain import binds a decorator) patches ``frx.jit`` with the compile-only
wrapper — the outermost zone call lowers + compiles into the persistent cache
and returns ``eval_shape``'d zeros with the real jit's committed placement —
and rebinds the GKR PoW absorb so the zeroed transcript flows. This module
then builds the entry's fixed-shape recursion shard
(:func:`~sp1_zorch.shard_prover.warm_recursion.build_recursion_shard`) and
drives the staged recursion chain
(:func:`~sp1_zorch.shard_prover.warm_recursion.prove_stage_chain`) through
that intercept, so every recursion-machine zone lands in ``FRX/JAX
_COMPILATION_CACHE_DIR`` with production-identical keys and no kernel ever
executes.

Run as a subprocess per entry from ``warm_recursion_cache``::

    python -m sp1_zorch.shard_prover.warm_recursion_worker <spec.json> <entry>
"""

import sys

# Patches frx.jit at ITS import top — must precede every chain import below.
from sp1_zorch.shard_prover import warm_worker as _intercept
from sp1_zorch.shard_prover import warm_recursion as _wr  # noqa: E402


def run_entry(spec_path: str, entry_name: str) -> int:
    """Warm one spec entry; return the number of failed zone compiles."""
    entries = {e.name: e for e in _wr.load_spec(spec_path)}
    if entry_name not in entries:
        raise ValueError(
            f"entry {entry_name!r} not in spec {spec_path} " f"(has {sorted(entries)})"
        )
    entry = entries[entry_name]
    shard = _wr.build_recursion_shard(entry, _wr.load_recursion_chips())
    _wr.prove_stage_chain(entry.machine(), shard)
    return _intercept._drain_compiles()


if __name__ == "__main__":
    n_failed = run_entry(sys.argv[1], sys.argv[2])
    print(
        f"=== recursion warm worker done: entry={sys.argv[2]} "
        f"{_intercept._stats['compiled']} zones compiled, {n_failed} failed ===",
        flush=True,
    )
    if n_failed:
        sys.exit(1)
