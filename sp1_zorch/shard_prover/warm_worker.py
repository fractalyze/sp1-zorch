# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Compile-only cache-fill worker: prove the given shards WITHOUT executing.

Thin subprocess shell over ``warm_mode`` — installs the compile-only
``frx.jit`` intercept before the chain imports, bypasses the value-dependent
host checks that zero'd outputs can't satisfy, and runs the prove chain under
``warm_mode.active()``. Peak device memory is the autotune scratch, not the
~29 GiB execute workspace — ~2 GiB at 46M area, ~18 GiB at 400M with the
default two compile threads. Run as a subprocess per shard from
``warm_shard_cache --warm``.

``WARM_TARGET_CONFIG`` (a GpuTargetConfigProto textproto path; dump one from a
real-device run's ``--xla_dump_to``) lowers zones against a PJRT compile-only
topology — zero VRAM, no CUDA device. Experimental: cache-key parity with
real-device proves is unresolved.
"""

import os
import sys

import frx  # establish the frx jax fork before anything imports `jax`

from sp1_zorch.shard_prover import warm_mode

_topo_dev = None
if (_cfg_path := os.environ.get("WARM_TARGET_CONFIG")):
    from frx.experimental import topologies

    with open(_cfg_path) as _f:
        _topo_dev = topologies.get_topology_desc(
            "warm-aot", "cuda", target_config=_f.read(),
            topology="1x1x1").devices[0]

warm_mode.install(
    compile_threads=int(os.environ.get("WARM_COMPILE_THREADS", "2")),
    lower_device=_topo_dev,
)

from sp1_zorch.shard_prover import verify_prove_shard as V  # noqa: E402
from sp1_zorch.logup_gkr import head as _head  # noqa: E402

# Bypass value-dependent HOST checks that zero'd compile-only outputs can't
# satisfy — they gate correctness, not compilation.
V.check_match = lambda *a, **k: True


def _grind_no_pow(self, carry, transcript):
    transcript, _ = transcript.check_witness(self._pow_bits, self._witness)
    return carry, transcript, self._witness


_head.GrindRound.__call__ = _grind_no_pow


if __name__ == "__main__":
    # argv[1] = comma-separated shard dirs; argv[2] (optional) = group manifest
    # so grouped-zerocheck compiles match the real prove's pinned class.
    shards = sys.argv[1]
    argv = ["warm_worker", f"--shard_dir={shards}", "--max_stage=4"]
    if len(sys.argv) > 2 and sys.argv[2]:
        argv.append(f"--group_manifest_json={sys.argv[2]}")
    sys.argv = argv
    n_failed = 0
    try:
        with warm_mode.active():
            try:
                V.app.run(V.main)
            except SystemExit:
                pass
    except warm_mode.WarmError as e:
        n_failed = 1
        print(f"=== {e} ===", flush=True)
    st = frx.local_devices()[0].memory_stats() or {}
    print(f"=== worker done: {warm_mode.zones_compiled()} zones compiled, "
          f"{n_failed} failed, "
          f"peak={st.get('peak_bytes_in_use', 0) / 2**30:.2f}GiB ===", flush=True)
    if n_failed:
        sys.exit(1)
