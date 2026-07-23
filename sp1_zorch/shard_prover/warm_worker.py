# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Compile-only cache-fill worker: prove the given shards WITHOUT executing.

Every heavy stage/zorch zone is a `frx.jit`. We intercept `frx.jit` so the
OUTERMOST call in the eager orchestration lowers+compiles the zone (writing the
persistent cache) and returns `eval_shape`'d zeros — the chain flows to the next
stage on correct shapes without ever running a kernel. A depth guard keeps
nested zone calls running the real jit, so each zone lowers with its nested jits
inlined exactly as the real prove compiles them (verified: a real prove then
hits every cache entry byte-for-byte). Peak device memory is the autotune
scratch, not the ~29 GiB execute workspace — ~2 GiB at 46M area, ~18 GiB at
400M with the default two compile threads.

`frx.jit` MUST be patched before the chain imports bind their decorators, so
this module patches at import top, before any sp1/zorch import. Run as a
subprocess per shard from ``warm_shard_cache --warm``.
"""

import concurrent.futures
import os
import sys

import frx  # establish the frx jax fork before anything imports `jax`
import jax
import frx.numpy as fnp

_real_jit = frx.jit
_depth = [0]
_stats = {"compiled": 0}
# Backend compiles are fire-and-forget for a warm: the chain flows on
# eval_shape zeros and never consumes the executable, so `compile()` (C++,
# GIL-released: LLVM, ptxas, autotune) parallelizes across zones. Tracing/
# lowering stays inline — it is GIL-bound and produces the next zone's shapes.
# Default 2: concurrent on-device autotune scratch is the binding resource
# (~13.5 GiB per 400M-area zone against the worker's pool cap).
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.environ.get("WARM_COMPILE_THREADS", "2")))
_futures: list = []

# Deviceless warm: WARM_TARGET_CONFIG points at a GpuTargetConfigProto textproto
# (dump one from a real-device run's --xla_dump_to, *_gpu_target_config.pbtxt).
# Zones then lower+compile against a PJRT compile-only topology — zero VRAM,
# no CUDA device needed. Run with JAX_PLATFORMS=cpu so the eager glue (dummy
# inputs, eval_shape zeros) stays on host; only the plugin's compiler runs.
_topo_dev = None
if (_cfg_path := os.environ.get("WARM_TARGET_CONFIG")):
    from frx.experimental import topologies  # noqa: E402

    with open(_cfg_path) as _f:
        _topo_dev = topologies.get_topology_desc(
            "warm-aot", "cuda", target_config=_f.read(),
            topology="1x1x1").devices[0]


def _compile_only_jit(fn=None, **kw):
    if fn is None:
        return lambda f: _compile_only_jit(f, **kw)
    jitted = _real_jit(fn, **kw)

    def wrapper(*args, **kwargs):
        # Nested (under an outer lower/eval_shape trace): run the real jit so it
        # inlines into the outer zone's module — never intercept a nested call.
        if _depth[0] > 0:
            return jitted(*args, **kwargs)
        _depth[0] += 1
        try:
            if _topo_dev is not None:
                with jax.default_device(_topo_dev):
                    lowered = jitted.lower(*args, **kwargs)
            else:
                lowered = jitted.lower(*args, **kwargs)
            out = jax.eval_shape(jitted, *args, **kwargs)
        finally:
            _depth[0] -= 1
        _futures.append(_pool.submit(lowered.compile))  # write cache, no execute
        return jax.tree_util.tree_map(lambda s: fnp.zeros(s.shape, s.dtype), out)

    return wrapper


def _drain_compiles() -> int:
    """Wait for queued backend compiles; count successes, report failures."""
    failed = 0
    for f in _futures:
        try:
            f.result()
            _stats["compiled"] += 1
        except Exception as e:  # noqa: BLE001 — surface every zone failure
            failed += 1
            print(f"=== zone compile FAILED: {type(e).__name__}: {e} ===",
                  flush=True)
    _futures.clear()
    return failed


frx.jit = _compile_only_jit

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
    try:
        V.app.run(V.main)
    except SystemExit:
        pass
    n_failed = _drain_compiles()
    st = frx.local_devices()[0].memory_stats() or {}
    print(f"=== worker done: {_stats['compiled']} zones compiled, "
          f"{n_failed} failed, "
          f"peak={st.get('peak_bytes_in_use', 0) / 2**30:.2f}GiB ===", flush=True)
    if n_failed:
        sys.exit(1)
