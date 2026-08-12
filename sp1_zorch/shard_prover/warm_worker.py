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

The zeros a zone returns carry the real jit's COMMITTED placement: jax commits
a jit's results to the execution device exactly when an input arrived
committed, and the cache key carries each parameter's committed/uncommitted
state (`allow_spmd_sharding_propagation_to_parameters`: a committed arg lowers
with its sharding specified, an uncommitted one as unspecified). Commitment
originates in the chain's own eager `device_put`s (the transcript host-FS
round trips) and propagates zone to zone through the data, so the wrapper
mirrors the rule — `device_put` the zeros when any arg leaf was committed,
plain zeros otherwise — and the warmed chain reproduces the staged prove's
per-parameter pattern arg-for-arg. Zeros that dropped the committed placement
would key every downstream zone all-unspecified and miss the staged prove's
entries on precisely the transcript-descended (cap/jagged/opening) zones.

`frx.jit` MUST be patched before the chain imports bind their decorators, so
this module patches at import top, before any sp1/zorch import. Run as a
subprocess per shard from ``warm_shard_cache --warm``.
"""

import concurrent.futures
import os
import sys
from collections.abc import Callable
from typing import Any

import frx  # establish the frx jax fork before anything imports `jax`
import frx.numpy as fnp
import jax
from jax.sharding import SingleDeviceSharding

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
    max_workers=int(os.environ.get("WARM_COMPILE_THREADS", "2"))
)
_futures: list = []

# Deviceless warm: WARM_TARGET_CONFIG points at a GpuTargetConfigProto textproto
# (dump one from a real-device run's --xla_dump_to, *_gpu_target_config.pbtxt).
# Zones then lower+compile against a PJRT compile-only topology — zero VRAM,
# no CUDA device needed. Run with JAX_PLATFORMS=cpu so the eager glue (dummy
# inputs, eval_shape zeros) stays on host; only the plugin's compiler runs.
_topo_dev = None
if _cfg_path := os.environ.get("WARM_TARGET_CONFIG"):
    from frx.experimental import topologies  # noqa: E402

    with open(_cfg_path) as _f:
        _topo_dev = topologies.get_topology_desc(
            "warm-aot", "cuda", target_config=_f.read(), topology="1x1x1"
        ).devices[0]

# The device committed zone outputs are placed on (a real jit commits its
# results to the execution device). Resolved lazily: importing this module
# must not initialize a backend.
_commit_dev = [None]


def _args_committed(args: tuple, kwargs: dict) -> bool:
    """Whether the real jit would commit this call's outputs: for the zones
    warmed here — no out_shardings, no context mesh, no in-jaxpr memory-kind
    transfers — jax commits a single-device jit's results exactly when an arg
    leaf arrived committed (`frx._src.interpreters.pxla` keys it on any
    specified in-sharding; those other commitment triggers would need
    modeling here if a zone ever grows one)."""
    return any(
        isinstance(x, jax.Array) and getattr(x, "committed", False)
        for x in jax.tree_util.tree_leaves((args, kwargs))
    )


def _zone_zeros(out_shapes: Any, committed: bool) -> Any:
    """Zeros for a zone's `eval_shape` outputs, committed to the local device
    exactly when the real prove's results would be."""
    if committed and _commit_dev[0] is None:
        _commit_dev[0] = jax.local_devices()[0]

    def zero(s: Any) -> Any:
        z = fnp.zeros(s.shape, s.dtype)
        return jax.device_put(z, _commit_dev[0]) if committed else z

    return jax.tree_util.tree_map(zero, out_shapes)


def _lowering_args(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """The (args, kwargs) a zone lowers on.

    Deviceless mode: committed leaves live on host CPU but stand in for the
    staged prove's committed CUDA arrays, so each becomes an abstract spec
    sharded on the topology device — the lowering keeps its committed
    (sharding-specified) keying without pulling the CPU device into the
    lowering. Uncommitted leaves pass through and lower as unspecified,
    matching the prove's host-value args. With a real device the args already
    carry the right shardings; no translation."""
    if _topo_dev is None:
        return args, kwargs

    def retarget(x: Any) -> Any:
        if isinstance(x, jax.Array) and getattr(x, "committed", False):
            return jax.ShapeDtypeStruct(
                x.shape,
                x.dtype,
                sharding=SingleDeviceSharding(_topo_dev),
                weak_type=x.aval.weak_type,
            )
        return x

    return jax.tree_util.tree_map(retarget, (args, kwargs))


def _compile_only_jit(fn: Callable[..., Any] | None = None, **kw: Any) -> Any:
    if fn is None:
        return lambda f: _compile_only_jit(f, **kw)
    jitted = _real_jit(fn, **kw)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Nested (under an outer lower/eval_shape trace): run the real jit so it
        # inlines into the outer zone's module — never intercept a nested call.
        if _depth[0] > 0:
            return jitted(*args, **kwargs)
        _depth[0] += 1
        try:
            largs, lkwargs = _lowering_args(args, kwargs)
            if _topo_dev is not None:
                with jax.default_device(_topo_dev):
                    lowered = jitted.lower(*largs, **lkwargs)
            else:
                lowered = jitted.lower(*largs, **lkwargs)
            out = jax.eval_shape(jitted, *args, **kwargs)
        finally:
            _depth[0] -= 1
        _futures.append(_pool.submit(lowered.compile))  # write cache, no execute
        return _zone_zeros(out, _args_committed(args, kwargs))

    return wrapper


def _chain_rc(code: object) -> int:
    """The shard chain's verdict from ``app.run``'s SystemExit code: absl
    always leaves via SystemExit, and the staged sweep exits with a
    failed-shards payload when any shard's chain died. 0/None is a clean
    pass; anything else (a nonzero int or a message string) is a dead chain
    the worker must report as its own failure."""
    return 0 if code in (0, None) else 1


def _drain_compiles() -> int:
    """Wait for queued backend compiles; count successes, report failures."""
    failed = 0
    for f in _futures:
        try:
            f.result()
            _stats["compiled"] += 1
        except Exception as e:  # noqa: BLE001 — surface every zone failure
            failed += 1
            print(f"=== zone compile FAILED: {type(e).__name__}: {e} ===", flush=True)
    _futures.clear()
    return failed


frx.jit = _compile_only_jit

from sp1_zorch.logup_gkr import prover as _gkr_prover  # noqa: E402
from sp1_zorch.shard_prover import staged_prove_shard as S  # noqa: E402

# Bypass value-dependent HOST checks that zero'd compile-only outputs can't
# satisfy — they gate correctness, not compilation.
S.check_match = lambda *a, **k: True


def _grind_no_pow(transcript: Any, pow_witness: Any, *, pow_bits: int = 0) -> Any:
    transcript, _ = transcript.check_witness(pow_witness, pow_bits=pow_bits)
    return transcript


# Rebinds the name in the module that calls it, the same way `check_match` is
# rebound above: the call sits deep inside the prover with no seam to inject
# through, and `from ... import absorb_grind` resolves it as a module global
# at call time.
_gkr_prover.absorb_grind = _grind_no_pow


if __name__ == "__main__":
    # argv[1] = comma-separated shard dirs; argv[2] (optional) = group manifest
    # so grouped-zerocheck compiles match the real prove's pinned class.
    # --noproof_sha256: bincode encoding is host work over the final proof; a
    # compile-only run's zeroed sections have nothing worth hashing.
    shards = sys.argv[1]
    argv = [
        "warm_worker",
        f"--shard_dir={shards}",
        "--max_phase=4",
        "--noproof_sha256",
    ]
    if len(sys.argv) > 2 and sys.argv[2]:
        argv.append(f"--group_manifest_json={sys.argv[2]}")
    sys.argv = argv
    # A swallowed failure payload would count a dead shard chain as warmed
    # while its tail zones never compiled (zkvm-prover#161, sp1-zorch#341).
    chain_rc = 0
    try:
        S.app.run(S.main)
    except SystemExit as e:
        chain_rc = _chain_rc(e.code)
    n_failed = _drain_compiles()
    st = frx.local_devices()[0].memory_stats() or {}
    print(
        f"=== worker done: {_stats['compiled']} zones compiled, "
        f"{n_failed} failed, harness={'FAILED' if chain_rc else 'ok'}, "
        f"peak={st.get('peak_bytes_in_use', 0) / 2**30:.2f}GiB ===",
        flush=True,
    )
    if n_failed or chain_rc:
        sys.exit(1)
