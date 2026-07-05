# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-chip zerocheck compile probe: localize the single-big-kernel cliff.

`prove_shard_zerocheck` folds every chip's K constraints under one challenge as
a `zorch.constraint_eval` composite, which lowers to one kernel per chip. On a
real shard the joint jit spends >12 min in LLVM-NVPTX codegen on a single giant
kernel; the suspect is the koalabear `Global` chip (~241 cols, ~10k-op formula).
This tool compiles each chip's `constraint_eval` in isolation -- exactly the
per-row fold the jagged prover builds (`prove_jagged_zerocheck`, jagged.py):
base-field trace, EF-RLC alpha, AND the runtime `live_width` row bound. That
bound is load-bearing: it is what makes the composite lower to the bounded
`zorch.constraint_eval_bounded` kernel. Omitting it compiles the *unbounded*
variant, which the compiler inlines to plain multiply/adds -- a different,
cliff-free body -- so the probe would never see the per-chip kernel it exists to
localize. Times `.compile()` per chip, so the per-chip codegen cost is
attributable.

Codegen is host-bound (LLVM-NVPTX, GPU ~0% util) and the kernel's instruction
count is driven by the constraint circuit and column count, not the row count,
so a tiny leading axis reproduces the kernel a real shard compiles while keeping
the GPU idle and the run cheap. Drive the suspect under `timeout` to bound the
cliff rather than sit through the full >12 min. `JAX_PLATFORMS=cuda` so the
zkx GPU emitter runs and a missing plugin errors instead of silently falling
back to CPU (the printed `backend=` line reports which ran); `--list` needs no
GPU:

    JAX_PLATFORMS=cuda bazel run //sp1_zorch/zerocheck:probe_chip_compile -- \
        --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --skip=Global
    timeout 240 env JAX_PLATFORMS=cuda bazel run \
        //sp1_zorch/zerocheck:probe_chip_compile -- \
        --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --only=Global

`--run` adds a RUNTIME measurement on the same compiled kernel (warm-up, then
`--run-iters` timed executions, ms/iter): unlike codegen, runtime scales with
rows, so pass `--rows` near the chip's real `num_real`. Zeros operands time
like real data — field-op cost is data-independent. This is the runtime arm of
compile/runtime A/Bs (e.g. the fractalyze/xla#200 interpreter-retirement gate);
see docs/testing.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from zk_dtypes import koalabearx4_mont

from sp1_zorch.shard_prover.fixture_loader import load_fixture_shard
from sp1_zorch.shard_prover.replay import shard_regions
from sp1_zorch.zerocheck.stage import bind_pv, probe_num_constraints
from zorch.constraint_eval import constraint_eval

_HDR = f"{'chip':<30}{'nc':>5}{'K':>6}{'num_real':>10}{'compile_s':>12}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-dir", required=True, help="rsp shard dump directory.")
    p.add_argument(
        "--rows",
        type=int,
        default=8,
        help="leading-axis size of the probe trace; codegen is row-independent, "
        "so this stays small to keep the run cheap.",
    )
    p.add_argument("--only", default=None, help="comma-separated chip names to probe.")
    p.add_argument("--skip", default=None, help="comma-separated chip names to skip.")
    p.add_argument(
        "--list",
        action="store_true",
        help="print each chip's nc/K/num_real and exit without compiling.",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="allocate a real (zeros) trace and time execution (ms/iter) of the "
        "compiled kernel. Runtime scales with rows, so pass --rows near the "
        "chip's real num_real for a representative number (codegen itself is "
        "row-independent, so compile-only probes keep the tiny default).",
    )
    p.add_argument("--run-iters", type=int, default=10, help="timed execution iters.")
    args = p.parse_args()

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    shard = load_fixture_shard(Path(args.shard_dir))
    main_region, prep_region = shard_regions(shard)
    chips = shard.main_trace_data.chips
    public_values = shard.main_trace_data.public_values
    chip_names = main_region.chip_names

    bf = main_region.dense.dtype  # koalabear_mont base field, the real trace dtype.
    ef = koalabearx4_mont  # the constraint-RLC alpha is an extension challenge.
    # Per-chip column count = the [main | prep] width chip_traces (stage.py) folds,
    # i.e. trace.shape[0] in prove_shard_zerocheck. Derived from the region widths
    # rather than by building chip_traces so this compile-only probe allocates no
    # real trace (it compiles against tiny synthetic shapes below).
    prep_w = {}
    if prep_region is not None:
        for k, n in enumerate(prep_region.chip_names):
            prep_w[n] = int(prep_region.chip_widths[k])

    print(
        f"backend={jax.default_backend()} rows={args.rows} chips={len(chip_names)}",
        flush=True,
    )
    print(_HDR, flush=True)

    timed: list[tuple[str, int, int, int, float]] = []
    for i, name in enumerate(chip_names):
        if only is not None and name not in only:
            continue
        if name in skip:
            continue
        nc = int(main_region.chip_widths[i]) + prep_w.get(name, 0)
        nr = int(main_region.chip_heights[i])
        if nc == 0:
            print(f"{name:<30}{nc:>5}{'-':>6}{nr:>10}{'(no cols)':>12}", flush=True)
            continue
        eval_fn = bind_pv(chips[name], public_values)
        k = probe_num_constraints(eval_fn, nc, ef)
        if k == 0:
            # Lookup-only chips (SP1's Byte / Program / Range) fold to nothing.
            print(f"{name:<30}{nc:>5}{k:>6}{nr:>10}{'(K=0)':>12}", flush=True)
            continue
        if args.list:
            print(f"{name:<30}{nc:>5}{k:>6}{nr:>10}{'(list)':>12}", flush=True)
            continue
        # `live_width` rides as a runtime scalar operand exactly as
        # `prove_jagged_zerocheck` passes `num_non_padded` (jagged.py) — its value
        # is irrelevant to codegen (the kernel is row-count-independent), but its
        # presence is what selects the bounded kernel over the inlined body.
        fn = jax.jit(lambda t, a, lw: constraint_eval(eval_fn, t, a, live_width=lw))
        print(f"[start] {name} nc={nc} K={k} ...", flush=True)
        lowered = fn.lower(
            jax.ShapeDtypeStruct((args.rows, nc), bf),
            jax.ShapeDtypeStruct((k,), ef),
            jax.ShapeDtypeStruct((), jnp.int32),
        )
        t0 = time.perf_counter()
        lowered.compile()
        dt = time.perf_counter() - t0
        print(f"{name:<30}{nc:>5}{k:>6}{nr:>10}{dt:>12.2f}", flush=True)
        timed.append((name, nc, k, nr, dt))

        if args.run:
            # Real (zeros) operands of the compiled shapes so the executable is
            # reused and the kernel actually runs hot.
            t = jnp.zeros((args.rows, nc), bf)
            a = jnp.zeros((k,), ef)
            lw = jnp.int32(args.rows)
            out = fn(t, a, lw)
            jax.block_until_ready(out)  # warm
            t0 = time.perf_counter()
            for _ in range(args.run_iters):
                out = fn(t, a, lw)
            jax.block_until_ready(out)
            dt_run = (time.perf_counter() - t0) / args.run_iters
            print(f"RUN {name} rows={args.rows}: {dt_run * 1000:.3f} ms/iter", flush=True)

    if len(timed) > 1:
        print("\n=== sorted by compile_s (desc) ===", flush=True)
        print(_HDR, flush=True)
        for name, nc, k, nr, dt in sorted(timed, key=lambda r: -r[4]):
            print(f"{name:<30}{nc:>5}{k:>6}{nr:>10}{dt:>12.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
