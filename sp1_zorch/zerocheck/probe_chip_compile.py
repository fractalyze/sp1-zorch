# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-chip zerocheck compile probe: localize the single-big-kernel cliff.

`prove_shard_zerocheck` folds every chip's K constraints under one challenge as
a `zorch.constraint_eval` composite, which lowers to one kernel per chip. On a
real shard the joint jit spends >12 min in LLVM-NVPTX codegen on a single giant
kernel; the suspect is the koalabear `Global` chip (~241 cols, ~10k-op formula).
This tool compiles each chip's `constraint_eval` in isolation -- exactly the
per-row fold `_fold_chip` builds, base-field trace and EF-RLC alpha -- and times
`.compile()` per chip, so the per-chip codegen cost is attributable.

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
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
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
        fn = jax.jit(lambda t, a: constraint_eval(eval_fn, t, a))
        print(f"[start] {name} nc={nc} K={k} ...", flush=True)
        lowered = fn.lower(
            jax.ShapeDtypeStruct((args.rows, nc), bf),
            jax.ShapeDtypeStruct((k,), ef),
        )
        t0 = time.perf_counter()
        lowered.compile()
        dt = time.perf_counter() - t0
        print(f"{name:<30}{nc:>5}{k:>6}{nr:>10}{dt:>12.2f}", flush=True)
        timed.append((name, nc, k, nr, dt))

    if len(timed) > 1:
        print("\n=== sorted by compile_s (desc) ===", flush=True)
        print(_HDR, flush=True)
        for name, nc, k, nr, dt in sorted(timed, key=lambda r: -r[4]):
            print(f"{name:<30}{nc:>5}{k:>6}{nr:>10}{dt:>12.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
