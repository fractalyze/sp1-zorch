# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Compile-time probe for the real jagged zerocheck round (sp1-zorch#123).

`probe_chip_compile` shows bare `constraint_eval` per chip is cheap (the Global
chip's kernel compiles in seconds). The real cliff, if it exists, is in
`prove_jagged_zerocheck`'s `lax.scan` body, where `constraint_eval` is inlined
(x3 t-points) alongside the eq sums, the GKR-column matmul, and the round-poly
machinery -- if the composite marker is fused into the monolithic scan body
instead of staying one reused kernel, that fused body is the wall.

This compiles `prove_jagged_zerocheck` for a chosen chip subset and times
`.compile()`. The trace operands are passed as `jax.ShapeDtypeStruct` (abstract
shapes, no data allocation, no GKR replay); every other operand
(`alphas / lambdas / zeta / transcript / beta / claims`) is a traced input the
way the shard prover jits it, so `zeta`->`eq` is computed in-kernel rather than
constant-folded. Bisect with `--only` / `--skip` to attribute the cliff.
`JAX_PLATFORMS=cuda` so the zkx GPU emitter runs and a missing plugin errors
instead of silently compiling on CPU (the printed `backend=` line reports which
ran):

    JAX_PLATFORMS=cuda bazel run //sp1_zorch/zerocheck:probe_round_compile -- \
        --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --skip=Global
    timeout -k 10 -s KILL 240 env JAX_PLATFORMS=cuda bazel run \
        //sp1_zorch/zerocheck:probe_round_compile -- \
        --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --only=Global
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from zk_dtypes import koalabearx4_mont

from sp1_zorch.shard_prover.fixture_loader import load_fixture_shard
from sp1_zorch.shard_prover.replay import MAX_LOG_ROW_COUNT, shard_regions
from sp1_zorch.zerocheck.jagged import prove_jagged_zerocheck
from sp1_zorch.zerocheck.stage import bind_pv, probe_num_constraints
from zorch.testkit.transcript import cheap_transcript


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-dir", required=True, help="rsp shard dump directory.")
    p.add_argument("--only", default=None, help="comma-separated chip names to keep.")
    p.add_argument("--skip", default=None, help="comma-separated chip names to drop.")
    p.add_argument(
        "--no-gkr",
        action="store_true",
        help="prove the pure zero-sum (no beta/claims column term).",
    )
    args = p.parse_args()

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    shard = load_fixture_shard(Path(args.shard_dir))
    main_region, prep_region = shard_regions(shard)
    chips = shard.main_trace_data.chips
    public_values = shard.main_trace_data.public_values
    bf = main_region.dense.dtype
    ef = koalabearx4_mont
    # Per-chip column count = the [main | prep] width chip_traces (stage.py) folds,
    # i.e. trace.shape[0] in prove_shard_zerocheck. Derived from the region widths
    # rather than by building chip_traces so this compile-only probe allocates no
    # real trace (the operands below are abstract shapes).
    prep_w = {}
    if prep_region is not None:
        for k, n in enumerate(prep_region.chip_names):
            prep_w[n] = int(prep_region.chip_widths[k])

    names, eval_fns, ncs, ks, nrs = [], [], [], [], []
    for i, name in enumerate(main_region.chip_names):
        if only is not None and name not in only:
            continue
        if name in skip:
            continue
        nc = int(main_region.chip_widths[i]) + prep_w.get(name, 0)
        if nc == 0:
            continue
        eval_fn = bind_pv(chips[name], public_values)
        names.append(name)
        eval_fns.append(eval_fn)
        ncs.append(nc)
        ks.append(probe_num_constraints(eval_fn, nc, ef))
        nrs.append(int(main_region.chip_heights[i]))

    num_chips = len(names)
    if num_chips == 0:
        raise SystemExit("no chips selected")
    num_vars = MAX_LOG_ROW_COUNT
    print(
        f"backend={jax.default_backend()} num_vars={num_vars} chips={num_chips}: "
        + ", ".join(f"{n}(nc={c},K={k},nr={r})" for n, c, k, r in zip(names, ncs, ks, nrs)),
        flush=True,
    )

    use_gkr = not args.no_gkr

    def f(traces, alphas, lambdas, zeta, transcript, beta, claims):
        return prove_jagged_zerocheck(
            eval_fns,
            traces,
            nrs,
            alphas,
            lambdas,
            zeta,
            transcript,
            beta=beta if use_gkr else None,
            claims=claims if use_gkr else None,
        )

    # Trace operands abstract (no alloc); everything else a real (tiny) traced
    # input so the round's eq/RLC are computed in-kernel, not folded as consts.
    traces = [ShapeDtypeStruct((ncs[i], nrs[i]), bf) for i in range(num_chips)]
    alphas = [jnp.zeros((ks[i],), ef) for i in range(num_chips)]
    lambdas = jnp.zeros((num_chips,), ef)
    zeta = jnp.zeros((num_vars,), ef)
    transcript = cheap_transcript(bf)
    beta = jnp.zeros((), ef)
    # Match the production signature: gkr_opening_claims returns a (num_chips,)
    # array, not a list of scalars — one input placeholder, same JIT graph.
    claims = jnp.zeros((num_chips,), ef)

    jf = jax.jit(f)
    print("[lower] tracing prove_jagged_zerocheck ...", flush=True)
    lowered = jf.lower(traces, alphas, lambdas, zeta, transcript, beta, claims)
    print("[compile] codegen ...", flush=True)
    t0 = time.perf_counter()
    lowered.compile()
    dt = time.perf_counter() - t0
    print(f"COMPILE prove_jagged_zerocheck [{','.join(names)}]: {dt:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
