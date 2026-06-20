# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Isolated phase benchmark for the jagged PCS evaluation stage.

Runs ONLY the jagged-eval sumcheck half (``JaggedEvalRound``) on the
``gpu_fibonacci`` fixture, so this stage can be iterated on without compiling or
running the whole shard proof (it is the proof's last stage, so a full run is a
slow way to reach it). The stage is split into its sub-phases so a measurement
shows which one dominates:

* ``outer_indicator`` -- ``build_outer_indicator``: materialize ``J̃`` over the
  ``2^n_d`` dense hypercube (one eager allocation).
* ``outer``           -- ``outer_sumcheck``: the LSB-first Hadamard sumcheck,
  ``log2(len(dense))`` rounds.
* ``inner``           -- ``inner_sumcheck``: the branching-program sumcheck,
  ``2·n_d`` rounds.
* ``total``           -- the whole ``JaggedEvalRound``.

Each op is timed cold (first call = trace + compile + run) and warm (mean of the
remaining calls = dispatch + execute), so the compile share is the gap. The
transcript is the production duplex challenger (``replay.fresh_transcript``,
poseidon2-koalabear16 rate 8), so the per-round Fiat-Shamir cost is counted the
way the real prove pays it.

    JAX_PLATFORMS=cuda bazel run -c opt \
        //sp1_zorch/jagged:bench_jagged_eval_phases -- --ops=total
    JAX_PLATFORMS=cuda bazel run -c opt \
        //sp1_zorch/jagged:bench_jagged_eval_phases -- \
        --ops=outer_indicator,outer,inner,total --iters=5

``JAX_PLATFORMS=cuda`` is load-bearing: without it jax silently falls back to CPU
(the plugin deps don't force the backend), giving misleading "GPU" timings.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from zk_dtypes import efinfo
from zk_dtypes import koalabear_mont as BF
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.jagged.prover import (
    JaggedEvalInputs,
    JaggedEvalRound,
    assemble_columns,
    build_outer_indicator,
    inner_sumcheck,
    outer_sumcheck,
    outer_sumcheck_claim,
)
from sp1_zorch.shard_prover.replay import fresh_transcript, from_u32
from zorch.pcs.jagged.poly import build_jagged_layout

_FIXTURE = Path(__file__).parent / "testdata" / "gpu_fibonacci"
# The packed dense lives with the zerocheck fixture (same shard); see
# jagged/prover_test.py for why it is reconstructed rather than re-dumped.
_ZC_INPUTS = (
    Path(__file__).parent.parent / "zerocheck" / "testdata" / "gpu_fibonacci" / "inputs"
)


def _raw_area(round_meta: dict) -> int:
    """Σ row_count·column_count -- the round's unpadded packed-dense length."""
    return sum(
        int(r) * int(c)
        for r, c in zip(
            round_meta["row_counts"], round_meta["column_counts"], strict=True
        )
    )


def _synthetic_ef(shape: tuple[int, ...], seed: int) -> jax.Array:
    """Deterministic EF array of the given shape (values are irrelevant to timing;
    only shapes drive the sumcheck cost)."""
    n = int(np.prod(shape, dtype=np.int64))
    vals = ((np.arange(n, dtype=np.uint32) + seed) % (1 << 20) + 1).reshape(shape)
    u = np.zeros((*shape, efinfo(EF).degree), np.uint32)
    u[..., 0] = vals
    return from_u32(u, EF)


def load_inputs_from_heights(start_indices_file: Path) -> JaggedEvalInputs:
    """Build shard-scale ``JaggedEvalInputs`` from a real dump's per-column
    heights (``data_start_indices.bin``: the cumulative dense offsets, so the
    per-column heights are its consecutive differences). Challenges and the dense
    buffer are synthetic — the jagged-eval cost is a function of the shapes
    (``L``, ``n_d``, ``n_r``), not the field values, so this faithfully times the
    real shard scale without needing the full processed fixture."""
    si = np.fromfile(start_indices_file, dtype=np.uint32).astype(np.int64)
    col_heights = [int(x) for x in np.diff(si)]
    L = len(col_heights)
    n_r = max(1, int(max(col_heights)).bit_length())
    _, cfg = build_jagged_layout(col_heights, L, n_r, EF)
    n_d, n_c = cfg.n_d, cfg.n_c
    # iota-mod base buffer: non-degenerate values so the fold is not optimized away.
    dense = from_u32(np.arange(1 << n_d, dtype=np.uint32) % (1 << 20) + 1, BF)
    return JaggedEvalInputs(
        col_heights=tuple(col_heights),
        all_claims=_synthetic_ef((L,), 3),
        z_row=_synthetic_ef((n_r,), 11),
        z_col=_synthetic_ef((n_c,), 23),
        dense=dense,
    )


def load_inputs() -> JaggedEvalInputs:
    """Build ``JaggedEvalInputs`` from the gpu_fibonacci fixture, matching
    jagged/prover_test.py's construction (real ``z_col``, reconstructed ``D``)."""
    meta = json.loads((_FIXTURE / "meta.json").read_text())
    row_counts_rounds = [[int(x) for x in r["row_counts"]] for r in meta["rounds"]]
    column_counts_rounds = [
        [int(x) for x in r["column_counts"]] for r in meta["rounds"]
    ]
    z_row = from_u32(np.load(_FIXTURE / "inputs" / "z_row.npy"), EF)
    claims = [
        from_u32(np.load(_FIXTURE / "inputs" / f"claims_r{r}.npy"), EF)
        for r in range(len(meta["rounds"]))
    ]
    z_col = from_u32(np.load(_FIXTURE / "outputs" / "challenges.npz")["z_col"], EF)
    prep = from_u32(
        np.load(_ZC_INPUTS / "prep_dense.npy")[: _raw_area(meta["rounds"][0])], BF
    )
    main = from_u32(
        np.load(_ZC_INPUTS / "main_dense.npy")[: _raw_area(meta["rounds"][1])], BF
    )
    dense = jnp.concatenate([prep, main])
    col_heights, all_claims = assemble_columns(
        row_counts_rounds, column_counts_rounds, claims, dtype=EF
    )
    return JaggedEvalInputs(
        col_heights=tuple(col_heights),
        all_claims=all_claims,
        z_row=z_row,
        z_col=z_col,
        dense=dense,
    )


def _time_op(name: str, fn, iters: int) -> None:
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    cold_ms = (time.perf_counter() - t0) * 1e3
    warm = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        warm.append((time.perf_counter() - t0) * 1e3)
    warm_ms = sum(warm) / len(warm) if warm else float("nan")
    print(
        f"  {name:16s}  cold={cold_ms:10.1f} ms   warm={warm_ms:10.1f} ms"
        f"   (warm n={iters})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops",
        default="outer_indicator,outer,inner,total",
        help="comma-separated phases to time",
    )
    parser.add_argument("--iters", type=int, default=3, help="warm iterations")
    parser.add_argument(
        "--measure",
        default="warm",
        choices=["warm", "compile"],
        help="warm = per-phase wall-clock; compile = @jit trace+compile of inner",
    )
    parser.add_argument(
        "--heights-file",
        default=None,
        help="path to a dump's data_start_indices.bin; times the real shard scale "
        "(synthetic challenges/dense) instead of the gpu_fibonacci fixture",
    )
    args = parser.parse_args()

    print(f"jax {jax.__version__}  devices={jax.devices()}")
    if args.heights_file:
        carry = load_inputs_from_heights(Path(args.heights_file))
        source = args.heights_file
    else:
        carry = load_inputs()
        source = "fixture gpu_fibonacci"
    n_d = (carry.dense.shape[0] - 1).bit_length()
    print(
        f"{source}: L={len(carry.col_heights)} columns  "
        f"len(D)={carry.dense.shape[0]}  n_d={n_d}  "
        f"outer_rounds={n_d}  inner_rounds={2 * n_d}"
    )

    # Intermediates so each sub-phase runs in isolation.
    claim = outer_sumcheck_claim(carry.all_claims, carry.z_col)
    indicator = build_outer_indicator(
        carry.col_heights, carry.z_row, carry.z_col, carry.dense.shape[0], dtype=EF
    )
    _, z_final, _, _ = outer_sumcheck(carry.dense, indicator, claim, fresh_transcript())

    ops = {
        "outer_indicator": lambda: build_outer_indicator(
            carry.col_heights, carry.z_row, carry.z_col, carry.dense.shape[0], dtype=EF
        ),
        "outer": lambda: outer_sumcheck(
            carry.dense, indicator, claim, fresh_transcript()
        ),
        "inner": lambda: inner_sumcheck(
            carry.col_heights, carry.z_row, carry.z_col, z_final, fresh_transcript(),
            dtype=EF,
        ),
        "total": lambda: JaggedEvalRound(dtype=EF)(carry, fresh_transcript()),
    }

    if args.measure == "compile":
        # Isolate trace+lower+compile from execution: col_heights rides as a
        # closure constant (static), so the @jit traces the round structure that
        # XLA then compiles. Rolling the per-round Python loops into lax.scan
        # collapses this trace from O(rounds) unrolled bodies to O(1).
        heights = carry.col_heights

        def _inner(z_row, z_col, z_trace, transcript):
            return inner_sumcheck(heights, z_row, z_col, z_trace, transcript, dtype=EF)

        jfn = jax.jit(_inner)
        targs = (carry.z_row, carry.z_col, z_final, fresh_transcript())
        t0 = time.perf_counter()
        lowered = jfn.lower(*targs)
        t1 = time.perf_counter()
        lowered.compile()
        t2 = time.perf_counter()
        print(
            f"  inner_sumcheck @jit compile:  trace+lower={ (t1 - t0) * 1e3:8.0f} ms"
            f"   xla_compile={ (t2 - t1) * 1e3:8.0f} ms"
            f"   total={ (t2 - t0) * 1e3:8.0f} ms"
        )
        return

    for name in args.ops.split(","):
        name = name.strip()
        if name not in ops:
            raise SystemExit(f"unknown op {name!r}; choose from {list(ops)}")
        _time_op(name, ops[name], args.iters)


if __name__ == "__main__":
    main()
