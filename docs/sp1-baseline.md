# SP1-vs-sp1-zorch wall-clock baseline (logup-gkr + trace commit)

Reproducible procedure to compare sp1-zorch's GPU prover against SP1's native
reference prover (`sp1-gpu`) on the **same** rsp shard, per stage. Both provers
must run the same shard shape; the helper `sp1_zorch/tools/sp1_baseline.py`
extracts each shard's real dimensions into SP1's bench-input formats.

> **Read this first — same-shard only.** A block's shards differ in size by
> >30×. For `rsp_21740136`: shard0 = 38.6 M first-layer rows, shard17 = 1.16 M
> (`gpu_first_layer.txt: height`, col_h units). Comparing sp1-zorch on one shard
> to SP1 on another is meaningless — always extract and run both on the *same*
> `--shard-dir`. (A relayed "SP1 ~81 ms" that turned out to be shard0 vs our
> shard17/223 ms is what motivated this doc.)

## 0. Prerequisites

- A shard dump, e.g. `/data/sp1_dumps/rsp_21740136_sp1/shard17`.
- SP1 reference checkout with `sp1-gpu` built (CUDA in `PATH`):
  `cargo build --release -p sp1-gpu-logup-gkr --bin logup_gkr_bench`
  and `-p sp1-gpu-merkle-tree --bin merkle_bench`.
- For the sp1-zorch side: a composite-capable plugin via `ZKX_GPU_PLUGIN_PATH`
  and `JAX_PLATFORMS=cuda` (the pinned wheel cannot legalize the poseidon2 / NTT
  composites). Capture `nvidia-smi` in-band; the GPU is shared.

## 1. Extract the shard's SP1 bench inputs (CPU, no prove)

```sh
JAX_PLATFORMS=cpu PYTHONPATH="$PWD:/abs/path/to/zorch" \
  python sp1_zorch/tools/sp1_baseline.py --shard-dir=<dump>/shard17 --stage logup_gkr
#  -> layer_workloads_shard17.json   (col_h interaction_row_counts, validated vs dump height)
JAX_PLATFORMS=cpu PYTHONPATH="$PWD:/abs/path/to/zorch" \
  python sp1_zorch/tools/sp1_baseline.py --shard-dir=<dump>/shard17 --stage commit
#  -> merkle_bench --sizes=23 --width=18
```

## 2. logup-gkr

**SP1 native** (real jagged structure; random `Felt` numerators = base-field
faithful). Copy the extracted JSON over SP1's input and run:

```sh
cp layer_workloads_shard17.json <sp1-gpu>/crates/logup_gkr/layer_workloads.json
cd <sp1-gpu> && ./target/release/logup_gkr_bench   # prints trace_gen + proof_gen
```

**sp1-zorch** (the production jagged prover, full stage incl. chip openings):

```sh
ZKX_GPU_PLUGIN_PATH=<plugin.so> JAX_PLATFORMS=cuda \
  bazel run -c opt //sp1_zorch/logup_gkr:bench_sp1_logup_gkr -- --shard-dir=<dump>/shard17
```

## 3. trace commit (SMCS Merkle)

The apples-to-apples is the SMCS Merkle commit (sp1-zorch#2's scope). SP1's
`merkle_bench` takes `--sizes=log2(num_leaves) --width=K` (the `--width` flag is
a one-line add to `merkle_bench.rs`, mirroring `--seed`; without it, rebuild with
`const WIDTH`). The RS-encode (NTT) is a separate axis (zkx `rs_encode` vs SP1's
NTT bench) and is not covered here.

```sh
cd <sp1-gpu> && ./target/release/merkle_bench --sizes=23 --width=18   # SP1 commit
# sp1-zorch: the smcs_commit op of bench_trace_commit (composite plugin):
ZKX_GPU_PLUGIN_PATH=<plugin.so> JAX_PLATFORMS=cuda \
  bazel run //sp1_zorch/commit:bench_trace_commit -- --shard_dir=<dump>/shard17 --phase runtime
```

## 4. Recorded baseline — `rsp_21740136`, RTX 5090, 2026-06-18

| stage | shard | rows / dims | SP1 native | sp1-zorch | ratio |
|---|---|---|---|---|---|
| logup-gkr (trace+proof) | shard17 | 1.16 M | **15.8 ms** (proof 15.1) | **~223 ms** | **~14×** |
| logup-gkr (trace+proof) | shard0 | 38.6 M | **57.9 ms** (proof 47.0) | OOM† | — |
| commit (SMCS Merkle, warm) | shard17 | 2²³ leaves × 18 | **12.6 ms** | **21.2 ms** | **1.68×** |
| commit (SMCS Merkle, warm) | shard0 | 2²³ leaves × 188 | **49.4 ms** | (run §3) | — |

- SP1 logup-gkr is one cold run (no warmup) — warm would be ≤, so the gap is a
  lower bound. SP1 scales sub-linearly (33× rows → 3.7× time). sp1-zorch's
  shard17 223 ms is fixed-overhead-bound (eager 20-layer circuit build + XLA
  dispatch + de-fusion), not the sumcheck math, which matches SP1 (eq
  factorization, Gruen 2-eval, base-field numerator all present).
- † sp1-zorch cannot prove shard0 with the current rolled harness unless the
  producer carries the `prove_jagged_pyramid` streaming memory bound (zorch#275);
  otherwise the stacked planes OOM a 32 GB GPU.
- The sp1-zorch commit is the `smcs_commit` (Merkle) op; the full `commit_region`
  is 33.96 ms (rs_encode/NTT 5.09 + smcs_commit/Merkle 21.19 + bind). At real
  shard17 scale the Merkle commit is **1.68×** SP1 — a 2¹⁶ toy reads ~parity
  (0.78 vs 0.71 ms, sp1-zorch#2) but understates it; scale matters. Commit's 1.68×
  is far smaller than logup-gkr's 14×, so logup-gkr is the dominant gap.
  shard0 sp1-zorch commit not yet measured (run §3).

## 5. Scope caveats
- SP1 `logup_gkr_bench` / `merkle_bench` use random values with the real shard
  *shape*; values don't change the op count, so timing is faithful (numerators
  are `Felt` = base field, exercising the small-value path).
- SP1's logup-gkr bench has no chip openings (synthetic GKR); sp1-zorch's
  `bench_sp1_logup_gkr` includes them. They are not ~200 ms, so the ~14×
  order of magnitude holds, but the two scopes are not identical.
- Commit comparison is Merkle-only; the NTT/RS-encode is a separate axis.
