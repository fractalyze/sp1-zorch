# `tools/fixture-gen/` — SP1-reference byte-match fixture generator

Reproducibly (re)generates the `gpu_fibonacci` byte-match fixtures from the
**pinned SP1 reference fork**, so the committed golden has a single authoritative
provenance instead of an ad-hoc hand-copied dump. Two steps: a cargo-built dump
(outside Bazel, GPU + SP1 toolchain) and a deterministic Python convert.

## Pin

SP1 fork `fractalyze/sp1` @ `ref` =
`945c23892db106d75686d4a63588d4f8106a6069`. The fork carries the
`SP1_DUMP_PHASES` / `SP1_DUMP_ZC_BUFFERS` instrumentation (absent from stock
succinctlabs/sp1), so it must be pinned. Determinism: the prove forces FRI
`pow_bits=0` under the dump env vars, so re-runs are byte-identical.

## Step 1 — dump (cargo, outside Bazel; needs CUDA GPU + SP1 toolchain)

```sh
# isolated worktree of the pinned fork:
git -C <sp1-clone> fetch <fork-remote> ref
git -C <sp1-clone> worktree add --detach <wt> 945c23892db106d75686d4a63588d4f8106a6069
cd <wt>
export CUDA_HOME=/usr/local/cuda-12.9 CUDA_PATH=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_ARCHS=120                 # sm_120 (RTX 5090); 89 Ada / 90 Hopper
export SP1_DUMP_PHASES=<dump_dir> SP1_DUMP_ZC_BUFFERS=<dump_dir>
cargo test -p sp1-gpu-shard-prover --release tests::test_prove_shard_fibonacci -- --nocapture
```

Emits `gpu_evaluation_proof.json` (the `JaggedPcsProof`), `phase4_*.txt`,
`data_dense_full.bin` (the packed dense `D`, Montgomery), and the per-chip
`gpu_traces/`.

## Step 2 — convert (deterministic, no GPU)

```sh
python tools/fixture-gen/convert.py --dump <dump_dir> --out <fixture_dir>
```

Emits the `gpu_fibonacci` layout (`meta.json`, `inputs/`, `outputs/`). The three
load-bearing conversions (each verified by byte-match against the dump's own
golden) are documented in `convert.py`:

1. SP1 JSON/txt field elements are **canonical** → convert canonical→Montgomery;
   the `.bin` device buffers are already Montgomery.
2. The sumcheck `point_and_eval[0]` is stored **reversed** vs the round feed
   order — `challenges.npz` carries feed order, `*_point` goldens the
   round-emitted (non-reversed) point.
3. `dense_eval` golden = `expected_eval`, not `point_and_eval[1]`.

## Bazel wiring

Pending (this slice lands the converter + recipe). The cargo dump step is wired
via the riscv-witness `requires-sp1-toolchain` pattern (cargo build outside
Bazel; a `genrule` copies the prebuilt binary; CI runs
`--test_tag_filters=-requires-sp1-toolchain` → build-only on PR, run post-merge).
The zerocheck + basefold-open halves of the shared fixture, and the unified
atomic refresh, are tracked on #76.
