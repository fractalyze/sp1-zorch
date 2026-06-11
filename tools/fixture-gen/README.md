# `tools/fixture-gen/` — SP1-reference byte-match fixture generator

Reproducibly (re)generates the `gpu_fibonacci` byte-match fixtures from the
**pinned SP1 reference fork**, so the committed golden has a single authoritative
provenance instead of an ad-hoc hand-copied dump. Two steps: a cargo-built dump
(outside Bazel, GPU + SP1 toolchain) and a deterministic Python convert.

## Pin

SP1 fork `fractalyze/sp1` @ `ref` =
`7abe1a690d2a71958f95c237166460c1a9e287cc`. The fork carries the
`SP1_DUMP_PHASES` / `SP1_DUMP_ZC_BUFFERS` instrumentation (absent from stock
succinctlabs/sp1) — including the open's `basefold_*` Fiat-Shamir dumps — so it
must be pinned. Determinism: the prove forces FRI `pow_bits=0` under the dump env
vars, so re-runs are byte-identical.

## Step 1 — dump (cargo, outside Bazel; needs CUDA GPU + SP1 toolchain)

```sh
# isolated worktree of the pinned fork:
git -C <sp1-clone> fetch <fork-remote> ref
git -C <sp1-clone> worktree add --detach <wt> 7abe1a690d2a71958f95c237166460c1a9e287cc
cd <wt>
export CUDA_HOME=/usr/local/cuda-12.9 CUDA_PATH=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_ARCHS=120                 # sm_120 (RTX 5090); 89 Ada / 90 Hopper
export SP1_DUMP_PHASES=<dump_dir> SP1_DUMP_ZC_BUFFERS=<dump_dir>
cargo test -p sp1-gpu-shard-prover --release tests::test_prove_shard_fibonacci -- --nocapture
```

Emits `gpu_evaluation_proof.json` (the `JaggedPcsProof`), `gpu_shard_proof.json`,
`phase3_*.txt` / `phase4_*.txt`, `data_dense_full.bin` (the packed dense `D`,
Montgomery), the per-chip `gpu_traces/`, and the open's Fiat-Shamir challenges
`basefold_{batch_challenges,fri_betas,pow_witness,query_indices}.txt` (these last
four come from the fork's `fri.rs` dump instrumentation — they aren't in the
proof, but the open replay needs them).

## Step 2 — convert (deterministic, no GPU)

```sh
# jagged-eval + stacked-open fixture (sp1_zorch/jagged/testdata/gpu_fibonacci):
python tools/fixture-gen/convert.py --dump <dump_dir> --out <jagged_fixture_dir>
# zerocheck fixture + the shared dense (sp1_zorch/zerocheck/testdata/gpu_fibonacci):
python tools/fixture-gen/convert.py --dump <dump_dir> --zerocheck-out <zc_fixture_dir>
```

`--out` emits the jagged-eval pieces **and** the stacked-open pieces (the open
augments the jagged dir's `meta.json` / `challenges.npz`); `--zerocheck-out`
emits the zerocheck chip schedule, transcript challenges, regions, and the shared
dense. The three load-bearing conversions (each verified by byte-match against
the dump's own golden) are documented in `convert.py`:

1. SP1 JSON/txt field elements are **canonical** → convert canonical→Montgomery;
   the `.bin` device buffers are already Montgomery.
2. The sumcheck `point_and_eval[0]` is stored **reversed** vs the round feed
   order — `challenges.npz` carries feed order, `*_point` goldens the
   round-emitted (non-reversed) point.
3. `dense_eval` golden = `expected_eval`, not `point_and_eval[1]`.

The zerocheck `chip_claims` are not dumped directly — they are the beta-power
weighting of each chip's `[main | prep]` GKR openings (`logup_evaluations.chip_openings`),
recomputed in the converter exactly as `sp1_zorch/zerocheck/stage.py` does.
`fri_raw_roots` is likewise not emitted: it is zorch's pre-binding Merkle digest,
an internal artifact with no SP1 reference (SP1 stores only the separator-bound
`fri_commitments`).

## Bazel wiring

Pending (this slice lands the converter + recipe). The cargo dump step is wired
via the riscv-witness `requires-sp1-toolchain` pattern (cargo build outside
Bazel; a `genrule` copies the prebuilt binary; CI runs
`--test_tag_filters=-requires-sp1-toolchain` → build-only on PR, run post-merge).
The converter emits all three stages (jagged via `--out`, zerocheck via
`--zerocheck-out`, open folded into `--out`); the **unified atomic refresh** of
the committed `gpu_fibonacci` to direct-SP1 provenance is tracked on #76 and
lands once the jagged generator (#72) merges.
