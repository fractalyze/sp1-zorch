# SP1-vs-sp1-zorch per-stage shard-prove benchmark

How to compare sp1-zorch's GPU prover against SP1's native reference on a real
rsp shard, **per stage**, on the premise that **both prove the same shard, at the
same scope, and produce the same output** (golden byte-match). Only under that
premise is a wall-clock comparison meaningful.

> **Why the premise matters (a correction).** An earlier version of this doc
> compared SP1's *synthetic* `logup_gkr_bench` (random values, real heights) to
> sp1-zorch's real-shard `bench_sp1_logup_gkr`. That is **invalid**: different
> data, different scope (sp1-zorch includes the per-chip openings + grind/head;
> the SP1 synthetic bench does not), and **no golden equivalence** between the two
> — they never prove the same instance. Numbers from that approach (a "14×"
> logup-gkr ratio) are scope-confounded and should not be quoted. Use the
> per-stage full-prove comparison below instead.

## The valid benchmark — same data, same scope, same output

Both sides run the **full shard prove** (trace commit → LogUp-GKR → zerocheck →
jagged eval) on the **same** rsp shard, and the outputs are byte-identical
(verified), so per-stage wall-clocks are comparing the same computation.

### sp1-zorch side — `verify_prove_shard` (per-stage + golden)

```bash
ZKX_GPU_PLUGIN_PATH=<composite-capable plugin.so> JAX_PLATFORMS=cuda \
  bazel run //sp1_zorch/shard_prover:verify_prove_shard -- \
    --shard_dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --ffi_verify --runs=2
```

- Runs `prove_shard_chain` (the `ProveChain` of `TraceCommitRound` → LogUp-GKR →
  zerocheck → jagged-eval) on the real shard.
- `_TimedRound` prints **per-stage wall-clock** in ms: `[stage TraceCommitRound]
  X.Yms`, `[stage <LogUpGkr…>] X.Yms`, `[stage <Zerocheck…>] X.Yms`,
  `[stage <JaggedEval…>] X.Yms`. `--runs=2` proves twice in one process: pass 1
  is cold (XLA/zkx compiles), pass 2 is **warm** (executables reused) — compare
  the warm pass against SP1.
- **Golden**: the chain's commitment must equal the dump's `main_commit`
  (`gpu_commitment.txt`), the zerocheck point must equal `gpu_z_row.txt`, the
  jagged claim must equal `phase4_sumcheck_claim`, and with `--ffi_verify` the
  assembled bincode proof is byte-verified through SP1's `sp1_verify_shard` FFI
  (`SP1_JAX_FFI_LIB` → `libsp1_gpu_jax_ffi.so`). So sp1-zorch's output is
  byte-identical to SP1's — the same-output premise holds.

### SP1 native side — `riscv-witness/tools/sp1/sp1_shard_prover`

The `sp1-shard-test` bin (standalone crate under
`riscv-witness/tools/sp1/sp1_shard_prover/`) proves one shard and prints each
stage's wall-clock — `[stage commit traces]`, `[stage logup gkr proof]`,
`[stage zerocheck]`, `[stage prove evaluation claims]` — via a timing layer over
SP1's own `debug_span!`s (no `RUST_LOG` tuning). The four span names map 1:1 to
the table rows below. It has a **GPU** path and a **CPU** path.

**GPU — use this (same hardware as sp1-zorch).** `no-exec-gpu-dump --gpu` (build
`--features gpu`) loads the shard's SP1 GPU phase-dump (`<shard_dir>/gpu_traces/`
+ `gpu_vk.txt` + `gpu_commitment.txt`, written by SP1's GPU prover under
`SP1_DUMP_PHASES`) and runs SP1's GPU prover **ELF-free** — no executor, no
ELF/stdin. It byte-matches the dump (`preprocessed_commit` vs `gpu_vk.txt`,
`main_commitment` vs `gpu_commitment.txt`), so the same-output premise holds:

```bash
# from riscv-witness/tools/sp1/sp1_shard_prover/ (RTX 5090 = sm_120):
cargo run --release --features gpu -- \
  no-exec-gpu-dump <sp1_dumps>/rsp_21740136_sp1/shard17 --gpu
```

shard17 (GPU, byte-matched): **commit 16.6 / logup-gkr 19.9 / zerocheck 156.9 /
eval 41.1 ms; wall 234.8 ms** (the GPU NoExec path was added in
riscv-witness#1971).

**CPU — reference / parity only.** Without `--gpu` (or via `NoExec` / `Prove` with
an ELF + stdin) the tool uses `CpuShardProver`: useful as the injection-validity
/ byte-match reference, but **not** the same hardware as sp1-zorch's GPU
`verify_prove_shard`. Keep CPU stage times out of the GPU-vs-GPU table below.

## Per-stage comparison (shard17)

| stage | SP1 GPU | sp1-zorch GPU | ratio | golden |
|---|---|---|---|---|
| trace commit | 16.6 ms | | | byte-match |
| LogUp-GKR | 19.9 ms | | | byte-match |
| zerocheck | **156.9 ms** | **218 ms** | **1.39×** | byte-match |
| jagged eval (PCS open) | 41.1 ms | | | byte-match |

The SP1 GPU column is from `no-exec-gpu-dump --gpu` above (warm, byte-matched).
The sp1-zorch zerocheck is the eq-fold-OFF baseline via
`//sp1_zorch/zerocheck:bench_sp1_zerocheck` (218 ms wall / 166.6 ms
`cuda_gpu_kern_sum`); fill the other sp1-zorch rows from a paired warm
`verify_prove_shard` run. The PCS opening proof IS in scope, so every stage is on
equal footing — same shard, same per-stage scope, byte-identical output, and now
**same hardware (GPU both sides)**. The 1.39× zerocheck gap is real and on-GPU:
sp1-zorch runs ~41 de-fused kernels/round vs SP1's one fused multi-block
`jaggedConstraintPolyEval`.

## Devenv provenance — measure shipped code

A bench or byte-match is only a baseline if it runs the code the team **ships**.
This devenv drives `zorch` through a dev-only `.bazelrc.user`
`--override_module=zorch=<checkout>` and (historically) a patched copy of the
pip-extracted `jax` — both can silently point at stale or superseded code.
sp1-zorch#153's first encode baseline was taken against a `zorch` override weeks
behind `origin/main` (predating the #220/#225 encode rework) and misread as the
shipped number; post-zkx#756 the shipped encode path is the GS/DIF rewriter on
**stock** `lax.bit_reverse(lax.fft(x))`, so a leftover override on the dead
`perf/encode-ntt-shard0` branch + a `bit_reverse_output`-patched jax would
measure a path we no longer ship.

The trace-commit tools (`bench_trace_commit`, `verify_trace_commit`) run a
preflight (`sp1_zorch/commit/bench_preflight.py`) that prints what they actually
loaded and warns when it is stale:

```text
=== sp1-zorch bench devenv provenance ===
  zorch : OVERRIDE /home/me/zorch @ main (abc1234)
  jax   : 0.0.5.dev20260622122155
  zkx   : /path/to/pjrt_c_api_gpu_plugin.so
  (clean: measuring shipped code)
```

It flags a `zorch` override that is off the `origin/main` lineage, behind
`origin/main`, or dirty, and a `jax` whose fft still carries the superseded
`bit_reverse_output` patch. **Capture a baseline only from a clean banner.** Make
a stale devenv a hard error with `--strict_devenv` (`verify_trace_commit`) or
`SP1_BENCH_STRICT_DEVENV=1` (`bench_trace_commit`). To clear it: point the
`.bazelrc.user` `@zorch` override at an `origin/main` checkout (or drop it for the
pinned wheel) and use stock jax.

## Shard size caveat (still applies)

A block's shards differ in size by >30×: for `rsp_21740136`, shard0 = 38.6 M
first-layer rows, shard17 = 1.16 M (`gpu_first_layer.txt: height`). Always run
**both provers on the same `--shard_dir`**; never compare across shards. (A
relayed "SP1 ~81 ms" was shard0; an earlier sp1-zorch number was shard17.)
