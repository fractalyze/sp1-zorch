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
    --shard_dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --ffi_verify
```

- Runs `prove_shard_chain` (the `ProveChain` of `TraceCommitRound` → LogUp-GKR →
  zerocheck → jagged-eval) on the real shard.
- `_TimedRound` prints **per-stage wall-clock**: `[stage TraceCommitRound] X.Ys`,
  `[stage <LogUpGkr…>] X.Ys`, `[stage <Zerocheck…>] X.Ys`, `[stage <JaggedEval…>]
  X.Ys`. (First run is cold-compile; warm via the JAX compilation cache or a
  second pass.)
- **Golden**: the chain's commitment must equal the dump's `main_commit`
  (`gpu_commitment.txt`), the zerocheck point must equal `gpu_z_row.txt`, the
  jagged claim must equal `phase4_sumcheck_claim`, and with `--ffi_verify` the
  assembled bincode proof is byte-verified through SP1's `sp1_verify_shard` FFI
  (`SP1_JAX_FFI_LIB` → `libsp1_gpu_jax_ffi.so`). So sp1-zorch's output is
  byte-identical to SP1's — the same-output premise holds.

### SP1 native side — `riscv-witness/tools/sp1/sp1_shard_prover`

```bash
RUST_LOG=info cargo run --release -p sp1-shard-test --bin sp1_shard_prover -- \
  Prove --traces=<dir with per-shard C++ traces> ...
```

- The streaming native SP1 prover (proves one shard, verifies, drops). Per-stage
  timing comes from SP1's own tracing spans inside the hypercube prover —
  `debug_span!("trace commit")`, `debug_span!("logup gkr proof")`,
  `debug_span!("zerocheck …")`, `prove_evaluations` (jagged opening). Run with a
  span-timing `tracing-subscriber` (or `RUST_LOG` + the tool's timing) and read
  the per-span durations. The shard must be the **same** one the dump above came
  from (same block, same shard index).

## Per-stage comparison (fill from a paired run)

| stage | SP1 native | sp1-zorch | ratio | golden |
|---|---|---|---|---|
| trace commit | | | | byte-match |
| LogUp-GKR | | | | byte-match |
| zerocheck | | | | byte-match |
| jagged eval (PCS open) | | | | byte-match |

The PCS opening proof IS in scope here (unlike the per-stage micro-benches), so
every stage is on equal footing. Same shard, same per-stage scope, byte-identical
output — this is the only ratio worth quoting.

## Shard size caveat (still applies)

A block's shards differ in size by >30×: for `rsp_21740136`, shard0 = 38.6 M
first-layer rows, shard17 = 1.16 M (`gpu_first_layer.txt: height`). Always run
**both provers on the same `--shard_dir`**; never compare across shards. (A
relayed "SP1 ~81 ms" was shard0; an earlier sp1-zorch number was shard17.)

## Secondary tool — `sp1_zorch/tools/sp1_baseline.py`

Extracts a shard's real first-layer counts / commit dims into SP1's *synthetic*
micro-bench inputs (`logup_gkr_bench` / `merkle_bench`). Useful as a
**structural** cross-check (does SP1's kernel cost scale as expected for this
shard's shape) — but it is **not** the apples-to-apples benchmark (synthetic
values, single-stage scope, no golden equivalence). The per-stage full-prove
comparison above is the source of truth.
