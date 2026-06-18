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
  is cold (XLA/ZKX compiles), pass 2 is **warm** (executables reused) — compare
  the warm pass against SP1.
- **Golden**: the chain's commitment must equal the dump's `main_commit`
  (`gpu_commitment.txt`), the zerocheck point must equal `gpu_z_row.txt`, the
  jagged claim must equal `phase4_sumcheck_claim`, and with `--ffi_verify` the
  assembled bincode proof is byte-verified through SP1's `sp1_verify_shard` FFI
  (`SP1_JAX_FFI_LIB` → `libsp1_gpu_jax_ffi.so`). So sp1-zorch's output is
  byte-identical to SP1's — the same-output premise holds.

### SP1 native side — `riscv-witness/tools/sp1/sp1_shard_prover`

Run from `riscv-witness/tools/sp1/sp1_shard_prover/` (standalone crate; the bin
is `sp1-shard-test`):

```bash
# Prove the same shard's C++ traces (no SP1 re-execution), per-stage timed:
cargo run --release -- NoExec <dir with per-shard C++ traces> <elf> <stdin>
# Or stream-execute the ELF and prove every shard (read the shard's line):
cargo run --release -- Prove <elf> <stdin>
```

- The tool prints each prove stage's wall-clock the same way the sp1-zorch side
  does — `[stage commit traces] X.Yms`, `[stage logup gkr proof] X.Yms`,
  `[stage zerocheck] X.Yms`, `[stage prove evaluation claims] X.Yms` — via a
  timing layer over SP1's own `debug_span!`s (no `RUST_LOG` tuning needed). The
  four span names map 1:1 to the table rows below. The shard must be the
  **same** one the dump above came from (same block, same shard index); `NoExec`
  proves exactly one shard, so its stage lines are unambiguous.
- **CPU caveat.** Every subcommand instantiates `CpuShardProver`, so these are
  SP1's **CPU** per-stage times, whereas sp1-zorch's `verify_prove_shard` runs
  on **GPU**. The comparison is same-shard / same-scope but **not**
  same-hardware — read the per-stage *shape* (where the time goes), not a raw
  GPU-vs-CPU ratio, until a GPU SP1 full-prove path is wired in.

## Per-stage comparison (fill from a paired run)

| stage | SP1 native | sp1-zorch | ratio | golden |
|---|---|---|---|---|
| trace commit | | | | byte-match |
| LogUp-GKR | | | | byte-match |
| zerocheck | | | | byte-match |
| jagged eval (PCS open) | | | | byte-match |

The PCS opening proof IS in scope here (unlike the per-stage micro-benches), so
every stage is on equal footing: same shard, same per-stage scope, byte-identical
output — the only **scope-honest** comparison. Mind the CPU caveat above, though:
the SP1 column is CPU and the sp1-zorch column GPU, so the `ratio` mixes hardware
with prover efficiency until a GPU SP1 full-prove path lands. Read the per-stage
shape first; quote a raw ratio only once both sides are on the same hardware.

## Shard size caveat (still applies)

A block's shards differ in size by >30×: for `rsp_21740136`, shard0 = 38.6 M
first-layer rows, shard17 = 1.16 M (`gpu_first_layer.txt: height`). Always run
**both provers on the same `--shard_dir`**; never compare across shards. (A
relayed "SP1 ~81 ms" was shard0; an earlier sp1-zorch number was shard17.)

## Reporting discipline — the provenance every number must carry

The premise above (same data / same scope / same output) is what makes a
*cross-implementation* comparison valid. Two further rules keep any single number
**citable and comparable across sessions, commits, and people**.

**1. Execution regime: eager ≠ jit'd — never compare across them.** A bench that
drives a prover stage directly (e.g. `bench_sp1_logup_gkr`) runs it **without an
outer `@jit`** by default. An eager warm number is **host-dispatch-bound**: the
wall-clock is the host orchestrating per-stage / per-transition dispatches (GPU can
sit ~0 % util with zero compiles in the timed window), not device sumcheck work. For
the *same* computation it can be 100×+ a device-bound (outer-`@jit`'d) number, so the
two **must never be subtracted or ratio'd**. A fusion/dispatch lever (e.g. a
per-transition `@jit` island) can even *invert* the ordering between the two regimes —
so measure such a lever under the regime it actually ships in.

**2. Every reported number carries its provenance tuple.** A number missing any field
is not comparable and should not be quoted:

- **bench target + file** — the canonical bench for that stage; anything else is
  non-canonical by definition.
- **shard identity** — dump name + shard index. Never compare across shards (see the
  size caveat above).
- **scope** — which stages, and whether openings / grind / head are in or out.
- **warm or cold** — and, if warm, the warmup count. Never report a cold number as warm.
- **execution mode** — eager or outer-`@jit` (rule 1).
- **A/B variable held** — exactly what changed, with everything else pinned. For a
  zorch-commit A/B: the two zorch SHAs, with the JAX wheel **and** the ZKX plugin
  build pinned identical across arms (ideally byte-identical — verify by `sha256`).
- **plugin + pin** — ZKX plugin provenance. A from-source `treatment.so` swapped via
  `ZKX_GPU_PLUGIN_PATH` is **not interchangeable** with a bumped-pin wheel number.
- **device** — the GPU model, plus a same-shell `nvidia-smi` before/after for any
  memory/perf-sensitive run.
- **golden status** — whether byte-match was verified for the measured config
  (`verify_gkr_prove`, `--ffi_verify`).

**3. A merged perf-lever change needs a completed warm A/B first.** "Byte-identical by
construction" is a *correctness* claim, not a perf measurement. A code or pin change
merged as a performance lever must have a completed warm A/B — carrying the tuple above
— posted to its tracking issue **before** the lever is treated as quantified. Deferring
the measurement past merge is how an unmeasured (or silently regressing) "optimization"
lands.
