# Development guide

Everything needed to build, test, and benchmark sp1-zorch: the environment
setup, the test conventions, the local shard-prove harness, and the
reproducible per-phase baseline against SP1. For architecture (Stage / Round) see
[architecture.md](architecture.md); for coding style see
[conventions.md](conventions.md).

## Development environment

Pure Python on frx (Field, Ring Accelerated), run against the
Fractalyze XLA GPU plugin. Bazel 9 (bzlmod). `sp1-zorch` consumes `zorch` as a
Bazel module, pinned in `MODULE.bazel` via `git_override` for reproducible
builds.

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
bazel test //...                 # hermetic, sandboxed; FRX_PLATFORMS=cpu default
```

For iterative dev outside Bazel — source the venv, then put the sp1-zorch and a
local `zorch` checkout on the path:

```sh
export PYTHONPATH="$PWD:/abs/path/to/zorch"
```

**Dev against a local `zorch` checkout** instead of the pinned commit — add to
`.bazelrc.user` (gitignored, holds an absolute path):

```text
common --override_module=zorch=/abs/path/to/your/zorch/checkout
```

**Bumping the `zorch` pin is a coupled change.** Keep the pin on `main`
commits so CI is reproducible, and move the frx family (`frx` = Fractalyze
Field, Ring Accelerated) in `requirements.in` + `requirements_lock_3_11.txt` to
match zorch's pin **in the same commit** — the two build against a shared
frxlib, so a lagging frx pin ABI-mismatches and **segfaults** the GPU tests
(`verify_shard_test`) rather than raising a clean `ImportError`. `sp1-zorch
main` is the reference for the matching `(zorch pin, frx)` pair.

**GPU-plugin gotcha.** A `py_binary` GPU runnable must dep
`requirement("frx_cuda12_plugin")` + `requirement("frx_cuda12_pjrt")` or frx
**silently falls back to CPU**. Run with `FRX_PLATFORMS=cuda` so a missing plugin
errors instead of silently degrading (`gpu` is wrong: it also initializes rocm
and dies). The Fractalyze XLA plugin loader takes no plugin-path env var; to
measure a locally built plugin you overwrite the wheel's bundled
`xla_cuda_plugin.so` — see [Measure shipped code](#measure-shipped-code) below.

### Bazel gotchas

- **Never run a second bazel command in a worktree that has a long one in
  flight.** The output base is a hash of the workspace path, so *different*
  worktrees are safely concurrent — but the same one shares a server and the
  second command interrupts the first. A `bazel run … --help` will kill a
  running benchmark. If you need same-worktree concurrency, pass an explicit
  `--output_base=…`.
- **`bazel test //sp1_zorch/...` does not exercise `py_binary` targets**, and
  `jagged_byte_match_test` is `gpu_only` — on CPU its wide `constraint_eval`
  compiles monolithically and never finishes. Use
  `bazel test //sp1_zorch/... --test_tag_filters=-gpu_only`, as CI does, and
  remember the suite can go green while `staged_prove_shard` (a `py_binary`)
  is broken. After changing anything it constructs, run it.

## Testing

Tests default to `FRX_PLATFORMS=cpu`. The SP1 FFI byte-match path needs a CUDA
GPU and is exercised through the per-stage `verify_*` `py_binary` tools and
the full-chain harness `//tools:staged_prove_shard`, not the unit suite.

### Test sizing & timeouts

`size` and `timeout` are independent knobs — set both deliberately on heavy
tests:

- **`size`** (`small`/`medium`/`large`) is a *resource* hint: roughly how much
  RAM/CPU the test needs, which governs how many run in parallel.
- **`timeout`** (`short`/`moderate`/`long`/`eternal` = 60/300/900/3600 s) is the
  wall-clock cap. When left unset it is *derived* from `size`
  (small→short, medium→moderate, large→long).

Declare a **`timeout` explicitly** on any heavy test rather than leaning on the
size-derived default. Why: a dependency bump (a wheel or the zorch pin)
invalidates the Bazel cache, so the whole suite re-runs **cold** on the shared
self-hosted CI runner — which is ~2–3× slower than a local box under parallel
test load. A test that finishes in 150 s locally can blow past the 300 s
`medium` cap on CI and fail as a `TIMEOUT` even though nothing is actually
wrong.

Heavy tests currently carrying explicit timeouts:
`shard_prover:prove_shard_test` (`long`),
`shard_prover:verify_shard_test` (`eternal`), `zerocheck:jagged_test` and
`zerocheck:verifier_test` (both `moderate`).

> A green CI on a branch with **no** recent dep bump is usually an all-cache-hit
> run (~20 s), not evidence the tests fit their caps — the cold path only
> surfaces after a bump. When you bump a dep, sanity-check the run actually
> re-ran the heavy tests.

### Fixtures

Reference fixtures byte-match the SP1 reference prover (Montgomery-form `u32`
bytes, no tolerances):

- **Vendored** small fixtures live per module under `testdata/` (e.g.
  `sp1_zorch/zerocheck/testdata/gpu_fibonacci`) and back the unit tests.
- **External** full-shard dumps are too large to vendor; they stay out of the
  repo and are checked via `--shard_dir` (GPU) — per stage with the `verify_*`
  `py_binary` tools, full-chain with `//tools:staged_prove_shard`. The CUDA
  FFI they call (`libsp1_gpu_jax_ffi`) lives in `whir-zorch` under
  `third_party/sp1/`.

## Running a local shard prove

`//tools:staged_prove_shard` is **the** local GPU harness: it proves one rsp
shard dump through the full chain (trace commit → LogUp-GKR → zerocheck →
jagged eval), byte-checking each phase against the dump's references the
instant that phase finishes — a phase-k mismatch aborts before phase k+1 pays
its multi-minute compile.

```bash
FRX_PLATFORMS=cuda,cpu \
  XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL" \
  bazel run //tools:staged_prove_shard -- \
    --shard_dir=/data/sp1_dumps/rsp_21740136_sp1/shard17
```

### Why the harness is staged

The harness drives the four phases stage by stage with a release point between
them — each stage's spent result object is dropped and gc runs before the next
stage allocates — instead of calling the composite `ShardProver.prove` once:

- **Memory.** A monolithic full-chain run keeps every stage's spent result
  live to the end of the chain: measured **89 GB host peak** on a core shard
  (shard6) — beyond any 32 GB dev box. Released stage by stage, one stage's
  working set is resident at a time, and a full rsp shard proves on a 32 GB
  card.
- **Production shape.** The staging order is the production zkvm pipeline
  driver's — the local harness rehearses the same stage-by-stage drive that
  ships.
- **Still byte-exact.** Same call order and transcript threading as the
  composite `ShardProver.prove`, so the Fiat-Shamir stream — and the proof —
  is the composite's byte for byte.

### Flags

| Flag | Meaning |
|---|---|
| `--shard_dir` | rsp shard dump dir. Comma-separate several to prove them sequentially in ONE process; same-class shards must then reuse the first shard's compiles (the shard-invariance check). |
| `--runs=N` | Prove the chain N times in one process: run 1 is cold (pays the XLA/zkx compiles), runs 2+ are warm (executables reused). Golden checks run on every pass. |
| `--max_phase=N` | Run + byte-check only phases 1..N (1 = trace commit … 4 = full, default). Skips the downstream compiles for a cheaper iteration loop. |
| `--ffi_verify` | Assemble the bincode wire and verify it through SP1's `sp1_verify_shard` FFI (needs `SP1_JAX_FFI_LIB`, below). |
| `--proof_sha256` | Default on: on a full run, bincode-encode the proof and print `PROOF_SHA256` — the cross-run/cross-stack byte-golden line. Disable for compile-only drivers. |
| `--zc_class_json` / `--gkr_class_json` | Global zerocheck / LogUp-GKR class pins (next section). |
| `--group_manifest_json` | Per-shard class pins for multi-group runs; overrides the global pins field by field. |
| `--jaxprof_dir` | Write an frx profiler trace of the last (warm) prove pass. |

The security-parameter flags (`--gkr_pow_bits`, `--open_num_queries`,
`--open_pow_bits`) default to SP1's core machine — leave them alone for
byte-match runs.

### Class pins and the group manifest

The heavy zones compile keyed on `(chip set, class)` — the shard's runtime
heights ride as traced values — so every shard of one class shares one
executable. Per shard the harness resolves its classes as, highest precedence
first:

1. the shard's entry in `--group_manifest_json`,
2. the global `--zc_class_json` / `--gkr_class_json` pins,
3. the shard's own a-priori-tight class (per-shard compile).

The single definition of the class math and this resolution is
`sp1_zorch.shard_prover.compile_classes`, shared with the `warm_shard_cache`
cache filler — the classes a warm fills are the classes a prove requests by
construction. Get a manifest from `warm_shard_cache` analyze
(`--out_manifest`; a `--warm` fill also writes `group_manifest.json` beside
the cache), or assemble a pin file as the per-field max of the class lines
below.

### Stdout contract

Bench and cache tooling parse these lines — keep their shape stable:

| Line | When | Meaning / consumer |
|---|---|---|
| `CHIP_HEIGHTS name:rows …` | per shard, pre-prove | census of real per-chip heights |
| `ZC_CLASS {"area_cap": N}` | per shard, pre-prove | the shard's tight zerocheck class; cross-shard pin = per-field max |
| `GKR_CLASS {"chip_heights": {…}, "slot_cap": N}` | per shard, pre-prove | the shard's tight GKR class; cross-shard pin = per-chip max |
| `JAGGED_CLASS {"L": …, "n_d": …, "K": …, "rlc_bits": …}` | per shard, pre-prove | the fully derived jagged class (no pin flag exists for it) |
| `[phase X] Yms mem=…GiB peak=…GiB` | per phase, per pass | phase wall-clock + device-pool telemetry; on a mid-phase OOM the *previous* phase's line is the resident set the failing alloc fought |
| `chain run: Yms` | per pass | full-chain wall |
| `prove_shard chain (phases 1..N) byte-match: ALL OK` | end of shard | every executed phase matched its golden reference |
| `PROOF_SHA256 <hex>` | full run only | cross-run/cross-stack byte-golden line |

### Environment contract

| Env | Setting | Why |
|---|---|---|
| `FRX_PLATFORMS` | `cuda,cpu` — not `cuda` alone | the long Fiat-Shamir absorbs run on the host sponge, which needs a registered CPU backend; without it the run dies in `frx.devices("cpu")` with `Unknown backend cpu`. (`gpu` is wrong too: it also initializes rocm and dies.) |
| `XLA_FLAGS` | `--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL` | captures each fused region (the whole-layer LogUp-GKR zone, the trace-commit tail) as a CUDA graph so the warm pass isn't host-dispatch-bound. **Do NOT add `--xla_gpu_graph_min_graph_size=1`**: it captures every 1-op region (~4.3k `wrapped_*` pyramid-transition ops) as its own resident CUDA graph, whose cross-pass buffer residency double-allocates the pyramid intermediate and OOMs a wide shard on `--runs>=2` (shard18 dies `RESOURCE_EXHAUSTED: allocate 3.77 GiB` on pass 2) — for no speedup, since the LogUp-GKR zone is already one big graph. |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | `cuda_async` on a wide shard | under the default BFC allocator shard0 (33 chips, `slot_cap` 77.3M) dies in the GKR phase with `RESOURCE_EXHAUSTED: allocate 11.62GiB` while only 10.57 GiB is in use on a 32 GiB card — fragmentation, not capacity; with `cuda_async` the same shard runs clean at 21.4 GiB peak and byte-matches. On a narrow shard `cuda_async` measured slightly *slower*, so it is not a blanket default. |
| `FRX_COMPILATION_CACHE_DIR` + `FRX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` + `FRX_RAISE_PERSISTENT_CACHE_ERRORS=true` | a **fresh per-toolchain dir, mounted on every run — byte-match gates included**. Seed it with `warm_shard_cache` (same `XLA_FLAGS` as the run: the flags are part of the compile key), or let the first run populate it. The only cache-less run is one measuring the cold-compile wall itself. | cuts the cold pass ~175 s → ~28 s (measured, shard17) and does not move the warm pass. Both extra flags matter: `min_compile_time` defaults to 1.0 s and admits nothing here (a cache dir alone caches zero modules), and cache errors are swallowed as warnings by default, so a broken cache looks like a working one — check the dir is non-empty. Gates stay valid because the per-phase golden byte-checks are the oracle — a cache-served wrong executable fails them loudly. What is forbidden is a **stale / shared / cross-toolchain** dir (one has served wrong executables): never carry a cache across a pin bump, `rm -rf` and reseed. |
| `SP1_JAX_FFI_LIB` | path to `libsp1_gpu_jax_ffi.so` | required by `--ffi_verify`; vendored in SP1 reference checkouts, e.g. whir-zorch `third_party/sp1/`. |
| `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=<idx>` | pin an idle card on a shared box | contending with another prove during CUDA init can hard-kill the run. |

## Per-phase baseline against SP1

A wall-clock comparison against SP1's native reference means something only
when both sides run the **full shard prove** (trace commit → LogUp-GKR →
zerocheck → jagged eval) on the **same** rsp shard and produce byte-identical
output. Both tools below satisfy that, so their per-phase numbers measure the
same computation.

> Comparing a *synthetic* SP1 bench (`logup_gkr_bench`: random values, real
> heights) against a sp1-zorch real-shard run breaks all three conditions —
> different data, different scope (sp1-zorch includes the per-chip openings and
> grind/head), and no golden equivalence, since the two never prove the same
> instance. Ratios obtained that way are scope-confounded; do not quote them.

#### sp1-zorch side — `staged_prove_shard` (per-phase + golden)

The [local shard-prove harness](#running-a-local-shard-prove) doubles as the
measurement tool — its warm `[phase X]` lines are the sp1-zorch column:

```bash
FRX_PLATFORMS=cuda,cpu \
  XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL" \
  bazel run //tools:staged_prove_shard -- \
    --shard_dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --ffi_verify --runs=5
```

Use `--runs=5`, not `--runs=2`: the **first** warm pass (pass 2) has not fully
settled and overstates a phase by ~10–15%, so read a converged pass (3–5). Pin
an idle card and set the rest of the
[environment contract](#environment-contract). This runs the GPU plugin
bundled in the pinned `frx-cuda12-pjrt` wheel; to measure a *locally built*
Fractalyze XLA plugin instead, see
[Measure shipped code](#measure-shipped-code).

**Host load is part of the measurement.** Warm LogUp-GKR inflates badly above
load ~5 — the same build reads 17 ms on an idle box and 27 ms at load 50 — and
one shard varies 134–151 ms across sessions at comparable load. Record the load
beside any number you quote, and never compare two arms measured in separate
sessions: interleave them in one run, alternating which arm goes first, or the
load trend becomes your result.

**Golden — the same-output premise holds.** Every phase byte-checks against
the dump as it finishes (commitment vs `main_commit`, the GKR evaluation
point's row tail vs `gpu_z_row.txt` — SP1's `zeta`, not the zerocheck point —
zerocheck's `final_eval`, the jagged outer sumcheck claim vs
`phase4_sumcheck_claim`), and `--ffi_verify` byte-verifies the assembled
bincode proof through SP1's `sp1_verify_shard` FFI. So sp1-zorch's output is
byte-identical to SP1's.

#### SP1 native side — `riscv-witness/tools/sp1/sp1_shard_prover`

The `sp1-shard-test` bin (standalone crate under
`riscv-witness/tools/sp1/sp1_shard_prover/`) proves one shard and prints each
phase's wall-clock — `[stage commit traces]`, `[stage logup gkr proof]`,
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
`staged_prove_shard`. Keep CPU phase times out of the GPU-vs-GPU table below.

### Per-phase comparison (shard17)

| Phase | SP1 GPU | sp1-zorch GPU | spread | ratio | golden |
|---|---|---|---|---|---|
| trace commit | 16.6 ms | 18.4 ms | 18.3–18.4 | 1.11× | byte-match |
| LogUp-GKR | 19.9 ms | **18.4 ms** | 18.1–18.7 | **0.92×** | byte-match |
| zerocheck | 156.9 ms | **52.7 ms** | 52.3–52.8 | **0.34×** | byte-match |
| jagged eval (PCS open) | 41.1 ms | **37.2 ms** | 37.1–37.3 | **0.91×** | byte-match |
| full chain (phase sum) | 234.8 ms | **124.6 ms** | — | **0.53×** | byte-match |

The sp1-zorch column was captured on `main` 23d2fff (2026-08-11). The full-chain
row is the sum of the four phase medians: since #330 the CLI's `chain run:`
print times the whole checked round — per-phase golden loads, device→host
readbacks and compares included (~5.2 s of I/O on shard17) — so that print is a
harness metric, not a prover latency, and is not comparable to the SP1 column.

**A `--max_phase` run reads a phase faster than the full chain does.** Truncated
runs are fine for A/B iteration, where both arms are truncated equally; they are
not comparable against the SP1 column, which is always full-chain.

The SP1 column is the SP1 GPU NoExec run. The sp1-zorch column is the median
of the six converged warm passes — passes 3–5 of two separate `--runs=5`
invocations — with the observed min–max beside it, on an RTX 5090, published
`frx` wheels (no locally built plugin), shard-invariant class routes on GKR,
zerocheck and the jagged open. Every phase byte-matches on every pass. Add
`--ffi_verify` to also byte-verify the assembled proof through SP1's
`sp1_verify_shard`; these figures do not include that leg.

**Read the spread before quoting a ratio.** A single pass is not evidence in
either direction; take several and quote the median with its min-max, and treat
the chain total the same way since it inherits every phase's variance.

### Per-phase comparison (shard14, keccak class)

shard17 above is a narrow shard; shard14 is a **wide keccak-class** shard
(`KeccakPermute` 122 k rows, `Program` 518 k, zerocheck `area_cap` 401 M) —
worth its own table because every phase is an order of magnitude heavier and
the allocator constraint changes. Same premise as shard17: same dump, same
RTX 5090, every sp1-zorch phase byte-checks against the dump as it finishes.

| Phase | SP1 GPU | sp1-zorch GPU | spread | ratio | golden |
|---|---|---|---|---|---|
| trace commit | 91.3 ms | 155.8 ms | 105.3–334.5 | 1.71× | byte-match |
| LogUp-GKR | 63.2 ms | **41.0 ms** | 34.9–53.9 | **0.65×** | byte-match |
| zerocheck | 947.5 ms | **693.8 ms** | 661.7–705.2 | **0.73×** | byte-match |
| jagged eval (PCS open) | 240.2 ms | **220.1 ms** | 209.2–240.8 | **0.92×** | byte-match |
| full chain (phase sum) | 1342.6 ms | **1110.7 ms** | — | **0.83×** | byte-match |

Captured on the #336/#337 tree (2026-08-12), default monolithic jagged scan
(no `--jagged_scan_cap`). The sp1-zorch column is the median of passes 3–5 of
**one** `--runs=5` invocation; the SP1 column is a single `no-exec-gpu-dump
--gpu` invocation. Both arms ran in the same session window at host load
29–38 — comparable to each other, but weaker sampling than the shard17 table,
and the load-sensitive rows show it (trace commit's 105–335 ms spread is load,
not the prover). Treat the per-phase ratios as first measurements; re-measure
on an idle box before quoting any single row.

Two shard14-specific gotchas:

- **`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` is required, not optional.** The
  monolithic run peaks at **17.51 GiB** and fits a 32 GiB card with no
  eviction under `cuda_async`. Under BFC the same shard dies of
  fragmentation — even after #336/#337 chunked the jagged-open temp arenas
  35.7 → 2.5 GB. This is the environment-contract wide-shard failure mode,
  one shard class heavier.
- **`sp1-shard-test` prints a hardcoded "shard17" tag** in its summary line
  regardless of the input dump (`gpu_dump_prover.rs`). Identify the shard by
  `--shard_dir` and the byte-match gates, never by the label.

### Measure shipped code

A per-phase number is only a baseline if it runs the code the team **ships**, so
before capturing one make sure the two knobs this repo lets you swap point at the
shipped path, not a stale local one:

- `zorch` is the `MODULE.bazel` pin — or, if you dev against a local checkout via
  a `.bazelrc.user` `--override_module=zorch=`, that checkout is on the same
  `origin/main` commit, not behind it and not dirty;
- the GPU plugin is the one you mean to measure. The Fractalyze XLA plugin loader
  (`frx_plugins/xla_cuda12/__init__.py` in the pinned `frx-cuda12-pjrt` wheel)
  reads no plugin-path env var — it loads the bundled `xla_cuda_plugin.so`. To
  measure a locally built Fractalyze XLA plugin, overwrite that bundled `.so` (back it up)
  and run the **prebuilt** binary directly — `bazel run` re-extracts the wheel
  and reverts the swap. Confirm which ran with
  `ldd <.so> | grep -c "not found"`: a published wheel reports 0, a locally
  built plugin reports 4 (nvshmem x3 + nccl). The `strings … hlo_verifier.cc`
  test does NOT discriminate — a pristine wheel prints `xla/service/…` too.

(sp1-zorch#153: a first encode baseline was taken against a `zorch` override
weeks behind `origin/main` and misread as the shipped number — the whole reason
this check exists.)

### Shard size caveat (still applies)

A block's shards differ in size by >30×: for `rsp_21740136`, shard0 = 38.6 M
first-layer rows, shard17 = 1.16 M (`gpu_first_layer.txt: height`). Always run
**both provers on the same `--shard_dir`**; never compare across shards. (A
relayed "SP1 ~81 ms" was shard0; an earlier sp1-zorch number was shard17.)
