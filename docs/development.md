# Development guide

Everything needed to build, test, and benchmark sp1-zorch: the environment
setup, the test conventions, and the reproducible per-stage baseline against
SP1. For architecture (ProveChain / Stage / Round / Bridge) see
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
bazel test //...                 # hermetic, sandboxed; JAX_PLATFORMS=cpu default
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

Bumping the `zorch` pin and its matching frx family is a coupled change — see the
"Dependency on zorch" section of [`../CLAUDE.md`](../CLAUDE.md) (a lagging frx pin
segfaults the GPU tests rather than raising a clean `ImportError`).

**GPU-plugin gotcha.** A `py_binary` GPU runnable must dep
`requirement("frx_cuda12_plugin")` + `requirement("frx_cuda12_pjrt")` or frx
**silently falls back to CPU**. Run with `JAX_PLATFORMS=cuda` so a missing plugin
errors instead of silently degrading (`gpu` is wrong: it also initializes rocm
and dies). The Fractalyze XLA plugin loader takes no plugin-path env var; to
measure a locally built plugin you overwrite the wheel's bundled
`xla_cuda_plugin.so` — see [Measure shipped code](#measure-shipped-code) below.

## Testing

Tests default to `JAX_PLATFORMS=cpu`. The SP1 FFI byte-match path needs a CUDA
GPU and is exercised through the `verify_*` `py_binary` tools, not the unit
suite.

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
`shard_prover:prove_shard_test`, `shard_prover:verify_shard_test`,
`jagged:verifier_test`, `logup_gkr:prover_test`, `zerocheck:jagged_test`.
(The `commit:*` tests jit their hashing, so they fit `medium` without one.)

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
  repo and are checked with the `verify_*` `py_binary` tools via `--shard_dir`
  (GPU). See the SP1 byte-match notes in [`../CLAUDE.md`](../CLAUDE.md).

## Per-stage baseline against SP1

How to compare sp1-zorch's GPU prover against SP1's native reference on a real
rsp shard, **per Stage**, on the premise that **both prove the same shard, at the
same scope, and produce the same output** (golden byte-match). Only under that
premise is a wall-clock comparison meaningful.

> **Why the premise matters (a correction).** An earlier baseline compared SP1's
> *synthetic* `logup_gkr_bench` (random values, real heights) to a sp1-zorch
> real-shard bench. That is **invalid**: different data, different scope
> (sp1-zorch includes the per-chip openings + grind/head; the SP1 synthetic
> bench does not), and **no golden equivalence** between the two — they never
> prove the same instance. Numbers from that approach (a "14×" logup-gkr ratio)
> are scope-confounded and should not be quoted. Use the per-stage full-prove
> comparison below instead.

### The valid benchmark — same data, same scope, same output

Both sides run the **full shard prove** (trace commit → LogUp-GKR → zerocheck →
jagged eval) on the **same** rsp shard, and the outputs are byte-identical
(verified), so per-stage wall-clocks are comparing the same computation.

#### sp1-zorch side — `verify_prove_shard` (per-stage + golden)

```bash
JAX_PLATFORMS=cuda \
  XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL" \
  bazel run //sp1_zorch/shard_prover:verify_prove_shard -- \
    --shard_dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --ffi_verify --runs=5
```

Use `--runs=5`, not `--runs=2`: the **first** warm pass (pass 2) has not fully
settled — LogUp-GKR's eager host-dispatch driver reads ~58 ms on pass 2 but
converges to ~38 ms by passes 3–5, so reading pass 2 overstates it ~50%. Take a
converged steady-state pass. Pin to an idle card on a shared box
(`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx>`) — contending with
another prove during CUDA init can hard-kill the run.

The `--xla_gpu_enable_command_buffer=...` flag captures each fused region (the
whole-layer LogUp-GKR zone, the trace-commit tail) as a CUDA graph so the warm
pass isn't host-dispatch-bound. **Do NOT add `--xla_gpu_graph_min_graph_size=1`.**
It additionally captures every 1-op region (the ~4.3k `wrapped_*` pyramid-transition
ops) as its own resident CUDA graph; their cross-pass buffer residency
double-allocates the pyramid intermediate on the warm pass and OOMs a wide shard —
`shard18 --runs≥2` dies with `RESOURCE_EXHAUSTED: allocate 3.77 GiB` on pass 2,
while a fresh single-pass prove succeeds. It also gives no speedup, since the
LogUp-GKR zone is already captured as one big graph.

`--ffi_verify` byte-verifies the assembled bincode proof through SP1's
`sp1_verify_shard` FFI; point `SP1_JAX_FFI_LIB` at `libsp1_gpu_jax_ffi.so`
(vendored in SP1 reference checkouts, e.g. whir-zorch `third_party/sp1/`). This
runs the GPU plugin bundled in the pinned `frx-cuda12-pjrt` wheel; to measure a
*locally built* Fractalyze XLA plugin instead, see [Measure shipped code](#measure-shipped-code).

- Runs `prove_shard_chain` (the `ProveChain` of `TraceCommitStage` → `LogupGkrStage`
  → `ZerocheckStage` → `JaggedPcsStage`) on the real shard.
- A `_TimedRound` wrapper prints **per-Stage wall-clock** in ms:
  `[stage TraceCommitStage] X.Yms`, and likewise for the other three. `--runs=5`
  proves five times in one process: pass 1 is cold (XLA compiles), passes
  2–5 are **warm** (executables reused); read a converged pass (3–5), not the
  first warm pass (see the run note above), and compare it against SP1.
- **Golden**: the chain's commitment must equal the dump's `main_commit`
  (`gpu_commitment.txt`), the zerocheck point must equal `gpu_z_row.txt`, the
  jagged claim must equal `phase4_sumcheck_claim`, and with `--ffi_verify` the
  assembled bincode proof is byte-verified through SP1's `sp1_verify_shard` FFI.
  So sp1-zorch's output is byte-identical to SP1's — the same-output premise
  holds.

#### SP1 native side — `riscv-witness/tools/sp1/sp1_shard_prover`

The `sp1-shard-test` bin (standalone crate under
`riscv-witness/tools/sp1/sp1_shard_prover/`) proves one shard and prints each
Stage's wall-clock — `[stage commit traces]`, `[stage logup gkr proof]`,
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

### Per-stage comparison (shard17)

| Stage | SP1 GPU | sp1-zorch GPU | ratio | golden |
|---|---|---|---|---|
| trace commit | 16.6 ms | 18.2 ms | 1.10× | byte-match |
| LogUp-GKR | 19.9 ms | 38.2 ms | 1.92× | byte-match |
| zerocheck | 156.9 ms | **141.7 ms** | **0.90×** | byte-match |
| jagged eval (PCS open) | 41.1 ms | **37.4 ms** | **0.91×** | byte-match |
| full chain | 234.8 ms | 242.6 ms | 1.03× | `sp1_verify_shard` ACCEPTED |

The two wall-clock columns are warm, byte-matched runs of the two tools above:
the sp1-zorch column is the converged warm steady state (passes 3–5 of the
`--runs=5` command, on an idle RTX 5090 — all four Stages byte-match and
`--ffi_verify` reports `sp1_verify_shard: ACCEPTED`); the SP1 column is the SP1
GPU NoExec run. zerocheck (0.90×) and the jagged-eval PCS open (0.91×) now edge
out SP1; the full chain is at ~parity (1.03×), with LogUp-GKR the one remaining
gap.

### Measure shipped code

A per-stage number is only a baseline if it runs the code the team **ships**, so
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
  `strings -a <.so> | grep service/hlo_verifier.cc` (`external/xla/…` = wheel,
  `xla/service/…` = your build).

(sp1-zorch#153: a first encode baseline was taken against a `zorch` override
weeks behind `origin/main` and misread as the shipped number — the whole reason
this check exists.)

### Shard size caveat (still applies)

A block's shards differ in size by >30×: for `rsp_21740136`, shard0 = 38.6 M
first-layer rows, shard17 = 1.16 M (`gpu_first_layer.txt: height`). Always run
**both provers on the same `--shard_dir`**; never compare across shards. (A
relayed "SP1 ~81 ms" was shard0; an earlier sp1-zorch number was shard17.)
