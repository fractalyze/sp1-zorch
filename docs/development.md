# Development guide

Everything needed to build, test, and benchmark sp1-zorch: the environment
setup, the test conventions, and the reproducible per-phase baseline against
SP1. For architecture (Stage / Round) see
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
  remember the suite can go green while `verify_prove_shard` (a `py_binary`)
  is broken. After changing anything it constructs, run it.

## Testing

Tests default to `FRX_PLATFORMS=cpu`. The SP1 FFI byte-match path needs a CUDA
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
  repo and are checked with the `verify_*` `py_binary` tools via `--shard_dir`
  (GPU). The CUDA FFI they call (`libsp1_gpu_jax_ffi`) lives in `whir-zorch`
  under `third_party/sp1/`.

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

#### sp1-zorch side — `verify_prove_shard` (per-phase + golden)

```bash
FRX_PLATFORMS=cuda,cpu \
  XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL" \
  bazel run //sp1_zorch/shard_prover:verify_prove_shard -- \
    --shard_dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --ffi_verify --runs=5
```

`FRX_PLATFORMS=cuda,cpu`, not `cuda` alone: the long Fiat-Shamir absorbs run on
the host sponge, which needs a CPU backend registered. Without it the run dies
in `frx.devices("cpu")` with `Unknown backend cpu`.

Use `--runs=5`, not `--runs=2`: the **first** warm pass (pass 2) has not fully
settled and overstates a phase by ~10–15%, so take a converged pass (3–5).
Pin to an idle card on a shared box
(`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx>`) — contending with
another prove during CUDA init can hard-kill the run.

**Host load is part of the measurement.** Warm LogUp-GKR inflates badly above
load ~5 — the same build reads 17 ms on an idle box and 27 ms at load 50 — and
one shard varies 134–151 ms across sessions at comparable load. Record the load
beside any number you quote, and never compare two arms measured in separate
sessions: interleave them in one run, alternating which arm goes first, or the
load trend becomes your result.

**Use the persistent compile cache while iterating.** It cuts the cold pass
from ~175 s to ~28 s (measured, shard17) and does **not** move the warm pass, so
it is safe for the numbers this tool exists to produce:

```bash
FRX_COMPILATION_CACHE_DIR=/data/<you>/frx-compile-cache \
  FRX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
  FRX_RAISE_PERSISTENT_CACHE_ERRORS=true
```

Both extra flags matter. `min_compile_time` defaults to **1.0 s** and admits
nothing here, so a cache dir on its own caches zero modules and silently
changes nothing; and cache errors are swallowed as warnings by default, so a
broken cache looks exactly like a working one. Check the dir is non-empty
before trusting that it did anything.

The `--xla_gpu_enable_command_buffer=...` flag captures each fused region (the
whole-layer LogUp-GKR zone, the trace-commit tail) as a CUDA graph so the warm
pass isn't host-dispatch-bound. **Do NOT add `--xla_gpu_graph_min_graph_size=1`.**
It additionally captures every 1-op region (the ~4.3k `wrapped_*` pyramid-transition
ops) as its own resident CUDA graph; their cross-pass buffer residency
double-allocates the pyramid intermediate on the warm pass and OOMs a wide shard —
`shard18 --runs≥2` dies with `RESOURCE_EXHAUSTED: allocate 3.77 GiB` on pass 2,
while a fresh single-pass prove succeeds. It also gives no speedup, since the
LogUp-GKR zone is already captured as one big graph.

**A wide shard needs `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.** Under the
default BFC allocator, shard0 (33 chips, `slot_cap` 77.3M) dies in the GKR phase
with `RESOURCE_EXHAUSTED: allocate 11.62GiB` while peak usage is only 10.57 GiB
on a 32 GiB card — fragmentation, not capacity. With `cuda_async` the same shard
runs clean at 21.4 GiB peak and byte-matches. This is separate from throughput:
on a narrow shard `cuda_async` measured slightly *slower*, which is why it was
once written off — but on a wide shard it is the difference between proving and
not.

`--ffi_verify` byte-verifies the assembled bincode proof through SP1's
`sp1_verify_shard` FFI; point `SP1_JAX_FFI_LIB` at `libsp1_gpu_jax_ffi.so`
(vendored in SP1 reference checkouts, e.g. whir-zorch `third_party/sp1/`). This
runs the GPU plugin bundled in the pinned `frx-cuda12-pjrt` wheel; to measure a
*locally built* Fractalyze XLA plugin instead, see [Measure shipped code](#measure-shipped-code).

- Runs `ShardProver` (`JaggedPcsProver.commit` → `LogupGkrProver`
  → `ZerocheckProver` → `JaggedPcsProver.prove`) on the real shard.
- A `_TimedRound` wrapper prints **per-phase wall-clock** in ms:
  `[phase TraceCommit] X.Yms`, and likewise for the other three. `--runs=5`
  proves five times in one process: pass 1 is cold (XLA compiles), passes
  2–5 are **warm** (executables reused); read a converged pass (3–5), not the
  first warm pass (see the run note above), and compare it against SP1.
- **Golden**: the chain's commitment must equal the dump's `main_commit`
  (`gpu_commitment.txt`), the GKR evaluation point's row tail must equal
  `gpu_z_row.txt` (SP1's `zeta`, not the zerocheck point), the
  jagged claim must equal `phase4_sumcheck_claim`, and with `--ffi_verify` the
  assembled bincode proof is byte-verified through SP1's `sp1_verify_shard` FFI.
  So sp1-zorch's output is byte-identical to SP1's — the same-output premise
  holds.

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
`verify_prove_shard`. Keep CPU phase times out of the GPU-vs-GPU table below.

### Per-phase comparison (shard17)

| Phase | SP1 GPU | sp1-zorch GPU | spread | ratio | golden |
|---|---|---|---|---|---|
| trace commit | 16.6 ms | 17.6 ms | 17.5–17.6 | 1.06× | byte-match |
| LogUp-GKR | 19.9 ms | **20.4 ms** | 20.4–20.6 | **1.03×** | byte-match |
| zerocheck | 156.9 ms | **50.8 ms** | 50.4–51.0 | **0.32×** | byte-match |
| jagged eval (PCS open) | 41.1 ms | **37.8 ms** | 36.3–37.9 | **0.92×** | byte-match |
| full chain | 234.8 ms | **134.0 ms** | 132.8–136.3 | **0.57×** | byte-match |

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
