# Testing

How tests are run, sized, and fixtured in sp1-zorch. For coding style see
[conventions.md](conventions.md); for the pipeline see
[shard-pipeline.md](shard-pipeline.md).

## Running

```sh
bazel test //...     # hermetic, sandboxed; JAX_PLATFORMS=cpu by default
```

Tests default to `JAX_PLATFORMS=cpu`. The SP1 FFI byte-match path needs a CUDA
GPU and is exercised through the `verify_*` `py_binary` tools, not the unit
suite. For iterative dev outside Bazel, see the environment notes in
[`../CLAUDE.md`](../CLAUDE.md).

## Test sizing & timeouts

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

## Fixtures

Reference fixtures byte-match the SP1 reference prover (Montgomery-form `u32`
bytes, no tolerances):

- **Vendored** small fixtures live per module under `testdata/` (e.g.
  `sp1_zorch/zerocheck/testdata/gpu_fibonacci`) and back the unit tests.
- **External** full-shard dumps are too large to vendor; they stay out of the
  repo and are checked with the `verify_*` `py_binary` tools via `--shard_dir`
  (GPU). See the SP1 byte-match notes in [`../CLAUDE.md`](../CLAUDE.md).

## Zerocheck perf probes (compile + runtime)

Two `py_binary` tools under `sp1_zorch/zerocheck` attribute
`constraint_eval_bounded` cost to a chip, on real shard-dump shapes, without
running the prover. They exist because whole-prove timings can't localize a
compile cliff (the ptxas/LLVM cost of one wide chip hides inside a joint jit),
and because emitter A/Bs — e.g. the decode-loop-interpreter retirement in
fractalyze/xla#200, whose acceptance gate was "runtime ≤ the interpreter's
s/exec at real rows" — need compile time AND execution time measured on the
same isolated kernel.

- **`probe_chip_compile`** compiles each chip's `constraint_eval` in isolation
  — with the runtime `live_width` bound, which is load-bearing: omitting it
  compiles the *unbounded* variant, a different cliff-free body. Times
  `.compile()` per chip. Codegen cost is row-independent, so the default
  `--rows` stays tiny.
- **`probe_round_compile`** compiles the real `prove_jagged_zerocheck` scan
  body (constraint_eval inlined ×3 t-points alongside the eq sums and
  round-poly machinery) from abstract `ShapeDtypeStruct` operands — the
  in-stage compile, where super-linear costs actually surface. Bisect with
  `--only` / `--skip`.

Both take `--run` / `--run-iters`, which re-execute the just-compiled kernel
on real (zeros) operands and print ms/iter — field-op cost is
data-independent, so zeros time like real data. Runtime (unlike codegen)
scales with rows: pair `--run` with `--rows` near the chip's real height in
`probe_chip_compile`.

```sh
# per-chip compile sweep (skip the known-heavy chip), GPU emitter
JAX_PLATFORMS=cuda bazel run //sp1_zorch/zerocheck:probe_chip_compile -- \
    --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --skip=Global

# one chip, compile + runtime at real height (CPU backend here)
JAX_PLATFORMS=cpu bazel run //sp1_zorch/zerocheck:probe_chip_compile -- \
    --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --only=Global \
    --rows=134048 --run --run-iters=5

# the full round body, compile + runtime
JAX_PLATFORMS=cpu bazel run //sp1_zorch/zerocheck:probe_round_compile -- \
    --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --only=Global --run
```

`JAX_PLATFORMS` must be explicit: `cuda` makes a missing GPU plugin error
instead of silently probing the CPU emitter (both print a `backend=` line —
trust that, not the flag). For emitter-path A/Bs on CPU, the
`XLA_CPU_CONSTRAINT_EVAL_CONE_PROGRAM_MIN_OPS` env var overrides the
cone-program floor (0 forces the cone-aware path, a huge value forces the
monolithic body; both arms compute the same result). Wheels pinned before
fractalyze/xla#202 spell it `XLA_CPU_CONSTRAINT_EVAL_LOOP_FORM_MIN_OPS`.
Measure cold: unset `JAX_COMPILATION_CACHE_DIR`, or compile numbers are cache
reads.
