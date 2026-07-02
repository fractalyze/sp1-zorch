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
