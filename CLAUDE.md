# Project context for Claude Code

- **Overview & quick start:** [`README.md`](README.md)
- **Conventions:** [`docs/conventions.md`](docs/conventions.md) — comments (why-not-what), external SP1 references (pinned permalinks).
- **Pipeline & terminology:** [`docs/shard-pipeline.md`](docs/shard-pipeline.md) — stages as Round compositions, SP1 dump "phase" vocabulary mapping.
- **Zerocheck protocol:** [`docs/zerocheck-protocol.md`](docs/zerocheck-protocol.md) — the joint claim, the three batchings, the round schedule, the point accounting.
- **Testing:** [`docs/testing.md`](docs/testing.md) — running tests, `size` vs `timeout` conventions, fixtures.
- **SP1 baseline:** [`docs/sp1-baseline.md`](docs/sp1-baseline.md) — reproducible SP1-vs-sp1-zorch per-stage wall-clock comparison (`shard_prover:verify_prove_shard` + SP1 `sp1_shard_prover`).
- **Current work:** [fractalyze/zorch#37](https://github.com/fractalyze/zorch/issues/37) — bootstrap + SMCS on zorch merkle blocks.

## One non-negotiable

- **SP1-specific only.** This repo holds the SP1 glue (domain separator, verify
  codes, heap proof layout, FFI byte-match). Anything scheme- or zkVM-agnostic
  belongs upstream in `zorch`, not here. If a generic block is missing, add it
  to `zorch` and depend on it — do not fork it into `sp1-zorch`.

## Dependency on zorch

`zorch` is a Bazel module, pinned in `MODULE.bazel` via `git_override` to a main
commit. For dev against a local working copy, add to `.bazelrc.user`
(gitignored — holds an absolute path):

```
common --override_module=zorch=/abs/path/to/your/zorch/checkout
```

Bump the pin when you need newer `zorch` blocks; keep it on `main` commits so CI
is reproducible.

## Development environment

Pure Python on JAX + the ZKX PJRT plugin. Bazel 9 (bzlmod). Tests default to
`JAX_PLATFORMS=cpu`; the SP1 FFI byte-match needs a CUDA GPU.

```sh
bazel test //...                 # hermetic, sandboxed
# iterative dev outside Bazel:
export PYTHONPATH="$PWD:/abs/path/to/zorch"
export ZKX_REPO_ROOT="$HOME/Workspace/zkx"   # dev against a local ZKX checkout
```

GPU runnables: a `py_binary` must dep `requirement("jax_cuda12_plugin")` +
`requirement("jax_cuda12_pjrt")` or jax **silently falls back to CPU** — run
with `JAX_PLATFORMS=cuda` so a missing plugin errors instead (`gpu` is wrong:
it also initializes rocm and dies). The new-jax loader takes no plugin-path env
var; to measure a locally built zkx plugin you overwrite the wheel's bundled
`xla_cuda_plugin.so` — see [`docs/sp1-baseline.md`](docs/sp1-baseline.md)
"Measure shipped code" for the procedure.

## SP1 byte-match

The commit/open/verify path byte-matches the SP1 reference prover. Reference
fixtures are vendored per module under `testdata/` (e.g.
`sp1_zorch/zerocheck/testdata/gpu_fibonacci`); full-shard dumps too large to
vendor stay external and are checked via the `verify_*` `py_binary` tools
(`--shard_dir`). The CUDA FFI (`libsp1_gpu_jax_ffi`) still lives in
`whir-zorch` (`third_party/sp1/`). Compare Montgomery-form `u32` bytes
directly, no tolerances.
