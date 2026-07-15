# Project context for Claude Code

- **Overview & quick start:** [`README.md`](README.md)
- **Conventions:** [`docs/conventions.md`](docs/conventions.md) — comments (why-not-what), external SP1 references (pinned permalinks).
- **Architecture & terminology:** [`docs/architecture.md`](docs/architecture.md) — the shard proof as a ProveChain of Stages, each running inner Rounds, threaded by a Bridge; SP1 dump "phase" vocabulary mapping.
- **Development:** [`docs/development.md`](docs/development.md) — environment setup, running tests (`size` vs `timeout`, fixtures), and the reproducible per-stage SP1 baseline (`shard_prover:verify_prove_shard` + SP1 `sp1_shard_prover`).

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
is reproducible. **Bump the frx family (`frx` = Fractalyze Field, Ring
Accelerated) in `requirements.in` + `requirements_lock_3_11.txt` to match zorch's
pin in the *same* commit** — zorch and sp1-zorch build against a shared frxlib,
so a lagging frx pin ABI-mismatches and **segfaults** the GPU tests
(`verify_shard_test`), not a clean `ImportError`. `sp1-zorch main` is the
reference for the matching `(zorch pin, frx)` pair.

## Development environment

Pure Python on frx (Field, Ring Accelerated), run against the
Fractalyze XLA GPU plugin. Bazel 9 (bzlmod). Tests default to
`JAX_PLATFORMS=cpu`; the SP1 FFI byte-match needs a CUDA GPU. Full setup, the
GPU-plugin gotcha (frx **silently falls back to CPU** without the cuda plugin
deps), test `size`/`timeout` conventions, and the per-stage SP1 baseline live in
[`docs/development.md`](docs/development.md).

## SP1 byte-match

The commit/open/verify path byte-matches the SP1 reference prover. Reference
fixtures are vendored per module under `testdata/` (e.g.
`sp1_zorch/zerocheck/testdata/gpu_fibonacci`); full-shard dumps too large to
vendor stay external and are checked via the `verify_*` `py_binary` tools
(`--shard_dir`). The CUDA FFI (`libsp1_gpu_jax_ffi`) still lives in
`whir-zorch` (`third_party/sp1/`). Compare Montgomery-form `u32` bytes
directly, no tolerances.
