# Project context for Claude Code

- **Overview & quick start:** [`README.md`](README.md)
- **Conventions:** [`docs/conventions.md`](docs/conventions.md) — comments (why-not-what), external SP1 references (pinned permalinks).
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

## SP1 byte-match (later slices)

The commit/open/verify path byte-matches the SP1 reference prover. The reference
fixtures and the vendored CUDA FFI (`libsp1_gpu_jax_ffi`) live in `whir-zorch`
today (`whir-zorch/sp1/testing`, `whir-zorch/third_party/sp1/`); how they're
vendored or referenced here is decided when the byte-match slice lands. Compare
Montgomery-form `u32` bytes directly, no tolerances.
