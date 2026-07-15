# sp1-zorch

A lean **SP1 prover** built on [`zorch`](https://github.com/fractalyze/zorch)'s
scheme-agnostic SNARK building blocks. `zorch` provides the reusable pieces
(hashing, Merkle commitment, sumcheck, fold, …); `sp1-zorch` adds only the
SP1-specific glue on top — domain separator, verify codes, heap proof layout,
and the FFI byte-match against the SP1 reference prover.

```
JAX  ──▶  zorch (scheme-/zkVM-agnostic blocks)  ──▶  sp1-zorch (SP1 glue)
```

Why a separate repo: most of a general WHIR prover's surface isn't on the SP1
path. Building directly on `zorch`'s blocks keeps this prover small, keeps
SP1-specific knowledge out of `zorch` (its hard rule), and gives a focused
target to grow SP1 glue and benchmark against the SP1 CUDA reference.

## Status

Early bootstrap. First slice: the single-matrix commitment scheme (SMCS, ≈ SP1's
`CudaTcsProver::commit_tensors`) ported onto `zorch`'s Sponge / Compression /
MerkleTree blocks. Tracking: [fractalyze/zorch#37](https://github.com/fractalyze/zorch/issues/37).

## Development

`sp1-zorch` is pure Python on JAX + the ZKX PJRT plugin, built with Bazel
(bzlmod). It consumes `zorch` as a Bazel module, pinned in `MODULE.bazel` via
`git_override` for reproducible builds.

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

**Dev against a local `zorch` checkout** instead of the pinned commit — create
`.bazelrc.user` (gitignored):

```
common --override_module=zorch=/abs/path/to/your/zorch/checkout
```

Run the tests (CPU is the default for correctness; the FFI byte-match against
the SP1 reference needs a CUDA GPU):

```sh
bazel test //...
```

## Documentation

See [`docs/`](docs/README.md) — the [architecture](docs/architecture.md)
(the shard proof as a ProveChain of Stages, each running inner Rounds, threaded
by a Bridge, plus the SP1 dump vocabulary), the
[development guide](docs/development.md) (environment, testing, and the
per-stage SP1 baseline), and the [conventions](docs/conventions.md).

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
