# sp1-zorch

A lean **SP1 prover** built on [`zorch`](https://github.com/fractalyze/zorch)'s
scheme-agnostic SNARK building blocks. `zorch` provides the reusable pieces
(hashing, Merkle commitment, sumcheck, fold, …); `sp1-zorch` adds only the
SP1-specific glue on top — domain separator, verify codes, heap proof layout,
and the FFI byte-match against the SP1 reference prover.

```text
frx  ──▶  zorch (scheme-/zkVM-agnostic blocks)  ──▶  sp1-zorch (SP1 glue)
```

## Status

The full SP1 shard proving scheme runs on `zorch` blocks: a `ProveChain` of
trace commit → LogUp-GKR → zerocheck → jagged PCS, byte-matching SP1's reference
prover end to end (its `sp1_verify_shard` accepts the assembled proof). See
[`docs/architecture.md`](docs/architecture.md).

## Development

`sp1-zorch` is pure Python on frx (Field, Ring Accelerated), run
against the Fractalyze XLA GPU plugin, built with Bazel (bzlmod). It consumes
`zorch` as a Bazel module, pinned in `MODULE.bazel` via
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
