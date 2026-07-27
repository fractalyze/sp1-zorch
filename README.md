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

## Installation

**Python 3.11 on Linux x86_64 only.**

### CPU

```sh
pip install sp1-zorch
```

### GPU (CUDA 12)

```sh
pip install sp1-zorch 'frx[cuda12]' \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The extra index carries the CUDA plugin wheels, which are too large for PyPI's
per-file limit. It is not needed for the CPU tier.

### Verify

```sh
python -c "import frx, sp1_zorch; print(frx.devices()); print(sp1_zorch.__version__)"
```

`[CpuDevice(id=0)]` means the CPU tier; a CUDA install prints the GPU devices.

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

Install the git hooks with both stages named. Plain `pre-commit install` wires
only the `pre-commit` stage, which leaves the commit-message linter inactive —
a malformed commit message then sails through to CI:

```sh
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org):
a valid type, a lowercase summary with no trailing period, a header of at most
80 characters, and a body on everything but `docs`. The scope is the package the
change lives in — `logup_gkr`, `poseidon2`, `shard_prover`, `zerocheck` — plus
`release` for the version in `sp1_zorch/__init__.py`. A change spanning several
packages takes no scope. The same linter runs in CI over every commit in a pull
request and over the PR title.

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
