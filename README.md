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

The full SP1 shard proving scheme runs on `zorch` blocks: a sequence of
trace commit → LogUp-GKR → zerocheck → jagged PCS, byte-matching SP1's reference
prover end to end (its `sp1_verify_shard` accepts the assembled proof). See
[`docs/architecture.md`](docs/architecture.md).

## Installation

**Python 3.11 on Linux x86_64, or macOS on Apple Silicon.** (`frxlib` ships a
cp311 wheel for those two platforms only — not 3.12/3.13, not Intel Macs.)

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

Pure Python on frx (Field, Ring Accelerated), built with Bazel (bzlmod), with
`zorch` consumed as a Bazel module pinned in `MODULE.bazel`.

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
bazel test //sp1_zorch/... --test_tag_filters=-gpu_only
```

Both `--hook-type` flags matter: plain `pre-commit install` wires only the
`pre-commit` stage, leaving the commit-message linter inactive so a malformed
message sails through to CI.

[`docs/development.md`](docs/development.md) has the rest — devving against a
local `zorch` checkout, the coupled zorch/frx pin, the GPU-plugin gotcha, test
conventions, and the per-phase SP1 baseline.

## Documentation

[`docs/`](docs/README.md) indexes the architecture, development, and
conventions guides.

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
