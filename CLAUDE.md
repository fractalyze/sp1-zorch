# Project context for Claude Code

| Read | For |
| --- | --- |
| [`README.md`](README.md) | What sp1-zorch is, and how to install it |
| [`docs/architecture.md`](docs/architecture.md) | Stage / Round, the PCS commit half, and the SP1 dump "phase" vocabulary |
| [`docs/development.md`](docs/development.md) | Environment, the coupled zorch/frx pin, bazel gotchas, testing, and the per-phase SP1 baseline |
| [`docs/conventions.md`](docs/conventions.md) | Comment scoping, commit messages, and how SP1 reference code is cited |

## One non-negotiable

**SP1-specific only.** This repo holds the SP1 glue — domain separator, verify
codes, heap proof layout, FFI byte-match. Anything scheme- or zkVM-agnostic
belongs upstream in `zorch`. If a generic block is missing, add it to `zorch`
and depend on it; do not fork it into `sp1-zorch`.
