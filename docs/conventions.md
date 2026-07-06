# Coding conventions

> Code, symbols, file paths, and comments are English.

sp1-zorch follows the team playbook and inherits zorch's conventions
(`fractalyze/zorch:docs/conventions.md` — `@jit`, naming, type annotations).
This file records the rules that bite hardest in an SP1 *consumer* repo:
how comments are scoped, how we cite the SP1 reference we mirror, and how
protocol verifiers are named apart from the byte-match harnesses. The
repo-level "SP1-specific only" rule lives in [`../CLAUDE.md`](../CLAUDE.md);
test `size`/`timeout` and fixture conventions live in [`testing.md`](testing.md).

## Comments & documentation

Comments and `docs/` prose carry only what the code can't: **WHY** — the
rationale for a non-obvious choice, a hidden constraint or invariant, a rule, an
external reference. They never restate **WHAT** the code does; the *what* already
lives in the names, the types, and the tests (which run every commit and can't
drift). This mirrors zorch's rule — keep them consistent.

- **WHY, not WHAT.** `# build the first layer` above `_first_layer(...)` is
  noise. `# Four distinct seeds so numerators/denominators don't alias` earns
  its line. If a name or type already says it, drop the comment.
- **Temporally neutral.** State the current permanent rule, not the journey. No
  "used to", "was blocked on", "lands in a follow-up" — `git log` carries the
  chronology; in-tree narration rots within a commit or two.
- **Self-contained.** A reader has only the source tree and history. No
  session/spec labels (`Q1:A`, `Approach D`), no uncommitted-file references, no
  home/scratch paths. Link rationale in a tracked file by its repo-relative
  path.
- **A docstring is rationale, not a signature echo.** `_prove`'s docstring says
  *why* it returns the last layer's round polys (the tail of the carry, so
  awaiting it times the whole prove), not that it "runs the prove."

[`sp1_zorch/logup_gkr/bench_sp1_logup_gkr.py`](../sp1_zorch/logup_gkr/bench_sp1_logup_gkr.py)
is the worked shape.

## External zkVM references

This repo byte-matches and benchmarks against the SP1 reference prover. When code
mirrors or is sized from upstream SP1 (or any external zkVM), leave a **pinned
GitHub permalink** so a future reader can diff against the exact source. Same
rule as `fractalyze/riscv-witness:docs/conventions/docs-and-refs.md`.

- **Short commit SHA, never a branch** — use git's abbreviation
  (`git rev-parse --short <sha>`); GitHub resolves it and it stays readable,
  while a branch link rots the moment upstream moves.
- **Pin a line range** (`#L80-L120`) when a specific span is the reference; a
  bare blob link is fine when the whole file is the scope.
- **Place it where the dependence lives** — the module docstring for file-wide
  sizing, directly above the adapted block for a local mirror.

```python
# Reference: https://github.com/fractalyze/sp1/blob/<sha>/path/to/file.rs#L10-L25
```

The LogUp-GKR bench sizes its first layer from SP1's `logup_gkr_zkbench`, pinned
in its module docstring:

```
https://github.com/fractalyze/sp1/blob/e2c02f376/sp1-gpu/crates/logup_gkr/bin/logup_gkr_zkbench.rs
```

## `verifier.py` vs `verify_*` runnables

Two artifact kinds share the "verif" prefix; they are different things:

- **`verifier.py`** (in a stage module) — product code: the stage's protocol
  verifier dual. It checks a *proof* — transcript replay plus oracle checks —
  and mirrors SP1's reference verifier.
- **`verify_*` py_binary runnables** (`verify_prove_shard`, `verify_zerocheck`,
  `verify_gkr_prove`, `verify_first_layer`) — dev harnesses: rsp byte-match of
  the *prover* against an external SP1 reference dump (`--shard_dir`), for
  full-shard dumps too large to vendor. Their cold-path units
  (`verify_*_test`) test the harness, not the protocol.

Name new files accordingly: a protocol dual is `verifier.py`; a dump-diff
harness gets the `verify_` prefix.
