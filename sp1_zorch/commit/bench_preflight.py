# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Devenv-provenance preflight for the trace-commit benches / verify tools.

A bench number is only a valid baseline if it measures **shipped** code. This
devenv drives ``zorch`` through a dev-only ``--override_module`` and (during the
#153 / zkx#751 work) a patched copy of the pip-extracted ``jax`` — both can
silently point at stale or superseded code:

  - sp1-zorch#153's first encode baseline was taken against a ``zorch`` override
    weeks behind ``origin/main`` (predating the #220/#225 encode rework) and
    misread as the shipped number.
  - Post-zkx#756 the shipped encode path is the GS/DIF rewriter firing on stock
    ``lax.bit_reverse(lax.fft(x))``; a leftover override on the dead
    ``perf/encode-ntt-shard0`` branch + a ``bit_reverse_output``-patched jax
    would measure a code path the team no longer ships.

``check_devenv`` resolves what the bench actually loaded — the ``zorch`` source
checkout's branch / HEAD / dirty state and how far it sits behind ``origin/main``,
plus whether ``jax``'s fft carries the superseded ``bit_reverse_output`` patch —
prints it as a banner, and (``strict=True``) aborts on a stale devenv so it can't
be reported as a baseline. See ``docs/sp1-baseline.md``.

The git / jax / env lookups are thin wrappers; the classification logic
(``parse_override_path``, ``source_warnings``, ``jax_warnings``) is pure and unit
tested.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TextIO

_ZORCH_MODULE = "zorch"
# The env var baseline scripts set to make a stale devenv a hard error.
_STRICT_ENV = "SP1_BENCH_STRICT_DEVENV"
# The superseded jax fft parameter (jax#226, closed) — its presence in the
# loaded jax source means a patched, non-shipped fft path.
_SUPERSEDED_JAX_FFT_TOKEN = "bit_reverse_output"


@dataclasses.dataclass(frozen=True)
class SourceState:
    """git state of a resolved ``zorch`` source checkout."""

    path: Path
    branch: str
    head: str
    dirty: bool
    behind: int  # commits behind local origin/main; -1 if unknown
    on_main_lineage: bool


# --------------------------------------------------------------------------- #
# Pure classification logic (unit tested)
# --------------------------------------------------------------------------- #
def parse_override_path(bazelrc_text: str, module: str = _ZORCH_MODULE) -> str | None:
    """Return the last active ``--override_module=<module>=PATH`` path, or None.

    Commented lines are ignored; later definitions win (bazel last-wins
    semantics), matching how ``.bazelrc.user`` overrides ``.bazelrc``.
    """
    token = f"--override_module={module}="
    found: str | None = None
    for raw in bazelrc_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        idx = line.find(token)
        if idx == -1:
            continue
        rest = line[idx + len(token) :].split()
        if rest:
            found = rest[0]
    return found


def source_warnings(state: SourceState) -> list[str]:
    """Staleness warnings for a resolved zorch override checkout."""
    out: list[str] = []
    if not state.on_main_lineage:
        out.append(
            f"zorch override is on '{state.branch}' — off origin/main lineage "
            "(likely a dead/feature branch; the shipped path is on main)"
        )
    elif state.behind > 0:
        out.append(
            f"zorch override is {state.behind} commit(s) behind origin/main "
            "(as of the last local fetch) — may predate shipped reworks"
        )
    if state.dirty:
        out.append(f"zorch override at {state.path} has uncommitted changes")
    return out


def jax_warnings(patched_fft: bool | None) -> list[str]:
    """Warn when jax carries the superseded fft patch (shipped needs stock jax)."""
    if patched_fft:
        return [
            f"jax fft carries the superseded `{_SUPERSEDED_JAX_FFT_TOKEN}` patch — "
            "the shipped encode path needs stock jax (zkx#756 rewrites "
            "lax.bit_reverse(lax.fft(x)) itself)"
        ]
    return []


def module_pin_commit(module_bazel_text: str, module: str = _ZORCH_MODULE) -> str | None:
    """Extract the ``git_override`` commit for ``module`` (for the pinned-wheel display)."""
    # Walk every git_override block; return the commit of the one for `module`.
    for m in re.finditer(r"git_override\((?P<body>.*?)\)", module_bazel_text, re.DOTALL):
        body = m.group("body")
        if re.search(rf'module_name\s*=\s*"{re.escape(module)}"', body):
            c = re.search(r'commit\s*=\s*"([0-9a-f]{7,40})"', body)
            return c.group(1) if c else None
    return None


# --------------------------------------------------------------------------- #
# IO wrappers (not unit tested — exercised via the bench)
# --------------------------------------------------------------------------- #
def _git(root: Path, *args: str) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_source_state(start: Path) -> SourceState | None:
    top = _git(start, "rev-parse", "--show-toplevel")
    if not top:
        return None
    root = Path(top)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "?"
    head = _git(root, "rev-parse", "--short", "HEAD") or "?"
    dirty = bool(_git(root, "status", "--porcelain"))
    is_ancestor = (
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", "HEAD", "origin/main"],
            capture_output=True,
        ).returncode
        == 0
    )
    behind_raw = _git(root, "rev-list", "--count", "HEAD..origin/main")
    behind = int(behind_raw) if behind_raw and behind_raw.isdigit() else -1
    return SourceState(
        path=root,
        branch=branch,
        head=head,
        dirty=dirty,
        behind=behind,
        on_main_lineage=(branch == "main") or is_ancestor,
    )


def _loaded_zorch_dir() -> Path | None:
    try:
        import zorch  # noqa: PLC0415 — lazy so the unit test needs no zorch/GPU

        f = getattr(zorch, "__file__", None)
        return Path(os.path.realpath(f)).parent if f else None
    except Exception:  # pragma: no cover - import-environment dependent
        return None


def _resolve_jax() -> tuple[str, bool | None]:
    try:
        import jax  # noqa: PLC0415 — lazy

        version = getattr(jax, "__version__", "?")
        pkg = Path(jax.__file__).parent
    except Exception:  # pragma: no cover
        return "?", None
    seen = False
    for rel in (("_src", "lax", "fft.py"), ("_src", "numpy", "fft.py")):
        try:
            text = (pkg.joinpath(*rel)).read_text()
        except OSError:
            continue
        seen = True
        if _SUPERSEDED_JAX_FFT_TOKEN in text:
            return version, True
    return version, (False if seen else None)


def _workspace() -> Path | None:
    # `bazel run` exports this to the sp1-zorch source root.
    ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    return Path(ws) if ws else None


def _resolve_zorch() -> tuple[str, list[str]]:
    """Return (display line, warnings) for the zorch the bench actually loaded."""
    state = None
    loaded = _loaded_zorch_dir()
    if loaded is not None:
        state = _resolve_source_state(loaded)
    # Fall back to the configured override path when runfiles aren't a git tree.
    ws = _workspace()
    if state is None and ws is not None:
        for name in (".bazelrc.user", ".bazelrc"):
            f = ws / name
            if not f.exists():
                continue
            override = parse_override_path(f.read_text())
            if override:
                state = _resolve_source_state(Path(override))
                if state is not None:
                    break
    if state is not None:
        # The dirty / behind / off-main detail is carried by source_warnings.
        line = f"OVERRIDE {state.path} @ {state.branch} ({state.head})"
        return line, source_warnings(state)
    pin = None
    if ws is not None:
        try:
            pin = module_pin_commit((ws / "MODULE.bazel").read_text())
        except OSError:
            pin = None  # preflight must never crash the bench it guards
    return ("pinned wheel" + (f" (zorch @ {pin})" if pin else "")), []


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def check_devenv(*, strict: bool | None = None, out: TextIO = sys.stderr) -> None:
    """Print the devenv provenance banner; warn (or abort, ``strict``) if stale.

    ``strict`` defaults to the ``SP1_BENCH_STRICT_DEVENV`` env var so baseline
    scripts can fail-closed without a CLI flag.
    """
    if strict is None:
        strict = os.environ.get(_STRICT_ENV, "").lower() in ("1", "true", "yes")

    zorch_line, warnings = _resolve_zorch()
    jax_version, patched = _resolve_jax()
    if patched:
        jax_note = "  [fft PATCHED: bit_reverse_output]"
    elif patched is None:
        jax_note = "  [fft patch state unknown]"
    else:
        jax_note = ""
    warnings = warnings + jax_warnings(patched)
    zkx = os.environ.get("ZKX_GPU_PLUGIN_PATH") or "wheel (no ZKX_GPU_PLUGIN_PATH)"

    print("=== sp1-zorch bench devenv provenance ===", file=out)
    print(f"  zorch : {zorch_line}", file=out)
    print(f"  jax   : {jax_version}{jax_note}", file=out)
    print(f"  zkx   : {zkx}", file=out)
    if warnings:
        print("  WARNING: stale devenv — this run may measure non-shipped code:", file=out)
        for w in warnings:
            print(f"    - {w}", file=out)
        print(
            "  Do NOT report this as a baseline. Reset the @zorch override to "
            "origin/main (or drop it) and use stock jax; see docs/sp1-baseline.md.",
            file=out,
        )
        if strict:
            raise SystemExit(
                "bench preflight: stale devenv with strict mode on "
                f"({_STRICT_ENV}=1 / --strict_devenv). Aborting."
            )
    else:
        print("  (clean: measuring shipped code)", file=out)
    print("=" * 42, file=out)
