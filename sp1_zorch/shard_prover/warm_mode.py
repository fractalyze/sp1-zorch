# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Toggleable compile-only mode for the prove chain — the warm_worker
intercept as a library, for hosts that warm and prove in one process.

``install()`` patches ``frx.jit`` with a dispatching wrapper. Outside
``active()`` the wrapper is a pass-through (one flag check per outer zone
call); inside, the OUTERMOST call in the eager chain lowers+compiles the zone
(persistent-cache write, no execute) and returns ``eval_shape``'d zeros so the
chain flows on shapes alone. A depth guard keeps nested zone calls inlined
into the outer module, exactly as a real prove compiles them.

zorch binds ``frx.jit`` at decoration (module import) time, so ``install()``
MUST run before any sp1_zorch/zorch prove module is imported — a context
manager alone cannot retrofit already-bound decorators. A long-lived prover
process calls ``install()`` at startup, warms under ``active()``, then proves
normally through the same wrappers.

Backend compiles are fire-and-forget onto a thread pool (the chain never
consumes the executable); ``active()`` drains them on exit and raises
``WarmError`` if any zone failed.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
from dataclasses import dataclass, field

import frx
import jax
import frx.numpy as fnp


class WarmError(RuntimeError):
    """One or more zone compiles failed during an ``active()`` block."""


@dataclass
class _State:
    real_jit: object = None
    on: bool = False
    depth: int = 0
    pool: concurrent.futures.ThreadPoolExecutor | None = None
    lower_device: object = None
    futures: list = field(default_factory=list)
    compiled: int = 0


_state = _State()


def install(compile_threads: int = 2, lower_device=None) -> None:
    """Patch ``frx.jit`` (idempotent). Call before prove-chain imports.

    ``compile_threads`` bounds concurrent zone compiles — on-device autotune
    scratch (~13.5 GiB per 400M-area zone) is the binding resource, not CPU.
    ``lower_device`` lowers zones under ``jax.default_device`` (a PJRT
    compile-only topology device makes the warm deviceless — experimental,
    cache-key parity with real-device proves is unresolved).
    """
    if _state.real_jit is not None:
        return
    _state.lower_device = lower_device
    _state.real_jit = frx.jit
    _state.pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=compile_threads)

    def dispatching_jit(fn=None, **kw):
        if fn is None:
            return lambda f: dispatching_jit(f, **kw)
        jitted = _state.real_jit(fn, **kw)

        def wrapper(*args, **kwargs):
            # Pass-through when warm mode is off, and for nested calls under
            # an outer lower/eval_shape trace (they must inline into the
            # outer zone's module).
            if not _state.on or _state.depth > 0:
                return jitted(*args, **kwargs)
            _state.depth += 1
            try:
                if _state.lower_device is not None:
                    with jax.default_device(_state.lower_device):
                        lowered = jitted.lower(*args, **kwargs)
                else:
                    lowered = jitted.lower(*args, **kwargs)
                out = jax.eval_shape(jitted, *args, **kwargs)
            finally:
                _state.depth -= 1
            _state.futures.append(_state.pool.submit(lowered.compile))
            return jax.tree_util.tree_map(
                lambda s: fnp.zeros(s.shape, s.dtype), out)

        return wrapper

    frx.jit = dispatching_jit


@contextlib.contextmanager
def active():
    """Compile-only mode for the enclosed chain run; drains compiles on exit.

    Raises ``WarmError`` on any failed zone compile. Not reentrant and not
    thread-safe — one warm at a time per process.
    """
    if _state.real_jit is None:
        raise RuntimeError("warm_mode.install() must run before prove imports")
    if _state.on:
        raise RuntimeError("warm_mode.active() is not reentrant")
    _state.on = True
    try:
        yield
    finally:
        _state.on = False
        errors = []
        for f in _state.futures:
            try:
                f.result()
                _state.compiled += 1
            except Exception as e:  # noqa: BLE001 — every failure surfaces below
                errors.append(f"{type(e).__name__}: {e}")
        _state.futures.clear()
    if errors:
        raise WarmError(
            f"{len(errors)} zone compile(s) failed: " + "; ".join(errors[:3]))


def zones_compiled() -> int:
    """Zones compiled by finished ``active()`` blocks in this process."""
    return _state.compiled
