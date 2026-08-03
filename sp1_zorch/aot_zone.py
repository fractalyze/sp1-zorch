# Copyright 2026 Fractalyze, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A jit zone split into its lowering and its executable, so the two can be
discarded independently.

A long-lived prover on a fixed card has to shed compiled executables to fit
its next prove — the keccak-class peak leaves no room for the classes already
resident. Dropping a ``jit`` cache entry sheds the executable *and* the
lowering with it, so the next prove of that class re-traces and re-lowers from
scratch. Nothing caches a lowering: the persistent compile cache stores
binaries only. Measured on the total-cap zone, that re-derivation is ~80% of
the reload — 11.7 s of the 14.5 s for a small class, and minutes for a big one.

The lowering is deterministic and shape-keyed, so re-deriving it after an
eviction buys nothing. :class:`AotZone` keeps it and treats only the executable
as disposable: a cleared zone recompiles from the retained lowering, which hits
the persistent compile cache and returns in milliseconds.

The call surface matches ``frx.jit`` for the ways this zone is used — call it,
ask for ``_cache_size()``, ``clear_cache()`` it — so callers that already drive
a jitted zone do not change.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

import frx

__all__ = ["AotZone", "aot_zone"]


def _aval_key(value: Any) -> Hashable:
    """A traced argument's contribution to the compile key.

    Flattened rather than walked by hand: a traced argument is any pytree the
    jit accepts — arrays, but also per-chip mappings of dataclasses — and only
    the structure plus each leaf's shape and dtype may enter the key. Hashing
    the leaves themselves would raise (arrays are unhashable) and keying on
    identity would compile once per call, since these containers are rebuilt
    every prove.
    """
    leaves, treedef = frx.tree_util.tree_flatten(value)
    return (
        treedef,
        tuple(
            (
                (tuple(leaf.shape), str(leaf.dtype))
                if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
                else leaf
            )
            for leaf in leaves
        ),
    )


def _is_traced(args: tuple[Any, ...]) -> bool:
    """Whether this call is happening inside an enclosing trace."""
    return any(
        isinstance(leaf, frx.core.Tracer) for leaf in frx.tree_util.tree_leaves(args)
    )


class AotZone:
    """``frx.jit`` driven through an explicit lower/compile split.

    Lowerings are keyed exactly as the jit would key them — the static kwargs
    plus every traced argument's shape and dtype — and are never evicted;
    executables live in a parallel map that :meth:`clear_cache` empties.
    """

    def __init__(self, fn: Callable[..., Any], *, static_argnames: tuple[str, ...]):
        self._jitted = frx.jit(fn, static_argnames=static_argnames)
        self._static_argnames = static_argnames
        self._lowered: dict[Hashable, Any] = {}
        self._compiled: dict[Hashable, Any] = {}

    def _key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
        statics = tuple(sorted((k, v) for k, v in kwargs.items()))
        return (statics, tuple(_aval_key(a) for a in args))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _is_traced(args):
            # Under an outer trace (the whole chain lowered as one program)
            # there is nothing to evict and no concrete signature to compile
            # for: an executable cannot accept tracers, whereas the jit simply
            # inlines. Compose like a jit and leave the split to the top level.
            return self._jitted(*args, **kwargs)
        key = self._key(args, kwargs)
        compiled = self._compiled.get(key)
        if compiled is None:
            lowered = self._lowered.get(key)
            if lowered is None:
                lowered = self._jitted.lower(*args, **kwargs)
                self._lowered[key] = lowered
            compiled = lowered.compile()
            self._compiled[key] = compiled
        return compiled(*args)

    def clear_cache(self) -> None:
        """Drop the executables, keep the lowerings.

        The caller's reason for clearing is device memory, which the
        executables hold; the lowerings are host-side and are exactly what a
        re-derivation would rebuild.
        """
        self._compiled.clear()

    def clear_lowerings(self) -> None:
        """Drop everything, including the wrapped jit's own trace cache — for
        a caller shedding host memory, which the retained lowerings do cost.
        Clearing only this object's map would leave the jit able to hand back
        a cached trace, so the next call would not actually re-derive."""
        self._compiled.clear()
        self._lowered.clear()
        self._jitted.clear_cache()

    def _cache_size(self) -> int:
        """Executables currently resident, the quantity callers gate memory
        decisions on. Retained lowerings are deliberately not counted."""
        return len(self._compiled)

    def lowering_count(self) -> int:
        return len(self._lowered)


def aot_zone(**jit_kwargs: Any) -> Callable[[Callable[..., Any]], AotZone]:
    """Decorator form: ``@aot_zone(static_argnames=(...))``."""

    def wrap(fn: Callable[..., Any]) -> AotZone:
        return AotZone(fn, **jit_kwargs)

    return wrap
