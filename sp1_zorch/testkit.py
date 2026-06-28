# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared test helpers for sp1-zorch.

``force_inline_composite_markers`` / ``inline_composite_markers`` exist to dodge
a zkx CPU-emitter bug: the kernel emitter in the pinned jaxlib wheel CHECK-fails
(``symbolic_map.cc:196``) when it compiles an engaged ``zorch.constraint_eval``
composite region on the CPU backend, aborting the process. The marker is
semantically transparent — a compiler that does not recognize it inlines the
decomposition to the identical tensor — so replacing ``lax.composite`` with an
inlining passthrough yields byte-identical output while sidestepping the
recogniser that trips the CHECK.

**The inline is surgical: only ``zorch.constraint_eval`` is inlined.** Every
other marker (notably ``zorch.poseidon2`` / ``zorch.poseidon2_sponge_hash``, the
SMCS Merkle leaf hash and fold compression) keeps the real ``lax.composite``
path. A *global* inline corrupts the Merkle commit: the inlining passthrough is
NOT byte-identical to the real composite for the poseidon2 permutation on this
wheel, so an inlined commit and an inlined reconstruct disagree and every honest
open's Merkle path fails to authenticate (``ROOT_MISMATCH``). The real composite
path for poseidon2 is exercised crash-free by ``commit/smcs_test`` and
``commit/trace_commit_test`` (neither inlines), so scoping the inline to the one
marker that actually trips the CHECK is both necessary and sufficient.

This is the successor to the ``zorch._composite._HAS_COMPOSITE_OP = False``
stanza these tests used for the earlier rank-1 ``linalg.broadcast`` variant of
the same emitter bug (fractalyze/zkx#605, fixed in fractalyze/zkx#652); zorch
dropped that flag when composite emission became unconditional
(fractalyze/zorch#329), so the toggle now lives here instead.

Re-engaging the markers on the CPU backend is tracked by fractalyze/sp1-zorch#62
— delete these helpers and their call sites once a published wheel embeds the
emitter fix.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

from jax import Array, lax

_Region = TypeVar("_Region")

# Captured at import so the context-manager form can restore the real emitter,
# and so the selective passthrough can defer to it for every non-inlined marker.
_real_composite = lax.composite

# The only marker that trips the CPU emitter CHECK (symbolic_map.cc:196). Inline
# exactly this one; route everything else (poseidon2 Merkle hash/compress,
# sumcheck) through the real composite, which a global inline would silently
# corrupt for the Merkle commit.
_INLINED_MARKERS = frozenset({"zorch.constraint_eval"})


def _inlining_composite(
    decomposition: Callable[..., _Region], *, name: str, version: int = 0
) -> Callable[..., _Region]:
    """Drop-in for ``jax.lax.composite`` that inlines only ``_INLINED_MARKERS``.

    Mirrors ``zorch._composite.composite``'s call shape
    (``lax.composite(decomposition, name=, version=)(*operands, **attrs)``). For a
    marker in ``_INLINED_MARKERS`` it runs ``decomposition`` directly, so no
    ``stablehlo.composite`` op reaches the backend; for every other marker it
    defers to the real ``lax.composite`` so the emitted region — and its lowered
    value — is unchanged.
    """
    if name not in _INLINED_MARKERS:
        return _real_composite(decomposition, name=name, version=version)

    def run(*operands: Array, **attrs: object) -> _Region:
        return decomposition(*operands, **attrs)

    return run


def force_inline_composite_markers() -> None:
    """Globally inline every zorch composite marker for this test process.

    Call once at module scope in a test that lowers/executes an engaged
    ``constraint_eval`` region on the CPU backend. Each Bazel ``py_test`` runs in
    its own process, so the patch is isolated to that target. Do NOT use this in
    a module that also asserts on ``stablehlo.composite`` text in lowered IR —
    use :func:`inline_composite_markers` to scope the inline to the executing
    tests there.
    """
    lax.composite = _inlining_composite  # type: ignore[assignment]


@contextmanager
def inline_composite_markers() -> Iterator[None]:
    """Scope the composite-marker inline to a ``with`` block, then restore.

    For tests that execute an engaged ``constraint_eval`` region but share a
    module with tests that assert the marker IS emitted in lowered IR.
    """
    lax.composite = _inlining_composite  # type: ignore[assignment]
    try:
        yield
    finally:
        lax.composite = _real_composite  # type: ignore[assignment]
