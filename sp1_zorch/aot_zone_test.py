# Copyright 2026 Fractalyze, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The lowering outlives the executable.

The point of the split is that a zone cleared for device memory does not pay
to re-derive its lowering on the way back, so the tests that matter are: a
clear keeps the lowerings, the recovered executable still computes the right
answer, and the keying matches what a jit would do — one executable per
(static args, input shape/dtype), not per call.
"""

from typing import Any

import frx
import frx.numpy as fnp
from absl.testing import absltest

from sp1_zorch.aot_zone import AotZone, aot_zone


def _scaled(x: Any, *, factor: int) -> Any:
    return x * factor


class AotZoneTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.traces = 0

        def counted(x: Any, *, factor: int) -> Any:
            self.traces += 1
            return _scaled(x, factor=factor)

        self.zone = AotZone(counted, static_argnames=("factor",))

    def test_matches_the_unwrapped_function(self) -> None:
        x = fnp.arange(4, dtype=fnp.float32)
        out = self.zone(x, factor=3)
        self.assertSequenceEqual(
            list(out), list(frx.jit(_scaled, static_argnames=("factor",))(x, factor=3))
        )

    def test_one_executable_per_static_and_shape(self) -> None:
        x4 = fnp.arange(4, dtype=fnp.float32)
        x8 = fnp.arange(8, dtype=fnp.float32)
        self.zone(x4, factor=2)
        self.zone(x4, factor=2)  # same key: reuses, no new trace
        self.assertEqual(self.zone._cache_size(), 1)
        self.assertEqual(self.traces, 1)

        self.zone(x4, factor=3)  # new static
        self.zone(x8, factor=2)  # new shape
        self.assertEqual(self.zone._cache_size(), 3)
        self.assertEqual(self.traces, 3)

    def test_clear_cache_drops_executables_but_keeps_lowerings(self) -> None:
        x = fnp.arange(4, dtype=fnp.float32)
        self.zone(x, factor=2)
        self.zone(x, factor=5)
        self.assertEqual(self.zone._cache_size(), 2)
        self.assertEqual(self.zone.lowering_count(), 2)

        self.zone.clear_cache()

        self.assertEqual(self.zone._cache_size(), 0)
        self.assertEqual(self.zone.lowering_count(), 2)

    def test_recompiles_from_the_retained_lowering_without_re_tracing(self) -> None:
        x = fnp.arange(4, dtype=fnp.float32)
        expected = list(self.zone(x, factor=7))
        self.zone.clear_cache()

        recovered = list(self.zone(x, factor=7))

        self.assertSequenceEqual(recovered, expected)
        self.assertEqual(self.traces, 1)  # the whole point: no re-derivation
        self.assertEqual(self.zone._cache_size(), 1)

    def test_clear_lowerings_forces_a_re_trace(self) -> None:
        x = fnp.arange(4, dtype=fnp.float32)
        self.zone(x, factor=2)
        self.zone.clear_lowerings()
        self.assertEqual(self.zone.lowering_count(), 0)

        self.zone(x, factor=2)

        self.assertEqual(self.traces, 2)

    def test_decorator_form(self) -> None:
        @aot_zone(static_argnames=("factor",))
        def doubled(x: Any, *, factor: int) -> Any:
            return x * factor

        self.assertIsInstance(doubled, AotZone)
        x = fnp.arange(3, dtype=fnp.float32)
        self.assertSequenceEqual(list(doubled(x, factor=2)), [0.0, 2.0, 4.0])

    def test_composes_under_an_outer_jit(self) -> None:
        # The chain is also lowered as one program; an executable cannot take
        # tracers, so a traced call has to fall back to the jit and inline.
        x = fnp.arange(4, dtype=fnp.float32)
        outer = frx.jit(lambda v: self.zone(v, factor=2) + 1)

        out = outer(x)

        self.assertSequenceEqual(list(out), [1.0, 3.0, 5.0, 7.0])
        self.assertEqual(self.zone._cache_size(), 0)  # nothing to evict here

    def test_mapping_arguments_key_on_leaves_not_identity(self) -> None:
        # Per-chip openings arrive as a fresh dict every prove; keying on
        # identity would compile once per call.
        def summed(d: Any, *, tag: str) -> Any:
            return d["a"] + d["b"]

        zone = AotZone(summed, static_argnames=("tag",))
        for _ in range(3):
            zone({"a": fnp.ones(4), "b": fnp.ones(4)}, tag="t")
        self.assertEqual(zone._cache_size(), 1)


if __name__ == "__main__":
    absltest.main()
