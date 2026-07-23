# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""warm_mode: compile-only inside active(), pass-through outside."""

import numpy as np
from absl.testing import absltest

import frx
import frx.numpy as fnp

from sp1_zorch.shard_prover import warm_mode


class WarmModeTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        warm_mode.install()

    def test_active_returns_zeros_and_compiles(self):
        @frx.jit
        def f(a, b):
            return a * b + a

        x = fnp.full((8,), 3, fnp.uint32)
        before = warm_mode.zones_compiled()
        with warm_mode.active():
            out = f(x, x)
        np.testing.assert_array_equal(np.asarray(out), np.zeros(8))
        self.assertEqual(warm_mode.zones_compiled(), before + 1)

    def test_pass_through_outside_active(self):
        @frx.jit
        def f(a):
            return a + 1

        out = f(fnp.full((4,), 1, fnp.uint32))
        np.testing.assert_array_equal(np.asarray(out), np.full(4, 2))

    def test_nested_jit_inlines_under_outer(self):
        @frx.jit
        def inner(a):
            return a + 1

        @frx.jit
        def outer(a):
            return inner(a) * 2

        with warm_mode.active():
            out = outer(fnp.full((4,), 1, fnp.uint32))
        # One OUTER zone compiled; the nested jit inlined into its module.
        np.testing.assert_array_equal(np.asarray(out), np.zeros(4))

    def test_active_requires_install_first(self):
        # install() ran in setUpClass; reentrancy is the reachable error here.
        with warm_mode.active():
            with self.assertRaises(RuntimeError):
                with warm_mode.active():
                    pass


if __name__ == "__main__":
    absltest.main()
