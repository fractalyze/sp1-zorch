# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""zerocheck coefficient layouts: descending-power RLC coeffs, shifted GKR
column weights, and a fold byte-equal to the plain dot it marks."""

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from sp1_zorch.zerocheck.coeffs import constraint_rlc, gkr_powers, rlc_coeffs


def _eval_fn(rows: fnp.ndarray, public_values: fnp.ndarray) -> fnp.ndarray:
    """A chip-eval stand-in: rows [N, num_cols] -> K=3 straight-line
    constraints [N, 3]. Self-contained so the test anchors on its own golden.
    Ignores ``public_values`` — a constraint that declares no pv_arg still
    takes the statement as a declared operand."""
    del public_values
    c0 = rows[:, 0] * rows[:, 1]
    c1 = rows[:, 1] + rows[:, 2]
    c2 = rows[:, 0] * rows[:, 2] + rows[:, 1]
    return fnp.stack([c0, c1, c2], axis=-1)


_ROWS = fnp.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [2, 3, 4],
        [5, 6, 7],
        [8, 9, 1],
        [3, 4, 5],
        [6, 7, 8],
    ],
    dtype=F,
)
_ALPHA = fnp.array(3, dtype=F)
_PV = fnp.zeros((8,), dtype=F)


class ConstraintRlcTest(absltest.TestCase):
    def test_rlc_coeffs_are_descending_powers(self) -> None:
        # SP1 folds constraint k with alpha^(K-1-k): [alpha^2, alpha, 1].
        want = fnp.stack([_ALPHA * _ALPHA, _ALPHA, fnp.ones((), dtype=F)])
        got = rlc_coeffs(_ALPHA, 3)
        self.assertTrue(bool(fnp.array_equal(got, want)), (got, want))

    def test_rlc_coeffs_empty_for_constraint_less_chip(self) -> None:
        # Lookup-only chips (SP1's Byte / Program / Range) carry K=0.
        self.assertEqual(rlc_coeffs(_ALPHA, 0).shape, (0,))
        with self.assertRaises(ValueError):
            rlc_coeffs(_ALPHA, -1)

    def test_gkr_powers_start_at_beta(self) -> None:
        # Column j carries beta**(j+1) — SP1's skip(1): [beta, beta^2, beta^3].
        want = fnp.stack([_ALPHA, _ALPHA * _ALPHA, _ALPHA * _ALPHA * _ALPHA])
        got = gkr_powers(_ALPHA, 3)
        self.assertTrue(bool(fnp.array_equal(got, want)), (got, want))
        # An empty column batch folds to nothing, mirroring rlc_coeffs' K=0.
        self.assertEqual(gkr_powers(_ALPHA, 0).shape, (0,))

    def test_folds_byte_equal_to_plain_dot(self) -> None:
        # The composite must inline to the identical result as the plain
        # `eval_fn(rows) @ alpha_slice` the SP1 reference computes.
        coeffs = rlc_coeffs(_ALPHA, 3)
        golden = _eval_fn(_ROWS, _PV) @ coeffs
        got = constraint_rlc(_eval_fn, _ROWS, _ALPHA, 3, _PV)
        self.assertTrue(bool(fnp.array_equal(got, golden)), (got, golden))


if __name__ == "__main__":
    absltest.main()
