# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The variant=sp1-zerocheck marker is byte-transparent: an unclaimed marker
(CPU) decomposes inline, reproducing the plain _reduce_and_assemble exactly."""
from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.poly.geq import VirtualGeq
from zorch.sumcheck.gruen import interp_matrix

from sp1_zorch.zerocheck._round_composite import zerocheck_round_poly
from sp1_zorch.zerocheck.coeffs import gkr_powers, rlc_coeffs
from sp1_zorch.zerocheck.jagged import (
    JaggedZerocheckSummand,
    _reduce_and_assemble,
    _summand_values,
)

BF, EF = koalabear_mont, koalabearx4_mont

# Witness chip: columns [a, b, c] with a == 1 on every real row, so both
# constraints vanish there while C(0_row) = [1, 0] keeps the padded-row
# correction live (adj = alpha_0 != 0) — mirrors jagged_test.py's fixture.
_NUM_COLS = 3
_K = 2
_PV = fnp.zeros((8,), dtype=BF)


def _eval_fn(trace: fnp.ndarray, public_values: fnp.ndarray) -> fnp.ndarray:
    del public_values
    a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
    one = fnp.ones((), trace.dtype)
    return fnp.stack([(a - one) * (c - one), (a - one) * b * c], axis=-1)


def _rand(seed: int, shape) -> fnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return fnp.array(ints, dtype=BF)


def _witness_trace(seed: int, nr: int) -> fnp.ndarray:
    ones = fnp.ones((1, nr), dtype=BF)
    return fnp.concatenate([ones, _rand(seed, (2, nr))], axis=0)


def _u32(a) -> np.ndarray:
    return np.asarray(frx.lax.bitcast_convert_type(a, fnp.uint32)).reshape(-1)


def _assert_bytes_equal(got, want, label: str = "") -> None:
    """Montgomery-form ``u32`` comparison — the repo's byte-exact convention
    (no float tolerance applies to field elements)."""
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


class MarkerByteTransparencyTest(absltest.TestCase):
    def test_marker_matches_plain_reduce(self):
        # Build a small live constrained chip's round inputs — pair width 4
        # (nr_live = 8, buffer width 8, so the live prefix is the whole
        # buffer and no truncation muddies the comparison).
        nr = 8
        trace = _witness_trace(0, nr)
        alpha = rlc_coeffs(_rand(99, ()), _K)
        beta = _rand(77, ())
        summand = JaggedZerocheckSummand(
            eval_fns=[_eval_fn],
            alphas=[alpha],
            lambdas=_rand(55, (1,)),
            beta=beta,
            public_values=_PV,
        )

        gkr = gkr_powers(beta, _NUM_COLS)
        # Row-major carry (fractalyze/sp1-zorch#242): `[rows, num_cols]` trace,
        # axis-0 stride-2 pair fold (the dual of the old last-axis split).
        trace_rm = trace.T
        p0, p1 = trace_rm[0::2], trace_rm[1::2]
        diff = p1 - p0
        eq = _rand(3, (nr // 2,))
        nr_live = fnp.asarray(nr, fnp.int32)
        claim = _rand(9, ())
        last = _rand(11, ())
        eq_adj = _rand(13, ())
        padded_row_adj = _eval_fn(fnp.zeros((1, _NUM_COLS), dtype=BF), _PV)[0] @ alpha
        vgeq = VirtualGeq(nr_live, fnp.ones((), BF), fnp.zeros((), BF))
        interp = interp_matrix((fnp.array(2, BF), fnp.array(4, BF)), last)
        is_round0 = fnp.array(False)

        term, alpha0 = summand._term_fns[0], summand.alphas[0]
        vals = _summand_values(term, alpha0, p0, diff, gkr, nr_live, is_round0)

        want = _reduce_and_assemble(
            vals, eq, interp, claim, last, eq_adj, padded_row_adj, nr_live, vgeq
        )
        got = zerocheck_round_poly(
            vals, eq, interp, claim, last, eq_adj, padded_row_adj, nr_live, vgeq
        )
        _assert_bytes_equal(got, want)


if __name__ == "__main__":
    absltest.main()
