# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The variant=zerocheck marker is byte-transparent: an unclaimed marker
(CPU) decomposes inline, reproducing the plain _reduce_and_assemble exactly."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.poly.geq import VirtualGeq
from zorch.sumcheck.gruen import interp_matrix
from zorch.sumcheck.prover import split_pairs

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
_PV = jnp.zeros((8,), dtype=BF)


def _eval_fn(trace: jnp.ndarray, public_values: jnp.ndarray) -> jnp.ndarray:
    del public_values
    a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
    one = jnp.ones((), trace.dtype)
    return jnp.stack([(a - one) * (c - one), (a - one) * b * c], axis=-1)


def _rand(seed: int, shape) -> jnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=BF)


def _witness_trace(seed: int, nr: int) -> jnp.ndarray:
    ones = jnp.ones((1, nr), dtype=BF)
    return jnp.concatenate([ones, _rand(seed, (2, nr))], axis=0)


def _u32(a) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


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
        p0, p1 = split_pairs(trace)
        diff = p1 - p0
        eq = _rand(3, (nr // 2,))
        nr_live = jnp.asarray(nr, jnp.int32)
        claim = _rand(9, ())
        last = _rand(11, ())
        eq_adj = _rand(13, ())
        padded_row_adj = _eval_fn(jnp.zeros((1, _NUM_COLS), dtype=BF), _PV)[0] @ alpha
        vgeq = VirtualGeq(nr_live, jnp.ones((), BF), jnp.zeros((), BF))
        interp = interp_matrix((jnp.array(2, BF), jnp.array(4, BF)), last)
        is_round0 = jnp.array(False)

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
