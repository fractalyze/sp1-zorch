# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Single-chip zerocheck sumcheck: self-verify a random-trace claim end to end.

A random (non-witness) trace gives a non-zero ``sum_x eq(zeta,x)*C_alpha(x)``;
the prover's round polynomials must reduce it back to the summand at the bound
point. Reuses zorch's summand-agnostic sumcheck verifier — it sees only the
round polynomials, so it validates the zerocheck summand without knowing it."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear as KB

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck import verifier
from zorch.transcript import StubTranscript
from zorch.verify import verify

from sp1_zorch.zerocheck.prover import rlc_coeffs
from sp1_zorch.zerocheck.round import ZerocheckRound, prove_zerocheck

# Two degree-2 constraints on 3 columns → round-poly degree 2 (constraint) + 1 (eq).
_NUM_COLS = 3
_K = 2
_DEGREE = 3


def _eval_fn(trace: jnp.ndarray) -> jnp.ndarray:
    a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
    return jnp.stack([a * b - c, b * c - a], axis=-1)


def _rand(seed: int, shape) -> jnp.ndarray:
    ints = np.random.default_rng(seed).integers(0, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=KB)


def _setup(num_vars: int):
    width = 1 << num_vars
    cols = [_rand(i, (width,)) for i in range(_NUM_COLS)]
    alpha = rlc_coeffs(_rand(99, ()), _K)
    zeta = _rand(7, (num_vars,))
    eq = expand_eq_to_hypercube(zeta, jnp.ones((), KB))
    claim = jnp.sum(eq * (_eval_fn(jnp.stack(cols, axis=-1)) @ alpha))
    return cols, alpha, zeta, claim


class ZerocheckRoundTest(absltest.TestCase):
    def test_random_trace_self_verifies(self) -> None:
        num_vars = 4
        cols, alpha, zeta, claim = _setup(num_vars)
        challenges = _rand(11, (num_vars,))

        final_state, _, msgs = prove_zerocheck(
            _eval_fn, cols, alpha, zeta, StubTranscript(challenges), degree=_DEGREE
        )
        point, final_claim, _, ok = verify(
            verifier.SumcheckRound(_DEGREE),
            claim,
            msgs.round_poly,
            StubTranscript(challenges),
        )

        self.assertTrue(bool(ok))
        self.assertTrue(bool(jnp.all(point == challenges)))
        # The verifier's reduced claim must equal the summand at the bound point,
        # i.e. the prover's fully-folded factors run back through the summand.
        want = ZerocheckRound(alpha=alpha, eval_fn=_eval_fn, degree=_DEGREE)._combine(
            *final_state
        )
        self.assertTrue(bool(final_claim == want.reshape(())))

    def test_wrong_claim_rejected(self) -> None:
        num_vars = 4
        cols, alpha, zeta, claim = _setup(num_vars)
        challenges = _rand(11, (num_vars,))
        _, _, msgs = prove_zerocheck(
            _eval_fn, cols, alpha, zeta, StubTranscript(challenges), degree=_DEGREE
        )
        _, _, _, ok = verify(
            verifier.SumcheckRound(_DEGREE),
            claim + jnp.array(1, KB),
            msgs.round_poly,
            StubTranscript(challenges),
        )
        self.assertFalse(bool(ok))


if __name__ == "__main__":
    absltest.main()
