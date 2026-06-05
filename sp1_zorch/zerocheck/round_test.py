# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Zerocheck sumcheck rounds: self-verify a random-trace claim end to end.

A random (non-witness) trace gives a non-zero ``sum_x eq(zeta,x)*C_alpha(x)``;
the prover's round polynomials must reduce it back to the summand at the bound
point. Reuses zorch's summand-agnostic sumcheck verifier — it sees only the
round polynomials, so it validates the zerocheck summand without knowing it.

Covers the single-chip round and the equal-height multi-chip joint round (the
λ-RLC across chips with one shared challenge folding every chip)."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as KB

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck import verifier
from zorch.testkit.transcript import cheap_transcript
from zorch.verify import verify

from sp1_zorch.zerocheck.prover import rlc_coeffs
from sp1_zorch.zerocheck.round import (
    MultiChipZerocheckRound,
    ZerocheckRound,
    prove_multi_chip_zerocheck,
    prove_zerocheck,
)

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

        # Real (cheap) Fiat-Shamir: prover and verifier each derive the same
        # challenge stream from the observed round polys.
        final_state, _, msgs = prove_zerocheck(
            _eval_fn, cols, alpha, zeta, cheap_transcript(KB), degree=_DEGREE
        )
        point, final_claim, _, ok = verify(
            verifier.SumcheckRound(_DEGREE),
            claim,
            msgs.round_poly,
            cheap_transcript(KB),
        )

        self.assertTrue(bool(ok))
        self.assertEqual(point.shape, (num_vars,))
        # The verifier's reduced claim must equal the summand at the bound point,
        # i.e. the prover's fully-folded factors run back through the summand.
        want = ZerocheckRound(alpha=alpha, eval_fn=_eval_fn, degree=_DEGREE)._combine(
            *final_state
        )
        self.assertTrue(bool(final_claim == want.reshape(())))

    def test_wrong_claim_rejected(self) -> None:
        num_vars = 4
        cols, alpha, zeta, claim = _setup(num_vars)
        _, _, msgs = prove_zerocheck(
            _eval_fn, cols, alpha, zeta, cheap_transcript(KB), degree=_DEGREE
        )
        _, _, _, ok = verify(
            verifier.SumcheckRound(_DEGREE),
            claim + jnp.array(1, KB),
            msgs.round_poly,
            cheap_transcript(KB),
        )
        self.assertFalse(bool(ok))


# Two chips of different shape jointly summed in one sumcheck. Chip A: 3 cols,
# two degree-2 constraints. Chip B: 2 cols, one degree-1 constraint (a lower
# degree, lifted to the shared domain). Round-poly degree = max chip degree (2)
# + 1 for the eq factor.
_MC_COLS = (3, 2)
_MC_K = (2, 1)
_MC_DEGREE = 3


def _chip_a_eval(trace: jnp.ndarray) -> jnp.ndarray:
    a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
    return jnp.stack([a * b - c, b * c - a], axis=-1)


def _chip_b_eval(trace: jnp.ndarray) -> jnp.ndarray:
    a, b = trace[:, 0], trace[:, 1]
    return jnp.stack([a - b], axis=-1)


def _mc_setup(num_vars: int):
    width = 1 << num_vars
    cols_a = [_rand(i, (width,)) for i in range(_MC_COLS[0])]
    cols_b = [_rand(10 + i, (width,)) for i in range(_MC_COLS[1])]
    alpha_a = rlc_coeffs(_rand(91, ()), _MC_K[0])
    alpha_b = rlc_coeffs(_rand(92, ()), _MC_K[1])
    # Cross-chip RLC coefficients [1, lam]; the exact SP1 orientation is pinned
    # by the byte-match brick, not this self-test.
    lam = _rand(93, ())
    lambdas = jnp.stack([jnp.ones_like(lam), lam])
    zeta = _rand(7, (num_vars,))
    eq = expand_eq_to_hypercube(zeta, jnp.ones((), KB))
    fold_a = _chip_a_eval(jnp.stack(cols_a, axis=-1)) @ alpha_a
    fold_b = _chip_b_eval(jnp.stack(cols_b, axis=-1)) @ alpha_b
    claim = jnp.sum(eq * (lambdas[0] * fold_a + lambdas[1] * fold_b))
    return [cols_a, cols_b], [alpha_a, alpha_b], lambdas, zeta, claim


class MultiChipZerocheckRoundTest(absltest.TestCase):
    def test_two_chip_random_trace_self_verifies(self) -> None:
        num_vars = 4
        chips_cols, alphas, lambdas, zeta, claim = _mc_setup(num_vars)
        eval_fns = (_chip_a_eval, _chip_b_eval)

        final_state, _, msgs = prove_multi_chip_zerocheck(
            eval_fns,
            chips_cols,
            alphas,
            lambdas,
            zeta,
            cheap_transcript(KB),
            degree=_MC_DEGREE,
        )
        point, final_claim, _, ok = verify(
            verifier.SumcheckRound(_MC_DEGREE),
            claim,
            msgs.round_poly,
            cheap_transcript(KB),
        )

        self.assertTrue(bool(ok))
        self.assertEqual(point.shape, (num_vars,))
        want = MultiChipZerocheckRound(
            alphas=tuple(alphas),
            lambdas=lambdas,
            eval_fns=eval_fns,
            col_counts=_MC_COLS,
            degree=_MC_DEGREE,
        )._combine(*final_state)
        self.assertTrue(bool(final_claim == want.reshape(())))

    def test_wrong_claim_rejected(self) -> None:
        num_vars = 4
        chips_cols, alphas, lambdas, zeta, claim = _mc_setup(num_vars)
        _, _, msgs = prove_multi_chip_zerocheck(
            (_chip_a_eval, _chip_b_eval),
            chips_cols,
            alphas,
            lambdas,
            zeta,
            cheap_transcript(KB),
            degree=_MC_DEGREE,
        )
        _, _, _, ok = verify(
            verifier.SumcheckRound(_MC_DEGREE),
            claim + jnp.array(1, KB),
            msgs.round_poly,
            cheap_transcript(KB),
        )
        self.assertFalse(bool(ok))

    def test_unequal_heights_rejected(self) -> None:
        # Equal height is the documented contract: a chip at half the width
        # must fail with a deterministic ValueError, not an opaque shape error
        # from inside the driver's scan.
        num_vars = 4
        chips_cols, alphas, lambdas, zeta, _ = _mc_setup(num_vars)
        chips_cols[1] = [col[: 1 << (num_vars - 1)] for col in chips_cols[1]]
        with self.assertRaises(ValueError):
            prove_multi_chip_zerocheck(
                (_chip_a_eval, _chip_b_eval),
                chips_cols,
                alphas,
                lambdas,
                zeta,
                cheap_transcript(KB),
                degree=_MC_DEGREE,
            )

    def test_no_chips_rejected(self) -> None:
        # Zero chips slips past the count agreement (0 == 0 == 0) and would
        # silently sumcheck just the eq factor.
        zeta = _rand(7, (4,))
        with self.assertRaises(ValueError):
            prove_multi_chip_zerocheck(
                (),
                [],
                [],
                jnp.zeros((0,), KB),
                zeta,
                cheap_transcript(KB),
                degree=_MC_DEGREE,
            )


if __name__ == "__main__":
    absltest.main()
