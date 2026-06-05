# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged zerocheck round vs a schedule-free brute-force reference.

The reference materializes every column, the eq vector, and the
``x >= num_real`` indicator at full hypercube width; per round it lifts each
to t in {0..4} (the constraint polynomial applies AFTER lifting — the summand
is not multilinear), subtracts the indicator-scaled zero-row constant, and
interpolates coefficients. The driver must reproduce those coefficients
through its truncated sums, claim identity, and virtual-geq correction.

Traces are WITNESS traces (constraints vanish on real rows, ``C(0_row) != 0``):
round 0's constraint-skip and the ``claim - y0`` identity are protocol-equal
only on witnesses, and a zero ``C(0_row)`` would leave the padded-row
correction unexercised."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as KB

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import compute_inv_vandermonde, eval_coeffs
from zorch.testkit.transcript import cheap_transcript

from sp1_zorch.zerocheck.jagged import DEGREE, prove_jagged_zerocheck
from sp1_zorch.zerocheck.prover import rlc_coeffs

# Witness chip: columns [a, b, c] with a == 1 on every real row, so both
# constraints vanish there while C(0_row) = [1, 0] keeps the padded-row
# correction live (adj = alpha_0 != 0).
_NUM_COLS = 3
_K = 2


def _eval_fn(trace: jnp.ndarray) -> jnp.ndarray:
    a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
    one = jnp.ones((), trace.dtype)
    return jnp.stack([(a - one) * (c - one), (a - one) * b * c], axis=-1)


def _rand(seed: int, shape) -> jnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=KB)


def _witness_trace(seed: int, nr: int) -> jnp.ndarray:
    if nr == 0:
        return jnp.zeros((_NUM_COLS, 0), dtype=KB)
    ones = jnp.ones((1, nr), dtype=KB)
    return jnp.concatenate([ones, _rand(seed, (2, nr))], axis=0)


class _ScriptedTranscript:
    """Transcript stub returning preset challenges — the forced-challenge seam
    for comparing the driver against the reference (the driver only calls
    ``observe_and_sample``)."""

    def __init__(self, challenges):
        self._next = list(challenges)

    def observe_and_sample(self, values, n=1):
        out = jnp.stack([self._next.pop(0) for _ in range(n)])
        return self, out


def _lift(v: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Bind the LSB variable at t: even + t*(odd - even) along the last axis."""
    return v[..., 0::2] + t * (v[..., 1::2] - v[..., 0::2])


def _zero_extend(trace: jnp.ndarray, width: int) -> jnp.ndarray:
    pad = width - trace.shape[1]
    return jnp.concatenate(
        [trace, jnp.zeros((trace.shape[0], pad), dtype=trace.dtype)], axis=1
    )


def _naive_round_polys(traces, num_reals, alphas, lambdas, zeta, challenges):
    n = int(zeta.shape[0])
    width = 1 << n
    one = jnp.ones((), KB)
    inv_vand = compute_inv_vandermonde(DEGREE, KB)

    cols = [_zero_extend(t, width) for t in traces]
    geqs = [
        jnp.concatenate([jnp.zeros(nr, dtype=KB), jnp.ones(width - nr, dtype=KB)])
        for nr in num_reals
    ]
    adjs = [_eval_fn(jnp.zeros((1, _NUM_COLS), dtype=KB))[0] @ a for a in alphas]
    e = expand_eq_to_hypercube(zeta, one)

    polys = []
    for r in challenges:
        evals = []
        for t in range(DEGREE + 1):
            tv = jnp.array(t, KB)
            et = _lift(e, tv)
            tot = jnp.zeros_like(et)
            for i in range(len(cols)):
                ct = _lift(cols[i], tv)
                cv = _eval_fn(ct.T) @ alphas[i]
                tot = tot + lambdas[i] * (cv - adjs[i] * _lift(geqs[i], tv))
            evals.append(jnp.sum(et * tot))
        polys.append(jnp.dot(inv_vand, jnp.stack(evals)))
        e = _lift(e, r)
        cols = [_lift(c, r) for c in cols]
        geqs = [_lift(g, r) for g in geqs]
    return polys


class JaggedZerocheckRoundTest(absltest.TestCase):
    def _check_against_reference(self, num_vars: int, num_reals, seed: int = 0):
        nchips = len(num_reals)
        traces = [_witness_trace(seed + i, nr) for i, nr in enumerate(num_reals)]
        alphas = [rlc_coeffs(_rand(99 + i, ()), _K) for i in range(nchips)]
        lambdas = _rand(55, (nchips,))
        zeta = _rand(7, (num_vars,))
        challenges = [_rand(1000 + r, ()) for r in range(num_vars)]

        _, _, msgs = prove_jagged_zerocheck(
            [_eval_fn] * nchips,
            traces,
            num_reals,
            alphas,
            lambdas,
            zeta,
            _ScriptedTranscript(challenges),
        )
        want = _naive_round_polys(traces, num_reals, alphas, lambdas, zeta, challenges)
        for r in range(num_vars):
            self.assertTrue(
                bool(jnp.all(msgs.round_poly[r] == want[r])), msg=f"round {r}"
            )

    def test_jagged_heights_match_reference(self) -> None:
        self._check_against_reference(num_vars=3, num_reals=[5, 8, 3])

    def test_full_height_matches_reference(self) -> None:
        self._check_against_reference(num_vars=3, num_reals=[8, 8])

    def test_zero_height_chip_among_live(self) -> None:
        self._check_against_reference(num_vars=3, num_reals=[5, 0, 3])

    def test_freeze_at_two_tail(self) -> None:
        self._check_against_reference(num_vars=4, num_reals=[3, 1])

    def test_single_chip(self) -> None:
        self._check_against_reference(num_vars=3, num_reals=[6])

    def test_transcript_invariants_and_final_fold(self) -> None:
        num_vars, num_reals = 3, [5, 8, 3]
        nchips = len(num_reals)
        traces = [_witness_trace(10 + i, nr) for i, nr in enumerate(num_reals)]
        alphas = [rlc_coeffs(_rand(90 + i, ()), _K) for i in range(nchips)]
        lambdas = _rand(50, (nchips,))
        zeta = _rand(3, (num_vars,))

        finals, _, msgs = prove_jagged_zerocheck(
            [_eval_fn] * nchips,
            traces,
            num_reals,
            alphas,
            lambdas,
            zeta,
            cheap_transcript(KB),
        )

        # Claim threading on the lambda-RLC'd coefficient polys: claim_0 = 0,
        # p_r(0) + p_r(1) == claim_r, claim_{r+1} = p_r(challenge_r).
        claim = jnp.zeros((), KB)
        for r in range(num_vars):
            coeffs = msgs.round_poly[r]
            self.assertTrue(
                bool(coeffs[0] + jnp.sum(coeffs) == claim), msg=f"round {r}"
            )
            claim = eval_coeffs(coeffs, msgs.challenge[r])

        # Final folded columns == naive even/odd binds of the zero-extended
        # traces at the transcript's challenges.
        width = 1 << num_vars
        for i in range(nchips):
            v = _zero_extend(traces[i], width)
            for r in range(num_vars):
                v = _lift(v, msgs.challenge[r])
            if finals[i].shape[1] > 0:
                self.assertTrue(
                    bool(jnp.all(finals[i][:, 0] == v[:, 0])), msg=f"chip {i}"
                )

    def test_validation_rejects_mismatched_height(self) -> None:
        trace = _witness_trace(0, 4)
        with self.assertRaisesRegex(ValueError, "num_reals"):
            prove_jagged_zerocheck(
                [_eval_fn],
                [trace],
                [5],
                [rlc_coeffs(_rand(99, ()), _K)],
                _rand(55, (1,)),
                _rand(7, (3,)),
                cheap_transcript(KB),
            )


if __name__ == "__main__":
    absltest.main()
