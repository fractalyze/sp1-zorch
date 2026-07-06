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
correction unexercised.

GKR-batched cases weight each chip's columns by ``beta**(j+1)`` in both
driver and reference, seeding the claims from the columns' MLE openings at
``zeta``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as KB
from zk_dtypes import koalabearx4_mont as EF

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import compute_inv_vandermonde, eval_coeffs
from zorch.sumcheck.prover import zero_extend
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import sample_challenge

from sp1_zorch.zerocheck.jagged import (
    DEGREE,
    JaggedZerocheckSummand,
    prove_jagged_zerocheck,
)
from sp1_zorch.zerocheck.coeffs import rlc_coeffs

# Witness chip: columns [a, b, c] with a == 1 on every real row, so both
# constraints vanish there while C(0_row) = [1, 0] keeps the padded-row
# correction live (adj = alpha_0 != 0).
_NUM_COLS = 3
_K = 2


def _eval_fn(trace: jnp.ndarray) -> jnp.ndarray:
    a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
    one = jnp.ones((), trace.dtype)
    return jnp.stack([(a - one) * (c - one), (a - one) * b * c], axis=-1)


def _eval_fn_empty(trace: jnp.ndarray) -> jnp.ndarray:
    """Lookup-only chip: no transition constraints (SP1's Byte / Program /
    Range shape) — ``(N, 0)``, so only the GKR column term contributes."""
    return jnp.zeros((trace.shape[0], 0), dtype=trace.dtype)


def _rand(seed: int, shape) -> jnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=KB)


def _rand_ef(seed: int, shape) -> jnp.ndarray:
    return jax.lax.bitcast_convert_type(_rand(seed, (*shape, 4)), EF)


def _witness_trace(seed: int, nr: int) -> jnp.ndarray:
    if nr == 0:
        return jnp.zeros((_NUM_COLS, 0), dtype=KB)
    ones = jnp.ones((1, nr), dtype=KB)
    return jnp.concatenate([ones, _rand(seed, (2, nr))], axis=0)


@partial(
    jax.tree_util.register_dataclass, data_fields=["challenges", "pos"], meta_fields=[]
)
@dataclass(frozen=True)
class _ScriptedTranscript:
    """Transcript stub returning preset challenges — the forced-challenge seam
    for comparing the driver against the reference (the driver only observes
    and samples). A registered pytree with the cursor as a leaf so it rides
    the round ``lax.scan`` carry; ``sample`` advances it with ``dynamic_slice``
    (a Python ``list.pop`` cannot be a scan carry)."""

    challenges: jnp.ndarray
    pos: jnp.ndarray

    @classmethod
    def replaying(cls, challenges) -> "_ScriptedTranscript":
        return cls(jnp.asarray(challenges), jnp.asarray(0, jnp.int32))

    def observe(self, values):
        del values
        return self

    def sample(self, n=1):
        out = jax.lax.dynamic_slice_in_dim(self.challenges, self.pos, n, axis=0)
        return _ScriptedTranscript(self.challenges, self.pos + n), out


def _lift(v: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Bind the LSB variable at t: even + t*(odd - even) along the last axis."""
    return v[..., 0::2] + t * (v[..., 1::2] - v[..., 0::2])


def _naive_round_polys(
    eval_fns, traces, num_reals, alphas, lambdas, zeta, challenges, gkr_powers=None
):
    n = int(zeta.shape[0])
    width = 1 << n
    one = jnp.ones((), KB)
    inv_vand = compute_inv_vandermonde(DEGREE, KB)

    cols = [zero_extend(t, width) for t in traces]
    geqs = [
        jnp.concatenate([jnp.zeros(nr, dtype=KB), jnp.ones(width - nr, dtype=KB)])
        for nr in num_reals
    ]
    adjs = [
        f(jnp.zeros((1, _NUM_COLS), dtype=KB))[0] @ a for f, a in zip(eval_fns, alphas)
    ]
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
                cv = eval_fns[i](ct.T) @ alphas[i]
                cv = cv + ct.T @ gkr_powers[i]
                tot = tot + lambdas[i] * (cv - adjs[i] * _lift(geqs[i], tv))
            evals.append(jnp.sum(et * tot))
        polys.append(jnp.dot(inv_vand, jnp.stack(evals)))
        e = _lift(e, r)
        cols = [_lift(c, r) for c in cols]
        geqs = [_lift(g, r) for g in geqs]
    return polys


def _gkr_inputs(beta, traces, zeta):
    """Per-chip ``beta**(j+1)`` weights and matching claims — each chip's
    ``sum_j beta**(j+1) * mle_j(zeta)`` over its zero-extended columns."""
    width = 1 << int(zeta.shape[0])
    pows = [beta]
    for _ in range(_NUM_COLS - 1):
        pows.append(pows[-1] * beta)
    gkr = jnp.stack(pows)
    e = expand_eq_to_hypercube(zeta, jnp.ones((), KB))
    claims = [gkr @ (zero_extend(t, width) @ e) for t in traces]
    return [gkr] * len(traces), claims


class JaggedZerocheckRoundTest(absltest.TestCase):
    def _assert_claim_thread(self, msgs, claim) -> None:
        """On the lambda-RLC'd coefficient polys: ``p_r(0) + p_r(1) == claim_r``
        and ``claim_{r+1} = p_r(challenge_r)``."""
        for r in range(msgs.round_poly.shape[0]):
            coeffs = msgs.round_poly[r]
            self.assertTrue(
                bool(coeffs[0] + jnp.sum(coeffs) == claim), msg=f"round {r}"
            )
            claim = eval_coeffs(coeffs, msgs.challenge[r])

    def _check_against_reference(
        self,
        num_vars: int,
        num_reals,
        seed: int = 0,
        *,
        constraint_free: frozenset[int] = frozenset(),
    ):
        nchips = len(num_reals)
        traces = [_witness_trace(seed + i, nr) for i, nr in enumerate(num_reals)]
        eval_fns = [
            _eval_fn_empty if i in constraint_free else _eval_fn for i in range(nchips)
        ]
        alphas = [
            rlc_coeffs(_rand(99 + i, ()), 0 if i in constraint_free else _K)
            for i in range(nchips)
        ]
        lambdas = _rand(55, (nchips,))
        zeta = _rand(7, (num_vars,))
        challenges = [_rand(1000 + r, ()) for r in range(num_vars)]
        beta = _rand(77, ())
        gkr_powers, claims = _gkr_inputs(beta, traces, zeta)

        _, _, msgs = prove_jagged_zerocheck(
            JaggedZerocheckSummand(
                eval_fns=eval_fns, alphas=alphas, lambdas=lambdas, beta=beta
            ),
            traces,
            num_reals,
            zeta,
            _ScriptedTranscript.replaying(challenges),
            claims=claims,
        )
        want = _naive_round_polys(
            eval_fns, traces, num_reals, alphas, lambdas, zeta, challenges, gkr_powers
        )
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

    def test_constraint_free_chip_among_live(self) -> None:
        self._check_against_reference(
            num_vars=3, num_reals=[5, 8, 3], constraint_free=frozenset({1})
        )

    def test_wider_mixed_heights_match_reference(self) -> None:
        # Stresses both eq fits: a narrow chip truncates the early wide eq
        # tables, and the late narrow tables zero-extend up to the widest
        # chip's pair width.
        self._check_against_reference(num_vars=4, num_reals=[9, 16, 2, 0, 5])

    def _constraint_markers(self, num_vars: int, num_reals):
        nchips = len(num_reals)
        traces = [_witness_trace(i, nr) for i, nr in enumerate(num_reals)]
        alphas = [rlc_coeffs(_rand(99 + i, ()), _K) for i in range(nchips)]
        challenges = [_rand(1000 + r, ()) for r in range(num_vars)]
        beta = _rand(77, ())
        _, claims = _gkr_inputs(beta, traces, _rand(7, (num_vars,)))

        def run(traces, alphas, lambdas, zeta):  # type: ignore[no-untyped-def]
            return prove_jagged_zerocheck(
                JaggedZerocheckSummand(
                    eval_fns=[_eval_fn] * nchips,
                    alphas=alphas,
                    lambdas=lambdas,
                    beta=beta,
                ),
                traces,
                num_reals,
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=claims,
            )[2].round_poly

        txt = (
            jax.jit(run)
            .lower(traces, alphas, _rand(55, (nchips,)), _rand(7, (num_vars,)))
            .as_text()
        )
        calls = re.findall(r'stablehlo\.composite "zorch\.constraint_eval".*', txt)
        bounded = [c for c in calls if "live_width_operand_idx" in c]
        return calls, bounded

    def test_constraint_markers_are_bounded_and_round_invariant(self) -> None:
        # The compile-time contract the fixed-width buffers + lax.scan exist
        # for: every per-pair constraint evaluation rides a live-width-bounded
        # marker whose trace operand keeps ONE shape, so a recognizing compiler
        # reuses one kernel per chip across all rounds.
        num_reals = [5, 2]
        nchips = len(num_reals)
        calls, bounded = self._constraint_markers(3, num_reals)
        # EVERY constraint_eval marker is live-width-bounded — including the
        # per-chip C_alpha(0_row) probe, so it routes to the loop-form emitter
        # instead of the monolithic CSE unroll (the Global compile cliff,
        # fractalyze/zkx#702).
        self.assertEqual(len(calls), len(bounded), calls)
        # One rolled round body: the 3 t-points per chip run as separate
        # live-width-bounded launches (the round-0 t=0 drop zeroes alpha per
        # t-point, so they cannot share one batched circuit), plus the one
        # C_alpha(0_row) probe per chip — 4 markers per chip, never one per
        # round.
        self.assertEqual(len(bounded), 4 * nchips)
        shapes = {re.search(r"tensor<(\d+x\d+)x", c).group(1) for c in bounded}  # type: ignore[union-attr]
        # Each t-point trace is the chip's [pair, nc] block — pad4(5)/2 = 4
        # and pad4(2)/2 = 2 rows. The C_alpha(0_row) probe is a clean 1-row
        # block (the loop-form emitter engages on a single-row trace as of
        # fractalyze/zkx#704).
        self.assertEqual(shapes, {"4x3", "2x3", "1x3"})

    def test_round_loop_stays_rolled(self) -> None:
        # The regression guard for the scan: the constraint-marker count (the
        # term that drove the unrolled compile wall) must be invariant in
        # num_vars. An accidental return to a Python round loop would scale it
        # with the round count.
        num_reals = [5, 2]
        calls3, bounded3 = self._constraint_markers(3, num_reals)
        calls6, bounded6 = self._constraint_markers(6, num_reals)
        self.assertEqual((len(calls6), len(bounded6)), (len(calls3), len(bounded3)))

    def _tail_dot_count(self, num_vars: int, num_reals) -> int:
        """``dot_general`` op count in the lowered prove. The GKR column term
        rides ``constraint_eval``'s ``column_weights``, so the per-chip dots
        are the column dot inside each t-point's composite decomposition
        (each composite call emits its own decomposition func — three per
        constrained chip, shape-divergent by design) plus the interpolation +
        RLC tail (the Gruen matrix product and the λ-RLC)."""
        nchips = len(num_reals)
        traces = [_witness_trace(i, nr) for i, nr in enumerate(num_reals)]
        alphas = [rlc_coeffs(_rand(99 + i, ()), _K) for i in range(nchips)]
        challenges = [_rand(1000 + r, ()) for r in range(num_vars)]
        beta = _rand(77, ())
        _, claims = _gkr_inputs(beta, traces, _rand(7, (num_vars,)))

        def run(traces, alphas, lambdas, zeta):  # type: ignore[no-untyped-def]
            return prove_jagged_zerocheck(
                JaggedZerocheckSummand(
                    eval_fns=[_eval_fn] * nchips,
                    alphas=alphas,
                    lambdas=lambdas,
                    beta=beta,
                ),
                traces,
                num_reals,
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=claims,
            )[2].round_poly

        txt = (
            jax.jit(run)
            .lower(traces, alphas, _rand(55, (nchips,)), _rand(7, (num_vars,)))
            .as_text()
        )
        return len(re.findall(r"stablehlo\.dot_general", txt))

    def test_interp_rlc_tail_batches_across_chips(self) -> None:
        # The interpolation + RLC tail is batched into one [num_chips, ...]
        # matmul each, so its dot_general count is INDEPENDENT of the chip
        # count. A per-chip unrolled tail scales it with nchips (~2 dots/chip)
        # — the tiny-launch cluster the batch collapses. Each added chip
        # brings only its three t-point composite decompositions, each
        # carrying the in-body column dot (3 dots per chip, by design); the
        # tail must contribute ZERO growth beyond them.
        self.assertEqual(
            self._tail_dot_count(3, [5, 2, 3, 4, 1]) - self._tail_dot_count(3, [5, 2]),
            3 * 3,
        )

    def test_gkr_claim_threading(self) -> None:
        # The claim thread starts at the lambda-RLC of the GKR claims and
        # follows the p(1) = claim - p(0) identity round to round.
        num_vars, num_reals = 3, [5, 8, 3]
        nchips = len(num_reals)
        traces = [_witness_trace(20 + i, nr) for i, nr in enumerate(num_reals)]
        alphas = [rlc_coeffs(_rand(80 + i, ()), _K) for i in range(nchips)]
        lambdas = _rand(40, (nchips,))
        zeta = _rand(5, (num_vars,))
        beta = _rand(70, ())
        _, claims = _gkr_inputs(beta, traces, zeta)

        _, _, msgs = prove_jagged_zerocheck(
            JaggedZerocheckSummand(
                eval_fns=[_eval_fn] * nchips, alphas=alphas, lambdas=lambdas, beta=beta
            ),
            traces,
            num_reals,
            zeta,
            cheap_transcript(KB),
            claims=claims,
        )

        claim = jnp.zeros((), KB)
        for i in range(nchips):
            claim = claim + lambdas[i] * claims[i]
        self._assert_claim_thread(msgs, claim)

    def test_transcript_invariants_and_final_fold(self) -> None:
        num_vars, num_reals = 3, [5, 8, 3]
        nchips = len(num_reals)
        traces = [_witness_trace(10 + i, nr) for i, nr in enumerate(num_reals)]
        alphas = [rlc_coeffs(_rand(90 + i, ()), _K) for i in range(nchips)]
        lambdas = _rand(50, (nchips,))
        zeta = _rand(3, (num_vars,))

        beta = _rand(60, ())
        _, claims = _gkr_inputs(beta, traces, zeta)
        finals, _, msgs = prove_jagged_zerocheck(
            JaggedZerocheckSummand(
                eval_fns=[_eval_fn] * nchips,
                alphas=alphas,
                lambdas=lambdas,
                beta=beta,
            ),
            traces,
            num_reals,
            zeta,
            cheap_transcript(KB),
            claims=claims,
        )

        # The claim thread starts at the lambda-RLC of the GKR claims.
        claim = jnp.zeros((), KB)
        for i in range(nchips):
            claim = claim + lambdas[i] * claims[i]
        self._assert_claim_thread(msgs, claim)

        # Final folded columns == naive even/odd binds of the zero-extended
        # traces at the transcript's challenges.
        width = 1 << num_vars
        for i in range(nchips):
            v = zero_extend(traces[i], width)
            for r in range(num_vars):
                v = _lift(v, msgs.challenge[r])
            if finals[i].shape[1] > 0:
                self.assertTrue(
                    bool(jnp.all(finals[i][:, 0] == v[:, 0])), msg=f"chip {i}"
                )

    def test_round_challenge_is_one_extension_sample(self) -> None:
        """The round challenge binds SP1's ``sample_ext_element`` rule: degree
        base squeezes reinterpreted as one extension element, pinned against
        the shared ``sample_challenge`` definition on a real transcript
        (fractalyze/sp1-zorch#88). The scripted replays above bypass the
        squeeze rule entirely, so they cannot catch a squeeze-count drift."""
        num_vars, nr = 2, 3
        traces = [_witness_trace(11, nr)]
        alphas = [rlc_coeffs(_rand_ef(99, ()), _K)]
        lambdas = _rand_ef(50, (1,))
        zeta = _rand_ef(3, (num_vars,))

        beta = _rand_ef(60, ())
        _, claims = _gkr_inputs(beta, traces, zeta)
        _, _, msgs = prove_jagged_zerocheck(
            JaggedZerocheckSummand(
                eval_fns=[_eval_fn], alphas=alphas, lambdas=lambdas, beta=beta
            ),
            traces,
            [nr],
            zeta,
            cheap_transcript(KB),
            claims=claims,
        )

        self.assertEqual(msgs.challenge.dtype, EF)
        t = cheap_transcript(KB)
        for r in range(num_vars):
            t = t.observe(msgs.round_poly[r])
            t, want = sample_challenge(t, EF, 4)
            self.assertTrue(
                bool(jnp.array_equal(want, msgs.challenge[r])), msg=f"round {r}"
            )

    def test_validation_rejects_mismatched_height(self) -> None:
        trace = _witness_trace(0, 4)
        with self.assertRaisesRegex(ValueError, "num_reals"):
            prove_jagged_zerocheck(
                JaggedZerocheckSummand(
                    eval_fns=[_eval_fn],
                    alphas=[rlc_coeffs(_rand(99, ()), _K)],
                    lambdas=_rand(55, (1,)),
                    beta=_rand(77, ()),
                ),
                [trace],
                [5],
                _rand(7, (3,)),
                cheap_transcript(KB),
                claims=[jnp.zeros((), KB)],
            )

    def test_validation_rejects_short_claims(self) -> None:
        with self.assertRaisesRegex(ValueError, "per chip"):
            prove_jagged_zerocheck(
                JaggedZerocheckSummand(
                    eval_fns=[_eval_fn],
                    alphas=[rlc_coeffs(_rand(99, ()), _K)],
                    lambdas=_rand(55, (1,)),
                    beta=_rand(77, ()),
                ),
                [_witness_trace(0, 4)],
                [4],
                _rand(7, (3,)),
                cheap_transcript(KB),
                claims=[jnp.zeros((), KB)] * 2,
            )


if __name__ == "__main__":
    absltest.main()
