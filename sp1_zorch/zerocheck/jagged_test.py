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

import frx
import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as KB
from zk_dtypes import koalabearx4_mont as EF

from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.geq import VirtualGeq
from zorch.poly.univariate import compute_inv_vandermonde, eval_coeffs
from zorch.sumcheck.gruen import interp_matrix, round_coeffs_from_matrix
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import sample_challenge

from sp1_zorch.zerocheck import jagged
from sp1_zorch.zerocheck.jagged import (
    DEGREE,
    JaggedZerocheckSummand,
    TotalCapClass,
    _reduce_and_assemble,
    _summand_values,
    prove_jagged_zerocheck,
)
from sp1_zorch.zerocheck.coeffs import gkr_powers, rlc_coeffs


def zero_extend(arr, width):
    """Oracle-local zero-extend of the last axis to `width` (zorch#412 removed
    `sumcheck.prover.zero_extend`; the engine keeps a private copy). Byte-equal
    to the old block: the padded rows are exact field zeros."""
    pad = width - arr.shape[-1]
    if pad == 0:
        return arr
    return jnp.concatenate([arr, jnp.zeros((*arr.shape[:-1], pad), arr.dtype)], axis=-1)


# Witness chip: columns [a, b, c] with a == 1 on every real row, so both
# constraints vanish there while C(0_row) = [1, 0] keeps the padded-row
# correction live (adj = alpha_0 != 0).
_NUM_COLS = 3
_K = 2


def _eval_fn(trace: jnp.ndarray, public_values: jnp.ndarray) -> jnp.ndarray:
    del public_values
    a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
    one = jnp.ones((), trace.dtype)
    return jnp.stack([(a - one) * (c - one), (a - one) * b * c], axis=-1)


def _eval_fn_empty(trace: jnp.ndarray, public_values: jnp.ndarray) -> jnp.ndarray:
    """Lookup-only chip: no transition constraints (SP1's Byte / Program /
    Range shape) — ``(N, 0)``, so only the GKR column term contributes."""
    del public_values
    return jnp.zeros((trace.shape[0], 0), dtype=trace.dtype)


# The eval_fns above ignore the statement (no constraint declares a pv_arg),
# but it still rides as a declared `constraint_eval` operand — so the summand
# and the reference thread a (dummy) pv through the 2-ary signature.
_PV = jnp.zeros((8,), dtype=KB)


def _rand(seed: int, shape) -> jnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=KB)


def _rand_ef(seed: int, shape) -> jnp.ndarray:
    return frx.lax.bitcast_convert_type(_rand(seed, (*shape, 4)), EF)


def _u32(a) -> np.ndarray:
    return np.asarray(frx.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


def _assert_bytes_equal(got, want, label: str = "") -> None:
    """Montgomery-form ``u32`` comparison — the repo's byte-exact convention
    (no float tolerance applies to field elements)."""
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


def _witness_trace(seed: int, nr: int) -> jnp.ndarray:
    if nr == 0:
        return jnp.zeros((_NUM_COLS, 0), dtype=KB)
    ones = jnp.ones((1, nr), dtype=KB)
    return jnp.concatenate([ones, _rand(seed, (2, nr))], axis=0)


@partial(
    frx.tree_util.register_dataclass, data_fields=["challenges", "pos"], meta_fields=[]
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
        out = frx.lax.dynamic_slice_in_dim(self.challenges, self.pos, n, axis=0)
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
        f(jnp.zeros((1, _NUM_COLS), dtype=KB), _PV)[0] @ a
        for f, a in zip(eval_fns, alphas)
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
                cv = eval_fns[i](ct.T, _PV) @ alphas[i]
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
                eval_fns=eval_fns,
                alphas=alphas,
                lambdas=lambdas,
                beta=beta,
                public_values=_PV,
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

    def test_reduce_and_assemble_matches_summand_slots(self) -> None:
        # `_reduce_and_assemble` fed by `_summand_values` is the same seam as
        # `chip_raw_evals` + `correct` + `round_coeffs_from_matrix`
        # (Task 2's refactor delegates the latter into the former) — one
        # constrained chip, one round, live prefix at the buffer's full pair
        # width (nr = 4, so no truncation muddies the comparison).
        nr = 4
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
        # Row-major carry (fractalyze/sp1-zorch#242): the trace is
        # `[rows, num_cols]` and the pair fold is an axis-0 stride-2 split
        # (the dual of the old column-major last-axis `split_pairs`).
        trace_rm = trace.T
        p0, p1 = trace_rm[0::2], trace_rm[1::2]
        diff = p1 - p0
        eq = _rand(3, (nr // 2,))
        nr_live = jnp.asarray(nr, jnp.int32)
        claim = _rand(9, ())
        last = _rand(11, ())
        eq_adj = _rand(13, ())
        padded_row_adj = _eval_fn(jnp.zeros((1, _NUM_COLS), dtype=KB), _PV)[0] @ alpha
        vgeq = VirtualGeq(nr_live, jnp.ones((), KB), jnp.zeros((), KB))
        interp = interp_matrix((jnp.array(2, KB), jnp.array(4, KB)), last)
        is_round0 = jnp.array(False)

        term, alpha0 = summand._term_fns[0], summand.alphas[0]
        vals = _summand_values(term, alpha0, p0, diff, gkr, nr_live, is_round0)
        got = _reduce_and_assemble(
            vals, eq, interp, claim, last, eq_adj, padded_row_adj, nr_live, vgeq
        )

        raws = summand.chip_raw_evals(
            0, p0, diff, eq, gkr, nr_live, is_zero=False, is_round0=is_round0
        )
        y0, y2, y4 = summand.correct(
            raws, eq, last, eq_adj, padded_row_adj, nr_live, vgeq, is_zero=False
        )
        want = round_coeffs_from_matrix(interp, y0, claim, (y2, y4))
        _assert_bytes_equal(got, want)

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
                    public_values=_PV,
                ),
                traces,
                num_reals,
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=claims,
            )[2].round_poly

        txt = (
            frx.jit(run)
            .lower(traces, alphas, _rand(55, (nchips,)), _rand(7, (num_vars,)))
            .as_text()
        )
        calls = re.findall(r'stablehlo\.composite "zorch\.constraint_eval".*', txt)
        bounded = [c for c in calls if "live_width_operand_idx" in c]
        return calls, bounded

    def test_constraint_markers_are_bounded_and_round_invariant(self) -> None:
        # The compile-time contract the shared total-cap buffer + shrink prefix
        # + lax.scan exist for: every per-pair constraint evaluation rides a
        # live-width-bounded marker whose trace operand keeps ONE shape per
        # round — the shared TALL half `[row_cap_r/2, MAX_COLS]` out of
        # `shrink_schedule` (fractalyze/sp1-zorch#242), so a recognizing
        # compiler reuses one kernel across all chips and t-points of a round.
        num_reals = [5, 2]
        nchips = len(num_reals)
        num_vars = 3
        unroll = min(jagged._SHRINK_ROUNDS, num_vars)
        calls, bounded = self._constraint_markers(num_vars, num_reals)
        # EVERY constraint_eval marker is live-width-bounded — including the
        # per-chip C_alpha(0_row) probe, so it routes to the loop-form emitter
        # instead of the monolithic CSE unroll (the Global compile cliff,
        # fractalyze/xla#702).
        self.assertEqual(len(calls), len(bounded), calls)
        # 3 t-point markers per chip per compiled round body (the round-0 t=0
        # drop zeroes alpha per t-point, so they cannot share one batched
        # circuit): one body per unrolled shrink round plus one rolled tail
        # scan, plus the one C_alpha(0_row) probe per chip.
        round_bodies = unroll + (1 if num_vars > unroll else 0)
        self.assertEqual(len(bounded), (3 * round_bodies + 1) * nchips)
        shapes = {re.search(r"tensor<(\d+(?:x\d+)?)x", c).group(1) for c in bounded}  # type: ignore[union-attr]
        # Every t-point trace is the round's shared FLAT jagged half
        # `[area_cap_r/2]`, halving per shrink round; each chip reads its
        # `[W_r, cols]` window IN PLACE via constraint_eval's
        # start_offset/col_stride (both runtime — the operand stays the one
        # flat buffer, so it keeps ONE shape across chips within a round). The
        # C_alpha(0_row) probe stays a clean rank-2 1-row block (the loop-form
        # emitter engages on a single-row trace, fractalyze/xla#704).
        cls = TotalCapClass.from_heights(num_reals, [_NUM_COLS] * nchips)
        caps_r, _ = cls.shrink_schedule(nchips * _NUM_COLS, jagged._SHRINK_ROUNDS)
        want_shapes = {
            f"{caps_r[r] // 2}" for r in range(round_bodies)
        } | {f"1x{_NUM_COLS}"}
        self.assertEqual(shapes, want_shapes)

    def test_round_loop_stays_rolled(self) -> None:
        # The regression guard for the scan tail: past the `_SHRINK_ROUNDS`
        # unrolled prefix the constraint-marker count must be invariant in
        # num_vars. An accidental return to a full Python round loop would
        # scale it with the round count.
        num_reals = [5, 2]
        tail3 = self._constraint_markers(jagged._SHRINK_ROUNDS + 3, num_reals)
        tail1 = self._constraint_markers(jagged._SHRINK_ROUNDS + 1, num_reals)
        self.assertEqual(
            (len(tail3[0]), len(tail3[1])), (len(tail1[0]), len(tail1[1]))
        )

    def _tail_dot_count(self, num_vars: int, num_reals) -> int:
        """``dot_general`` op count in the lowered prove. The GKR column term
        rides ``constraint_eval``'s ``column_weights``, so the per-chip dots
        are the column dot inside each t-point's composite decomposition
        (each composite call emits its own decomposition func — three per
        constrained chip, shape-divergent by design) plus, since the
        ``variant=zerocheck`` round marker (sp1-zorch#242), the per-chip
        Gruen interpolation dot inside each live chip's own
        `zerocheck_round_poly` decomposition (one more per constrained
        chip) — the cross-chip λ-RLC (`combine_chips`) stays a single batched
        dot regardless of chip count."""
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
                    public_values=_PV,
                ),
                traces,
                num_reals,
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=claims,
            )[2].round_poly

        txt = (
            frx.jit(run)
            .lower(traces, alphas, _rand(55, (nchips,)), _rand(7, (num_vars,)))
            .as_text()
        )
        return len(re.findall(r"stablehlo\.dot_general", txt))

    def test_interp_rlc_tail_scales_per_chip_marker(self) -> None:
        # Since the `variant=zerocheck` round marker (sp1-zorch#242) the
        # Gruen interpolation assembly is de-batched: it runs INSIDE each
        # live chip's own `zerocheck_round_poly` decomposition, not one
        # shared [num_chips, ...] matmul — required so the round reduce has
        # a per-chip marker boundary (`jagged.py`'s `round_step`). Each added
        # constrained chip therefore brings its three t-point composite
        # decompositions (the in-body column dot, 3 dots/chip, by design)
        # PLUS one Gruen dot inside its own round marker: 4 dots/chip per
        # compiled round body (each unrolled shrink round plus the rolled
        # tail), always growing linearly with the chip count (never O(1),
        # unlike the old batched tail this replaces). The cross-chip λ-RLC
        # stays a single dot per round body regardless of chip count,
        # contributing zero extra growth.
        num_vars = 3
        round_bodies = min(jagged._SHRINK_ROUNDS, num_vars) + (
            1 if num_vars > jagged._SHRINK_ROUNDS else 0
        )
        self.assertEqual(
            self._tail_dot_count(num_vars, [5, 2, 3, 4, 1])
            - self._tail_dot_count(num_vars, [5, 2]),
            4 * 3 * round_bodies,
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
                eval_fns=[_eval_fn] * nchips,
                alphas=alphas,
                lambdas=lambdas,
                beta=beta,
                public_values=_PV,
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
                public_values=_PV,
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
                eval_fns=[_eval_fn],
                alphas=alphas,
                lambdas=lambdas,
                beta=beta,
                public_values=_PV,
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
                    public_values=_PV,
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
                    public_values=_PV,
                ),
                [_witness_trace(0, 4)],
                [4],
                _rand(7, (3,)),
                cheap_transcript(KB),
                claims=[jnp.zeros((), KB)] * 2,
            )


class TotalCapTracedTest(absltest.TestCase):
    """The sp1-zorch#242 deliverable: the total-Σ-heights-cap round with TRACED
    heights is shard-invariant — two shards whose per-chip heights differ but
    whose ``TotalCapClass`` matches trace ONCE (the row offsets are the cumsum of
    the traced heights; every buffer shape is a class constant). And it stays
    byte-exact: each traced prove matches the static per-shard total-cap path
    (itself reference-checked above), so the runtime-offset packing is correct."""

    _NUM_VARS = 4
    _NCHIPS = 2
    # A class bounding both shards below: W = max ceil(h/2) over {5,2} and {3,4}
    # is 3; each 3-column `_witness_trace` chip's area is 3*evenpad(h), so
    # Σ areas = 24 for both shards and area_cap ≥ 24 + 2W = 30 — 32 is a valid
    # class bound.
    _CLASS = TotalCapClass(area_cap=32, window=3)

    def _summand(self, alphas, lambdas, beta):
        return JaggedZerocheckSummand(
            eval_fns=[_eval_fn] * self._NCHIPS,
            alphas=alphas,
            lambdas=lambdas,
            beta=beta,
            public_values=_PV,
        )

    @staticmethod
    def _pad_rows(trace, height):
        """Column-major ``[cols, nr]`` -> ``[cols, height]``, zeros past nr."""
        return jnp.pad(trace, ((0, 0), (0, height - trace.shape[1])))

    def test_traced_total_cap_shares_one_compile_and_byte_matches(self) -> None:
        num_vars, nchips = self._NUM_VARS, self._NCHIPS
        row_block = 2 * self._CLASS.window
        alphas = [rlc_coeffs(_rand(99 + i, ()), _K) for i in range(nchips)]
        lambdas = _rand(55, (nchips,))
        beta = _rand(77, ())
        zeta = _rand(7, (num_vars,))
        challenges = [_rand(1000 + r, ()) for r in range(num_vars)]

        # Inputs ride as jitted arrays (not closed-over constants), so two
        # different-height shards hit the SAME executable iff their shapes match.
        def run(traces, heights, claims):  # type: ignore[no-untyped-def]
            _, _, msgs = prove_jagged_zerocheck(
                self._summand(alphas, lambdas, beta),
                [traces[i] for i in range(nchips)],
                [heights[i] for i in range(nchips)],
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=[claims[i] for i in range(nchips)],
                total_cap_class=self._CLASS,
            )
            return msgs.round_poly, msgs.challenge

        jrun = frx.jit(run)

        for shard_heights in ([5, 2], [3, 4]):
            exact = [_witness_trace(i, nr) for i, nr in enumerate(shard_heights)]
            _, claims = _gkr_inputs(beta, exact, zeta)
            padded = [self._pad_rows(t, row_block) for t in exact]

            got_poly, got_chal = jrun(
                padded,
                jnp.asarray(shard_heights, jnp.int32),
                jnp.stack(claims),
            )

            # Static per-shard total-cap path (width_caps=None, class derived
            # from the shard's own heights): the byte oracle.
            _, _, want = prove_jagged_zerocheck(
                self._summand(alphas, lambdas, beta),
                exact,
                shard_heights,
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=claims,
            )
            label = f"heights {shard_heights}"
            _assert_bytes_equal(got_poly, want.round_poly, f"{label} round_poly")
            _assert_bytes_equal(got_chal, want.challenge, f"{label} challenge")

        # THE shard-invariance assertion: distinct traced-height shards of one
        # class compiled exactly once.
        self.assertEqual(jrun._cache_size(), 1)

    def test_traced_total_cap_handles_runtime_empty_chip(self) -> None:
        # A chip live at compile time but empty at run time (height 0) must fall
        # to the trivial claim identity and zero finals, sharing the compile.
        num_vars, nchips = self._NUM_VARS, self._NCHIPS
        row_block = 2 * self._CLASS.window
        alphas = [rlc_coeffs(_rand(99 + i, ()), _K) for i in range(nchips)]
        lambdas = _rand(55, (nchips,))
        beta = _rand(77, ())
        zeta = _rand(7, (num_vars,))
        challenges = [_rand(1000 + r, ()) for r in range(num_vars)]

        for shard_heights in ([5, 0], [0, 4]):
            exact = [_witness_trace(i, nr) for i, nr in enumerate(shard_heights)]
            _, claims = _gkr_inputs(beta, exact, zeta)
            padded = [self._pad_rows(t, row_block) for t in exact]

            finals_t, _, msgs_t = prove_jagged_zerocheck(
                self._summand(alphas, lambdas, beta),
                padded,
                [jnp.asarray(nr, jnp.int32) for nr in shard_heights],
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=claims,
                total_cap_class=self._CLASS,
            )
            # Static reference with the same (host-int) heights.
            finals_s, _, msgs_s = prove_jagged_zerocheck(
                self._summand(alphas, lambdas, beta),
                exact,
                shard_heights,
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=claims,
            )
            label = f"heights {shard_heights}"
            _assert_bytes_equal(msgs_t.round_poly, msgs_s.round_poly, label)
            for i, (ft, fs) in enumerate(zip(finals_t, finals_s, strict=True)):
                _assert_bytes_equal(ft, fs, f"{label} finals[{i}]")


if __name__ == "__main__":
    absltest.main()
