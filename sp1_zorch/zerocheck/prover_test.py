# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`prove_shard_zerocheck` glue vs the hand-derived SP1 zerocheck-stage recipe.

The stage's job is pure derivation glue: sample the three stage challenges in
SP1's order (batching -> GKR opening batch -> lambda), slice zeta off the GKR
point, weight each chip's GKR openings into its claim, assemble the
``[main | prep]`` column-major traces from the regions, and hand everything to
`prove_jagged_zerocheck`. The test replays exactly that recipe by hand on the
same deterministic sponge and demands byte-identical outputs — any drift in
sampling order, claim weighting, or trace assembly desynchronizes the two
Fiat-Shamir streams and fails loudly.

Reference: whir-zorch ``sp1/shard_prover/prover.py``, its zerocheck (SP1
"phase 3") block; vocabulary in ``docs/shard-pipeline.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import sample_challenge

from zorch.pcs.jagged.region import JaggedRegion
from sp1_zorch.logup_gkr.prover import ChipEvaluation
from sp1_zorch.zerocheck.jagged import (
    JaggedZerocheckSummand,
    prove_jagged_zerocheck,
)
from sp1_zorch.zerocheck.coeffs import gkr_powers, rlc_coeffs
from sp1_zorch.zerocheck.prover import (
    OpenedValuesRound,
    prove_shard_zerocheck,
    split_opened_values,
)


BF = koalabear_mont
EF = koalabearx4_mont

_MAX_LOG_ROW_COUNT = 3


class _WitnessChip:
    """Witness-shaped stub (a == 1 on real rows, so constraints vanish there
    while ``C(0_row) != 0`` keeps the padded-row correction live) whose second
    constraint folds in ``public_values[0]`` — the pv-binding seam the stage
    must thread through.

    Reads the rw-constraints export layout: flat rows in ``[prep | main]``
    order (prep width 2, so the 3 main value columns sit at flat indices
    2..4), while the stage's traces and openings are wire-order
    ``[main | prep]`` — the ``export_order_eval_fn`` rotation is what lines
    them up."""

    def eval_constraints(self, trace, public_values):
        a, b, c = trace[:, 2], trace[:, 3], trace[:, 4]
        one = jnp.ones((), trace.dtype)
        pv0 = jnp.concatenate([public_values[:1], jnp.zeros((3,), BF)]).view(EF)[0]
        return jnp.stack([(a - one) * (c - one), (a - one) * (b + pv0)], axis=-1)


class _LookupChip:
    """Constraint-less chip (SP1's Byte / Program / Range shape)."""

    def eval_constraints(self, trace, public_values):
        return jnp.zeros((trace.shape[0], 0), dtype=trace.dtype)


def _rand_bf(seed: int, shape) -> jnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=BF)


def _rand_ef(seed: int, shape) -> jnp.ndarray:
    return _rand_bf(seed, tuple(shape) + (4,)).view(EF).reshape(shape)


def _u32(a) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


def _assert_bytes_equal(got, want, label: str = "") -> None:
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


class ProveZerocheckTest(absltest.TestCase):
    """One stage run vs one hand replay; each test byte-compares one output."""

    @classmethod
    def setUpClass(cls):
        # alpha: 3 main cols x 5 real rows (col a == 1), prep 2 cols x 3 rows
        # (shorter than num_real — exercises the prep zero-pad). lookup: one
        # main col x 3 rows, no prep, no constraints.
        alpha_main = jnp.concatenate(
            [jnp.ones((5, 1), dtype=BF), _rand_bf(1, (5, 2))], axis=1
        )
        alpha_prep = _rand_bf(2, (3, 2))
        lookup_main = _rand_bf(3, (3, 1))

        main_region = JaggedRegion.from_chips(
            [alpha_main, lookup_main],
            log_stacking_height=4,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=("alpha", "lookup"),
        )
        prep_region = JaggedRegion.from_chips(
            [alpha_prep],
            log_stacking_height=4,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=("alpha",),
        )

        chips = {"alpha": _WitnessChip(), "lookup": _LookupChip()}
        chip_openings = {
            "alpha": ChipEvaluation(
                main=_rand_ef(4, (3,)), preprocessed=_rand_ef(5, (2,))
            ),
            "lookup": ChipEvaluation(main=_rand_ef(6, (1,)), preprocessed=None),
        }
        public_values = _rand_bf(7, (8,))
        # Longer than max_log_row_count so the zeta slice is observable.
        eval_point = _rand_ef(8, (5,))

        transcript = cheap_transcript(BF)
        cls.got_transcript, cls.proof = prove_shard_zerocheck(
            chips,
            main_region,
            prep_region,
            public_values,
            eval_point,
            chip_openings,
            transcript,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
        )

        # Hand replay of the Phase-3 recipe on the same sponge.
        t, alpha = sample_challenge(transcript, EF, 4)
        t, beta = sample_challenge(t, EF, 4)
        t, lambda_ = sample_challenge(t, EF, 4)
        cls.alpha, cls.beta, cls.lambda_ = alpha, beta, lambda_

        zeta = eval_point[-_MAX_LOG_ROW_COUNT:]
        claims = [
            jnp.sum(
                gkr_powers(beta, 5)
                * jnp.concatenate(
                    [chip_openings["alpha"].main, chip_openings["alpha"].preprocessed]
                )
            ),
            jnp.sum(gkr_powers(beta, 1) * chip_openings["lookup"].main),
        ]
        # [main | prep] column-major, prep zero-padded to num_real.
        alpha_trace = jnp.concatenate(
            [
                alpha_main.T,
                jnp.concatenate([alpha_prep.T, jnp.zeros((2, 2), dtype=BF)], axis=1),
            ],
            axis=0,
        )
        traces = [alpha_trace, lookup_main.T]
        # The chips' 2-ary ``eval_constraints`` are the eval_fns; the statement
        # is threaded to the summand as a declared operand, not closed over.
        # The stage hands the summand export-order rows: [main | prep] trace
        # rows rotate to the exporter's [prep | main] before the chip sees
        # them (alpha: main width 3 of 5 columns; lookup is main-only).
        eval_fns = [
            lambda tr, pv: chips["alpha"].eval_constraints(
                jnp.concatenate([tr[..., 3:], tr[..., :3]], axis=-1), pv
            ),
            chips["lookup"].eval_constraints,
        ]
        alphas = [rlc_coeffs(alpha, 2), rlc_coeffs(alpha, 0)]
        lambdas = rlc_coeffs(lambda_, 2)

        want_finals, t, want_msgs = prove_jagged_zerocheck(
            JaggedZerocheckSummand(
                eval_fns=eval_fns,
                alphas=alphas,
                lambdas=lambdas,
                beta=beta,
                public_values=public_values,
            ),
            traces,
            [5, 3],
            zeta,
            t,
            claims=claims,
        )
        # The stage's transcript tail replayed raw — the deliberate second
        # writing of the opened-values absorb schedule (chip count, then per
        # chip the length-prefixed [prep | main] evaluations at the sumcheck
        # point; a prep-less chip absorbs a bare zero length). Finals stack
        # [main | prep], so alpha's prep is rows 3:5 of its column stack.
        alpha_vals = want_finals[0][:, 0]
        lookup_vals = want_finals[1][:, 0]
        t = t.observe(jnp.array(2, BF))
        t = t.observe(jnp.array(2, BF))
        t = t.observe(alpha_vals[3:5])
        t = t.observe(jnp.array(3, BF))
        t = t.observe(alpha_vals[:3])
        t = t.observe(jnp.array(0, BF))
        t = t.observe(jnp.array(1, BF))
        t = t.observe(lookup_vals[:1])
        cls.want_finals, cls.want_transcript, cls.want_msgs = want_finals, t, want_msgs
        cls.zeta = zeta
        cls.want_claims = claims

    def test_round_polys_byte_match_hand_replay(self):
        _assert_bytes_equal(self.proof.msgs.round_poly, self.want_msgs.round_poly)

    def test_claimed_sum_is_lambda_horner_fold_of_chip_claims(self):
        _assert_bytes_equal(
            self.proof.claimed_sum,
            self.want_claims[0] * self.lambda_ + self.want_claims[1],
        )

    def test_finals_byte_match_hand_replay(self):
        self.assertEqual(len(self.proof.finals), len(self.want_finals))
        for i, (got, want) in enumerate(zip(self.proof.finals, self.want_finals)):
            _assert_bytes_equal(got, want, f"chip {i} finals")

    def test_zeta_is_eval_point_tail(self):
        _assert_bytes_equal(self.proof.zeta, self.zeta)

    def test_zeta_is_not_the_sumcheck_point(self):
        # verify_zerocheck checks SP1's gpu_z_row.txt -- which is zeta, the
        # eval_point row-tail -- against `proof.zeta`, NOT against the zerocheck
        # sumcheck point (`msgs.challenge`). Both index the row variables, so on
        # a real shard they share a length and an earlier harness conflated
        # them: the shape check passed while the values mismatched. Pin that
        # they are distinct vectors so the check can't regress to the sumcheck
        # point.
        zeta = _u32(self.proof.zeta)
        sumcheck_point = _u32(self.proof.msgs.challenge[::-1])
        self.assertEqual(zeta.shape, sumcheck_point.shape)
        self.assertFalse(np.array_equal(zeta, sumcheck_point))

    def test_challenges_sampled_in_sp1_order(self):
        for name, got, want in (
            ("batching_challenge", self.proof.batching_challenge, self.alpha),
            (
                "gkr_opening_batch_challenge",
                self.proof.gkr_opening_batch_challenge,
                self.beta,
            ),
            ("lambda_", self.proof.lambda_, self.lambda_),
        ):
            _assert_bytes_equal(got, want, name)

    def test_transcript_streams_converge(self):
        _, got = sample_challenge(self.got_transcript, EF, 4)
        _, want = sample_challenge(self.want_transcript, EF, 4)
        _assert_bytes_equal(got, want)

    def test_opened_values_are_the_finals_split(self):
        opened = self.proof.opened_values
        _assert_bytes_equal(opened["alpha"].main, self.want_finals[0][:3, 0])
        _assert_bytes_equal(opened["alpha"].preprocessed, self.want_finals[0][3:5, 0])
        _assert_bytes_equal(opened["lookup"].main, self.want_finals[1][:1, 0])
        self.assertIsNone(opened["lookup"].preprocessed)


class OpenedValuesRoundGuardTest(absltest.TestCase):
    """The absorb order is the caller's statement; a mapping that does not
    cover it exactly is rejected before anything reaches the transcript —
    the mapping is proof-controlled once the verifier dual drives the
    Round."""

    def test_extra_chip_raises(self) -> None:
        opened = {
            "alpha": ChipEvaluation(main=jnp.zeros((1,), EF), preprocessed=None),
            "stowaway": ChipEvaluation(main=jnp.zeros((1,), EF), preprocessed=None),
        }
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            OpenedValuesRound(opened, ("alpha",))(None, cheap_transcript(BF))


class SplitOpenedValuesTest(absltest.TestCase):
    """Pins the finals split directly: position 0 of each ``[main | prep]``
    column stack is the column's evaluation at the sumcheck point, sliced
    main-first per the ``chip_traces`` order; a chip absent from the prep
    region gets ``preprocessed=None``."""

    def test_splits_main_then_prep(self) -> None:
        # "alpha": 2 main cols + 1 prep col; "lookup": 1 main col, no prep.
        main_region = JaggedRegion(
            dense=jnp.zeros(8, dtype=BF),
            chip_starts=(0, 6, 8),
            row_counts=(3, 2, 4, 1),
            column_counts=(2, 1, 1, 1),
            log_stacking_height=2,
            chip_names=("alpha", "lookup"),
        )
        prep_region = JaggedRegion(
            dense=jnp.zeros(3, dtype=BF),
            chip_starts=(0, 3),
            row_counts=(3, 4, 1),
            column_counts=(1, 1, 1),
            log_stacking_height=2,
            chip_names=("alpha",),
        )
        finals = [
            jnp.array([[31, 0], [32, 0], [33, 0]], dtype=EF),
            jnp.array([[41, 0]], dtype=EF),
        ]

        opened = split_opened_values(finals, main_region, prep_region)

        _assert_bytes_equal(opened["alpha"].main, finals[0][:2, 0], "alpha main")
        _assert_bytes_equal(
            opened["alpha"].preprocessed, finals[0][2:3, 0], "alpha prep"
        )
        _assert_bytes_equal(opened["lookup"].main, finals[1][:1, 0], "lookup main")
        self.assertIsNone(opened["lookup"].preprocessed)

    def test_no_prep_region_means_no_preprocessed_anywhere(self) -> None:
        main_region = JaggedRegion(
            dense=jnp.zeros(8, dtype=BF),
            chip_starts=(0, 8),
            row_counts=(4, 4, 1),
            column_counts=(2, 1, 1),
            log_stacking_height=2,
            chip_names=("alpha",),
        )
        finals = [jnp.array([[31, 0], [32, 0]], dtype=EF)]

        opened = split_opened_values(finals, main_region, None)

        _assert_bytes_equal(opened["alpha"].main, finals[0][:, 0], "alpha main")
        self.assertIsNone(opened["alpha"].preprocessed)


if __name__ == "__main__":
    absltest.main()
