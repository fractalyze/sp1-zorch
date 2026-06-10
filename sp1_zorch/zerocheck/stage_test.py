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

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.logup_gkr.prover import ChipEvaluation
from sp1_zorch.zerocheck.jagged import prove_jagged_zerocheck
from sp1_zorch.zerocheck.prover import gkr_powers, rlc_coeffs
from sp1_zorch.zerocheck.stage import prove_shard_zerocheck

# The pinned jaxlib wheel's embedded zkx CPU emitter CHECK-fails on the rank-1
# linalg.broadcast inside an engaged zorch.constraint_eval region
# (fractalyze/zkx#605), so run every marker's inline decomposition instead —
# byte-identical output, only the fusion marker is dropped. Tracked removal:
# fractalyze/sp1-zorch#62.
import zorch._composite as _zorch_composite

_zorch_composite._HAS_COMPOSITE_OP = False

BF = koalabear_mont
EF = koalabearx4_mont

_MAX_LOG_ROW_COUNT = 3


class _WitnessChip:
    """Witness-shaped stub (a == 1 on real rows, so constraints vanish there
    while ``C(0_row) != 0`` keeps the padded-row correction live) whose second
    constraint folds in ``public_values[0]`` — the pv-binding seam the stage
    must thread through."""

    def eval_constraints(self, trace, public_values):
        a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
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
        eval_fns = [
            lambda tr: chips["alpha"].eval_constraints(tr, public_values),
            lambda tr: chips["lookup"].eval_constraints(tr, public_values),
        ]
        alphas = [rlc_coeffs(alpha, 2), rlc_coeffs(alpha, 0)]
        lambdas = rlc_coeffs(lambda_, 2)

        cls.want_finals, cls.want_transcript, cls.want_msgs = prove_jagged_zerocheck(
            eval_fns,
            traces,
            [5, 3],
            alphas,
            lambdas,
            zeta,
            t,
            beta=beta,
            claims=claims,
        )
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


if __name__ == "__main__":
    absltest.main()
