# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""``verify_shard_zerocheck`` vs ``prove_shard_zerocheck`` — the stage duality.

Accept: a proof off the prover verifies and leaves the sponge in the
prover's exact state, so the two streams enter the evaluation stage in
sync. The fixture's GKR openings are honest column-MLE evaluations at zeta
(unlike ``prover_test``'s random ones): the dual's final oracle check closes
the claim chain back to the opened row, which only an internally consistent
proof satisfies. Reject: each tamper lands on the acceptance leg that owns
it — the wire's challenge/claim copies, the round-poly replay, the sampled
point copy, and the opened values through the oracle check (nothing samples
after the openings absorb inside this stage, so without that check an
openings tamper would only surface downstream).
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.testkit.transcript import cheap_transcript

from zorch.pcs.jagged.region import JaggedRegion
from sp1_zorch.logup_gkr.prover import ChipEvaluation, _open_chip
from sp1_zorch.zerocheck.prover import prove_shard_zerocheck
from sp1_zorch.zerocheck.verifier import verify_shard_zerocheck


BF = koalabear_mont
EF = koalabearx4_mont

_MAX_LOG_ROW_COUNT = 3
_CHIP_NAMES = ("alpha", "lookup")
_CHIP_HEIGHTS = {"alpha": 5, "lookup": 3}


class _WitnessChip:
    """Witness-shaped stub (a == 1 on real rows, so constraints vanish there
    while ``C(0_row) != 0`` keeps the padded-row correction live) whose second
    constraint folds in ``public_values[0]`` — the pv-binding seam the dual
    must thread through its oracle check."""

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


class VerifyShardZerocheckTest(absltest.TestCase):
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

        cls.main_region = JaggedRegion.from_chips(
            [alpha_main, lookup_main],
            log_stacking_height=4,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=_CHIP_NAMES,
        )
        cls.prep_region = JaggedRegion.from_chips(
            [alpha_prep],
            log_stacking_height=4,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=("alpha",),
        )

        cls.chips = {"alpha": _WitnessChip(), "lookup": _LookupChip()}
        cls.public_values = _rand_bf(7, (8,))
        # Longer than max_log_row_count so the zeta slice is observable.
        cls.eval_point = _rand_ef(8, (5,))
        zeta = cls.eval_point[-_MAX_LOG_ROW_COUNT:]

        # Honest GKR openings: every [main | prep] column's zero-extended MLE
        # evaluated at zeta (the eq order of the round engine), so each
        # chip's claim equals the true batched sum the sumcheck proves.
        rev = zeta[::-1]
        cls.chip_openings = {
            "alpha": ChipEvaluation(
                main=_open_chip(alpha_main, rev, 5),
                preprocessed=_open_chip(alpha_prep, rev, 3),
            ),
            "lookup": ChipEvaluation(
                main=_open_chip(lookup_main, rev, 3), preprocessed=None
            ),
        }

        cls.prover_transcript, cls.proof = prove_shard_zerocheck(
            cls.chips,
            cls.main_region,
            cls.prep_region,
            cls.public_values,
            cls.eval_point,
            cls.chip_openings,
            cls._pre_stage_transcript(),
            max_log_row_count=_MAX_LOG_ROW_COUNT,
        )

    @staticmethod
    def _pre_stage_transcript():
        """A sponge with prior absorbs, as the real pipeline's preamble + GKR
        stages leave it. A FRESH cheap sponge squeezes zero challenges, which
        degenerates the stage (beta = lambda = 0 weights every opened value
        and every chip but the last out of the oracle check) — tampers would
        be accepted not because the dual is loose but because the statement
        stops depending on the tampered values."""
        return cheap_transcript(BF).observe(_rand_bf(9, (4,)))

    @classmethod
    def _verify(cls, proof):
        return verify_shard_zerocheck(
            cls.chips,
            _CHIP_NAMES,
            _CHIP_HEIGHTS,
            cls.public_values,
            cls.eval_point,
            cls.chip_openings,
            proof,
            cls._pre_stage_transcript(),
            max_log_row_count=_MAX_LOG_ROW_COUNT,
        )

    def test_accepts_and_matches_the_prover_stream(self) -> None:
        transcript, point, ok = self._verify(self.proof)
        self.assertTrue(bool(ok))
        _assert_bytes_equal(point, self.proof.msgs.challenge, "point seam")
        _, got = transcript.sample(2)
        _, want = self.prover_transcript.sample(2)
        _assert_bytes_equal(got, want, "post-stage sample")

    def test_tampered_round_poly_rejected(self) -> None:
        polys = self.proof.msgs.round_poly
        bad = replace(
            self.proof,
            msgs=replace(
                self.proof.msgs,
                round_poly=polys.at[1, 0].add(jnp.ones((), polys.dtype)),
            ),
        )
        _, _, ok = self._verify(bad)
        self.assertFalse(bool(ok))

    def test_tampered_challenge_copies_rejected(self) -> None:
        for field in ("batching_challenge", "gkr_opening_batch_challenge", "lambda_"):
            value = getattr(self.proof, field)
            bad = replace(self.proof, **{field: value + jnp.ones((), value.dtype)})
            _, _, ok = self._verify(bad)
            self.assertFalse(bool(ok), field)

    def test_tampered_zeta_copy_rejected(self) -> None:
        bad_zeta = self.proof.zeta.at[0].add(jnp.ones((), self.proof.zeta.dtype))
        _, _, ok = self._verify(replace(self.proof, zeta=bad_zeta))
        self.assertFalse(bool(ok))

    def test_tampered_claimed_sum_rejected(self) -> None:
        bad = self.proof.claimed_sum + jnp.ones((), self.proof.claimed_sum.dtype)
        _, _, ok = self._verify(replace(self.proof, claimed_sum=bad))
        self.assertFalse(bool(ok))

    def test_tampered_point_copy_rejected(self) -> None:
        ch = self.proof.msgs.challenge
        bad = replace(
            self.proof,
            msgs=replace(self.proof.msgs, challenge=ch.at[0].add(jnp.ones((), ch.dtype))),
        )
        _, _, ok = self._verify(bad)
        self.assertFalse(bool(ok))

    def test_tampered_opened_values_rejected_by_the_oracle_check(self) -> None:
        opening = self.proof.opened_values["alpha"]
        bad_openings = dict(self.proof.opened_values)
        bad_openings["alpha"] = ChipEvaluation(
            main=opening.main.at[0].add(jnp.ones((), opening.main.dtype)),
            preprocessed=opening.preprocessed,
        )
        _, _, ok = self._verify(replace(self.proof, opened_values=bad_openings))
        self.assertFalse(bool(ok))

    def test_missing_chip_opening_raises(self) -> None:
        openings = {"alpha": self.proof.opened_values["alpha"]}
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            self._verify(replace(self.proof, opened_values=openings))

    def test_wrong_round_count_raises(self) -> None:
        bad = replace(
            self.proof,
            msgs=replace(self.proof.msgs, round_poly=self.proof.msgs.round_poly[:-1]),
        )
        with self.assertRaisesRegex(ValueError, "per row variable"):
            self._verify(bad)


if __name__ == "__main__":
    absltest.main()
