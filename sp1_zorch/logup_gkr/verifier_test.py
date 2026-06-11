# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""``verify_logup_gkr`` vs ``prove_logup_gkr`` — the stage duality.

Accept: a proof straight off the prover verifies and leaves the sponge in
the prover's exact state, so the two streams enter the next stage in sync.
Reject: each tamper lands on the acceptance leg that owns it — the grind
gate, the layer replay, the wire's point copy, the leaf check on the opened
values. The leaf check is what makes an openings tamper a *stage-local*
reject: nothing samples after the openings absorb inside this stage, so
without it the tamper would only surface downstream.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from absl.testing import absltest
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.prover import ChipEvaluation, prove_logup_gkr
from sp1_zorch.logup_gkr.verifier import verify_logup_gkr, virtual_padding_geq
from zorch.logup_gkr.circuit import LogUpGkrOutput
from zorch.poly.multilinear import eval_mle
from zorch.testkit.transcript import cheap_transcript

_NUM_BETAS = 3
_NUM_ROW_VARIABLES = 4
_CHIP_HEIGHTS = {"A": 24, "B": 4}
_CHIP_NAMES = tuple(_CHIP_HEIGHTS)


def _interaction(mult_col: int, val_col: int, *, kind: int = 3) -> Interaction:
    return Interaction(
        values=(VirtualPairCol.single_main(val_col),),
        multiplicity=VirtualPairCol.single_main(mult_col),
        kind=kind,
        is_send=True,
    )


def _main(height: int, width: int = 2, offset: int = 0) -> jnp.ndarray:
    return (
        jnp.arange(offset, offset + height * width, dtype=jnp.uint32)
        .reshape(height, width)
        .view(F)
    )


def _gkr_chips() -> list[GkrChip]:
    # Three interactions: an odd count, so the leaf MLE pads one slot to the
    # power of two — the accept test then exercises the (num 0, den 1)
    # padding semantics against the prover's neutral interaction slots.
    return [
        GkrChip("A", (_interaction(0, 1), _interaction(1, 0, kind=4))),
        GkrChip("B", (_interaction(0, 1, kind=5),)),
    ]


def _prove(*, pow_bits: int = 0, witness=None):
    region = JaggedRegion.from_chips(
        [_main(_CHIP_HEIGHTS["A"]), _main(_CHIP_HEIGHTS["B"], offset=100)],
        log_stacking_height=3,
        max_log_row_count=5,
        chip_names=_CHIP_NAMES,
    )
    return prove_logup_gkr(
        _gkr_chips(),
        region,
        None,
        cheap_transcript(F),
        num_betas=_NUM_BETAS,
        num_row_variables=_NUM_ROW_VARIABLES,
        pow_bits=pow_bits,
        witness=witness,
    )


def _verify(proof, *, pow_bits: int = 0):
    return verify_logup_gkr(
        _gkr_chips(),
        _CHIP_NAMES,
        _CHIP_HEIGHTS,
        proof,
        cheap_transcript(F),
        num_betas=_NUM_BETAS,
        num_row_variables=_NUM_ROW_VARIABLES,
        pow_bits=pow_bits,
    )


class VerifyLogupGkrTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prover_transcript, cls.proof = _prove()

    def test_accepts_and_matches_the_prover_stream(self) -> None:
        transcript, eval_point, ok = _verify(self.proof)
        self.assertTrue(bool(ok))
        self.assertTrue(bool(jnp.all(eval_point == self.proof.eval_point)))
        _, got = transcript.sample(2)
        _, want = self.prover_transcript.sample(2)
        self.assertTrue(bool(jnp.all(got == want)))

    def test_tampered_round_poly_rejected(self) -> None:
        rp = self.proof.round_proofs[1]
        bad_polys = rp.round_polys.at[0, 0].add(jnp.ones((), rp.round_polys.dtype))
        bad_rounds = list(self.proof.round_proofs)
        bad_rounds[1] = replace(rp, round_polys=bad_polys)
        _, _, ok = _verify(replace(self.proof, round_proofs=bad_rounds))
        self.assertFalse(bool(ok))

    def test_tampered_circuit_output_rejected(self) -> None:
        out = self.proof.circuit_output
        bad_num = out.numerator.at[0].add(jnp.ones((), out.numerator.dtype))
        bad = LogUpGkrOutput(numerator=bad_num, denominator=out.denominator)
        _, _, ok = _verify(replace(self.proof, circuit_output=bad))
        self.assertFalse(bool(ok))

    def test_tampered_chip_opening_rejected_by_the_leaf_check(self) -> None:
        opening = self.proof.chip_openings["A"]
        bad_main = opening.main.at[0].add(jnp.ones((), opening.main.dtype))
        bad_openings = dict(self.proof.chip_openings)
        bad_openings["A"] = ChipEvaluation(
            main=bad_main, preprocessed=opening.preprocessed
        )
        _, _, ok = _verify(replace(self.proof, chip_openings=bad_openings))
        self.assertFalse(bool(ok))

    def test_tampered_eval_point_copy_rejected(self) -> None:
        bad_point = self.proof.eval_point.at[0].add(
            jnp.ones((), self.proof.eval_point.dtype)
        )
        _, _, ok = _verify(replace(self.proof, eval_point=bad_point))
        self.assertFalse(bool(ok))

    def test_wrong_grind_witness_rejected(self) -> None:
        # The transcript's own grind finds a witness whose one-bit gate
        # passes on the stage's fresh sponge; prove under it, then verify
        # with a failing witness: the pow leg must reject.
        _, passing = cheap_transcript(F).grind(1)
        failing = next(
            w
            for w in (jnp.array(i, F) for i in range(16))
            if not bool(cheap_transcript(F).check_witness(1, w)[1])
        )

        _, proof = _prove(pow_bits=1, witness=passing)
        _, _, ok = _verify(proof, pow_bits=1)
        self.assertTrue(bool(ok))
        _, _, ok = _verify(replace(proof, witness=failing), pow_bits=1)
        self.assertFalse(bool(ok))

    def test_wrong_layer_count_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "one layer proof per"):
            _verify(replace(self.proof, round_proofs=self.proof.round_proofs[:-1]))

    def test_missing_chip_opening_raises(self) -> None:
        openings = {"A": self.proof.chip_openings["A"]}
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            _verify(replace(self.proof, chip_openings=openings))

    def test_extra_chip_opening_raises(self) -> None:
        openings = dict(self.proof.chip_openings)
        openings["stowaway"] = openings["A"]
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            _verify(replace(self.proof, chip_openings=openings))


class VirtualPaddingGeqTest(absltest.TestCase):
    def test_matches_the_materialized_indicator(self) -> None:
        # The fold must agree with the brute-force MLE of the 0/1 indicator
        # at every threshold shape: empty padding region (full-height chip),
        # power-of-two and ragged thresholds, all-padding.
        # An arbitrary full-limb EF point (small fixed affine limbs, all
        # well below the field modulus).
        limbs = jnp.arange(24, dtype=jnp.uint32).reshape(6, 4) * 48271 + 7
        point = jax.lax.bitcast_convert_type(limbs, EF)
        size = 1 << point.shape[0]
        for threshold in (0, 4, 24, 32, size):
            indicator = (jnp.arange(size) >= threshold).astype(EF)
            want = eval_mle(indicator, point)
            got = virtual_padding_geq(threshold, point)
            self.assertTrue(
                bool(jnp.all(got == want)), f"threshold {threshold} diverged"
            )


if __name__ == "__main__":
    absltest.main()
