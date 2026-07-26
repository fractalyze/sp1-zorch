# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""``ShardVerifier`` vs ``ShardProver`` — the structural mirror.

The verifier's guarantee is structural before it is cryptographic: one
verifier Stage per prover stage, so a proof whose message list misaligns with
the schedule is rejected loudly by ``verify_rounds`` itself rather than
accepted on a desynced stream. These tests pin that alignment plus all four
stage duals against a full prover run (the shared ``shard_testkit``
fixture): same Fiat-Shamir stream, seams derived for the downstream
duals, a tampered proof section rejected end to end (the per-leg
tamper coverage is each stage's own verifier test), and the zerocheck
dual's opening-shape statement checks.
"""

from __future__ import annotations

from dataclasses import replace

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont

from zorch.testkit.transcript import cheap_transcript

from sp1_zorch.shard_prover.types import ChipShape, TraceShape
from sp1_zorch.shard_prover.verify_shard import (
    ZerocheckVerifier,
)


from sp1_zorch.shard_prover.prove_shard import ZerocheckClaim

from sp1_zorch.shard_prover.shard_testkit import (
    CHIP_HEIGHT,
    CHIP_WIDTH,
    MAX_LOG_ROW_COUNT,
    small_shard_fixture,
)

BF = koalabear_mont


def _u32(a) -> np.ndarray:
    return np.asarray(frx.lax.bitcast_convert_type(a, fnp.uint32)).reshape(-1)


def _assert_bytes_equal(got, want, label: str = "") -> None:
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


class VerifyShardChainTest(absltest.TestCase):
    """A full four-phase prove vs one full verify, every phase dual checked
    cryptographically end to end."""

    @classmethod
    def setUpClass(cls):
        fx = small_shard_fixture()
        cls.fx = fx
        proof = fx.proof
        cls.commitment = proof.commitment
        cls.gkr_proof = proof.gkr
        cls.zc_proof = proof.zerocheck
        cls.je_proof = proof.jagged
        verified = fx.verifier.verify(fx.claim, proof, cheap_transcript(BF))
        cls.dual_transcript, cls.dual_ok = verified.transcript, verified.ok
        # The same phases again, kept apart, so a test can assert on one
        # phase's reduced claim or drive one dual directly.
        t, cls.roots = fx.verifier.absorber.absorb(
            fx.claim, proof.commitment, cheap_transcript(BF)
        )
        gkr = fx.verifier.gkr.verify(fx.claim, proof.gkr, t)
        cls.gkr_reduced = gkr.reduced_claim
        cls.zc_claim = ZerocheckClaim(fx.claim.public_values, gkr.reduced_claim)
        cls.zc_transcript = gkr.transcript
        zc = fx.verifier.zerocheck.verify(
            cls.zc_claim, proof.zerocheck, gkr.transcript
        )
        cls.zc_reduced = zc.reduced_claim

    def test_both_roles_derive_the_same_seams(self) -> None:
        """The mirror that matters: every claim the prover reduces to, the
        verifier re-derives from the proof alone and gets the same value.

        A phase wired to the wrong claim, or a dual replaying a different
        schedule, breaks this even when both sides still accept — which is
        what a name-level symmetry check would miss."""
        fx = self.fx
        t, _, _, _ = fx.prover.committer.commit(
            fx.claim, fx.witness, cheap_transcript(BF)
        )
        p_gkr = fx.prover.gkr.prove(fx.claim, fx.witness, t)
        p_zc = fx.prover.zerocheck.prove(
            ZerocheckClaim(fx.claim.public_values, p_gkr.reduced_claim),
            fx.witness,
            p_gkr.transcript,
        )

        _assert_bytes_equal(
            self.gkr_reduced.eval_point,
            p_gkr.reduced_claim.eval_point,
            "gkr eval_point",
        )
        for name, want in p_gkr.reduced_claim.chip_openings.items():
            _assert_bytes_equal(
                self.gkr_reduced.chip_openings[name].main, want.main, f"gkr {name}"
            )
        _assert_bytes_equal(
            self.zc_reduced.point, p_zc.reduced_claim.point, "zerocheck point"
        )
        for name, want in p_zc.reduced_claim.opened_values.items():
            _assert_bytes_equal(
                self.zc_reduced.opened_values[name].main, want.main, f"zerocheck {name}"
            )

    def test_live_duals_match_the_prover_stream(self) -> None:
        """The verifier accepts and its output transcript byte-matches the
        prover's post-stage-4 one, so the two Fiat-Shamir streams agree
        through every stage, glue included."""
        self.assertTrue(bool(self.dual_ok))
        _, got = self.dual_transcript.sample(1)
        _, want = self.fx.prover_transcript.sample(1)
        _assert_bytes_equal(got, want, "post-stage-4 sample")

    def test_gkr_dual_writes_the_zerocheck_seams(self) -> None:
        """The point is the dual's own derivation (pinned against the wire
        copy inside the stage), the openings the proof's leaf-checked values
        — what the zerocheck dual reads, surviving to the verifier's output."""
        _assert_bytes_equal(
            self.gkr_reduced.eval_point, self.gkr_proof.eval_point, "point"
        )
        _assert_bytes_equal(
            self.gkr_reduced.chip_openings["alpha"].main,
            self.gkr_proof.chip_openings["alpha"].main,
            "openings",
        )

    def test_tampered_gkr_message_rejected_through_the_chain(self) -> None:
        """One representative stage-2 tamper rejecting end to end;
        the per-leg coverage is the stage's own test file."""
        rp = self.gkr_proof.round_proofs[0]
        bad_polys = rp.round_polys.at[0, 0].add(fnp.ones((), rp.round_polys.dtype))
        bad_rounds = [replace(rp, round_polys=bad_polys)] + list(
            self.gkr_proof.round_proofs[1:]
        )
        bad_proof = replace(self.gkr_proof, round_proofs=bad_rounds)
        ok = self.fx.verifier.verify(
            self.fx.claim,
            replace(self.fx.proof, gkr=bad_proof),
            cheap_transcript(BF),
        ).ok
        self.assertFalse(bool(ok))

    def test_zerocheck_dual_writes_the_eval_seams(self) -> None:
        """The point is the dual's own sampled challenges (the prover's
        ``msgs.challenge`` order), the opened values the proof's
        oracle-checked ones — what the jagged-eval dual reads, surviving to
        the verifier's output."""
        _assert_bytes_equal(
            self.zc_reduced.point, self.zc_proof.msgs.challenge, "point"
        )
        _assert_bytes_equal(
            self.zc_reduced.opened_values["alpha"].main,
            self.zc_proof.opened_values["alpha"].main,
            "opened values",
        )

    def test_tampered_zerocheck_message_rejected_through_the_chain(self) -> None:
        """One representative stage-3 tamper rejecting end to end;
        the per-leg coverage is the stage's own test file."""
        bad_sum = self.zc_proof.claimed_sum + fnp.ones(
            (), self.zc_proof.claimed_sum.dtype
        )
        ok = self.fx.verifier.verify(
            self.fx.claim,
            replace(
                self.fx.proof, zerocheck=replace(self.zc_proof, claimed_sum=bad_sum)
            ),
            cheap_transcript(BF),
        ).ok
        self.assertFalse(bool(ok))

    def test_tampered_jagged_eval_message_rejected_through_the_chain(self) -> None:
        """One representative stage-4 tamper rejecting end to end;
        the per-leg coverage is the stage's own verifier test."""
        bad_eval = replace(
            self.je_proof.eval,
            dense_eval=self.je_proof.eval.dense_eval
            + fnp.ones((), self.je_proof.eval.dense_eval.dtype),
        )
        ok = self.fx.verifier.verify(
            self.fx.claim,
            replace(self.fx.proof, jagged=replace(self.je_proof, eval=bad_eval)),
            cheap_transcript(BF),
        ).ok
        self.assertFalse(bool(ok))

    def test_truncated_main_opening_rejected(self) -> None:
        """The opening-shape check (SP1's ``verify_opening_shape``): the
        statement owns the widths, so an opening that disagrees is a loud
        structural reject at the zerocheck dual — the verifier absorbs the
        proof's opened values, so a shape lie never desyncs Fiat-Shamir and
        only the statement check catches it."""
        ev = self.zc_proof.opened_values["alpha"]
        bad = replace(
            self.zc_proof,
            opened_values={"alpha": replace(ev, main=ev.main[:-1])},
        )
        with self.assertRaisesRegex(ValueError, "main claim per statement"):
            self.fx.verifier.zerocheck.verify(
                self.zc_claim, bad, self.zc_transcript
            )

    def test_unexpected_preprocessed_opening_rejected(self) -> None:
        """A statement with no preprocessed trace rejects a proof that opens
        one."""
        ev = self.zc_proof.opened_values["alpha"]
        bad = replace(
            self.zc_proof,
            opened_values={"alpha": replace(ev, preprocessed=ev.main[:1])},
        )
        with self.assertRaisesRegex(ValueError, "no preprocessed trace"):
            self.fx.verifier.zerocheck.verify(
                self.zc_claim, bad, self.zc_transcript
            )

    def test_missing_preprocessed_opening_rejected(self) -> None:
        """A statement whose chip carries a preprocessed trace rejects a
        proof that opens none (SP1's preprocessed-chips-appear-in-the-proof
        check). The seams are the post-prove ones, so the
        call exercises only the shape check, which raises before any
        cryptographic work."""
        stage = ZerocheckVerifier(
            self.fx.chips,
            chip_names=("alpha",),
            chip_shapes={
                "alpha": ChipShape(
                    TraceShape(CHIP_HEIGHT, CHIP_WIDTH),
                    prep=TraceShape(CHIP_HEIGHT, 1),
                )
            },
            max_log_row_count=MAX_LOG_ROW_COUNT,
        )
        with self.assertRaisesRegex(ValueError, "preprocessed claim per statement"):
            stage.verify(self.zc_claim, self.zc_proof, self.zc_transcript)

    def test_trace_commit_dual_derives_commitment_roots(self) -> None:
        """[prep (from the vk), main (from the message)] — the order of SP1's
        round_evaluation_claims, read skip-level by the stacked-open dual."""
        roots = self.roots
        _assert_bytes_equal(
            roots[0], self.fx.vk.preprocessed_commit, "prep root"
        )
        _assert_bytes_equal(roots[1], self.commitment, "main root")


if __name__ == "__main__":
    absltest.main()
