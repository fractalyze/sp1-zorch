# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""``verify_shard_chain`` vs ``prove_shard_chain`` — the structural mirror.

The dual chain's guarantee is structural before it is cryptographic: one
verifier Round per prover stage, so a proof whose message list misaligns with
the schedule is rejected loudly by ``VerifyChain`` itself rather than
accepted on a desynced stream. These tests pin that alignment plus all four
stage duals against a full prover run (the shared ``chain_testkit``
fixture): same Fiat-Shamir stream, carry seams written for the downstream
duals, a tampered stage message rejected through the chain (the per-leg
tamper coverage is each stage's own verifier test), and the zerocheck
dual's opening-shape statement checks.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont

from zorch.testkit.transcript import cheap_transcript

from sp1_zorch.shard_prover.types import ChipShape, TraceShape
from sp1_zorch.shard_prover.verify_shard import (
    ShardVerifierCarry,
    ShardZerocheckVerifierRound,
)

from sp1_zorch.shard_prover.chain_testkit import (
    CHIP_HEIGHT,
    CHIP_WIDTH,
    MAX_LOG_ROW_COUNT,
    small_shard_chain_fixture,
)

BF = koalabear_mont


def _u32(a) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


def _assert_bytes_equal(got, want, label: str = "") -> None:
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


class VerifyShardChainTest(absltest.TestCase):
    """A full four-stage prover run vs one full dual-chain run, every stage
    dual checked cryptographically through the chain."""

    @classmethod
    def setUpClass(cls):
        fx = small_shard_chain_fixture()
        cls.fx = fx
        cls.commitment, cls.gkr_proof, cls.zc_proof, cls.je_proof = fx.messages
        cls.dual_carry, cls.dual_transcript, cls.dual_ok = fx.dual(
            ShardVerifierCarry(fx.public_values),
            fx.messages,
            cheap_transcript(BF),
        )

    def test_one_verifier_round_per_prover_stage(self) -> None:
        self.assertLen(self.fx.dual.rounds, len(self.fx.prove_chain.rounds))

    def test_round_count_mismatch_fails_loud(self) -> None:
        """A message list one short of the schedule is a structural reject —
        ``VerifyChain``'s own check, before any stage dual runs."""
        with self.assertRaisesRegex(ValueError, "one message per round"):
            self.fx.dual(
                ShardVerifierCarry(self.fx.public_values),
                self.fx.messages[:3],
                cheap_transcript(BF),
            )

    def test_live_duals_match_the_prover_stream(self) -> None:
        """The chain accepts and its output transcript byte-matches the
        prover's post-stage-4 one, so the two Fiat-Shamir streams agree
        through every stage, glue included."""
        self.assertTrue(bool(self.dual_ok))
        _, got = self.dual_transcript.sample(1)
        _, want = self.fx.prover_transcript.sample(1)
        _assert_bytes_equal(got, want, "post-stage-4 sample")

    def test_gkr_dual_writes_the_zerocheck_seams(self) -> None:
        """The point is the dual's own derivation (pinned against the wire
        copy inside the stage), the openings the proof's leaf-checked values
        — what the zerocheck dual reads, surviving to the chain output."""
        _assert_bytes_equal(
            self.dual_carry.gkr_eval_point, self.gkr_proof.eval_point, "point"
        )
        _assert_bytes_equal(
            self.dual_carry.gkr_chip_openings["alpha"].main,
            self.gkr_proof.chip_openings["alpha"].main,
            "openings",
        )

    def test_tampered_gkr_message_rejected_through_the_chain(self) -> None:
        """One representative stage-2 tamper rejecting at the chain level;
        the per-leg coverage is the stage's own test file."""
        rp = self.gkr_proof.round_proofs[0]
        bad_polys = rp.round_polys.at[0, 0].add(jnp.ones((), rp.round_polys.dtype))
        bad_rounds = [replace(rp, round_polys=bad_polys)] + list(
            self.gkr_proof.round_proofs[1:]
        )
        bad_proof = replace(self.gkr_proof, round_proofs=bad_rounds)
        _, _, ok = self.fx.dual(
            ShardVerifierCarry(self.fx.public_values),
            [self.commitment, bad_proof, self.zc_proof, self.je_proof],
            cheap_transcript(BF),
        )
        self.assertFalse(bool(ok))

    def test_zerocheck_dual_writes_the_eval_seams(self) -> None:
        """The point is the dual's own sampled challenges (the prover's
        ``msgs.challenge`` order), the opened values the proof's
        oracle-checked ones — what the jagged-eval dual reads, surviving to
        the chain output."""
        _assert_bytes_equal(
            self.dual_carry.zc_sumcheck_point, self.zc_proof.msgs.challenge, "point"
        )
        _assert_bytes_equal(
            self.dual_carry.zc_opened_values["alpha"].main,
            self.zc_proof.opened_values["alpha"].main,
            "opened values",
        )

    def test_tampered_zerocheck_message_rejected_through_the_chain(self) -> None:
        """One representative stage-3 tamper rejecting at the chain level;
        the per-leg coverage is the stage's own test file."""
        bad_sum = self.zc_proof.claimed_sum + jnp.ones(
            (), self.zc_proof.claimed_sum.dtype
        )
        _, _, ok = self.fx.dual(
            ShardVerifierCarry(self.fx.public_values),
            [
                self.commitment,
                self.gkr_proof,
                replace(self.zc_proof, claimed_sum=bad_sum),
                self.je_proof,
            ],
            cheap_transcript(BF),
        )
        self.assertFalse(bool(ok))

    def test_tampered_jagged_eval_message_rejected_through_the_chain(self) -> None:
        """One representative stage-4 tamper rejecting at the chain level;
        the per-leg coverage is the stage's own verifier test."""
        bad_eval = replace(
            self.je_proof.eval,
            dense_eval=self.je_proof.eval.dense_eval
            + jnp.ones((), self.je_proof.eval.dense_eval.dtype),
        )
        _, _, ok = self.fx.dual(
            ShardVerifierCarry(self.fx.public_values),
            [
                self.commitment,
                self.gkr_proof,
                self.zc_proof,
                replace(self.je_proof, eval=bad_eval),
            ],
            cheap_transcript(BF),
        )
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
            self.fx.dual.rounds[2](self.dual_carry, bad, self.dual_transcript)

    def test_unexpected_preprocessed_opening_rejected(self) -> None:
        """A statement with no preprocessed trace rejects a proof that opens
        one."""
        ev = self.zc_proof.opened_values["alpha"]
        bad = replace(
            self.zc_proof,
            opened_values={"alpha": replace(ev, preprocessed=ev.main[:1])},
        )
        with self.assertRaisesRegex(ValueError, "no preprocessed trace"):
            self.fx.dual.rounds[2](self.dual_carry, bad, self.dual_transcript)

    def test_missing_preprocessed_opening_rejected(self) -> None:
        """A statement whose chip carries a preprocessed trace rejects a
        proof that opens none (SP1's preprocessed-chips-appear-in-the-proof
        check). The carry is the post-chain one (every seam written), so the
        call exercises only the shape check, which raises before any
        cryptographic work."""
        round_ = ShardZerocheckVerifierRound(
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
            round_(self.dual_carry, self.zc_proof, self.dual_transcript)

    def test_trace_commit_dual_writes_commitment_roots(self) -> None:
        """[prep (from the vk), main (from the message)] — the order of SP1's
        round_evaluation_claims, read skip-level by the stacked-open dual."""
        roots = self.dual_carry.commitment_roots
        _assert_bytes_equal(
            roots[0], self.fx.vk.preprocessed_commit, "prep root"
        )
        _assert_bytes_equal(roots[1], self.commitment, "main root")

    def test_verifier_carry_flattens_to_array_leaves(self) -> None:
        """``ShardVerifierCarry`` is a pytree like the prover's carry: the
        public values and written roots are its array leaves, so the dual
        chain can cross a ``@jit`` boundary as one argument."""
        leaves = jax.tree_util.tree_leaves(self.dual_carry)
        self.assertNotEmpty(leaves)
        for leaf in leaves:
            self.assertIsInstance(leaf, jax.Array)


if __name__ == "__main__":
    absltest.main()
