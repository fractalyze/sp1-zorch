# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-leg tamper coverage for the stage-4 dual.

One honest four-stage prover run (the shared ``chain_testkit`` fixture)
feeds one ``ShardJaggedEvalVerifierRound`` instance; each test tampers one
leg of the ``ShardJaggedEvalProof`` and asserts the dual rejects it — the
eval half's wire pins and oracle checks, the structure rebind, and the
stacked open's transcript-bound and query-phase legs. The honest run
accepting (and byte-matching the prover's stream) is the chain-level test
(``shard_prover/verify_shard_test``)."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.testkit.transcript import cheap_transcript

from sp1_zorch.jagged.verifier import stacked_basefold_verify
from sp1_zorch.shard_prover.prove_shard import ShardJaggedEvalProof
from sp1_zorch.shard_prover.verify_shard import ShardVerifierCarry

from sp1_zorch.shard_prover.chain_testkit import small_shard_chain_fixture

BF = koalabear_mont


class ShardJaggedEvalVerifierRoundTest(absltest.TestCase):
    """The shared four-stage fixture, with the first three duals run once to
    position the carry and transcript at stage 4; each test replays only the
    stage-4 dual on a tampered message."""

    @classmethod
    def setUpClass(cls):
        fx = small_shard_chain_fixture()
        cls.smcs = fx.smcs
        cls.je_proof = fx.messages[3]
        carry = ShardVerifierCarry(fx.public_values)
        transcript = cheap_transcript(BF)
        for dual_round, msg in zip(fx.dual.rounds[:3], fx.messages[:3]):
            carry, transcript, _ = dual_round(carry, msg, transcript)
        cls.carry, cls.transcript = carry, transcript
        cls.round = fx.dual.rounds[3]

    def _ok(self, proof: ShardJaggedEvalProof) -> bool:
        _, _, ok = self.round(self.carry, proof, self.transcript)
        return bool(ok)

    def _bump(self, value: jnp.ndarray, index=None) -> jnp.ndarray:
        one = jnp.ones((), value.dtype)
        return value + one if index is None else value.at[index].add(one)

    def test_honest_proof_accepts(self) -> None:
        self.assertTrue(self._ok(self.je_proof))

    def test_tampered_outer_claim_rejected(self) -> None:
        """The wire claim pin — the dual's own derivation from the carry's
        opened values disagrees with a drifted copy."""
        ev = self.je_proof.eval
        bad = replace(
            ev, outer_sumcheck_claim=self._bump(ev.outer_sumcheck_claim)
        )
        self.assertFalse(self._ok(replace(self.je_proof, eval=bad)))

    def test_tampered_outer_poly_rejected(self) -> None:
        ev = self.je_proof.eval
        bad = replace(
            ev, outer_sumcheck_polys=self._bump(ev.outer_sumcheck_polys, (0, 0))
        )
        self.assertFalse(self._ok(replace(self.je_proof, eval=bad)))

    def test_tampered_inner_claimed_sum_rejected(self) -> None:
        """J̃'s claimed value feeds the product check and the leaf check —
        a consistent forgery of one breaks the other."""
        ev = self.je_proof.eval
        bad = replace(ev, inner_claimed_sum=self._bump(ev.inner_claimed_sum))
        self.assertFalse(self._ok(replace(self.je_proof, eval=bad)))

    def test_tampered_inner_poly_rejected(self) -> None:
        ev = self.je_proof.eval
        bad = replace(
            ev, inner_sumcheck_polys=self._bump(ev.inner_sumcheck_polys, (0, 0))
        )
        self.assertFalse(self._ok(replace(self.je_proof, eval=bad)))

    def test_tampered_dense_eval_rejected(self) -> None:
        ev = self.je_proof.eval
        bad = replace(ev, dense_eval=self._bump(ev.dense_eval))
        self.assertFalse(self._ok(replace(self.je_proof, eval=bad)))

    def test_tampered_component_commitment_rejected(self) -> None:
        """The structure rebind: a proof commitment that does not rebind to
        the preamble-observed root is rejected even if internally
        consistent."""
        op = self.je_proof.open
        bad = replace(
            op,
            component_commitments=[self._bump(op.component_commitments[0], 0)],
        )
        self.assertFalse(self._ok(replace(self.je_proof, open=bad)))

    def test_tampered_batch_eval_rejected(self) -> None:
        op = self.je_proof.open
        bad = replace(op, batch_evals=[self._bump(op.batch_evals[0], 0)])
        self.assertFalse(self._ok(replace(self.je_proof, open=bad)))

    def test_tampered_fold_root_rejected(self) -> None:
        op = self.je_proof.open
        bad = replace(op, fri_commitments=self._bump(op.fri_commitments, (0, 0)))
        self.assertFalse(self._ok(replace(self.je_proof, open=bad)))

    def test_tampered_final_poly_rejected(self) -> None:
        op = self.je_proof.open
        bad = replace(op, final_poly=self._bump(op.final_poly))
        self.assertFalse(self._ok(replace(self.je_proof, open=bad)))

    def test_zero_stacking_variables_rejected(self) -> None:
        """``log_stacking_height == 0`` (no fold layers, no query openings)
        is outside the SP1 wire — rejected up front, not an index error."""
        ev = self.je_proof.eval
        code = BitReversedReedSolomon(message_len=1, blowup=2, dtype=BF)
        with self.assertRaisesRegex(ValueError, "stacking variable"):
            stacked_basefold_verify(
                self.smcs,
                code,
                [1],
                jnp.zeros((1,), ev.dense_eval.dtype),
                ev.dense_eval,
                0,
                self.je_proof.open,
                cheap_transcript(BF),
                num_queries=1,
                pow_bits=0,
            )

    def test_tampered_query_row_rejected(self) -> None:
        """A forged opened leaf fails its Merkle rebuild."""
        op = self.je_proof.open
        rows, paths = op.query_openings[0]
        bad_openings = [(self._bump(rows, (0, 0)), paths)] + list(
            op.query_openings[1:]
        )
        bad = replace(op, query_openings=bad_openings)
        self.assertFalse(self._ok(replace(self.je_proof, open=bad)))


if __name__ == "__main__":
    absltest.main()
