# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""``verify_shard_chain`` vs ``prove_shard_chain`` — the structural mirror.

The dual chain's guarantee is structural before it is cryptographic: one
verifier Round per prover stage, so a proof whose message list misaligns with
the schedule is rejected loudly by ``VerifyChain`` itself rather than
accepted on a desynced stream. These tests pin that alignment plus the live
stage duals — trace commit and LogUp-GKR — against a two-stage prover run:
same Fiat-Shamir stream, carry seams written for the downstream duals, and a
tampered stage-2 message rejected through the chain (the per-leg tamper
coverage is ``logup_gkr/verifier_test``).
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont

from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.testkit.transcript import cheap_transcript

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.prove_shard import (
    ShardCarry,
    preamble_chip_metadata,
    prove_shard_chain,
)
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from sp1_zorch.shard_prover.verify_shard import (
    ShardVerifierCarry,
    verify_shard_chain,
)

BF = koalabear_mont

_MAX_LOG_ROW_COUNT = 5
_NUM_ROW_VARIABLES = _MAX_LOG_ROW_COUNT - 1
_NUM_BETAS = 3
_CHIP_HEIGHT = 4


def _rand_bf(seed: int, shape) -> jnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=BF)


def _u32(a) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


def _assert_bytes_equal(got, want, label: str = "") -> None:
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


class VerifyShardChainTest(absltest.TestCase):
    """A two-stage prover run vs one full dual-chain run; the live duals
    (trace commit, LogUp-GKR) are checked cryptographically, the rest
    structurally (round count) — stages 3-4 are still placeholders, so the
    zerocheck Round's chips stay empty: its constructor just stores them."""

    @classmethod
    def setUpClass(cls):
        main_region = JaggedRegion.from_chips(
            [_rand_bf(1, (_CHIP_HEIGHT, 2))],
            log_stacking_height=4,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=("alpha",),
        )
        public_values = _rand_bf(30, (8,))
        vk = MachineVerifyingKey(
            preprocessed_commit=_rand_bf(31, (8,)),
            pc_start=_rand_bf(32, (3,)),
            cum_sum_x=_rand_bf(33, (7,)),
            cum_sum_y=_rand_bf(34, (7,)),
            enable_untrusted=0,
        )
        metadata = preamble_chip_metadata(("alpha",), [_CHIP_HEIGHT], dtype=BF)
        gkr_chips = (
            GkrChip(
                "alpha",
                (
                    Interaction(
                        values=(VirtualPairCol.single_main(1),),
                        multiplicity=VirtualPairCol.single_main(0),
                        kind=3,
                        is_send=True,
                    ),
                ),
            ),
        )

        perm = Poseidon2(koalabear16_params())
        smcs = SingleMatrixCommitmentScheme(
            Sponge(perm, SpongeParams(rate=8, out=8)),
            Compression(perm, CompressionParams(arity=2, chunk=8)),
        )
        cls.prove_chain = prove_shard_chain(
            smcs=smcs,
            log_blowup=1,
            vk=vk,
            chip_metadata=metadata,
            gkr_chips=gkr_chips,
            chips={},
            num_betas=_NUM_BETAS,
            num_row_variables=_NUM_ROW_VARIABLES,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            open_num_queries=2,
        )
        carry = ShardCarry(main_region, None, public_values)
        carry, transcript, cls.commitment = cls.prove_chain.rounds[0](
            carry, cheap_transcript(BF)
        )
        _, cls.prover_transcript, cls.gkr_proof = cls.prove_chain.rounds[1](
            carry, transcript
        )

        cls.dual = verify_shard_chain(
            vk=vk,
            chip_metadata=metadata,
            gkr_chips=gkr_chips,
            chip_names=("alpha",),
            chip_heights={"alpha": _CHIP_HEIGHT},
            num_betas=_NUM_BETAS,
            num_row_variables=_NUM_ROW_VARIABLES,
        )
        cls.dual_carry, cls.dual_transcript, cls.dual_ok = cls.dual(
            ShardVerifierCarry(public_values),
            [cls.commitment, cls.gkr_proof, None, None],
            cheap_transcript(BF),
        )
        cls.vk = vk
        cls.public_values = public_values

    def test_one_verifier_round_per_prover_stage(self) -> None:
        self.assertLen(self.dual.rounds, len(self.prove_chain.rounds))

    def test_round_count_mismatch_fails_loud(self) -> None:
        """A message list one short of the schedule is a structural reject —
        ``VerifyChain``'s own check, before any stage dual runs."""
        with self.assertRaisesRegex(ValueError, "one message per round"):
            self.dual(
                ShardVerifierCarry(self.public_values),
                [self.commitment, self.gkr_proof, None],
                cheap_transcript(BF),
            )

    def test_live_duals_match_the_prover_stream(self) -> None:
        """The chain-output transcript byte-matches the prover's post-stage-2
        one, so the two Fiat-Shamir streams enter stage 3 in sync — and the
        placeholder stages provably leave the stream untouched."""
        self.assertTrue(bool(self.dual_ok))
        _, got = self.dual_transcript.sample(1)
        _, want = self.prover_transcript.sample(1)
        _assert_bytes_equal(got, want, "post-stage-2 sample")

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
        _, _, ok = self.dual(
            ShardVerifierCarry(self.public_values),
            [self.commitment, bad_proof, None, None],
            cheap_transcript(BF),
        )
        self.assertFalse(bool(ok))

    def test_trace_commit_dual_writes_commitment_roots(self) -> None:
        """[prep (from the vk), main (from the message)] — the order of SP1's
        round_evaluation_claims, read skip-level by the stacked-open dual; the
        write survives the placeholder stages to the chain output."""
        roots = self.dual_carry.commitment_roots
        _assert_bytes_equal(roots[0], self.vk.preprocessed_commit, "prep root")
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
