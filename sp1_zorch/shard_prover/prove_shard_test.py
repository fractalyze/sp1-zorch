# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`prove_shard_chain` vs the hand-threaded stage sequence.

The chain is wiring, not math — each stage function is gated by its own
tests — so this test demands the ``ProveChain`` composition is byte-identical
to calling commit, LogUp-GKR, and zerocheck by hand on the same sponge: same
messages, same carry products, same Fiat-Shamir stream afterwards. Any drift
in stage order, carry threading, or preamble encoding desynchronizes the two
streams and fails loudly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.testkit.transcript import cheap_transcript

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.commit.trace_commit import commit_region
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.prover import prove_logup_gkr
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.prove_shard import (
    ShardCarry,
    ShardZerocheckRound,
    preamble_chip_metadata,
    prove_shard_chain,
)
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from sp1_zorch.zerocheck.stage import prove_shard_zerocheck

BF = koalabear_mont
EF = koalabearx4_mont

_MAX_LOG_ROW_COUNT = 5
_NUM_ROW_VARIABLES = _MAX_LOG_ROW_COUNT - 1
_NUM_BETAS = 3
_LOG_BLOWUP = 1


class _WitnessChip:
    """Witness-shaped stub (a == 1 on real rows) whose second constraint folds
    in ``public_values[0]`` — the pv-binding seam the stage threads through."""

    def eval_constraints(self, trace, public_values):
        a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
        one = jnp.ones((), trace.dtype)
        pv0 = jnp.concatenate([public_values[:1], jnp.zeros((3,), BF)]).view(EF)[0]
        return jnp.stack([(a - one) * (c - one), (a - one) * (b + pv0)], axis=-1)


class _LookupChip:
    """Constraint-less chip (SP1's Byte / Program / Range shape)."""

    def eval_constraints(self, trace, public_values):
        return jnp.zeros((trace.shape[0], 0), dtype=trace.dtype)


def _interaction(mult_col: int, val_col: int, *, kind: int = 3) -> Interaction:
    return Interaction(
        values=(VirtualPairCol.single_main(val_col),),
        multiplicity=VirtualPairCol.single_main(mult_col),
        kind=kind,
        is_send=True,
    )


def _rand_bf(seed: int, shape) -> jnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=BF)


def _u32(a) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


def _assert_bytes_equal(got, want, label: str = "") -> None:
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


class ProveShardChainTest(absltest.TestCase):
    """One chain run vs one hand replay; each test byte-compares one stage."""

    @classmethod
    def setUpClass(cls):
        # alpha: 3 main cols x 6 real rows (col 0 == 1 keeps the interaction
        # multiplicities and the witness constraints live; GKR's even/odd row
        # split needs even heights), prep 2 cols x 3 rows (shorter than
        # num_real — exercises the prep zero-pad). lookup: 2 main cols x 4
        # rows, no prep, no constraints, receive interaction.
        alpha_main = jnp.concatenate(
            [jnp.ones((6, 1), dtype=BF), _rand_bf(1, (6, 2))], axis=1
        )
        alpha_prep = _rand_bf(2, (3, 2))
        lookup_main = jnp.concatenate(
            [jnp.ones((4, 1), dtype=BF), _rand_bf(3, (4, 1))], axis=1
        )

        names = ("alpha", "lookup")
        main_region = JaggedRegion.from_chips(
            [alpha_main, lookup_main],
            log_stacking_height=4,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=names,
        )
        prep_region = JaggedRegion.from_chips(
            [alpha_prep],
            log_stacking_height=4,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=("alpha",),
        )

        chips = {"alpha": _WitnessChip(), "lookup": _LookupChip()}
        gkr_chips = [
            GkrChip("alpha", (_interaction(0, 1),)),
            GkrChip("lookup", (_interaction(0, 1, kind=5),)),
        ]
        public_values = _rand_bf(7, (8,))

        perm = Poseidon2(koalabear16_params())
        smcs = SingleMatrixCommitmentScheme(
            Sponge(perm, SpongeParams(rate=8, out=8)),
            Compression(perm, CompressionParams(arity=2, chunk=8)),
        )
        vk = MachineVerifyingKey(
            preprocessed_commit=_rand_bf(9, (8,)),
            pc_start=_rand_bf(10, (3,)),
            cum_sum_x=_rand_bf(11, (7,)),
            cum_sum_y=_rand_bf(12, (7,)),
            enable_untrusted=0,
        )
        metadata = preamble_chip_metadata(names, [6, 4], dtype=BF)

        # Hand-threaded reference: the replay-style stage sequence.
        bound, _ = commit_region(main_region, smcs, log_blowup=_LOG_BLOWUP)
        t = vk.observe_into(cheap_transcript(BF))
        t = t.observe(public_values)
        t = t.observe(bound)
        t = t.observe(metadata)
        t, gkr_proof = prove_logup_gkr(
            gkr_chips,
            main_region,
            prep_region,
            t,
            num_betas=_NUM_BETAS,
            num_row_variables=_NUM_ROW_VARIABLES,
        )
        t, zc_proof = prove_shard_zerocheck(
            chips,
            main_region,
            prep_region,
            public_values,
            gkr_proof.eval_point,
            gkr_proof.chip_openings,
            t,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
        )
        cls.want_commitment = bound
        cls.want_gkr = gkr_proof
        cls.want_zc = zc_proof
        cls.want_transcript = t

        chain = prove_shard_chain(
            smcs=smcs,
            log_blowup=_LOG_BLOWUP,
            vk=vk,
            chip_metadata=metadata,
            gkr_chips=gkr_chips,
            chips=chips,
            num_betas=_NUM_BETAS,
            num_row_variables=_NUM_ROW_VARIABLES,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
        )
        cls.carry, cls.got_transcript, cls.msgs = chain(
            ShardCarry(main_region, prep_region, public_values),
            cheap_transcript(BF),
        )

    def test_chain_emits_one_message_per_stage(self) -> None:
        self.assertLen(self.msgs, 3)

    def test_commitment_message_matches(self) -> None:
        _assert_bytes_equal(self.msgs[0], self.want_commitment, "commitment")

    def test_gkr_message_matches(self) -> None:
        got = self.msgs[1]
        _assert_bytes_equal(got.eval_point, self.want_gkr.eval_point, "eval_point")
        _assert_bytes_equal(got.witness, self.want_gkr.witness, "witness")
        for name, want in self.want_gkr.chip_openings.items():
            _assert_bytes_equal(got.chip_openings[name].main, want.main, name)
            if want.preprocessed is not None:
                _assert_bytes_equal(
                    got.chip_openings[name].preprocessed, want.preprocessed, name
                )

    def test_zerocheck_message_matches(self) -> None:
        got, want = self.msgs[2], self.want_zc
        _assert_bytes_equal(got.batching_challenge, want.batching_challenge, "alpha")
        _assert_bytes_equal(
            got.gkr_opening_batch_challenge, want.gkr_opening_batch_challenge, "beta"
        )
        _assert_bytes_equal(got.lambda_, want.lambda_, "lambda")
        _assert_bytes_equal(got.zeta, want.zeta, "zeta")
        for i, (g, w) in enumerate(zip(got.finals, want.finals, strict=True)):
            _assert_bytes_equal(g, w, f"finals[{i}]")
        _assert_bytes_equal(got.msgs.round_poly, want.msgs.round_poly, "round_poly")
        _assert_bytes_equal(got.msgs.challenge, want.msgs.challenge, "challenge")

    def test_carry_threads_stage_outputs(self) -> None:
        _assert_bytes_equal(
            self.carry.gkr_eval_point, self.want_gkr.eval_point, "gkr_eval_point"
        )

    def test_transcript_streams_stay_in_sync(self) -> None:
        _, got = self.got_transcript.sample(1)
        _, want = self.want_transcript.sample(1)
        _assert_bytes_equal(got, want, "post-chain sample")

    def test_zerocheck_round_rejects_a_chain_without_gkr(self) -> None:
        round_ = ShardZerocheckRound(
            {"alpha": _WitnessChip()}, max_log_row_count=_MAX_LOG_ROW_COUNT
        )
        carry = ShardCarry(
            self.carry.main_region, self.carry.prep_region, self.carry.public_values
        )
        with self.assertRaisesRegex(ValueError, "LogUp-GKR"):
            round_(carry, cheap_transcript(BF))


if __name__ == "__main__":
    absltest.main()
