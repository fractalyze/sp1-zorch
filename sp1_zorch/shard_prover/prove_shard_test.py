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

import dataclasses
from dataclasses import replace

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
    PreambleRound,
    ShardCarry,
    ShardZerocheckRound,
    preamble_chip_metadata,
    prove_shard_chain,
)
from sp1_zorch.shard_prover.types import MachineVerifyingKey
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

_MAX_LOG_ROW_COUNT = 5
_NUM_ROW_VARIABLES = _MAX_LOG_ROW_COUNT - 1
_NUM_BETAS = 3
_LOG_BLOWUP = 1
_OPEN_NUM_QUERIES = 2


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


def _flatten_arrays(x) -> list:
    """Every array leaf under ``x`` in deterministic order, recursing through
    lists/tuples, dicts, and dataclasses — so a proof message flattens without
    each proof type being a registered pytree. Static scalars (counts, flags)
    are identical by construction and contribute nothing."""
    if isinstance(x, (jax.Array, np.ndarray)):
        return [x]
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [a for e in x for a in _flatten_arrays(e)]
    if isinstance(x, dict):
        # Sort keys so the leaf order is independent of dict construction order
        # (the two chains compared here build their openings dicts identically,
        # but a sorted walk makes the helper robust regardless).
        return [a for k in sorted(x) for a in _flatten_arrays(x[k])]
    if dataclasses.is_dataclass(x):
        return [
            a for f in dataclasses.fields(x) for a in _flatten_arrays(getattr(x, f.name))
        ]
    return []


def _assert_proof_byte_equal(got, want, label: str) -> None:
    gs, ws = _flatten_arrays(got), _flatten_arrays(want)
    assert len(gs) == len(ws), f"{label}: {len(gs)} vs {len(ws)} array leaves"
    for i, (g, w) in enumerate(zip(gs, ws, strict=True)):
        _assert_bytes_equal(g, w, f"{label}[{i}]")


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
            open_num_queries=_OPEN_NUM_QUERIES,
        )
        # The full four-stage chain runs eagerly on CPU: the jagged-eval
        # stage's base->extension embeds keep its EF->PF converts off the CPU
        # path (jax#168), so the open executes. The hand replay stops at
        # zerocheck, so snapshot the transcript there for the in-sync check;
        # the open's SP1 byte-match is the GPU verify_prove_shard harness's job.
        carry = ShardCarry(main_region, prep_region, public_values)
        transcript = cheap_transcript(BF)
        msgs = []
        for i, stage in enumerate(chain.rounds):
            carry, transcript, msg = stage(carry, transcript)
            msgs.append(msg)
            if i == 2:  # after zerocheck, where the hand replay stops
                cls.got_transcript = transcript
        cls.carry, cls.msgs, cls.jagged = carry, msgs, msgs[-1]
        cls.chain = chain
        cls.main_region = main_region
        cls.prep_region = prep_region
        cls.public_values = public_values
        # Retained so the offload-parity test can rebuild the same chain with
        # ``offload_commit_rounds=True``.
        cls.smcs = smcs
        cls.vk = vk
        cls.gkr_chips = gkr_chips
        cls.chips = chips
        cls.metadata = metadata

    def test_chain_emits_one_message_per_stage(self) -> None:
        self.assertLen(self.msgs, 4)  # commit, LogUp-GKR, zerocheck, jagged eval

    def test_jagged_eval_stage_opens_each_committed_round(self) -> None:
        """The fourth stage runs the outer/inner jagged sumcheck to D(z_final)
        and the stacked BaseFold open over the [prep, main] committed rounds.
        The open exposes one batch-eval set and one component opening per
        committed round; SP1 byte-correctness is the GPU verify_prove_shard
        harness's job, so here we pin the executed shape."""
        msg = self.jagged
        self.assertEqual(msg.eval.dense_eval.shape, ())  # scalar D(z_final)
        n_rounds = len(self.carry.commit_rounds)
        self.assertLen(msg.open.batch_evals, n_rounds)
        self.assertLen(msg.open.component_openings, n_rounds)
        self.assertNotEmpty(msg.open.query_openings)  # one per FRI fold layer

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

    def test_chain_lowers_under_single_jit(self) -> None:
        """The whole chain traces as one ``@jit`` region: no stage forces a
        host sync that would split it. The carry is built inside the traced
        function (so this needs no pytree registration of ``ShardCarry``);
        backend compile stays GPU's job — poseidon2 has no CPU fusion emitter
        and CPU jit miscompiles field dots (fractalyze/jax#168) — so the smoke
        stops at StableHLO lowering."""

        def run(dense, public_values, transcript):
            carry = ShardCarry(
                replace(self.main_region, dense=dense), self.prep_region, public_values
            )
            _, out_transcript, _ = self.chain(carry, transcript)
            return out_transcript

        lowered = jax.jit(run).lower(
            self.main_region.dense, self.public_values, cheap_transcript(BF)
        )
        self.assertIn("func", lowered.as_text())

    def test_jaggedregion_is_a_pytree_with_only_dense_as_leaf(self) -> None:
        """JaggedRegion registers ``dense`` as its sole array leaf; the layout
        counts are static aux data, so a region crosses a ``@jit`` boundary
        without leaking the count tuples into the traced graph."""
        leaves, treedef = jax.tree_util.tree_flatten(self.main_region)
        self.assertLen(leaves, 1)
        self.assertIs(leaves[0], self.main_region.dense)
        # The layout counts ride in the treedef as static aux, not as leaves.
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertIs(rebuilt.dense, self.main_region.dense)
        self.assertEqual(rebuilt.row_counts, self.main_region.row_counts)

    def test_shardcarry_flattens_to_its_array_buffers(self) -> None:
        """ShardCarry is a pytree: its leaves are exactly the region dense
        buffers and the public values — the ``None`` stage-output fields
        contribute no leaves — so the whole carry crosses a ``@jit`` boundary
        as one argument."""
        carry = ShardCarry(self.main_region, self.prep_region, self.public_values)
        leaves = jax.tree_util.tree_leaves(carry)
        self.assertEqual(
            [id(x) for x in leaves],
            [
                id(self.main_region.dense),
                id(self.prep_region.dense),
                id(self.public_values),
            ],
        )

    def test_carry_crosses_jit_as_a_donated_argument(self) -> None:
        """With ShardCarry a pytree, the chain runs under a single ``@jit``
        that takes the carry as a *donated* argument (vs the closed-over carry
        in ``test_chain_lowers_under_single_jit``), letting XLA reuse its input
        buffers. Stops at StableHLO lowering: CPU can't execute field dots
        (fractalyze/jax#168), GPU owns backend compile."""

        def run(carry, transcript):
            _, out_transcript, _ = self.chain(carry, transcript)
            return out_transcript

        carry = ShardCarry(self.main_region, self.prep_region, self.public_values)
        lowered = jax.jit(run, donate_argnums=0).lower(carry, cheap_transcript(BF))
        self.assertIn("func", lowered.as_text())

    def test_populated_carry_flattens_to_array_leaves_only(self) -> None:
        """The carry threaded out of the chain holds the GKR stage outputs —
        the evaluation point and the per-chip ChipEvaluation openings — yet
        still flattens to array leaves only. Every carry-component type
        (region, opening) is a pytree, so a mid-chain populated carry can
        cross a ``@jit`` boundary too, not just the initial one."""
        self.assertIsNotNone(self.carry.gkr_chip_openings)
        leaves = jax.tree_util.tree_leaves(self.carry)
        self.assertNotEmpty(leaves)
        for leaf in leaves:
            self.assertIsInstance(leaf, jax.Array)

    def test_trace_commit_round_carries_stacked_open_witness(self) -> None:
        """TraceCommitRound retains each region's stacked witness — the
        ``[S, K]`` message matrix and the committed ``[S*blowup, K]``
        bit-reversed codeword — on the carry as ``[prep, main]``, so the
        jagged-eval open stage reproves them without recommitting."""
        rounds = self.carry.commit_rounds
        self.assertIsNotNone(rounds)
        self.assertLen(rounds, 2)  # prep, then main
        S = 1 << self.main_region.log_stacking_height
        blowup = 1 << _LOG_BLOWUP
        for rd in rounds:
            self.assertEqual(rd.mle.shape[0], S)
            self.assertEqual(rd.codeword.shape[0], S * blowup)
            self.assertEqual(rd.mle.shape[1], rd.codeword.shape[1])

    def test_zerocheck_round_carries_the_eval_point(self) -> None:
        """ShardZerocheckRound threads its sumcheck point onto the carry as the
        jagged-eval open's z_row (the accumulated per-round challenges, not the
        GKR zeta), so the eval stage opens the trace at the right point."""
        _assert_bytes_equal(
            self.carry.zc_sumcheck_point, self.want_zc.msgs.challenge, "z_row"
        )

    def test_carry_threads_stage_outputs(self) -> None:
        _assert_bytes_equal(
            self.carry.gkr_eval_point, self.want_gkr.eval_point, "gkr_eval_point"
        )

    def test_zerocheck_round_carries_opened_values(self) -> None:
        """ShardZerocheckRound threads the stage's per-chip opened values onto
        the carry — the jagged-eval stage's per-column claims (SP1's
        round_evaluation_claims, the trace evaluations at the zerocheck point,
        NOT the GKR-point openings) and the wire's ShardOpenedValues read
        them there."""
        got = self.carry.zc_opened_values
        self.assertIsNotNone(got)
        for name, want in self.want_zc.opened_values.items():
            _assert_bytes_equal(got[name].main, want.main, f"{name} main")
            if want.preprocessed is None:
                self.assertIsNone(got[name].preprocessed, name)
            else:
                _assert_bytes_equal(
                    got[name].preprocessed, want.preprocessed, f"{name} prep"
                )

    def test_transcript_streams_stay_in_sync(self) -> None:
        _, got = self.got_transcript.sample(1)
        _, want = self.want_transcript.sample(1)
        _assert_bytes_equal(got, want, "post-chain sample")

    def test_offload_commit_rounds_is_byte_identical(self) -> None:
        """``offload_commit_rounds=True`` parks the committed witness on host
        through the GKR + zerocheck stages and reloads it for the open -- the
        fractalyze/sp1-zorch#55/#124 GPU-OOM lever. It is a pure device<->host
        round-trip, so every stage message -- including the jagged-eval open,
        the one stage that reads ``commit_rounds`` -- must be byte-identical to
        the in-device chain, and the parked witness must be host-resident
        (numpy) yet bit-equal to the device one (lever active AND lossless)."""
        chain = prove_shard_chain(
            smcs=self.smcs,
            log_blowup=_LOG_BLOWUP,
            vk=self.vk,
            chip_metadata=self.metadata,
            gkr_chips=self.gkr_chips,
            chips=self.chips,
            num_betas=_NUM_BETAS,
            num_row_variables=_NUM_ROW_VARIABLES,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            open_num_queries=_OPEN_NUM_QUERIES,
            offload_commit_rounds=True,
        )
        carry = ShardCarry(self.main_region, self.prep_region, self.public_values)
        transcript = cheap_transcript(BF)
        got_msgs = []
        for stage in chain.rounds:
            carry, transcript, msg = stage(carry, transcript)
            got_msgs.append(msg)

        for i, (got, want) in enumerate(zip(got_msgs, self.msgs, strict=True)):
            _assert_proof_byte_equal(got, want, f"stage{i} message")

        # The parked witness left the chain on host, bit-equal to the device one.
        self.assertIsNotNone(carry.commit_rounds)
        for parked, resident in zip(
            carry.commit_rounds, self.carry.commit_rounds, strict=True
        ):
            # Every retained array (mle, codeword, each digest layer) is parked
            # on host (numpy, not device); a partial offload that left any on
            # device would pin buffers through GKR + zerocheck and fail here.
            leaves = _flatten_arrays(parked)
            self.assertNotEmpty(leaves)
            for leaf in leaves:
                self.assertIsInstance(leaf, np.ndarray)
            _assert_proof_byte_equal(parked, resident, "parked witness")

    def test_zerocheck_round_rejects_a_chain_without_gkr(self) -> None:
        round_ = ShardZerocheckRound(
            {"alpha": _WitnessChip()}, max_log_row_count=_MAX_LOG_ROW_COUNT
        )
        carry = ShardCarry(
            self.carry.main_region, self.carry.prep_region, self.carry.public_values
        )
        with self.assertRaisesRegex(ValueError, "LogUp-GKR"):
            round_(carry, cheap_transcript(BF))


class PreambleRoundTest(absltest.TestCase):
    """Pins ``PreambleRound`` against a raw transcript walk — the one
    deliberate second writing of the preamble schedule, so an accidental
    reorder in the Round fails here instead of two tools later in a
    byte-match hunt."""

    def test_matches_raw_walk(self) -> None:
        vk = MachineVerifyingKey(
            preprocessed_commit=_rand_bf(20, (8,)),
            pc_start=_rand_bf(21, (3,)),
            cum_sum_x=_rand_bf(22, (7,)),
            cum_sum_y=_rand_bf(23, (7,)),
            enable_untrusted=0,
        )
        public_values = _rand_bf(24, (8,))
        commitment = _rand_bf(25, (8,))
        metadata = preamble_chip_metadata(("ab", "c"), (6, 4), dtype=BF)

        sentinel = object()
        carry, got_t, msg = PreambleRound(
            vk=vk,
            public_values=public_values,
            commitment=commitment,
            chip_metadata=metadata,
        )(sentinel, cheap_transcript(BF))

        self.assertIs(carry, sentinel)  # carry-agnostic pass-through
        _assert_bytes_equal(msg, commitment, "message")

        want_t = vk.observe_into(cheap_transcript(BF))
        want_t = want_t.observe(public_values)
        want_t = want_t.observe(commitment)
        want_t = want_t.observe(metadata)
        _, got = got_t.sample(1)
        _, want = want_t.sample(1)
        _assert_bytes_equal(got, want, "post-preamble sample")


class PreambleChipMetadataTest(absltest.TestCase):
    """Pins the chip-metadata layout directly: the chain test only exercises it
    on both TraceCommitRound and replay.py's preamble at once, where a layout
    bug would cancel out."""

    def test_flat_layout(self) -> None:
        got = preamble_chip_metadata(("ab", "c"), (6, 4), dtype=BF)
        want = jnp.array([2, 6, 2, ord("a"), ord("b"), 4, 1, ord("c")], dtype=BF)
        _assert_bytes_equal(got, want, "chip metadata")


if __name__ == "__main__":
    absltest.main()
