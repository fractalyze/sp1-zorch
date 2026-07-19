# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`prove_shard_chain` vs the hand-threaded stage sequence.

The chain is wiring, not math — each stage function is gated by its own
tests — so this test demands the ``ProveChain`` composition is byte-identical
to calling commit, LogUp-GKR, and zerocheck by hand on the same sponge: same
messages, same bridge products, same Fiat-Shamir stream afterwards. Any drift
in stage order, bridge threading, or preamble encoding desynchronizes the two
streams and fails loudly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.testkit.transcript import cheap_transcript

from zorch.pcs.jagged.region import JaggedRegion
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.pcs.jagged.commit import commit_region
from sp1_zorch.logup_gkr.circuit import (
    GkrCapClass,
    GkrChip,
    _chip_first_layer_capped,
)
from sp1_zorch.logup_gkr.prover import (
    ChipEvaluation,
    open_traces_capped,
    prove_logup_gkr,
)
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.prove_shard import (
    JaggedPcsStage,
    LogupGkrStage,
    PreambleStage,
    ShardBridge,
    ZerocheckStage,
    _jagged_eval_jit,
    _region_mle,
    preamble_chip_metadata,
    prove_shard_chain,
)
from sp1_zorch.shard_prover.replay import JitPermutation
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from sp1_zorch.zerocheck.jagged import TotalCapClass
from sp1_zorch.zerocheck.prover import prove_shard_zerocheck
from zorch.utils.bits import log2_ceil_usize


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
        one = fnp.ones((), trace.dtype)
        pv0 = fnp.concatenate([public_values[:1], fnp.zeros((3,), BF)]).view(EF)[0]
        return fnp.stack([(a - one) * (c - one), (a - one) * (b + pv0)], axis=-1)


class _LookupChip:
    """Constraint-less chip (SP1's Byte / Program / Range shape)."""

    def eval_constraints(self, trace, public_values):
        return fnp.zeros((trace.shape[0], 0), dtype=trace.dtype)


def _interaction(mult_col: int, val_col: int, *, kind: int = 3) -> Interaction:
    return Interaction(
        values=(VirtualPairCol.single_main(val_col),),
        multiplicity=VirtualPairCol.single_main(mult_col),
        kind=kind,
        is_send=True,
    )


def _rand_bf(seed: int, shape) -> fnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return fnp.array(ints, dtype=BF)


def _u32(a) -> np.ndarray:
    return np.asarray(frx.lax.bitcast_convert_type(a, fnp.uint32)).reshape(-1)


def _assert_bytes_equal(got, want, label: str = "") -> None:
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


def _flatten_arrays(x) -> list:
    """Every array leaf under ``x`` in deterministic order, recursing through
    lists/tuples, dicts, and dataclasses — so a proof message flattens without
    each proof type being a registered pytree. Static scalars (counts, flags)
    are identical by construction and contribute nothing."""
    if isinstance(x, (frx.Array, np.ndarray)):
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
            a
            for f in dataclasses.fields(x)
            for a in _flatten_arrays(getattr(x, f.name))
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
        alpha_main = fnp.concatenate(
            [fnp.ones((6, 1), dtype=BF), _rand_bf(1, (6, 2))], axis=1
        )
        alpha_prep = _rand_bf(2, (3, 2))
        lookup_main = fnp.concatenate(
            [fnp.ones((4, 1), dtype=BF), _rand_bf(3, (4, 1))], axis=1
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

        # This machine's constraint circuit reads public_values. That is now
        # jit-legal — the statement rides `zorch.constraint_eval`'s aux_operands
        # operand, not a closure the composite would reject (the lowering smokes
        # below exercise it). This executed reference stays eager because CPU
        # cannot execute the jitted field dots (fractalyze/frx#168); the jitted
        # prove is GPU's, byte-checked there.
        # The zerocheck stage is always the traced total-cap route (jitted,
        # CPU-runnable); with no class pinned it derives the shard's own
        # a-priori-tight class and stays byte-identical to an eager exact
        # prove.
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
            jit=False,
        )
        # A jit=True twin for the lowering smoke (sp1-zorch#119): same wiring,
        # with the commit/zerocheck/jagged-eval stages jitted (LogUp-GKR stays eager).
        cls.jit_chain = prove_shard_chain(
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
            jit=True,
        )
        # The full four-stage chain runs eagerly on CPU: the jagged-eval
        # stage's base->extension embeds keep its EF->PF converts off the CPU
        # path (frx#168), so the open executes. The hand replay stops at
        # zerocheck, so snapshot the transcript there for the in-sync check;
        # the open's SP1 byte-match is the GPU verify_prove_shard harness's job.
        bridge = ShardBridge(main_region, prep_region, public_values)
        transcript = cheap_transcript(BF)
        msgs = []
        for i, stage in enumerate(chain.rounds):
            bridge, transcript, msg = stage(bridge, transcript)
            msgs.append(msg)
            if i == 2:  # after zerocheck, where the hand replay stops
                cls.got_transcript = transcript
        cls.bridge, cls.msgs, cls.jagged = bridge, msgs, msgs[-1]
        cls.chain = chain
        cls.main_region = main_region
        cls.prep_region = prep_region
        cls.public_values = public_values
        # Retained so tests can rebuild the same chain.
        cls.smcs = smcs
        cls.vk = vk
        cls.gkr_chips = gkr_chips
        cls.chips = chips
        cls.metadata = metadata

    def test_chain_emits_one_message_per_stage(self) -> None:
        self.assertLen(self.msgs, 4)  # commit, LogUp-GKR, zerocheck, jagged eval

    def test_dense_eval_folds_the_full_committed_dense(self) -> None:
        # dense_eval = D(z_final) must fold the FULL stacking-aligned dense (the
        # same D the stacked open reconstructs), not region.dense[:raw_size]:
        # dropping a region's pad shortens and misaligns D, so both the point and
        # the eval differ. Both regions here carry a real internal pad.
        self.assertGreater(self.prep_region.dense.shape[0], self.prep_region.raw_size)
        self.assertGreater(self.main_region.dense.shape[0], self.main_region.raw_size)

        regions = [r for r in (self.prep_region, self.main_region) if r is not None]
        d = fnp.concatenate([r.dense for r in regions])
        d = fnp.pad(d, (0, (1 << log2_ceil_usize(d.shape[0])) - d.shape[0]))
        # Fold LSB-first by the round-order challenges (the point is reversed).
        for alpha in self.jagged.eval.outer_sumcheck_point[::-1]:
            d = d[0::2] + alpha * (d[1::2] - d[0::2])
        _assert_bytes_equal(self.jagged.eval.dense_eval, d[0], "dense_eval")

    def test_jagged_eval_stage_opens_each_committed_round(self) -> None:
        """The fourth stage runs the outer/inner jagged sumcheck to D(z_final)
        and the stacked BaseFold open over the [prep, main] committed rounds.
        The open exposes one batch-eval set and one component opening per
        committed round; SP1 byte-correctness is the GPU verify_prove_shard
        harness's job, so here we pin the executed shape."""
        msg = self.jagged
        self.assertEqual(msg.eval.dense_eval.shape, ())  # scalar D(z_final)
        n_rounds = len(self.bridge.commit_digest_layers)
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

    def _assert_chain_lowers(self, chain) -> None:
        # The bridge is built inside the traced function (so this needs no pytree
        # registration of ``ShardBridge``); backend compile stays GPU's job --
        # poseidon2 has no CPU fusion emitter and CPU jit miscompiles field dots
        # (fractalyze/frx#168) -- so the smoke stops at StableHLO lowering.
        def run(dense, public_values, transcript):
            bridge = ShardBridge(
                replace(self.main_region, dense=dense), self.prep_region, public_values
            )
            _, out_transcript, _ = chain(bridge, transcript)
            return out_transcript

        lowered = frx.jit(run).lower(
            self.main_region.dense, self.public_values, cheap_transcript(BF)
        )
        self.assertIn("func", lowered.as_text())

    def test_chain_lowers_under_single_jit(self) -> None:
        """The whole chain traces as one ``@jit`` region: no stage forces a host
        sync that would split it."""
        self._assert_chain_lowers(self.chain)

    def test_jit_chain_lowers_with_logup_gkr_eager(self) -> None:
        """``prove_shard_chain(jit=True)`` jits the commit/zerocheck/jagged-eval
        stages while LogUp-GKR stays eager (sp1-zorch#264): the chain still lowers
        cleanly, with no stray host-side ``bool(ok)`` leaking from the eager grind."""
        self._assert_chain_lowers(self.jit_chain)

    def test_jaggedregion_is_a_pytree_with_only_dense_as_leaf(self) -> None:
        """JaggedRegion registers ``dense`` as its sole array leaf; the layout
        counts are static aux data, so a region crosses a ``@jit`` boundary
        without leaking the count tuples into the traced graph."""
        leaves, treedef = frx.tree_util.tree_flatten(self.main_region)
        self.assertLen(leaves, 1)
        self.assertIs(leaves[0], self.main_region.dense)
        # The layout counts ride in the treedef as static aux, not as leaves.
        rebuilt = frx.tree_util.tree_unflatten(treedef, leaves)
        self.assertIs(rebuilt.dense, self.main_region.dense)
        self.assertEqual(rebuilt.row_counts, self.main_region.row_counts)

    def test_shardbridge_flattens_to_its_array_buffers(self) -> None:
        """ShardBridge is a pytree: its leaves are exactly the region dense
        buffers and the public values — the ``None`` stage-output fields
        contribute no leaves — so the whole bridge crosses a ``@jit`` boundary
        as one argument."""
        bridge = ShardBridge(self.main_region, self.prep_region, self.public_values)
        leaves = frx.tree_util.tree_leaves(bridge)
        self.assertEqual(
            [id(x) for x in leaves],
            [
                id(self.main_region.dense),
                id(self.prep_region.dense),
                id(self.public_values),
            ],
        )

    def test_bridge_crosses_jit_as_a_donated_argument(self) -> None:
        """With ShardBridge a pytree, the chain runs under a single ``@jit``
        that takes the bridge as a *donated* argument (vs the closed-over bridge
        in ``test_chain_lowers_under_single_jit``), letting XLA reuse its input
        buffers. Stops at StableHLO lowering: CPU can't execute field dots
        (fractalyze/frx#168), GPU owns backend compile."""

        def run(bridge, transcript):
            _, out_transcript, _ = self.chain(bridge, transcript)
            return out_transcript

        bridge = ShardBridge(self.main_region, self.prep_region, self.public_values)
        lowered = frx.jit(run, donate_argnums=0).lower(bridge, cheap_transcript(BF))
        self.assertIn("func", lowered.as_text())

    def test_populated_bridge_flattens_to_array_leaves_only(self) -> None:
        """The bridge threaded out of the chain holds the GKR stage outputs —
        the evaluation point and the per-chip ChipEvaluation openings — yet
        still flattens to array leaves only. Every bridge-component type
        (region, opening) is a pytree, so a mid-chain populated bridge can
        cross a ``@jit`` boundary too, not just the initial one."""
        self.assertIsNotNone(self.bridge.gkr_chip_openings)
        leaves = frx.tree_util.tree_leaves(self.bridge)
        self.assertNotEmpty(leaves)
        for leaf in leaves:
            self.assertIsInstance(leaf, frx.Array)

    def test_trace_commit_stage_threads_digest_layers_not_mle(self) -> None:
        """TraceCommitStage retains only each region's digest tree on the bridge
        as ``[prep, main]`` — NOT the trace-sized ``[S, K]`` mle, which the
        jagged-eval open recomputes from the region dense (``_region_mle``) so a
        trace-sized copy never rides the card through GKR + zerocheck
        (fractalyze/sp1-zorch#264). The recompute yields the stacked shape."""
        digests = self.bridge.commit_digest_layers
        self.assertIsNotNone(digests)
        self.assertLen(digests, 2)  # prep, then main
        S = 1 << self.main_region.log_stacking_height
        self.assertEqual(_region_mle(self.main_region).shape[0], S)

    def test_zerocheck_stage_threads_the_eval_point(self) -> None:
        """ZerocheckStage threads its sumcheck point onto the bridge as the
        jagged-eval open's z_row (the accumulated per-round challenges, not the
        GKR zeta), so the eval stage opens the trace at the right point."""
        _assert_bytes_equal(
            self.bridge.zc_sumcheck_point, self.want_zc.msgs.challenge, "z_row"
        )

    def test_bridge_threads_stage_outputs(self) -> None:
        _assert_bytes_equal(
            self.bridge.gkr_eval_point, self.want_gkr.eval_point, "gkr_eval_point"
        )

    def test_zerocheck_stage_threads_opened_values(self) -> None:
        """ZerocheckStage threads the stage's per-chip opened values onto
        the bridge — the jagged-eval stage's per-column claims (SP1's
        round_evaluation_claims, the trace evaluations at the zerocheck point,
        NOT the GKR-point openings) and the wire's ShardOpenedValues read
        them there."""
        got = self.bridge.zc_opened_values
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

    def test_zerocheck_stage_rejects_a_chain_without_gkr(self) -> None:
        stage = ZerocheckStage(
            {"alpha": _WitnessChip()},
            max_log_row_count=_MAX_LOG_ROW_COUNT,
        )
        bridge = ShardBridge(
            self.bridge.main_region, self.bridge.prep_region, self.bridge.public_values
        )
        with self.assertRaisesRegex(ValueError, "LogUp-GKR"):
            stage(bridge, cheap_transcript(BF))


class PreambleStageTest(absltest.TestCase):
    """Pins ``PreambleStage`` against a raw transcript walk — the one
    deliberate second writing of the preamble schedule, so an accidental
    reorder in the Stage fails here instead of two tools later in a
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
        bridge, got_t, msg = PreambleStage(
            vk=vk,
            public_values=public_values,
            commitment=commitment,
            chip_metadata=metadata,
        )(sentinel, cheap_transcript(BF))

        self.assertIs(bridge, sentinel)  # bridge-agnostic pass-through
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
    on both TraceCommitStage and replay.py's preamble at once, where a layout
    bug would cancel out."""

    def test_flat_layout(self) -> None:
        got = preamble_chip_metadata(("ab", "c"), (6, 4), dtype=BF)
        want = fnp.array([2, 6, 2, ord("a"), ord("b"), 4, 1, ord("c")], dtype=BF)
        _assert_bytes_equal(got, want, "chip metadata")


class JitPermutationTest(absltest.TestCase):
    """JitPermutation rides as a static meta_field in DuplexTranscript, so it
    keys the jit cache whenever a transcript is a jit argument (the jitted stage
    bodies: zerocheck, jagged-eval, and the GKR inner zones). Value-equality forwarded from the
    inner Poseidon2 is what lets a fresh same-config transcript reuse the
    compiled stage instead of recompiling per prove -- without it every
    fresh_transcript() is a new cache key."""

    def test_same_config_wrappers_are_equal_and_hash_equal(self) -> None:
        a = JitPermutation(Poseidon2(koalabear16_params()))
        b = JitPermutation(Poseidon2(koalabear16_params()))
        self.assertIsNot(a, b)  # distinct objects, as fresh_transcript() builds
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_distinct_from_a_non_jitpermutation(self) -> None:
        p = Poseidon2(koalabear16_params())
        self.assertNotEqual(JitPermutation(p), p)


class LogupGkrStageCapClassTest(absltest.TestCase):
    """LogupGkrStage's class plumbing: one Stage with a
    pinned ``GkrCapClass`` proves two shards of different heights
    byte-identically to the exact ``prove_logup_gkr``, and the class-keyed
    inner zones compile once for the class, not once per shard."""

    def test_cap_class_stage_matches_exact_and_shares_compiles(self) -> None:
        gkr_chips = (GkrChip("alpha", (_interaction(0, 1),)),)
        # The pinned class exceeds BOTH shard heights: the tight-class oracle
        # proves below (prove_logup_gkr with no class routes through the same
        # capped builders) then cannot pre-warm the class executables the
        # compile-count window measures.
        stage = LogupGkrStage(
            gkr_chips,
            num_betas=_NUM_BETAS,
            num_row_variables=_NUM_ROW_VARIABLES,
            gkr_cap_class=GkrCapClass((12,)),
        )
        shards = []
        for seed, rows in ((60, 6), (70, 10)):
            main_region = JaggedRegion.from_chips(
                [_rand_bf(seed, (rows, 2))],
                log_stacking_height=4,
                max_log_row_count=_MAX_LOG_ROW_COUNT,
                chip_names=("alpha",),
            )
            public_values = _rand_bf(seed + 1, (8,))
            _, want = prove_logup_gkr(
                gkr_chips,
                main_region,
                None,
                cheap_transcript(BF),
                num_betas=_NUM_BETAS,
                num_row_variables=_NUM_ROW_VARIABLES,
            )
            shards.append((rows, main_region, public_values, want))

        build_before = _chip_first_layer_capped._cache_size()
        open_before = open_traces_capped._cache_size()
        for rows, main_region, public_values, want in shards:
            bridge = ShardBridge(main_region, None, public_values)
            got_bridge, _, got = stage(bridge, cheap_transcript(BF))

            label = f"rows {rows}"
            _assert_proof_byte_equal(got, want, label)
            _assert_bytes_equal(
                got_bridge.gkr_eval_point, want.eval_point, label
            )
            _assert_bytes_equal(
                got_bridge.gkr_chip_openings["alpha"].main,
                want.chip_openings["alpha"].main,
                label,
            )
        # One first-layer compile per chip and one open compile across both
        # shards: the class shapes + traced heights keep shard 2 a cache hit.
        self.assertEqual(
            _chip_first_layer_capped._cache_size() - build_before,
            len(gkr_chips),
        )
        self.assertEqual(open_traces_capped._cache_size() - open_before, 1)


class ZerocheckStageTotalCapTest(absltest.TestCase):
    """ZerocheckStage's total-cap plumbing: the eager flat pack + traced
    heights feed the class-level jit body, and two shards of one
    ``TotalCapClass`` share its compile while byte-matching an eager
    exact-heights prove (``prove_shard_zerocheck`` with per-shard heights)."""

    class _PvFreeChip:
        def eval_constraints(self, trace, public_values):
            a, b = trace[:, 0], trace[:, 1]
            one = fnp.ones((), trace.dtype)
            return fnp.stack([(a - one) * (b - one)], axis=-1)

    def _rand_ef(self, seed: int, shape) -> fnp.ndarray:
        return _rand_bf(seed, tuple(shape) + (4,)).view(EF).reshape(shape)

    def test_total_cap_stage_matches_exact_and_shares_one_compile(self) -> None:
        # The sp1-zorch#242 stage-level deliverable: a `total_cap_class` Stage
        # repacks each chip to `2*window` rows, rides the shard's heights as one
        # traced int32 vector into the single shared-buffer stage, and two shards
        # of one class share its compile while byte-matching an exact-heights
        # prove. Class bounds both shards: window >= max ceil(rows/2) over {5, 9}
        # = 5; the 2-column chip's area is 2*evenpad(rows) <= 20, so
        # area_cap >= 20 + 2*window = 30.
        chips = {"alpha": self._PvFreeChip()}
        total = ZerocheckStage(
            chips,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            total_cap_class=TotalCapClass(area_cap=30, window=5),
        )
        before = ZerocheckStage._jit_body_totalcap_traced._cache_size()
        for seed, rows in ((40, 5), (50, 9)):
            main_region = JaggedRegion.from_chips(
                [
                    fnp.concatenate(
                        [fnp.ones((rows, 1), dtype=BF), _rand_bf(seed, (rows, 1))],
                        axis=1,
                    )
                ],
                log_stacking_height=4,
                max_log_row_count=_MAX_LOG_ROW_COUNT,
                chip_names=("alpha",),
            )
            public_values = _rand_bf(seed + 1, (8,))
            gkr_eval_point = self._rand_ef(seed + 2, (7,))
            gkr_chip_openings = {
                "alpha": ChipEvaluation(
                    main=self._rand_ef(seed + 3, (2,)), preprocessed=None
                )
            }
            bridge = replace(
                ShardBridge(main_region, None, public_values),
                gkr_eval_point=gkr_eval_point,
                gkr_chip_openings=gkr_chip_openings,
            )
            _, want = prove_shard_zerocheck(
                chips,
                main_region,
                None,
                public_values,
                gkr_eval_point,
                gkr_chip_openings,
                cheap_transcript(BF),
                max_log_row_count=_MAX_LOG_ROW_COUNT,
            )
            got_bridge, _, got = total(bridge, cheap_transcript(BF))

            label = f"rows {rows}"
            _assert_bytes_equal(got.msgs.round_poly, want.msgs.round_poly, label)
            _assert_bytes_equal(got.msgs.challenge, want.msgs.challenge, label)
            _assert_bytes_equal(got.claimed_sum, want.claimed_sum, label)
            _assert_bytes_equal(got.finals[0], want.finals[0], label)
            _assert_bytes_equal(
                got.opened_values["alpha"].main,
                want.opened_values["alpha"].main,
                label,
            )
            _assert_bytes_equal(
                got_bridge.zc_sumcheck_point, got.msgs.challenge, label
            )
        self.assertEqual(
            ZerocheckStage._jit_body_totalcap_traced._cache_size() - before, 1
        )


class JaggedPcsStageClassTest(absltest.TestCase):
    """JaggedPcsStage's shard-invariant eval zone (sp1-zorch#274): the eager
    prologue folds per-shard heights into traced array values, so two shards of
    one layout class — same chip set (fixes L) and same stacking-aligned area
    tier (fixes n_d and the padded dense) — share ONE ``_jagged_eval_jit``
    compile while byte-matching the eager ``eval_round_core`` path."""

    def _rand_ef(self, seed: int, shape) -> fnp.ndarray:
        return _rand_bf(seed, tuple(shape) + (4,)).view(EF).reshape(shape)

    def test_eval_zone_matches_eager_and_shares_one_compile(self) -> None:
        perm = Poseidon2(koalabear16_params())
        smcs = SingleMatrixCommitmentScheme(
            Sponge(perm, SpongeParams(rate=8, out=8)),
            Compression(perm, CompressionParams(arity=2, chunk=8)),
        )
        stage = JaggedPcsStage(
            smcs,
            log_blowup=_LOG_BLOWUP,
            num_queries=_OPEN_NUM_QUERIES,
            pow_bits=0,
            jit=True,
        )
        eager = JaggedPcsStage(
            smcs,
            log_blowup=_LOG_BLOWUP,
            num_queries=_OPEN_NUM_QUERIES,
            pow_bits=0,
            jit=False,
        )
        before = _jagged_eval_jit._cache_size()
        # Heights 5 and 7 share the layout class: 2 columns each (same L), both
        # areas stacking-align to one 16-element block (same K, tier, padded
        # dense). Only the height VALUES differ — exactly what must not key the
        # compile.
        for seed, rows in ((80, 5), (90, 7)):
            main_region = JaggedRegion.from_chips(
                [_rand_bf(seed, (rows, 2))],
                log_stacking_height=4,
                max_log_row_count=_MAX_LOG_ROW_COUNT,
                chip_names=("alpha",),
            )
            _, commit_data = commit_region(
                main_region, smcs, log_blowup=_LOG_BLOWUP, jit=False
            )
            bridge = replace(
                ShardBridge(main_region, None, _rand_bf(seed + 1, (8,))),
                commit_digest_layers=(commit_data.digest_layers,),
                zc_sumcheck_point=self._rand_ef(seed + 2, (_MAX_LOG_ROW_COUNT,)),
                zc_opened_values={
                    "alpha": ChipEvaluation(
                        main=self._rand_ef(seed + 3, (2,)), preprocessed=None
                    )
                },
            )
            _, got_t, got = stage(bridge, cheap_transcript(BF))
            _, want_t, want = eager(bridge, cheap_transcript(BF))

            label = f"rows {rows}"
            _assert_proof_byte_equal(got.eval, want.eval, f"{label} eval")
            _assert_proof_byte_equal(got.open, want.open, f"{label} open")
            _, got_s = got_t.sample(1)
            _, want_s = want_t.sample(1)
            _assert_bytes_equal(got_s, want_s, f"{label} post-stage sample")
        # ONE eval-zone compile across both shards: the heights ride as traced
        # array values, so shard 2 is a cache hit.
        self.assertEqual(_jagged_eval_jit._cache_size() - before, 1)


if __name__ == "__main__":
    absltest.main()
