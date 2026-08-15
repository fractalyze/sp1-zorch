# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Layered-prove glue invariants.

The value-level check is the rsp byte-match (``verify_gkr_prove``, run
against the capture separately); these tests pin the glue's transcript
shape -- that the stream ``prove_logup_gkr`` emits is exactly what the
zorch jagged verifier dual replays -- plus the SP1 floor handling and the
beta-count rule.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import fields
from types import SimpleNamespace

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import Array
from rw_constraints import Chip, Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF
from zorch.logup_gkr.circuit import JaggedGkrLayer, jagged_layer_transition
from zorch.logup_gkr.jagged_prover import JaggedLayerProof
from zorch.logup_gkr.jagged_verifier import JaggedGkrLayerRound as VerifierRound
from zorch.pcs.jagged.region import JaggedRegion
from zorch.round import verify_rounds
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import sample_challenge

from sp1_zorch.logup_gkr.circuit import (
    GkrCapClass,
    GkrChip,
    pack_gkr_arrival,
    region_statics,
)
from sp1_zorch.logup_gkr.head import (
    EF_CHALLENGES,
    EF_LIMBS,
    absorb_grind,
    bind_circuit_output,
    sample_head_challenges,
)
from sp1_zorch.logup_gkr.prover import (
    _OPEN_FOLD_CHUNK_CELLS,
    ChipOpeningsRound,
    _fold_chunk_rows,
    _open_chip,
    _open_chip_zone,
    absorb_chip_openings,
    extract_sp1_outputs,
    flat_openings_absorb,
    num_beta_values,
    open_traces_capped,
    prove_logup_gkr,
    resolve_witness_and_grind,
    select_openings,
)
from sp1_zorch.shard_prover.chip_loader import make_chip_stub
from sp1_zorch.types import (
    ChipEvaluation,
    LogupGkrProof,
    ShardWitness,
)


def _interaction(mult_col: int, val_col: int, *, kind: int = 3) -> Interaction:
    return Interaction(
        values=(VirtualPairCol.single_main(val_col),),
        multiplicity=VirtualPairCol.single_main(mult_col),
        kind=kind,
        is_send=True,
    )


def _region(*chips: Array, names: Sequence[str]) -> JaggedRegion:
    return JaggedRegion.from_chips(
        list(chips),
        log_stacking_height=3,
        max_log_row_count=5,
        chip_names=names,
    )


def _main(height: int, width: int = 2, offset: int = 0) -> fnp.ndarray:
    return (
        fnp.arange(offset, offset + height * width, dtype=fnp.uint32)
        .reshape(height, width)
        .view(F)
    )


def _jagged(
    row_counts: Sequence[int],
    n0: Sequence[int],
    n1: Sequence[int],
    d0: Sequence[int],
    d1: Sequence[int],
) -> JaggedGkrLayer:
    return JaggedGkrLayer(
        numerator_0=fnp.array(n0, F),
        numerator_1=fnp.array(n1, F),
        denominator_0=fnp.array(d0, F),
        denominator_1=fnp.array(d1, F),
        row_counts=fnp.asarray(row_counts, fnp.int32),
    )


class NumBetaValuesTest(absltest.TestCase):
    def _chip(self, name: str, widths: Sequence[int]) -> Chip:
        chip = make_chip_stub(name, 2)
        chip._interaction_info = {
            f"f{i}": SimpleNamespace(kind="send", tuple_width=w)
            for i, w in enumerate(widths)
        }
        return chip

    def test_max_tuple_width_plus_one(self) -> None:
        chips = {"A": self._chip("A", [3, 7]), "B": self._chip("B", [5])}
        self.assertEqual(num_beta_values(chips), 8)

    def test_no_interactions_defaults_to_one(self) -> None:
        self.assertEqual(num_beta_values({"A": self._chip("A", [])}), 1)


class ExtractSp1OutputsTest(absltest.TestCase):
    def test_saturated_floor_folds_once_then_interleaves(self) -> None:
        # Two interactions at two slots each; the fold combines each pair
        # into one fraction and the interleave routes children.
        layer = _jagged(
            (2, 2),
            [1, 0, 2, 0],
            [3, 0, 4, 0],
            [5, 1, 6, 1],
            [7, 1, 8, 1],
        )
        out = extract_sp1_outputs(layer)
        folded = jagged_layer_transition(layer, (1, 1))
        self.assertTrue(
            bool(
                fnp.all(
                    out.numerator
                    == fnp.stack(
                        [folded.numerator_0, folded.numerator_1], axis=-1
                    ).flatten()
                )
            )
        )
        self.assertEqual(out.numerator.shape, (4,))

    def test_all_ones_floor_rejected(self) -> None:
        # Counts are traced, so the contract narrowed to the saturated
        # all-2s floor; an already-folded floor extracts via zorch's
        # extract_jagged_outputs directly, and the width gate refuses it
        # here rather than silently folding its children.
        layer = _jagged((1, 1), [1, 2], [3, 4], [5, 6], [7, 8])
        with self.assertRaises(ValueError):
            extract_sp1_outputs(layer)

    def test_mixed_floor_rejected_by_width(self) -> None:
        layer = _jagged((2, 1), [1, 0, 2], [3, 0, 4], [5, 1, 6], [7, 1, 8])
        with self.assertRaises(ValueError):
            extract_sp1_outputs(layer)


def _bytes_of(arr: Array) -> bytes:
    return np.ascontiguousarray(np.asarray(fnp.asarray(arr))).tobytes()


class OpenFoldChunkingTest(absltest.TestCase):
    """Chunked open fold vs the monolithic fold -- exact byte equality.

    An LSB-first bind never pairs rows across an aligned power-of-two
    window, so each chunk-local generation is the monolithic fold's own
    restriction to its window; field ops are exact, so equality is
    byte-exact, never approximate."""

    def _rev_point(self, n: int) -> Array:
        base = fnp.arange(3, 3 + n, dtype=fnp.uint32).view(F)
        return fnp.ones((), EF) * base

    def test_chunked_matches_mono_across_chunk_sizes(self) -> None:
        # Height 24 pads to 32: rows 4/8/16 all divide the padded height,
        # and the 16-row plan's tail chunk mixes real and pad rows.
        trace = _main(24, width=3)
        rev_point = self._rev_point(6)
        mono = _bytes_of(_open_chip(trace, rev_point, 24))
        for rows in (4, 8, 16):
            with self.subTest(rows=rows):
                chunked = _open_chip(trace, rev_point, 24, chunk_cells=rows * 3)
                self.assertEqual(_bytes_of(chunked), mono)

    def test_chunked_matches_mono_under_jit(self) -> None:
        trace = _main(24, width=3)
        rev_point = self._rev_point(6)
        jitted = frx.jit(
            _open_chip, static_argnums=(2,), static_argnames=("chunk_cells",)
        )
        self.assertEqual(
            _bytes_of(jitted(trace, rev_point, 24, chunk_cells=12)),
            _bytes_of(_open_chip(trace, rev_point, 24)),
        )

    def test_height_not_divisible_by_chunk(self) -> None:
        # Real height 6 pads to 8; 4-row chunks put the second chunk half
        # in the zero pad.
        trace = _main(6, width=2)
        rev_point = self._rev_point(4)
        self.assertEqual(
            _bytes_of(_open_chip(trace, rev_point, 6, chunk_cells=8)),
            _bytes_of(_open_chip(trace, rev_point, 6)),
        )

    def test_single_chunk_degenerate_is_monolithic(self) -> None:
        # A budget admitting the whole padded height plans no chunking, and
        # the bytes agree with the unbudgeted fold.
        self.assertIsNone(_fold_chunk_rows(3, 32, 32 * 3))
        trace = _main(24, width=3)
        rev_point = self._rev_point(6)
        self.assertEqual(
            _bytes_of(_open_chip(trace, rev_point, 24, chunk_cells=32 * 3)),
            _bytes_of(_open_chip(trace, rev_point, 24)),
        )

    def test_fold_chunk_rows_plan(self) -> None:
        self.assertIsNone(_fold_chunk_rows(3, 32, None))
        self.assertEqual(_fold_chunk_rows(3, 32, 24), 8)
        self.assertEqual(_fold_chunk_rows(3, 32, 30), 8)  # power-of-two floor
        self.assertEqual(_fold_chunk_rows(64, 1 << 21, 1), 4)  # 4-row minimum
        self.assertIsNone(_fold_chunk_rows(1, 4, 4))  # rows >= padded height
        # The default budget is None (chunking disabled -- the scan
        # formulation regresses GPU memory; see _OPEN_FOLD_CHUNK_CELLS).
        self.assertIsNone(_OPEN_FOLD_CHUNK_CELLS)
        # An engaged budget must sit below the per-chip cell counts it is
        # meant to split: at 2^24 cells both registry-shaped chips (33
        # columns x 2^21 padded rows; 241 columns x 2^19 padded rows) plan
        # chunks.
        self.assertEqual(_fold_chunk_rows(33, 1 << 21, 1 << 24), 1 << 18)
        self.assertEqual(_fold_chunk_rows(241, 1 << 19, 1 << 24), 1 << 16)

    def test_capped_open_and_absorb_byte_identical_chunked(self) -> None:
        # The full openings zone + absorb, chunked vs monolithic: same
        # openings, same flat message, same post-absorb transcript stream
        # (a squeeze depends on the whole absorbed history).
        main_a, main_b = _main(24, width=2), _main(6, width=2, offset=100)
        region = _region(main_a, main_b, names=("A", "B"))
        cap_class = GkrCapClass.from_heights([24, 6])
        main_flat, prep_flat, _ = pack_gkr_arrival(region, None, cap_class)
        chip_names, main_widths, _ = region_statics(region)
        eval_point = self._rev_point(6)

        def open_with(
            fold_chunk_cells: int | None,
        ) -> tuple[dict[str, ChipEvaluation], Array]:
            return open_traces_capped(
                main_flat,
                prep_flat,
                eval_point,
                fold_chunk_cells=fold_chunk_cells,
                trace_dimension=6,
                cap_class=cap_class,
                chip_names=chip_names,
                main_widths=main_widths,
                prep_names=(),
                prep_widths=(),
                prep_heights=(),
            )

        opened_mono = open_with(fold_chunk_cells=None)
        opened_chunked = open_with(fold_chunk_cells=8)
        self.assertEqual(_bytes_of(opened_chunked[1]), _bytes_of(opened_mono[1]))
        for name in chip_names:
            self.assertEqual(
                _bytes_of(opened_chunked[0][name].main),
                _bytes_of(opened_mono[0][name].main),
            )
        t_mono, _ = absorb_chip_openings(cheap_transcript(F), opened_mono)
        t_chunked, _ = absorb_chip_openings(cheap_transcript(F), opened_chunked)
        _, c_mono = sample_challenge(t_mono, EF, EF_LIMBS)
        _, c_chunked = sample_challenge(t_chunked, EF, EF_LIMBS)
        self.assertEqual(_bytes_of(c_chunked), _bytes_of(c_mono))


# Golden digest of the rolled prove output (the sole prove path). Regenerate with
# `print(_proof_digest(ProveLogupGkrTest()._prove()))` when the prove output
# legitimately changes (e.g. a frx/zkx wheel bump that alters the field encoding).
_ROLLED_PYRAMID_GOLDEN = (
    "af801e4f09ae9c3a375f9cdc4613282ecd753e212129f5f91196a1494cd0cce4"
)


def _proof_digest(proof: LogupGkrProof) -> str:
    """SHA-256 over the proof's field bytes in a fixed order -- a compact CPU
    regression guard. The full field-level oracle is the SP1 reference byte-match
    (``verify_gkr_prove``, a GPU runnable)."""
    leaves: dict[str, object] = {
        "eval_point": proof.eval_point,
        "numerator": proof.circuit_output.numerator,
        "denominator": proof.circuit_output.denominator,
    }
    for i, rp in enumerate(proof.round_proofs):
        for f in fields(JaggedLayerProof):
            leaves[f"round_proofs.{i}.{f.name}"] = getattr(rp, f.name)
    for name in sorted(proof.chip_openings):
        leaves[f"chip_openings.{name}.main"] = proof.chip_openings[name].main
    h = hashlib.sha256()
    for key, arr in sorted(leaves.items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(np.asarray(fnp.asarray(arr))).tobytes())
    return h.hexdigest()


class ProveLogupGkrTest(absltest.TestCase):
    def _prove(self, *, pow_witness: Array | None = None) -> LogupGkrProof:
        main_a, main_b = _main(24), _main(4, offset=100)
        gkr_chips = [
            GkrChip("A", (_interaction(0, 1),)),
            GkrChip("B", (_interaction(0, 1, kind=5),)),
        ]
        region = _region(main_a, main_b, names=("A", "B"))
        transcript = cheap_transcript(F)
        transcript, proof = prove_logup_gkr(
            gkr_chips,
            ShardWitness(region, None),
            transcript,
            num_betas=3,
            num_row_variables=4,
            pow_witness=pow_witness,
        )
        return proof

    def test_rolled_pyramid_matches_golden(self) -> None:
        # The rolled prove (prove_jagged_pyramid) is the sole prove path; pin its
        # output to a captured golden as the fast CPU regression guard. The
        # independent value-level oracle is the SP1 reference byte-match
        # (verify_gkr_prove), per the module docstring.
        self.assertEqual(_proof_digest(self._prove()), _ROLLED_PYRAMID_GOLDEN)

    def test_stream_replays_through_the_zorch_verifier_dual(self) -> None:
        # The glue's per-layer carry threading must be byte-for-byte the
        # jagged verifier round's: replay the head through the shared glue
        # Rounds (their raw-schedule pin is head_test) and drive the verifier
        # chain off the same fresh sponge -- every layer must accept and land
        # on the same point.
        proof = self._prove()

        transcript = cheap_transcript(F)
        transcript = absorb_grind(transcript, proof.pow_witness)
        transcript, _ = sample_head_challenges(transcript, 3)
        transcript, carry = bind_circuit_output(transcript, proof.circuit_output)

        (num_eval, den_eval, point), _, ok = verify_rounds(
            [VerifierRound(EF_CHALLENGES) for _ in proof.round_proofs],
            carry,
            proof.round_proofs,
            transcript,
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(fnp.all(point == proof.eval_point)))
        del num_eval, den_eval

    def test_round_proofs_carry_layer_points(self) -> None:
        # The wire's per-layer point_and_eval reads rp.point; pin its coherence
        # with the carry-produced eval_point this proof also carries. The
        # per-round point invariant itself is zorch's contract, tested there.
        proof = self._prove()
        self.assertTrue(
            bool(fnp.all(proof.round_proofs[-1].point == proof.eval_point[:-1]))
        )

    def test_round_claims_recorded_per_layer(self) -> None:
        proof = self._prove()
        self.assertEqual(len(proof.round_proofs), 4)
        for rp in proof.round_proofs:
            self.assertEqual(rp.lam.dtype, EF)
            self.assertEqual(rp.claim.dtype, EF)

    def test_opens_every_chip_at_full_width(self) -> None:
        proof = self._prove()
        self.assertEqual(set(proof.chip_openings), {"A", "B"})
        for ev in proof.chip_openings.values():
            self.assertEqual(ev.main.shape, (2,))  # one eval per column
            self.assertEqual(ev.main.dtype, EF)
            self.assertIsNone(ev.preprocessed)

    def test_pow_without_witness_grinds(self) -> None:
        # No witness + pow_bits > 0: prove_logup_gkr now GRINDS for the witness
        # (sp1-zorch#197) instead of rejecting. Returning at all means the ground
        # witness passed the internal GrindRound pow gate (it raises otherwise);
        # the proof carries that witness.
        gkr_chips = [GkrChip("A", (_interaction(0, 1),))]
        region = _region(_main(8), names=("A",))
        _, proof = prove_logup_gkr(
            gkr_chips,
            ShardWitness(region, None),
            cheap_transcript(F),
            num_betas=3,
            num_row_variables=3,
            pow_bits=8,
        )
        self.assertIsNotNone(proof.pow_witness)

    def test_negative_pow_bits_rejected(self) -> None:
        # Fail closed at the stage boundary -- a negative bit count would
        # otherwise fall through to the zero-bit replay path.
        gkr_chips = [GkrChip("A", (_interaction(0, 1),))]
        region = _region(_main(8), names=("A",))
        with self.assertRaises(ValueError):
            prove_logup_gkr(
                gkr_chips,
                ShardWitness(region, None),
                cheap_transcript(F),
                num_betas=3,
                num_row_variables=3,
                pow_bits=-1,
            )

    def test_pow_bits_zero_keeps_a_passed_witness(self) -> None:
        # pow_bits == 0 with no witness defaults to a zero that only advances
        # the stream. A *passed* witness at pow_bits == 0 is a recorded-witness
        # replay: the zero-bit GrindRound still observes it into the transcript
        # (only the proof-of-work verdict host-read is gated on pow_bits > 0),
        # so it must reach the sponge. Zeroing a passed witness diverged that
        # replay from the judged pow_bits > 0 path.
        zero = self._prove()
        self.assertTrue(bool(fnp.all(zero.pow_witness == fnp.zeros((), F))))

        passed = fnp.ones((), F)
        proof = self._prove(pow_witness=passed)
        # Kept, not discarded: the proof carries exactly the witness observed.
        self.assertTrue(bool(fnp.all(proof.pow_witness == passed)))
        # And observing it perturbs the post-grind sponge, so the head
        # challenges -- and the eval_point they drive -- diverge from the
        # zero-witness run.
        self.assertFalse(bool(fnp.all(proof.eval_point == zero.eval_point)))


class CappedProveTest(absltest.TestCase):
    """The class-shaped prove against the exact prove — the linchpin of the
    class contract: the class layout only adds fold-neutral slots,
    so every proof field, opening, and the transcript must be byte-identical
    on every shard the class admits."""

    _CHIPS = [
        GkrChip("A", (_interaction(0, 1),)),
        GkrChip("B", (_interaction(0, 1, kind=5),)),
    ]

    def _shards(self) -> list[JaggedRegion]:
        return [
            _region(_main(24), _main(4, offset=100), names=("A", "B")),
            _region(_main(6, offset=7), _main(16, offset=50), names=("A", "B")),
        ]

    def _class_of(self, shards: Sequence[JaggedRegion]) -> GkrCapClass:
        return GkrCapClass.union(
            *(
                GkrCapClass.from_heights([int(h) for h in s.chip_heights])
                for s in shards
            )
        )

    def _assert_proofs_byte_equal(
        self, got: LogupGkrProof, want: LogupGkrProof
    ) -> None:
        self.assertEqual(_proof_digest(got), _proof_digest(want))
        # The digest skips prep openings and the witness; compare directly.
        for name, ev in want.chip_openings.items():
            got_prep = got.chip_openings[name].preprocessed
            if ev.preprocessed is None:
                self.assertIsNone(got_prep)
            else:
                self.assertTrue(bool(fnp.all(got_prep == ev.preprocessed)))
        self.assertTrue(bool(fnp.all(got.pow_witness == want.pow_witness)))

    def test_capped_prove_matches_exact_across_one_class(self) -> None:
        shards = self._shards()
        cap_class = self._class_of(shards)
        for shard in shards:
            _, exact = prove_logup_gkr(
                self._CHIPS,
                ShardWitness(shard, None),
                cheap_transcript(F),
                num_betas=3,
                num_row_variables=4,
            )
            capped_t, capped = prove_logup_gkr(
                self._CHIPS,
                ShardWitness(shard, None),
                cheap_transcript(F),
                num_betas=3,
                num_row_variables=4,
                cap_class=cap_class,
            )
            self._assert_proofs_byte_equal(capped, exact)
            # The advanced transcripts agree too: same next challenge.
            exact_t, _ = prove_logup_gkr(
                self._CHIPS,
                ShardWitness(shard, None),
                cheap_transcript(F),
                num_betas=3,
                num_row_variables=4,
            )
            _, c_exact = exact_t.sample(1)
            _, c_capped = capped_t.sample(1)
            self.assertTrue(bool(fnp.all(c_exact == c_capped)))

    def test_slot_cap_class_matches_exact(self) -> None:
        # The total-cap half of the class: slot_cap pinned to the max MEMBER
        # tight total — strictly below the per-chip-max-derived bound, so
        # the pyramid capacity really is narrower than the union layout —
        # must stay byte-identical (capacity only moves dead-zero tail).
        shards = self._shards()
        base = self._class_of(shards)
        names = ("A", "B")
        member_totals = [
            sum(
                GkrCapClass.from_heights([int(h) for h in s.chip_heights]).slot_counts(
                    self._CHIPS, names
                )
            )
            for s in shards
        ]
        derived = sum(base.slot_counts(self._CHIPS, names))
        self.assertLess(max(member_totals), derived)
        cap_class = GkrCapClass(base.chip_heights, max(member_totals))
        for shard in shards:
            _, exact = prove_logup_gkr(
                self._CHIPS,
                ShardWitness(shard, None),
                cheap_transcript(F),
                num_betas=3,
                num_row_variables=4,
            )
            _, capped = prove_logup_gkr(
                self._CHIPS,
                ShardWitness(shard, None),
                cheap_transcript(F),
                num_betas=3,
                num_row_variables=4,
                cap_class=cap_class,
            )
            self._assert_proofs_byte_equal(capped, exact)

    def test_slot_cap_rejects_oversized_shard(self) -> None:
        # Admission is two-sided: per-chip bounds AND the tight-total bound.
        shards = self._shards()
        cap_class = GkrCapClass(self._class_of(shards).chip_heights, 4)
        with self.assertRaisesRegex(ValueError, "slot_cap"):
            prove_logup_gkr(
                self._CHIPS,
                ShardWitness(shards[0], None),
                cheap_transcript(F),
                num_betas=3,
                num_row_variables=4,
                cap_class=cap_class,
            )

    def test_capped_open_shares_one_compile_across_the_class(self) -> None:
        shards = self._shards()
        cap_class = self._class_of(shards)
        before = _open_chip_zone._cache_size()
        first = None
        for shard in shards:
            prove_logup_gkr(
                self._CHIPS,
                ShardWitness(shard, None),
                cheap_transcript(F),
                num_betas=3,
                num_row_variables=4,
                cap_class=cap_class,
            )
            if first is None:
                first = _open_chip_zone._cache_size() - before
        # The class shapes compile per-chip zones on the first shard and
        # every later shard of the class is a pure cache hit.
        self.assertGreaterEqual(first, 1)
        self.assertEqual(_open_chip_zone._cache_size() - before, first)

    def test_capped_prove_with_prep_matches_exact(self) -> None:
        # A prep chip under a class wider than the shard: the class bound
        # stands in for the exact build's trim-to-main and the prep opens at
        # its keygen height on both paths.
        prep = _main(8, width=1, offset=200)
        inter = Interaction(
            values=(VirtualPairCol(constant=0, column_weights=((0, True, 1),)),),
            multiplicity=VirtualPairCol.single_main(0),
            kind=2,
            is_send=True,
        )
        chips = [GkrChip("A", (inter,))]
        shard = _region(_main(4), names=("A",))
        prep_region = _region(prep, names=("A",))
        _, exact = prove_logup_gkr(
            chips,
            ShardWitness(shard, prep_region),
            cheap_transcript(F),
            num_betas=3,
            num_row_variables=3,
        )
        _, capped = prove_logup_gkr(
            chips,
            ShardWitness(shard, prep_region),
            cheap_transcript(F),
            num_betas=3,
            num_row_variables=3,
            cap_class=GkrCapClass((6,)),
        )
        self._assert_proofs_byte_equal(capped, exact)


class ChipOpeningsRoundTest(absltest.TestCase):
    def test_round_reproduces_the_raw_absorb_schedule(self) -> None:
        # The round is the single in-tree definition of SP1's chip-openings
        # absorb; write the schedule out a second time as raw transcript ops
        # (count, then per chip prep-before-main, each eval length-prefixed)
        # and pin that the round leaves the sponge in the same state.
        prep = fnp.arange(3, dtype=fnp.uint32).view(F).astype(EF)
        main_a = fnp.arange(10, 12, dtype=fnp.uint32).view(F).astype(EF)
        main_b = fnp.arange(20, 24, dtype=fnp.uint32).view(F).astype(EF)
        openings = {
            "A": ChipEvaluation(main=main_a, preprocessed=prep),
            "B": ChipEvaluation(main=main_b, preprocessed=None),
        }

        _, transcript, msg = ChipOpeningsRound(openings, ("A", "B"))(
            None, cheap_transcript(F)
        )

        raw = cheap_transcript(F)
        raw = raw.observe(fnp.array(2, F))
        for ev in (prep, main_a, main_b):
            raw = raw.observe(fnp.array(ev.shape[0], F))
            raw = raw.observe(ev)

        _, round_next = transcript.sample(1)
        _, raw_next = raw.sample(1)
        self.assertTrue(bool(fnp.all(round_next == raw_next)))
        self.assertIs(msg, openings)

    def test_absorb_bytes_golden(self) -> None:
        # Absolute Fiat-Shamir golden, captured before flat_openings_absorb
        # moved its eager path to host numpy: the flat message's raw
        # Montgomery limbs and the post-absorb challenge stream on fixed
        # inputs. The equivalence test above cannot catch a change that
        # shifts prover AND verifier together (both share this builder);
        # this pins the byte stream itself, so the numpy `.view` EF->BF
        # reinterpretation is proven identical to the old device bitcast.
        prep = fnp.arange(3, dtype=fnp.uint32).view(F).astype(EF)
        openings = {
            "A": ChipEvaluation(
                main=fnp.arange(10, 12, dtype=fnp.uint32).view(F).astype(EF),
                preprocessed=prep,
            ),
            "B": ChipEvaluation(
                main=fnp.arange(20, 24, dtype=fnp.uint32).view(F).astype(EF),
                preprocessed=None,
            ),
        }

        flat = flat_openings_absorb(
            select_openings(openings, ("A", "B")), empty_prep_absorbs_zero=False
        )
        self.assertEqual(
            np.asarray(flat).view(np.uint32).tobytes().hex(),
            "fcffff03faffff05000000000000000000000000000000000100000000000000"
            "000000000000000002000000000000000000000000000000fcffff030a000000"
            "0000000000000000000000000b000000000000000000000000000000f8ffff07"
            "1400000000000000000000000000000015000000000000000000000000000000"
            "1600000000000000000000000000000017000000000000000000000000000000",
        )

        _, transcript, _ = ChipOpeningsRound(openings, ("A", "B"))(
            None, cheap_transcript(F)
        )
        _, challenge = transcript.sample(8)
        self.assertEqual(
            np.asarray(challenge).view(np.uint32).tobytes().hex(),
            "176991221769912217699122ca7bb66d" "f249a31ef249a31ef249a31ec64d856b",
        )


class LiveGrindTest(absltest.TestCase):
    """``resolve_witness_and_grind`` must SEARCH for the witness when none is
    supplied (sp1-zorch#197) -- producing a witness the ``GrindRound`` gate
    accepts, and a transcript byte-identical to replaying that witness."""

    def test_grind_finds_gate_passing_witness_and_replays_identically(self) -> None:
        # Small pow_bits: fast to grind, still exercises the real search + gate.
        pow_bits = 6
        orig = cheap_transcript(F)
        # pow_witness=None -> must grind. resolve runs GrindRound(pow_bits) internally,
        # which raises unless the witness passes the pow_bits gate; returning at
        # all proves the search found a gate-passing witness.
        t_grind, found = resolve_witness_and_grind(
            orig, pow_bits=pow_bits, pow_witness=None, bf_dtype=F
        )
        self.assertIsNotNone(found)
        # Replaying the found witness (the recorded-dump path) must reproduce the
        # exact post-grind stream: same resolved witness, same next challenge.
        t_replay, w_replay = resolve_witness_and_grind(
            orig, pow_bits=pow_bits, pow_witness=found, bf_dtype=F
        )
        self.assertTrue(bool(fnp.all(w_replay == found)))
        _, c_grind = t_grind.sample(1)
        _, c_replay = t_replay.sample(1)
        self.assertTrue(bool(fnp.all(c_grind == c_replay)))

    def test_zero_pow_bits_defaults_to_zero_witness(self) -> None:
        _, witness = resolve_witness_and_grind(
            cheap_transcript(F), pow_bits=0, pow_witness=None, bf_dtype=F
        )
        self.assertTrue(bool(fnp.all(witness == fnp.zeros((), witness.dtype))))

    def test_negative_pow_bits_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_witness_and_grind(
                cheap_transcript(F), pow_bits=-1, pow_witness=None, bf_dtype=F
            )


if __name__ == "__main__":
    absltest.main()
