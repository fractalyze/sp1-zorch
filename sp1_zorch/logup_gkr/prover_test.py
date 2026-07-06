# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Layered-prove glue invariants.

The value-level check is the rsp byte-match (``verify_gkr_prove``, run
against the capture separately); these tests pin the glue's transcript
shape -- that the stream ``prove_logup_gkr`` emits is exactly what the
zorch jagged verifier dual replays -- plus the SP1 floor handling and the
beta-count rule.
"""

import hashlib
from dataclasses import fields
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.jagged.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.head import (
    EF_LIMBS,
    GrindRound,
    HeadChallengesRound,
    OutputBindRound,
)
from sp1_zorch.logup_gkr.prover import (
    ChipEvaluation,
    ChipOpeningsRound,
    extract_sp1_outputs,
    num_beta_values,
    prove_logup_gkr,
    resolve_witness_and_grind,
)
from sp1_zorch.shard_prover.chip_loader import make_chip_stub
from zorch.logup_gkr.circuit import JaggedGkrLayer, jagged_layer_transition
from zorch.logup_gkr.jagged_prover import JaggedLayerProof
from zorch.logup_gkr.jagged_verifier import JaggedGkrLayerRound as VerifierRound
from zorch.round import VerifyChain
from zorch.testkit.transcript import cheap_transcript


def _interaction(mult_col: int, val_col: int, *, kind: int = 3) -> Interaction:
    return Interaction(
        values=(VirtualPairCol.single_main(val_col),),
        multiplicity=VirtualPairCol.single_main(mult_col),
        kind=kind,
        is_send=True,
    )


def _region(*chips, names) -> JaggedRegion:
    return JaggedRegion.from_chips(
        list(chips),
        log_stacking_height=3,
        max_log_row_count=5,
        chip_names=names,
    )


def _main(height: int, width: int = 2, offset: int = 0) -> jnp.ndarray:
    return (
        jnp.arange(offset, offset + height * width, dtype=jnp.uint32)
        .reshape(height, width)
        .view(F)
    )


def _jagged(row_counts, n0, n1, d0, d1):
    return JaggedGkrLayer(
        numerator_0=jnp.array(n0, F),
        numerator_1=jnp.array(n1, F),
        denominator_0=jnp.array(d0, F),
        denominator_1=jnp.array(d1, F),
        row_counts=row_counts,
    )


class NumBetaValuesTest(absltest.TestCase):
    def _chip(self, name, widths):
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
                jnp.all(
                    out.numerator
                    == jnp.stack(
                        [folded.numerator_0, folded.numerator_1], axis=-1
                    ).flatten()
                )
            )
        )
        self.assertEqual(out.numerator.shape, (4,))

    def test_all_ones_floor_passes_through(self) -> None:
        layer = _jagged((1, 1), [1, 2], [3, 4], [5, 6], [7, 8])
        out = extract_sp1_outputs(layer)
        self.assertTrue(bool(jnp.all(out.numerator == jnp.array([1, 3, 2, 4], F))))

    def test_mixed_floor_rejected(self) -> None:
        layer = _jagged((2, 1), [1, 0, 2], [3, 0, 4], [5, 1, 6], [7, 1, 8])
        with self.assertRaises(ValueError):
            extract_sp1_outputs(layer)


# Golden digest of the rolled prove output (the sole prove path). Regenerate with
# `print(_proof_digest(ProveLogupGkrTest()._prove()))` when the prove output
# legitimately changes (e.g. a jax/zkx wheel bump that alters the field encoding).
_ROLLED_PYRAMID_GOLDEN = (
    "af801e4f09ae9c3a375f9cdc4613282ecd753e212129f5f91196a1494cd0cce4"
)


def _proof_digest(proof: object) -> str:
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
        h.update(np.ascontiguousarray(np.asarray(jnp.asarray(arr))).tobytes())
    return h.hexdigest()


class ProveLogupGkrTest(absltest.TestCase):
    def _prove(self, *, witness=None):
        main_a, main_b = _main(24), _main(4, offset=100)
        gkr_chips = [
            GkrChip("A", (_interaction(0, 1),)),
            GkrChip("B", (_interaction(0, 1, kind=5),)),
        ]
        region = _region(main_a, main_b, names=("A", "B"))
        transcript = cheap_transcript(F)
        transcript, proof = prove_logup_gkr(
            gkr_chips,
            region,
            None,
            transcript,
            num_betas=3,
            num_row_variables=4,
            witness=witness,
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
        _, transcript, _ = GrindRound(proof.witness)(None, transcript)
        _, transcript, _ = HeadChallengesRound(3)(None, transcript)
        carry, transcript, _ = OutputBindRound(proof.circuit_output)(
            None, transcript
        )

        chain = VerifyChain([VerifierRound(EF_LIMBS) for _ in proof.round_proofs])
        (num_eval, den_eval, point), _, ok = chain(
            carry, proof.round_proofs, transcript
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(jnp.all(point == proof.eval_point)))
        del num_eval, den_eval

    def test_round_proofs_carry_layer_points(self) -> None:
        # The wire's per-layer point_and_eval reads rp.point; pin its coherence
        # with the carry-produced eval_point this proof also carries. The
        # per-round point invariant itself is zorch's contract, tested there.
        proof = self._prove()
        self.assertTrue(
            bool(jnp.all(proof.round_proofs[-1].point == proof.eval_point[:-1]))
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
            region,
            None,
            cheap_transcript(F),
            num_betas=3,
            num_row_variables=3,
            pow_bits=8,
        )
        self.assertIsNotNone(proof.witness)

    def test_negative_pow_bits_rejected(self) -> None:
        # Fail closed at the stage boundary -- a negative bit count would
        # otherwise fall through to the zero-bit replay path.
        gkr_chips = [GkrChip("A", (_interaction(0, 1),))]
        region = _region(_main(8), names=("A",))
        with self.assertRaises(ValueError):
            prove_logup_gkr(
                gkr_chips,
                region,
                None,
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
        self.assertTrue(bool(jnp.all(zero.witness == jnp.zeros((), F))))

        passed = jnp.ones((), F)
        proof = self._prove(witness=passed)
        # Kept, not discarded: the proof carries exactly the witness observed.
        self.assertTrue(bool(jnp.all(proof.witness == passed)))
        # And observing it perturbs the post-grind sponge, so the head
        # challenges -- and the eval_point they drive -- diverge from the
        # zero-witness run.
        self.assertFalse(bool(jnp.all(proof.eval_point == zero.eval_point)))


class ChipOpeningsRoundTest(absltest.TestCase):
    def test_round_reproduces_the_raw_absorb_schedule(self) -> None:
        # The round is the single in-tree definition of SP1's chip-openings
        # absorb; write the schedule out a second time as raw transcript ops
        # (count, then per chip prep-before-main, each eval length-prefixed)
        # and pin that the round leaves the sponge in the same state.
        prep = jnp.arange(3, dtype=jnp.uint32).view(F).astype(EF)
        main_a = jnp.arange(10, 12, dtype=jnp.uint32).view(F).astype(EF)
        main_b = jnp.arange(20, 24, dtype=jnp.uint32).view(F).astype(EF)
        openings = {
            "A": ChipEvaluation(main=main_a, preprocessed=prep),
            "B": ChipEvaluation(main=main_b, preprocessed=None),
        }

        _, transcript, msg = ChipOpeningsRound(openings, ("A", "B"))(
            None, cheap_transcript(F)
        )

        raw = cheap_transcript(F)
        raw = raw.observe(jnp.array(2, F))
        for ev in (prep, main_a, main_b):
            raw = raw.observe(jnp.array(ev.shape[0], F))
            raw = raw.observe(ev)

        _, round_next = transcript.sample(1)
        _, raw_next = raw.sample(1)
        self.assertTrue(bool(jnp.all(round_next == raw_next)))
        self.assertIs(msg, openings)


class LiveGrindTest(absltest.TestCase):
    """``resolve_witness_and_grind`` must SEARCH for the witness when none is
    supplied (sp1-zorch#197) -- producing a witness the ``GrindRound`` gate
    accepts, and a transcript byte-identical to replaying that witness."""

    def test_grind_finds_gate_passing_witness_and_replays_identically(self) -> None:
        # Small pow_bits: fast to grind, still exercises the real search + gate.
        pow_bits = 6
        orig = cheap_transcript(F)
        # witness=None -> must grind. resolve runs GrindRound(pow_bits) internally,
        # which raises unless the witness passes the pow_bits gate; returning at
        # all proves the search found a gate-passing witness.
        t_grind, witness = resolve_witness_and_grind(
            orig, pow_bits=pow_bits, witness=None, bf_dtype=F
        )
        self.assertIsNotNone(witness)
        # Replaying the found witness (the recorded-dump path) must reproduce the
        # exact post-grind stream: same resolved witness, same next challenge.
        t_replay, w_replay = resolve_witness_and_grind(
            orig, pow_bits=pow_bits, witness=witness, bf_dtype=F
        )
        self.assertTrue(bool(jnp.all(w_replay == witness)))
        _, c_grind = t_grind.sample(1)
        _, c_replay = t_replay.sample(1)
        self.assertTrue(bool(jnp.all(c_grind == c_replay)))

    def test_zero_pow_bits_defaults_to_zero_witness(self) -> None:
        _, witness = resolve_witness_and_grind(
            cheap_transcript(F), pow_bits=0, witness=None, bf_dtype=F
        )
        self.assertTrue(bool(jnp.all(witness == jnp.zeros((), witness.dtype))))

    def test_negative_pow_bits_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_witness_and_grind(
                cheap_transcript(F), pow_bits=-1, witness=None, bf_dtype=F
            )


if __name__ == "__main__":
    absltest.main()
