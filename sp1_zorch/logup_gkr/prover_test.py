# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Layered-prove glue invariants.

The value-level check is the rsp byte-match (``verify_gkr_prove``, run
against the capture separately); these tests pin the glue's transcript
shape -- that the stream ``prove_logup_gkr`` emits is exactly what the
zorch jagged verifier dual replays -- plus the SP1 floor handling and the
beta-count rule.
"""

from dataclasses import fields
from types import SimpleNamespace

import jax.numpy as jnp
from absl.testing import absltest
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.prover import (
    extract_sp1_outputs,
    num_beta_values,
    prove_logup_gkr,
)
from sp1_zorch.shard_prover.chip_loader import make_chip_stub
from zorch.logup_gkr.circuit import JaggedGkrLayer, jagged_layer_transition
from zorch.logup_gkr.jagged_verifier import JaggedGkrLayerRound as VerifierRound
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.round import VerifyChain
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import sample_challenge
from zorch.utils.bits import log2_ceil_usize

_EF_LIMBS = 4


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


class ProveLogupGkrTest(absltest.TestCase):
    def _prove(self, *, jit: bool = False):
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
            jit=jit,
        )
        return proof

    def test_stream_replays_through_the_zorch_verifier_dual(self) -> None:
        # The glue's per-layer carry threading must be byte-for-byte the
        # jagged verifier round's: replay the head (witness, challenges,
        # output binding) and drive the verifier chain off the same fresh
        # sponge -- every layer must accept and land on the same point.
        proof = self._prove()

        transcript = cheap_transcript(F)
        transcript = transcript.observe(proof.witness)
        transcript, _ = transcript.sample(1)
        transcript, _alpha = sample_challenge(transcript, EF, _EF_LIMBS)
        for _ in range(log2_ceil_usize(3)):
            transcript, _ = sample_challenge(transcript, EF, _EF_LIMBS)
        transcript, _ = sample_challenge(transcript, EF, _EF_LIMBS)

        num = proof.circuit_output.numerator
        den = proof.circuit_output.denominator
        transcript = transcript.observe(jnp.array(num.shape[0], F))
        transcript = transcript.observe(num)
        transcript = transcript.observe(jnp.array(den.shape[0], F))
        transcript = transcript.observe(den)
        coords = []
        for _ in range(2):  # niv + 1
            transcript, c = sample_challenge(transcript, EF, _EF_LIMBS)
            coords.append(c)
        z1 = jnp.stack(coords)
        eq_z1 = expand_eq_to_hypercube(z1, jnp.ones((), EF))
        carry = (jnp.sum(num * eq_z1), jnp.sum(den * eq_z1), z1)

        chain = VerifyChain([VerifierRound(_EF_LIMBS) for _ in proof.round_proofs])
        (num_eval, den_eval, point), _, ok = chain(
            carry, proof.round_proofs, transcript
        )
        self.assertTrue(bool(ok))
        self.assertTrue(bool(jnp.all(point == proof.eval_point)))
        del num_eval, den_eval

    def test_jit_prove_matches_eager(self) -> None:
        # jit=True must be a pure dispatch change: the whole proof stream
        # byte-identical to the eager prove. The cheap transcript keeps the
        # layers unmarked -- compiling the marked `zorch.sumcheck` composite
        # is a multi-minute XLA CPU compile, and jit(marked) parity already
        # follows from zorch's JaggedGkrLayerRoundJitTest composed with its
        # marked-vs-plain coverage.
        eager = self._prove()
        jitted = self._prove(jit=True)
        self.assertTrue(bool(jnp.all(eager.eval_point == jitted.eval_point)))
        self.assertTrue(
            bool(
                jnp.all(
                    eager.circuit_output.numerator == jitted.circuit_output.numerator
                )
            )
        )
        self.assertEqual(len(eager.round_proofs), len(jitted.round_proofs))
        for i, (e, j) in enumerate(zip(eager.round_proofs, jitted.round_proofs)):
            for f in fields(e):
                self.assertTrue(
                    bool(jnp.all(getattr(e, f.name) == getattr(j, f.name))),
                    f"round_proofs[{i}].{f.name} diverged under jit",
                )
        for name, ev in eager.chip_openings.items():
            self.assertTrue(bool(jnp.all(ev.main == jitted.chip_openings[name].main)))

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

    def test_pow_without_witness_rejected(self) -> None:
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
                pow_bits=12,
            )


if __name__ == "__main__":
    absltest.main()
