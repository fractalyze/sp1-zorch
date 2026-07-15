# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Head-schedule glue invariants.

The rounds in ``head`` are the single in-tree definition of SP1's GKR head
stream; this test writes the schedule out a second time as raw transcript
ops and pins that the rounds reproduce it value-for-value and leave the
sponge in the same state. The value-level anchor against SP1 itself is the
rsp byte-match (``verify_gkr_prove``).
"""

import frx.numpy as jnp
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.logup_gkr.head import (
    EF_LIMBS,
    GrindRound,
    HeadChallengesRound,
    OutputBindRound,
)
from zorch.logup_gkr.circuit import LogUpGkrOutput
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import sample_challenge
from zorch.utils.bits import log2_ceil_usize


def _ef(values) -> jnp.ndarray:
    """EF elements from per-limb u32 rows."""
    return lax.bitcast_convert_type(jnp.array(values, jnp.uint32), EF)


def _output() -> LogUpGkrOutput:
    num = _ef([[i + 1, 0, 0, 0] for i in range(4)])
    den = _ef([[i + 9, 0, 0, 0] for i in range(4)])
    return LogUpGkrOutput(numerator=num, denominator=den)


class HeadStreamTest(absltest.TestCase):
    def test_rounds_reproduce_the_raw_schedule(self) -> None:
        witness = jnp.zeros((), F)
        output = _output()
        num_betas = 3

        transcript = cheap_transcript(F)
        _, transcript, witness_msg = GrindRound(witness)(None, transcript)
        _, transcript, head = HeadChallengesRound(num_betas)(None, transcript)
        carry, transcript, z1_msg = OutputBindRound(output)(None, transcript)

        raw = cheap_transcript(F)
        raw = raw.observe(witness)
        raw, _ = raw.sample(1)
        raw, alpha = sample_challenge(raw, EF, EF_LIMBS)
        seeds = []
        for _ in range(log2_ceil_usize(num_betas)):
            raw, seed = sample_challenge(raw, EF, EF_LIMBS)
            seeds.append(seed)
        raw, pv_challenge = sample_challenge(raw, EF, EF_LIMBS)  # the pv challenge
        num, den = output.numerator, output.denominator
        raw = raw.observe(jnp.array(num.shape[0], F))
        raw = raw.observe(num)
        raw = raw.observe(jnp.array(den.shape[0], F))
        raw = raw.observe(den)
        coords = []
        for _ in range(2):  # log2(len(num))
            raw, c = sample_challenge(raw, EF, EF_LIMBS)
            coords.append(c)
        z1 = jnp.stack(coords)

        self.assertTrue(bool(jnp.all(witness_msg == witness)))
        self.assertTrue(bool(jnp.all(head.alpha == alpha)))
        self.assertTrue(bool(jnp.all(head.beta_seeds == jnp.stack(seeds))))
        self.assertTrue(
            bool(
                jnp.all(
                    head.betas
                    == expand_eq_to_hypercube(jnp.stack(seeds), jnp.ones((), EF))
                )
            )
        )
        self.assertTrue(bool(jnp.all(head.pv_challenge == pv_challenge)))
        self.assertTrue(bool(jnp.all(z1_msg == z1)))
        num_eval, den_eval, carry_z1 = carry
        self.assertTrue(bool(jnp.all(carry_z1 == z1)))
        self.assertTrue(bool(jnp.all(num_eval == eval_mle(num, z1))))
        self.assertTrue(bool(jnp.all(den_eval == eval_mle(den, z1))))

        # Same sponge state after both walks: the next squeeze agrees.
        _, rounds_next = transcript.sample(1)
        _, raw_next = raw.sample(1)
        self.assertTrue(bool(jnp.all(rounds_next == raw_next)))

    def test_single_beta_has_no_seeds_and_unit_expansion(self) -> None:
        transcript = cheap_transcript(F)
        _, _, head = HeadChallengesRound(1)(None, transcript)
        self.assertEqual(head.beta_seeds.shape[0], 0)
        self.assertTrue(bool(jnp.all(head.betas == jnp.ones((1,), EF))))


class GrindGateTest(absltest.TestCase):
    def test_pow_gate_judges_the_sampled_bit(self) -> None:
        # Find witnesses whose post-grind squeeze is odd (gate fails at one
        # bit) and even (gate passes), so both arms run deterministically.
        odd = even = None
        for w in range(16):
            witness = jnp.array(w, F)
            _, sample = cheap_transcript(F).observe(witness).sample(1)
            if int(sample[0]) & 1:
                odd = witness
            else:
                even = witness
            if odd is not None and even is not None:
                break
        self.assertIsNotNone(odd)
        self.assertIsNotNone(even)

        with self.assertRaises(ValueError):
            GrindRound(odd, pow_bits=1)(None, cheap_transcript(F))
        GrindRound(even, pow_bits=1)(None, cheap_transcript(F))
        # pow_bits == 0 never gates — replay callers rely on it.
        GrindRound(odd)(None, cheap_transcript(F))


if __name__ == "__main__":
    absltest.main()
