# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match the jagged-eval sumcheck Round against the SP1 pipeline dump.

Drives ``JaggedEvalRound`` through ``ProveChain`` (the composition path) with a
scripted transcript replaying the dumped outer + inner challenges, and
byte-matches the full sumcheck half: the outer Hadamard sumcheck ``Σ D·J̃``
(round polys, folded point, ``dense_eval``) and the inner branching-program
sumcheck. The committed dense buffer ``D`` is not re-dumped — it is the same
shard packing the zerocheck stage commits, reconstructed here from the shared
``zerocheck`` dense fixture (the eval dump carries only its own outputs).
Mont-u32, no tolerances.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.round import ProveChain
from zorch.transcript import sample_challenge

from sp1_zorch.jagged.prover import (
    JaggedEvalInputs,
    JaggedEvalRound,
    assemble_columns,
)

BF = koalabear_mont
EF = koalabearx4_mont
_FIXTURE = Path(__file__).parent / "testdata" / "gpu_fibonacci"
# The packed dense lives with the zerocheck fixture (same shard); the jagged
# eval reproves Σ D·J̃ over it. Wired in via the //sp1_zorch/zerocheck:
# shard_dense_fixture filegroup.
_ZC_INPUTS = Path(__file__).parent.parent / "zerocheck" / "testdata" / "gpu_fibonacci" / "inputs"


def _from_u32(u32, dtype):
    return jax.lax.bitcast_convert_type(jnp.asarray(u32, dtype=jnp.uint32), dtype)


def _u32(a) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


def _raw_area(round_meta) -> int:
    """Σ row_count·column_count — the round's unpadded packed-dense length."""
    return sum(
        int(r) * int(c)
        for r, c in zip(round_meta["row_counts"], round_meta["column_counts"], strict=True)
    )


class _ScriptedTranscript:
    """Replays the dumped per-round challenges — the byte-match reproduces the
    reference run's Fiat-Shamir outcomes rather than re-deriving them (the duplex
    encoding is the pipeline's concern, not this round's). Mirrors
    ``zerocheck/jagged_byte_match_test``.

    Stores the extension challenges as base-field limbs: the prover binds each
    round via ``sample_challenge``, which squeezes ``degree`` base elements and
    views them as one extension element, so the stub returns those limbs and
    the reinterpret reassembles the dumped challenge."""

    def __init__(self, challenges):
        flat = jax.lax.bitcast_convert_type(jnp.stack(challenges), BF).reshape(-1)
        self._next = list(flat)

    def observe(self, values):
        del values
        return self

    def sample(self, n=1):
        return self, jnp.stack([self._next.pop(0) for _ in range(n)])


class JaggedEvalRoundByteMatchTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        meta = json.loads((_FIXTURE / "meta.json").read_text())
        row_counts_rounds = [[int(x) for x in r["row_counts"]] for r in meta["rounds"]]
        column_counts_rounds = [
            [int(x) for x in r["column_counts"]] for r in meta["rounds"]
        ]

        z_row = _from_u32(np.load(_FIXTURE / "inputs" / "z_row.npy"), EF)
        claims = [
            _from_u32(np.load(_FIXTURE / "inputs" / f"claims_r{r}.npy"), EF)
            for r in range(len(meta["rounds"]))
        ]
        ch = np.load(_FIXTURE / "outputs" / "challenges.npz")
        z_col = _from_u32(ch["z_col"], EF)
        outer_alphas = _from_u32(ch["outer_alphas"], EF)
        inner_alphas = _from_u32(ch["inner_alphas"], EF)

        # Reconstruct the committed dense buffer D: per round, strip the
        # stacking pad to the raw packed area, concat in round order
        # (prep, main), matching SP1's _build_combined_dense. The two rounds'
        # raw areas sum to a power of two here, so no extra pad is needed.
        prep = _from_u32(
            np.load(_ZC_INPUTS / "prep_dense.npy")[: _raw_area(meta["rounds"][0])], BF
        )
        main = _from_u32(
            np.load(_ZC_INPUTS / "main_dense.npy")[: _raw_area(meta["rounds"][1])], BF
        )
        dense = jnp.concatenate([prep, main])

        col_heights, all_claims = assemble_columns(
            row_counts_rounds, column_counts_rounds, claims, dtype=EF
        )
        carry = JaggedEvalInputs(
            col_heights=tuple(col_heights),
            all_claims=all_claims,
            z_row=z_row,
            z_col=z_col,
            dense=dense,
        )
        # Run via ProveChain — the round must compose, not just stand alone.
        # The outer sumcheck samples its 23 alphas first, then the inner its 48.
        chain = ProveChain([JaggedEvalRound(dtype=EF)])
        script = list(outer_alphas) + list(inner_alphas)
        _, _, msgs = chain(carry, _ScriptedTranscript(script))
        cls.msg = msgs[0]

    def _expect(self, name):
        return np.load(_FIXTURE / "outputs" / name).reshape(-1)

    def _assert_match(self, got, name):
        exp = self._expect(name)
        self.assertGreater(int(exp.sum()), 0, "degenerate fixture")
        got = _u32(got)
        self.assertEqual(got.shape, exp.shape)
        mism = np.nonzero(got != exp)[0]
        self.assertEqual(mism.size, 0, f"{name} diverged at u32 {mism[:8]}")

    def test_outer_sumcheck_claim(self):
        self._assert_match(self.msg.outer_sumcheck_claim, "outer_sumcheck_claim.npy")

    def test_outer_sumcheck_polys(self):
        self._assert_match(self.msg.outer_sumcheck_polys, "outer_sumcheck_polys.npy")

    def test_outer_sumcheck_point(self):
        self._assert_match(self.msg.outer_sumcheck_point, "outer_sumcheck_point.npy")

    def test_dense_eval(self):
        self._assert_match(self.msg.dense_eval, "dense_eval.npy")

    def test_inner_claimed_sum(self):
        self._assert_match(self.msg.inner_claimed_sum, "inner_claimed_sum.npy")

    def test_inner_sumcheck_polys(self):
        self._assert_match(self.msg.inner_sumcheck_polys, "inner_sumcheck_polys.npy")

    def test_inner_point(self):
        self._assert_match(self.msg.inner_point, "inner_point.npy")


class OuterSumcheckSqueezeRuleTest(absltest.TestCase):
    """Pins SP1's per-round squeeze rule on a real transcript: each sumcheck
    round binds ONE extension element (efinfo(ef).degree base squeezes —
    p3's sample_ext_element), not one base squeeze. The scripted byte-match
    above replays dumped challenges, so it is blind to this. ``inner_sumcheck``
    binds its rounds through the identical ``observe`` + ``sample_challenge``
    mechanic, so this also covers it."""

    def test_outer_round_advances_one_extension_challenge(self):
        from zorch.testkit.transcript import cheap_transcript

        from sp1_zorch.jagged.prover import outer_sumcheck

        n = 8
        dense = _from_u32(np.arange(1, n + 1, dtype=np.uint32), BF)
        indicator = _from_u32(
            np.arange(1, 4 * n + 1, dtype=np.uint32).reshape(n, 4), EF
        )
        claim = _from_u32(np.array([5, 6, 7, 8], dtype=np.uint32), EF)

        t0 = cheap_transcript(BF)
        polys, _, _, t_engine = outer_sumcheck(dense, indicator, claim, t0)

        degree = 4  # efinfo(koalabearx4_mont).degree
        t = t0
        for coef in polys:
            t = t.observe(coef)
            t, _ = sample_challenge(t, EF, degree)
        _, ef_next = t.sample(1)
        _, engine_next = t_engine.sample(1)
        self.assertTrue(
            bool(jnp.array_equal(ef_next, engine_next)),
            "each round must advance the transcript by one EF challenge",
        )

        t = t0
        for coef in polys:
            t, _ = t.observe_and_sample(coef, 1)
        _, base_next = t.sample(1)
        self.assertFalse(
            bool(jnp.array_equal(base_next, engine_next)),
            "one base squeeze per round must not match the EF rule",
        )


if __name__ == "__main__":
    absltest.main()
