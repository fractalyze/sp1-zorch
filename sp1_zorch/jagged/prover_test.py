# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match the jagged-eval sumcheck Round against the SP1 pipeline dump.

Drives ``JaggedEvalRound`` through ``ProveChain`` (the composition path) with a
scripted transcript replaying the dumped inner challenges, and byte-matches the
dense-free outputs — the outer column claim and the inner branching-program
sumcheck. The outer round polys / ``dense_eval`` need the committed dense buffer,
not vendored in this fixture, so they are out of scope here. Mont-u32, no
tolerances.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabearx4_mont

from zorch.round import ProveChain

from sp1_zorch.jagged.prover import (
    JaggedEvalInputs,
    JaggedEvalRound,
    assemble_columns,
)

EF = koalabearx4_mont
_FIXTURE = Path(__file__).parent / "testdata" / "gpu_fibonacci"


def _from_u32(u32, dtype):
    return jax.lax.bitcast_convert_type(jnp.asarray(u32, dtype=jnp.uint32), dtype)


def _u32(a) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


class _ScriptedTranscript:
    """Replays the dumped per-round challenges — the byte-match reproduces the
    reference run's Fiat-Shamir outcomes rather than re-deriving them (the duplex
    encoding is the pipeline's concern, not this round's). Mirrors
    ``zerocheck/jagged_byte_match_test``."""

    def __init__(self, challenges):
        self._next = list(challenges)

    def observe_and_sample(self, values, n=1):
        out = jnp.stack([self._next.pop(0) for _ in range(n)])
        return self, out


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
        inner_alphas = _from_u32(ch["inner_alphas"], EF)
        # z_trace = the dumped outer sumcheck point (z_final); replayed.
        z_trace = _from_u32(
            np.load(_FIXTURE / "outputs" / "outer_sumcheck_point.npy"), EF
        )

        col_heights, all_claims = assemble_columns(
            row_counts_rounds, column_counts_rounds, claims, dtype=EF
        )
        carry = JaggedEvalInputs(
            col_heights=tuple(col_heights),
            all_claims=all_claims,
            z_row=z_row,
            z_col=z_col,
            z_trace=z_trace,
        )
        # Run via ProveChain — the round must compose, not just stand alone.
        chain = ProveChain([JaggedEvalRound(dtype=EF)])
        _, _, msgs = chain(carry, _ScriptedTranscript(list(inner_alphas)))
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

    def test_inner_claimed_sum(self):
        self._assert_match(self.msg.inner_claimed_sum, "inner_claimed_sum.npy")

    def test_inner_sumcheck_polys(self):
        self._assert_match(self.msg.inner_sumcheck_polys, "inner_sumcheck_polys.npy")

    def test_inner_point(self):
        self._assert_match(self.msg.inner_point, "inner_point.npy")


if __name__ == "__main__":
    absltest.main()
