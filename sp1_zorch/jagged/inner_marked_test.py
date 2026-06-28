# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Envelope tests for the inner jagged-eval sumcheck's ``zorch.sumcheck`` marker.

When the threaded transcript carries a dedicated-fusion permutation (real
poseidon2), ``inner_sumcheck`` wraps its round scan in a ``zorch.sumcheck``
composite a vendor codegens register-resident; a non-fusion transcript stays on
the plain scan. These tests assert the marker's recognition contract — name,
version, and shape attributes — off the jaxpr (no compile, so no zkx emitter is
needed: the byte-identity of the marked vs. scan path is the GPU-fused path's
gate, exercised once the ``jagged_eval`` emitter lands). The companion byte-match
is ``prover_test`` (scripted transcript -> scan path)."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from absl.testing import absltest
from jaxlib.mlir.dialects import stablehlo
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.pcs.jagged.poly import build_jagged_layout
from zorch.sumcheck.prover import SUMCHECK_MARKER, SUMCHECK_MARKER_VERSION
from zorch.testkit.transcript import cheap_transcript
from zorch.utils.bits import log2_ceil_usize

from sp1_zorch.jagged.prover import inner_sumcheck
from sp1_zorch.shard_prover.replay import fresh_transcript

BF = koalabear_mont
EF = koalabearx4_mont
_HAS_COMPOSITE_OP = hasattr(stablehlo, "CompositeOp")

# A small jagged instance: two columns of heights 2 and 3. `n_r` covers the
# tallest column; the layout derives `n_d` (the prefix-bit width); `n_c` is the
# column-bit width ceil(log2 l_max). (Mirrors zorch's test-only `oracle_cfg`,
# which is a private target — recompute it here from the public poly API.)
_COL_HEIGHTS = [2, 3]
_N_R = 2


def _layout():
    _, n_d = build_jagged_layout(_COL_HEIGHTS, len(_COL_HEIGHTS), EF)
    return SimpleNamespace(
        n_r=_N_R, n_c=log2_ceil_usize(len(_COL_HEIGHTS)), n_d=n_d
    )


def _ef(seed: int, n: int):
    # Distinct EF points for tracing — these tests only `make_jaxpr` / `lower`,
    # which never execute the field arithmetic, so any in-range Montgomery limbs
    # suffice (no reduction runs over them).
    limbs = ((jnp.arange(n * 4, dtype=jnp.uint32) + seed) % 1000 + 1).reshape(n, 4)
    return jax.lax.bitcast_convert_type(limbs, EF)


def _points(cfg):
    return _ef(1, cfg.n_r), _ef(2, cfg.n_c), _ef(3, cfg.n_d)


def _sumcheck_composites(jaxpr):
    return [
        e
        for e in jaxpr.eqns
        if e.primitive.name == "composite" and e.params["name"] == SUMCHECK_MARKER
    ]


@absltest.skipUnless(_HAS_COMPOSITE_OP, "jaxlib lacks stablehlo.CompositeOp")
class InnerMarkedTest(absltest.TestCase):
    def test_dedicated_fusion_emits_sumcheck_composite_envelope(self) -> None:
        cfg = _layout()
        z_row, z_col, z_trace = _points(cfg)
        t0 = fresh_transcript()  # real poseidon2 -> has_dedicated_fusion
        jaxpr = jax.make_jaxpr(
            lambda zr, zc, zt: inner_sumcheck(_COL_HEIGHTS, zr, zc, zt, t0, dtype=EF)
        )(z_row, z_col, z_trace).jaxpr

        composites = _sumcheck_composites(jaxpr)
        self.assertLen(composites, 1)
        eqn = composites[0]
        self.assertEqual(eqn.params["version"], SUMCHECK_MARKER_VERSION)
        attrs = {k: leaves[0] for k, leaves, _ in eqn.params["attributes"]}
        self.assertEqual(int(attrs["degree"]), 2)
        self.assertEqual(int(attrs["num_vars"]), 2 * cfg.n_d)
        self.assertEqual(int(attrs["num_factors"]), 1)
        self.assertEqual(attrs["fold_order"], "lsb")
        self.assertEqual(attrs["poly_form"], "jagged_eval")

    def test_non_fusion_transcript_stays_unmarked(self) -> None:
        # has_dedicated_fusion=False keeps the gate shut: plain scan, no composite.
        # Base-field transcript (like production koalabear16): EF challenges come
        # from reinterpreting `ef_limbs` base squeezes, so the field is base.
        cfg = _layout()
        z_row, z_col, z_trace = _points(cfg)
        t0 = cheap_transcript(BF)
        jaxpr = jax.make_jaxpr(
            lambda zr, zc, zt: inner_sumcheck(_COL_HEIGHTS, zr, zc, zt, t0, dtype=EF)
        )(z_row, z_col, z_trace).jaxpr
        self.assertEmpty(_sumcheck_composites(jaxpr))

    def test_marked_lowers_with_nested_poseidon2_marker(self) -> None:
        # The composite lowers (its scan body + nested Fiat-Shamir survive), and
        # the poseidon2 permutation rides inside as the nested marker the
        # recognizer threads its hash params from.
        cfg = _layout()
        z_row, z_col, z_trace = _points(cfg)
        t0 = fresh_transcript()
        text = (
            jax.jit(
                lambda zr, zc, zt: inner_sumcheck(
                    _COL_HEIGHTS, zr, zc, zt, t0, dtype=EF
                )
            )
            .lower(z_row, z_col, z_trace)
            .as_text()
        )
        self.assertIn(SUMCHECK_MARKER, text)
        self.assertIn("zorch.poseidon2", text)


if __name__ == "__main__":
    absltest.main()
