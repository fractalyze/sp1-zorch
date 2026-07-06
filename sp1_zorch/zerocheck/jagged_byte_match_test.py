# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match ``prove_jagged_zerocheck`` against the SP1 pipeline's zc_dump.

The ``testdata/gpu_fibonacci`` fixture is a one-time dump of the exact
post-logup_gkr state flowing into SP1's zerocheck — the jagged main/prep
regions, the transcript challenges (zeta / lambda / alpha / beta), the
per-chip GKR claims — plus the outputs of a known-good reference run
(round_polys, per-chip final openings, the per-round challenges). The test
reconstructs every chip's constraints from ``rw_constraints``, derives the
driver inputs from the dumped challenges, replays the round engine with the
dumped per-round challenges as a scripted transcript, and compares
Montgomery-form u32 bytes — no tolerances.

Fixture edge coverage (one shard, 35 chips, 22 rounds): chips with and
without preprocessed traces, PV-aware chips, constraint-less lookup chips
(Byte / Program / Range), heights from 0 (cluster fillers) through 2^16
including non-powers-of-two, and the freeze-at-2 fold tail.

The fixture layout is whir-zorch's ``sp1.zerocheck.testing.dump`` format;
regenerate via the dump recipe in whir-zorch ``sp1/zerocheck/testing/
prove_test.py`` and copy the directory under ``testdata/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.poly.eq import eval_eq

from zorch.pcs.jagged.region import JaggedRegion
from sp1_zorch.shard_prover.chip_loader import load_sp1_chips, sp1_name_to_rw
from sp1_zorch.shard_prover.types import PROOF_MAX_NUM_PVS
from sp1_zorch.zerocheck.jagged import (
    DEGREE,
    JaggedZerocheckSummand,
    prove_jagged_zerocheck,
)
from sp1_zorch.zerocheck.coeffs import rlc_coeffs
from sp1_zorch.zerocheck.prover import chip_traces


BF = koalabear_mont
EF = koalabearx4_mont

_FIXTURE = Path(__file__).parent / "testdata" / "gpu_fibonacci"


def _from_u32(u32: np.ndarray, dtype) -> jnp.ndarray:
    """Raw u32 Mont bitpatterns -> field array (EF collapses a trailing 4)."""
    return jax.lax.bitcast_convert_type(jnp.asarray(u32, dtype=jnp.uint32), dtype)


def _load_npy(name: str, dtype) -> jnp.ndarray:
    return _from_u32(np.load(_FIXTURE / "inputs" / name), dtype)


def _load_region(name: str, dense: jnp.ndarray) -> JaggedRegion:
    meta = json.loads((_FIXTURE / "inputs" / name).read_text())
    return JaggedRegion(
        dense=dense,
        chip_starts=tuple(int(x) for x in meta["chip_starts"]),
        row_counts=tuple(int(x) for x in meta["row_counts"]),
        column_counts=tuple(int(x) for x in meta["column_counts"]),
        log_stacking_height=int(meta["log_stacking_height"]),
        chip_names=tuple(meta["chip_names"]),
    )


def _chip_eval_fn(chip, public_values):
    """Bind the padded public-values vector; ``eval_constraints`` ignores it
    for constraints that declare no ``pv_arg``."""
    return lambda trace: chip.eval_constraints(trace, public_values)


@partial(
    jax.tree_util.register_dataclass, data_fields=["challenges", "pos"], meta_fields=[]
)
@dataclass(frozen=True)
class _ScriptedTranscript:
    """Returns the dumped per-round challenges — the byte-match replays the
    reference run's Fiat-Shamir outcomes rather than re-deriving them (the
    duplex-sponge encoding is the pipeline integration's concern, not the
    round engine's). A registered pytree with the cursor as a leaf so it rides
    the round ``lax.scan`` carry; ``sample`` advances it with
    ``dynamic_slice``."""

    challenges: jnp.ndarray
    pos: jnp.ndarray

    @classmethod
    def replaying(cls, challenges) -> "_ScriptedTranscript":
        # The engine squeezes base limbs and reassembles each EF challenge
        # (one extension element = degree base squeezes, the
        # ``sample_challenge`` rule — fractalyze/sp1-zorch#88), so the script
        # stores flat base limbs.
        flat = jax.lax.bitcast_convert_type(jnp.asarray(challenges), BF).reshape(-1)
        return cls(flat, jnp.asarray(0, jnp.int32))

    def observe(self, values):
        del values
        return self

    def sample(self, n=1):
        out = jax.lax.dynamic_slice_in_dim(self.challenges, self.pos, n, axis=0)
        return _ScriptedTranscript(self.challenges, self.pos + n), out


def _u32(a) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


class JaggedZerocheckByteMatchTest(absltest.TestCase):
    """One shared replay of the fixture; each test byte-compares one output."""

    @classmethod
    def setUpClass(cls):
        meta = json.loads((_FIXTURE / "meta.json").read_text())
        chip_names = list(meta["chip_names"])
        num_reals = [int(meta["num_reals"][n]) for n in chip_names]

        zeta = _load_npy("zeta.npy", EF)
        lambda_ = _load_npy("lambda.npy", EF)
        alpha = _load_npy("batching_challenge.npy", EF)
        beta = _load_npy("gkr_opening_batch_challenge.npy", EF)
        chip_claims = _load_npy("chip_claims.npy", EF)
        public_values = _load_npy("public_values.npy", BF)

        main_region = _load_region("main_region.json", _load_npy("main_dense.npy", BF))
        prep_region = _load_region("prep_region.json", _load_npy("prep_dense.npy", BF))
        assert tuple(chip_names) == tuple(main_region.chip_names)

        traces = chip_traces(chip_names, num_reals, main_region, prep_region)

        # Reconstruct each chip's constraint set; a missing or width-drifted
        # manifest entry must fail loudly — a stub would silently break the
        # byte-match.
        rw_names = [sp1_name_to_rw(n) for n in chip_names]
        all_chips = load_sp1_chips(chip_names=rw_names)
        chips = [all_chips[rw] for rw in rw_names]
        for chip, trace in zip(chips, traces):
            assert chip.num_cols == trace.shape[0], (chip.name, chip.num_cols)

        pv = jnp.concatenate(
            [
                public_values,
                jnp.zeros((PROOF_MAX_NUM_PVS - public_values.shape[0],), dtype=BF),
            ]
        )
        eval_fns = [_chip_eval_fn(c, pv) for c in chips]

        # Constraint-RLC vector per chip: descending powers of the dumped
        # batching challenge, one per constraint. The count comes from a
        # one-row probe — a chip's constraint functions may emit several
        # columns each, so it is not readable off the manifest.
        alphas = [
            rlc_coeffs(alpha, fn(jnp.zeros((1, t.shape[0]), dtype=EF)).shape[-1])
            for fn, t in zip(eval_fns, traces)
        ]

        # The reference folds chips by Horner under lambda, i.e. descending
        # lambda powers in chip order.
        lambdas = rlc_coeffs(lambda_, len(chips))

        cls.expected_round_polys = np.load(_FIXTURE / "outputs" / "round_polys.npy")
        cls.expected_finals = _from_u32(
            np.load(_FIXTURE / "outputs" / "chip_final_states.npy"), EF
        )
        cls.expected_final_lens = [
            int(x) for x in np.load(_FIXTURE / "outputs" / "chip_final_lens.npy")
        ]
        cls.zc_sumcheck_point = _from_u32(
            np.load(_FIXTURE / "outputs" / "zc_sumcheck_point.npy"), EF
        )
        cls.zeta = zeta
        cls.main_widths = [int(w) for w in main_region.chip_widths]

        # ``zc_sumcheck_point`` is the challenge list reversed (SP1's
        # ``jagged_point`` order); un-reverse to feed rounds in order.
        cls.finals, _, cls.msgs = prove_jagged_zerocheck(
            JaggedZerocheckSummand(
                eval_fns=eval_fns, alphas=alphas, lambdas=lambdas, beta=beta
            ),
            traces,
            num_reals,
            zeta,
            _ScriptedTranscript.replaying(cls.zc_sumcheck_point[::-1]),
            claims=list(chip_claims),
        )

    def test_round_polys_byte_match_reference(self):
        expected = self.expected_round_polys.reshape(-1)
        self.assertGreater(int(expected.sum()), 0, "degenerate fixture")
        got = _u32(self.msgs.round_poly)
        self.assertEqual(got.shape, expected.shape)
        mismatch = np.nonzero(got != expected)[0]
        round_stride = (DEGREE + 1) * 4  # u32 lanes per round poly
        self.assertEqual(
            mismatch.size,
            0,
            f"round_polys diverged from the dumped reference at flat u32 "
            f"indices {mismatch[:8]} "
            f"(round ~{mismatch[0] // round_stride if mismatch.size else '-'})",
        )

    def test_final_folded_openings_byte_match_reference(self):
        """Each chip's final per-column scalars, permuted back to the
        reference's ``[prep, main, eq_final]`` order (the driver folds
        ``[main | prep]`` traces)."""
        eq_final = eval_eq(self.zeta, self.zc_sumcheck_point)
        for i, final in enumerate(self.finals):
            nc = final.shape[0]
            mw = self.main_widths[i]
            self.assertEqual(self.expected_final_lens[i], nc + 1, f"chip {i}")
            if final.shape[1] > 0:
                vals = final[:, 0]
            else:
                vals = jnp.zeros((nc,), dtype=EF)
            got = jnp.concatenate([vals[mw:], vals[:mw], eq_final.reshape(1)])
            expected = self.expected_finals[i, : nc + 1]
            self.assertTrue(
                bool(np.array_equal(_u32(got), _u32(expected))),
                f"chip {i} final openings diverged",
            )


if __name__ == "__main__":
    absltest.main()
