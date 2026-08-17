# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The precondition for evaluating a wide AIR one SP1 ``BlockAir`` block at a time.

SP1 splits KeccakPermute and the Weierstrass add/double AIRs into 11-12 blocks
and dispatches one work item per (row, block)
(``sp1-gpu/crates/zerocheck/src/lib.rs`` ``initialize_dense_info``:
``tot_len += height * num_air_blocks``). Doing the same here is only legal if
the split is a pure REASSOCIATION of the α-fold: ``rlc_coeffs`` assigns
descending powers by constraint-column index, so block ``b``'s coefficients
must be the contiguous slice ``alpha[s_b:e_b]`` of the whole chip's.

Two things must hold, and both are properties of the rw-constraints manifest
rather than of anything sp1-zorch controls — hence this guard:

1. ``eval_constraints(block_idx=b)`` for ``b`` in order concatenates to
   ``eval_constraints()`` exactly (contiguous, order-preserving partition);
2. summing the per-block ``constraint_eval`` values reproduces the unsplit one
   bit for bit, with the GKR column term carried by exactly one block.

KoalaBear addition is associative and exact, so (2) follows from (1) — the test
pins it against a manifest re-export reordering the blocks.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF
from zorch.constraint_eval import constraint_eval

from sp1_zorch.shard_prover.chip_loader import load_sp1_chips
from sp1_zorch.types import PROOF_MAX_NUM_PVS
from sp1_zorch.zerocheck.coeffs import gkr_powers, rlc_coeffs
from sp1_zorch.zerocheck.prover import export_order_eval_fn, probe_num_constraints

_P = 2130706433
_WINDOW = 16


def _multiblock_chips() -> dict:
    return {
        name: chip
        for name, chip in load_sp1_chips().items()
        if getattr(chip, "num_blocks", 1) > 1 and chip._constraints
    }


class BlockSplitTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.chips = _multiblock_chips()
        # KeccakPermute + secp256{k1,r1}_{add,double}: if the manifest stops
        # exporting per-constraint blocks the split silently becomes a no-op,
        # so an empty set is a failure, not a skip.
        self.assertNotEmpty(self.chips)
        self.rng = np.random.default_rng(0xB10C)

    def _f(self, shape: tuple[int, ...] | tuple[()]) -> "fnp.ndarray":
        return fnp.asarray(self.rng.integers(0, _P, size=shape, dtype=np.uint32)).view(
            F
        )

    def _ef_scalar(self) -> "fnp.ndarray":
        return fnp.asarray(self.rng.integers(0, _P, size=(1, 4), dtype=np.uint32)).view(
            EF
        )[0]

    def test_blocks_partition_the_constraint_columns_in_order(self) -> None:
        for name, chip in self.chips.items():
            with self.subTest(chip=name):
                trace = self._f((8, chip.num_cols))
                pv = self._f((PROOF_MAX_NUM_PVS,))
                whole = np.asarray(chip.eval_constraints(trace, pv).view(fnp.uint32))
                parts = [
                    np.asarray(
                        chip.eval_constraints(trace, pv, block_idx=b).view(fnp.uint32)
                    )
                    for b in range(chip.num_blocks)
                ]
                cat = np.concatenate([p for p in parts if p.shape[-1]], axis=-1)
                np.testing.assert_array_equal(whole, cat)

    def test_summed_block_folds_are_byte_identical_to_the_unsplit_fold(self) -> None:
        for name, chip in self.chips.items():
            with self.subTest(chip=name):
                nc = chip.num_cols
                pv = self._f((PROOF_MAX_NUM_PVS,))
                p0 = self._f((_WINDOW * nc,))
                diff = self._f((_WINDOW * nc,))
                weights = gkr_powers(self._ef_scalar(), nc)
                # The round body's operand set verbatim (`jagged.
                # _round_constraint_eval_cached`), so the test exercises the
                # windowed/bounded form the prover actually emits.
                kw = dict(
                    live_width=fnp.asarray(_WINDOW, fnp.int32),
                    start_offset=fnp.asarray(0, fnp.int32),
                    window_rows=_WINDOW,
                    col_stride=fnp.asarray(_WINDOW, fnp.int32),
                    num_cols=nc,
                    delta=diff,
                    fold_coeff=self._f(()),
                    aux_operands=(pv,),
                )

                whole_fn = export_order_eval_fn(chip, nc, nc)
                k = probe_num_constraints(whole_fn, nc, EF, pv)
                alpha = rlc_coeffs(self._ef_scalar(), k)
                unsplit = constraint_eval(
                    whole_fn, p0, alpha, column_weights=weights, **kw
                )

                acc = None
                start = 0
                for b in range(chip.num_blocks):
                    fn_b = export_order_eval_fn(chip, nc, nc, block_idx=b)
                    k_b = probe_num_constraints(fn_b, nc, EF, pv)
                    if k_b == 0:
                        continue
                    # The column term belongs to the per-row value once, not
                    # once per block; the first block carries it.
                    v = constraint_eval(
                        fn_b,
                        p0,
                        alpha[start : start + k_b],
                        column_weights=(weights if acc is None else None),
                        **kw,
                    )
                    acc = v if acc is None else acc + v
                    start += k_b

                self.assertEqual(start, k)
                np.testing.assert_array_equal(
                    np.asarray(unsplit.view(fnp.uint32)),
                    np.asarray(acc.view(fnp.uint32)),
                )

    def test_block_eval_fns_have_stable_identities(self) -> None:
        # The round body's jit zone keys on eval_fn identity; a fresh closure
        # per call would bust the trace cache process-wide.
        name, chip = next(iter(self.chips.items()))
        nc = chip.num_cols
        for b in range(chip.num_blocks):
            self.assertIs(
                export_order_eval_fn(chip, nc, nc, block_idx=b),
                export_order_eval_fn(chip, nc, nc, block_idx=b),
                msg=f"{name} block {b}",
            )
        self.assertIsNot(
            export_order_eval_fn(chip, nc, nc, block_idx=0),
            export_order_eval_fn(chip, nc, nc),
        )


if __name__ == "__main__":
    absltest.main()
