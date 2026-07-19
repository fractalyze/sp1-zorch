# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""``eval_public_values`` — the SP1 public-values predicate port.

Accumulator: a real, valid shard's public-values vector drives the constraint
accumulator to zero, the way SP1's ``verify_public_values`` requires. This
pins the whole port at once — the ``#[repr(C)]`` field offsets, every ``eval_*``
constraint, and the canonical field constants — against ground truth, since a
single wrong offset, sign, or constant makes some constraint non-zero and the
Horner fold lands off zero.

Digest: a single interaction's signed bus mass matches the trace-side
fingerprint (``generate_interaction_vals_batch``, already byte-matched against
SP1), so the public-values digest shares one fingerprint definition with the
trace interactions — the property SP1's shared ``perm_challenges`` encode.
"""

from __future__ import annotations

from pathlib import Path

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.logup_gkr.circuit import generate_interaction_vals_batch
from sp1_zorch.logup_gkr.public_values import (
    SP1_PROOF_NUM_PV_ELTS,
    _Folder,
    eval_public_values,
)

_PV_NPY = (
    Path(__file__).resolve().parent.parent
    / "zerocheck"
    / "testdata"
    / "gpu_fibonacci"
    / "inputs"
    / "public_values.npy"
)


def _ef(seed: int, n: int) -> fnp.ndarray:
    """A deterministic EF vector with small, distinct, in-range limbs."""
    limbs = (np.arange(n * 4) * 48271 + seed * 7 + 1).reshape(n, 4) % 0x7F000001
    return frx.lax.bitcast_convert_type(
        fnp.asarray(limbs, dtype=fnp.uint32), EF
    )


class EvalPublicValuesTest(absltest.TestCase):
    def test_valid_shard_pv_drives_accumulator_to_zero(self) -> None:
        pv_u32 = np.load(_PV_NPY)
        public_values = frx.lax.bitcast_convert_type(
            fnp.asarray(pv_u32, dtype=fnp.uint32), F
        )
        self.assertGreaterEqual(public_values.shape[0], SP1_PROOF_NUM_PV_ELTS)

        # The accumulator is independent of the fingerprint challenges; any
        # in-range alpha/betas (>= 16 for GlobalAccumulation) exercises the
        # same constraint fold SP1's verifier checks.
        pv_challenge = _ef(1, 1)[0]
        alpha = _ef(2, 1)[0]
        betas = _ef(3, 16)
        accumulator, _ = eval_public_values(public_values, pv_challenge, alpha, betas)
        self.assertTrue(
            bool(accumulator == fnp.zeros((), EF)),
            f"valid shard PV left accumulator non-zero: {accumulator}",
        )

    def test_send_digest_matches_trace_fingerprint(self) -> None:
        # The public-values digest fingerprint must agree with the trace-side
        # one (already byte-matched against SP1). Build a one-row main trace
        # whose columns carry an interaction's multiplicity and values, take
        # its (mult, fingerprint) via the trace path, and check a folder send
        # of the same values yields +mult / fingerprint.
        alpha = _ef(5, 1)[0]
        betas = _ef(6, 16)
        kind = 5  # Byte
        # Column 0 holds the multiplicity, columns 1..5 the four values.
        main = fnp.arange(1, 6, dtype=fnp.uint32).reshape(1, 5).view(F)
        interaction = Interaction(
            values=tuple(VirtualPairCol.single_main(i) for i in range(1, 5)),
            multiplicity=VirtualPairCol.single_main(0),
            kind=kind,
            is_send=True,
        )
        mult_row, fingerprint_row = generate_interaction_vals_batch(
            interaction, None, main, alpha, betas
        )
        want = (mult_row[0] * fnp.ones((), EF)) / fingerprint_row[0]

        values = [main[0, i] * fnp.ones((), EF) for i in range(1, 5)]
        folder = _Folder(_ef(9, 1)[0], alpha, betas)
        folder.send(values, main[0, 0] * fnp.ones((), EF), kind)
        self.assertTrue(bool(folder.local_interaction_digest == want))

    def test_pv_tamper_shifts_the_digest(self) -> None:
        # The bus-balance leg rejects an unbalanced re-prove because the
        # public-values digest is sensitive to the public values: a re-prove
        # that ships a different global interaction count (index 129) produces
        # a different digest, so a circuit output that cancelled the honest
        # digest no longer balances. Pin that sensitivity directly.
        pv_u32 = np.load(_PV_NPY)
        public_values = frx.lax.bitcast_convert_type(
            fnp.asarray(pv_u32, dtype=fnp.uint32), F
        )
        pv_challenge, alpha, betas = _ef(1, 1)[0], _ef(2, 1)[0], _ef(3, 16)
        _, digest = eval_public_values(public_values, pv_challenge, alpha, betas)

        tampered = public_values.at[129].add(fnp.ones((), F))  # global_count
        _, tampered_digest = eval_public_values(
            tampered, pv_challenge, alpha, betas
        )
        self.assertFalse(bool(digest == tampered_digest))

    def test_send_then_receive_cancels(self) -> None:
        alpha = _ef(11, 1)[0]
        betas = _ef(12, 16)
        values = list(_ef(13, 5))
        mult = _ef(14, 1)[0]
        folder = _Folder(_ef(15, 1)[0], alpha, betas)
        folder.send(values, mult, 7)
        folder.receive(values, mult, 7)
        self.assertTrue(bool(folder.local_interaction_digest == fnp.zeros((), EF)))


if __name__ == "__main__":
    absltest.main()
