# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Cold-path units of the rsp zerocheck harness.

The GKR replay feeding the harness costs hours; a cache or parser bug that
only fires after it would burn a full run per iteration. These tests pin the
two pieces that run before/after that replay: the GKR-output cache roundtrip
(including the live sponge state) and the phase3 dump parser.
"""

from __future__ import annotations

import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont, koalabearx4_mont

from zorch.transcript import sample_challenge

from sp1_zorch.logup_gkr.prover import ChipEvaluation
from sp1_zorch.shard_prover.replay import (
    fresh_transcript,
    load_gkr_cache,
    save_gkr_cache,
    to_u32,
)
from sp1_zorch.zerocheck.verify_zerocheck import _parse_phase3

BF = koalabear_mont
EF = koalabearx4_mont


# Inlined zorch.testkit.random_field.rand_ext_field — its bazel target is not
# visible outside the zorch workspace at the current pin.
def _rand_ef(seed: int, shape) -> fnp.ndarray:
    ints = np.random.default_rng(seed).integers(
        0, 1 << 30, size=tuple(shape) + (4,), dtype=np.int64
    )
    return fnp.array(ints, dtype=BF).view(EF).reshape(shape)


def _u32(a) -> np.ndarray:
    return to_u32(a).reshape(-1)


class GkrCacheRoundtripTest(absltest.TestCase):
    def test_cache_preserves_outputs_and_transcript_stream(self):
        # A transcript mid-stream: absorbed values and a consumed sample leave
        # non-trivial buffer positions, the state a naive cache would drop.
        t = fresh_transcript()
        t = t.observe(fnp.arange(1, 6, dtype=fnp.uint32).view(BF))
        t, _ = t.sample(3)

        eval_point = _rand_ef(1, (5,))
        openings = {
            "A": ChipEvaluation(main=_rand_ef(2, (3,)), preprocessed=_rand_ef(3, (2,))),
            "B": ChipEvaluation(main=_rand_ef(4, (1,)), preprocessed=None),
        }
        path = pathlib.Path(self.create_tempdir().full_path) / "cache.npz"
        save_gkr_cache(path, eval_point, openings, t)
        ep2, op2, t2 = load_gkr_cache(path)

        np.testing.assert_array_equal(_u32(ep2), _u32(eval_point))
        self.assertEqual(sorted(op2), sorted(openings))
        for name, ev in openings.items():
            np.testing.assert_array_equal(_u32(op2[name].main), _u32(ev.main))
            if ev.preprocessed is None:
                self.assertIsNone(op2[name].preprocessed)
            else:
                np.testing.assert_array_equal(
                    _u32(op2[name].preprocessed), _u32(ev.preprocessed)
                )

        # The streams must stay interchangeable through both absorb and
        # squeeze: observe fresh data on each and demand the same challenge.
        probe = fnp.arange(7, 10, dtype=fnp.uint32).view(BF)
        _, want = sample_challenge(t.observe(probe), EF, 4)
        _, got = sample_challenge(t2.observe(probe), EF, 4)
        np.testing.assert_array_equal(_u32(got), _u32(want))


class ParsePhase3Test(absltest.TestCase):
    def test_prep_and_main_blocks(self):
        text = (
            "chip Add:\n"
            "  prep_len=0\n"
            "  main_len=2\n"
            "  main[0]=BinomialExtensionField { value: [1, 2, 3, 4] }\n"
            "  main[1]=BinomialExtensionField { value: [5, 6, 7, 8] }\n"
            "chip Byte:\n"
            "  prep_len=1\n"
            "  prep[0]=BinomialExtensionField { value: [9, 10, 11, 12] }\n"
            "  main_len=1\n"
            "  main[0]=BinomialExtensionField { value: [13, 14, 15, 16] }\n"
        )
        path = pathlib.Path(self.create_tempdir().full_path) / "phase3.txt"
        path.write_text(text)
        parsed = _parse_phase3(path)

        def _ef(lo: int, hi: int) -> np.ndarray:
            # The dump carries canonical limbs; the parser Mont-encodes them.
            return _u32(fnp.arange(lo, hi, dtype=fnp.int32).astype(BF).view(EF))

        self.assertEqual(sorted(parsed), ["Add", "Byte"])
        np.testing.assert_array_equal(_u32(parsed["Add"]["main"]), _ef(1, 9))
        self.assertEqual(parsed["Add"]["prep"].shape, (0,))
        np.testing.assert_array_equal(_u32(parsed["Byte"]["prep"]), _ef(9, 13))
        np.testing.assert_array_equal(_u32(parsed["Byte"]["main"]), _ef(13, 17))

    def test_unrecognized_line_fails_loudly(self):
        path = pathlib.Path(self.create_tempdir().full_path) / "phase3.txt"
        path.write_text(
            "chip Add:\n"
            "  prep_len=0\n"
            "  main_len=1\n"
            "  main[0]=BinomialExtensionField { value: [1, 2, 3, 4] }\n"
            "  eq=BinomialExtensionField { value: [5, 6, 7, 8] }\n"
        )
        with self.assertRaisesRegex(ValueError, "eq="):
            _parse_phase3(path)

    def test_count_mismatch_fails_loudly(self):
        path = pathlib.Path(self.create_tempdir().full_path) / "phase3.txt"
        path.write_text(
            "chip Add:\n"
            "  prep_len=0\n"
            "  main_len=2\n"
            "  main[0]=BinomialExtensionField { value: [1, 2, 3, 4] }\n"
        )
        with self.assertRaisesRegex(ValueError, "Add.*main"):
            _parse_phase3(path)


if __name__ == "__main__":
    absltest.main()
