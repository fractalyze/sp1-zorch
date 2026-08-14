# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-identity of the chunked total-cap prefix vs the monolithic path.

The chunked path (`TotalCapChunkClass`) must be uint32-exact against the
monolithic flat total-cap round — round polys, challenges, AND finals — across
chunk depths, chunk caps that split a chip's rows with a non-dividing tail,
odd live heights, runtime-empty and statically-empty chips, constraint-free
chips, full-depth chunking (no monolithic rounds left), and extension-field
challenges over a base-field arrival (the production dtype mix). The
monolithic flat path is itself byte-anchored to the exact static path
(`TotalCapTracedTest`); one case here re-anchors directly.

Mirrors the `OpenFoldChunkingTest` / `ChunkedEvalByteIdentityTest` pattern:
no tolerances, Montgomery-form u32 equality only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF
from zorch.poly.eq import expand_eq_to_hypercube

from sp1_zorch.zerocheck import jagged
from sp1_zorch.zerocheck.coeffs import gkr_powers, rlc_coeffs
from sp1_zorch.zerocheck.jagged import (
    JaggedZerocheckSummand,
    TotalCapChunkClass,
    TotalCapClass,
    pack_flat_arrival,
    prove_jagged_zerocheck,
)

# Witness chip: columns [a, b, c] with a == 1 on every real row, so both
# constraints vanish there while C(0_row) != 0 keeps the padded-row
# correction live (the same fixture shape as jagged_test).
_NUM_COLS = 3
_K = 2


def _eval_fn(trace: fnp.ndarray, public_values: fnp.ndarray) -> fnp.ndarray:
    del public_values
    a, b, c = trace[:, 0], trace[:, 1], trace[:, 2]
    one = fnp.ones((), trace.dtype)
    return fnp.stack([(a - one) * (c - one), (a - one) * b * c], axis=-1)


def _eval_fn_empty(trace: fnp.ndarray, public_values: fnp.ndarray) -> fnp.ndarray:
    """Lookup-only chip: no transition constraints (SP1's Byte / Program /
    Range shape) — only the GKR column term contributes."""
    del public_values
    return fnp.zeros((trace.shape[0], 0), dtype=trace.dtype)


_PV = fnp.zeros((8,), dtype=F)


def _rand(seed: int, shape: tuple[int, ...]) -> fnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return fnp.array(ints, dtype=F)


def _rand_ef(seed: int, shape: tuple[int, ...]) -> fnp.ndarray:
    return frx.lax.bitcast_convert_type(_rand(seed, (*shape, 4)), EF)


def _witness_trace(seed: int, nr: int, num_cols: int = _NUM_COLS) -> fnp.ndarray:
    if nr == 0:
        return fnp.zeros((num_cols, 0), dtype=F)
    ones = fnp.ones((1, nr), dtype=F)
    return fnp.concatenate([ones, _rand(seed, (num_cols - 1, nr))], axis=0)


def zero_extend(arr: Array, width: int) -> Array:
    pad = width - arr.shape[-1]
    if pad == 0:
        return arr
    return fnp.concatenate([arr, fnp.zeros((*arr.shape[:-1], pad), arr.dtype)], axis=-1)


@partial(
    frx.tree_util.register_dataclass, data_fields=["challenges", "pos"], meta_fields=[]
)
@dataclass(frozen=True)
class _ScriptedTranscript:
    """Preset-challenge transcript stub (the forced-challenge seam): stores
    flat BASE limbs, so `sample_challenge` reassembles either base or
    extension challenges — both dtype arms of the driver run one script."""

    challenges: fnp.ndarray
    pos: fnp.ndarray

    @classmethod
    def replaying(cls, challenges: Sequence[Array]) -> "_ScriptedTranscript":
        flat = frx.lax.bitcast_convert_type(fnp.asarray(challenges), F).reshape(-1)
        return cls(flat, fnp.asarray(0, fnp.int32))

    def observe(self, values: Array) -> "_ScriptedTranscript":
        del values
        return self

    def sample(self, n: int = 1) -> Any:
        out = frx.lax.dynamic_slice_in_dim(self.challenges, self.pos, n, axis=0)
        return _ScriptedTranscript(self.challenges, self.pos + n), out


def _u32(a: Array) -> np.ndarray:
    return np.asarray(frx.lax.bitcast_convert_type(a, fnp.uint32)).reshape(-1)


def _assert_bytes_equal(got: Array, want: Array, label: str = "") -> None:
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


def _claims(beta: Array, traces: Sequence[Array], zeta: Array) -> list[Array]:
    """Per-chip GKR opening claims over mixed column counts: each chip's
    ``beta**(j+1)``-weighted column MLE openings at zeta (the engine's own
    `gkr_powers` weighting, per-chip sliced)."""
    width = 1 << int(zeta.shape[0])
    e = expand_eq_to_hypercube(zeta, fnp.ones((), zeta.dtype))
    return [
        (
            gkr_powers(beta, t.shape[0]) @ (zero_extend(t, width) @ e)
            if t.shape[0]
            else fnp.zeros((), zeta.dtype)
        )
        for t in traces
    ]


class TotalCapChunkedByteIdentityTest(parameterized.TestCase):
    """uint32-exact equality of the chunked prefix vs the monolithic flat
    total-cap path (and, once, vs the exact static path)."""

    def _run_pair(
        self,
        num_vars: int,
        heights: Sequence[int | None],
        caps: Sequence[int],
        depth: int,
        chunk_cap: int,
        *,
        num_cols: Sequence[int] | None = None,
        constraint_free: frozenset[int] = frozenset(),
        ef_challenges: bool = False,
        also_static_oracle: bool = False,
    ) -> None:
        """``heights`` entries are ints (traced at the driver seam) or None
        for a STATICALLY empty chip (host 0). Compares the chunked run to the
        monolithic traced flat run on the same class and arrival."""
        nchips = len(heights)
        cols = list(num_cols) if num_cols is not None else [_NUM_COLS] * nchips
        h_ints = [0 if h is None else int(h) for h in heights]
        traces = [_witness_trace(11 + i, h_ints[i], cols[i]) for i in range(nchips)]
        rand_c = _rand_ef if ef_challenges else _rand
        eval_fns = [
            _eval_fn_empty if i in constraint_free else _eval_fn for i in range(nchips)
        ]
        alphas = [
            rlc_coeffs(rand_c(99 + i, ()), 0 if i in constraint_free else _K)
            for i in range(nchips)
        ]
        lambdas = rand_c(55, (nchips,))
        beta = rand_c(77, ())
        zeta = rand_c(7, (num_vars,))
        challenges = [rand_c(1000 + r, ()) for r in range(num_vars)]
        claims = _claims(beta, traces, zeta)
        summand = JaggedZerocheckSummand(
            eval_fns=eval_fns,
            alphas=alphas,
            lambdas=lambdas,
            beta=beta,
            public_values=_PV,
        )
        cap_class = TotalCapClass.from_heights(h_ints, cols)
        flat = pack_flat_arrival(traces, h_ints, cap_class, num_vars)
        # None stays a HOST 0 (statically empty chip); ints ride traced.
        num_reals = [0 if h is None else fnp.asarray(h, fnp.int32) for h in heights]

        def run(chunk_class: TotalCapChunkClass | None):  # type: ignore[no-untyped-def]
            return prove_jagged_zerocheck(
                summand,
                [],
                num_reals,
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=claims,
                total_cap_class=cap_class,
                flat_arrival=flat,
                num_cols=cols,
                chunk_class=chunk_class,
            )

        finals_w, _, want = run(None)
        finals_g, _, got = run(
            TotalCapChunkClass(
                depth=depth, chip_height_caps=tuple(caps), chunk_cap=chunk_cap
            )
        )
        label = f"heights={h_ints} depth={depth} chunk_cap={chunk_cap}"
        _assert_bytes_equal(got.round_poly, want.round_poly, f"{label} polys")
        _assert_bytes_equal(got.challenge, want.challenge, f"{label} challenge")
        for i, (fg, fw) in enumerate(zip(finals_g, finals_w, strict=True)):
            _assert_bytes_equal(fg, fw, f"{label} finals[{i}]")

        if also_static_oracle:
            # Anchor to the exact static path (itself reference-checked in
            # jagged_test): heights as host ints, per-shard tight class.
            finals_s, _, want_s = prove_jagged_zerocheck(
                summand,
                traces,
                h_ints,
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=claims,
            )
            _assert_bytes_equal(got.round_poly, want_s.round_poly, "static polys")
            for i, (fg, fs) in enumerate(zip(finals_g, finals_s, strict=True)):
                _assert_bytes_equal(fg, fs, f"static finals[{i}]")

    @parameterized.named_parameters(
        ("depth1", 1, 0),
        ("depth2", 2, 0),
        ("depth3", 3, 0),
        ("depth1_tail", 1, 7),
        ("depth2_tail", 2, 7),
    )
    def test_small_matches_monolithic(self, depth: int, chunk_cap: int) -> None:
        # chunk_cap=7 gives non-power window splits with a non-dividing tail
        # (3 cols: round-0 windows of 1 pair row over a 4-pair cap).
        self._run_pair(4, [5, 2], [8, 4], depth, chunk_cap)

    def test_static_oracle_anchor(self) -> None:
        self._run_pair(4, [5, 3], [8, 4], 2, 5, also_static_oracle=True)

    def test_odd_heights_and_height_one(self) -> None:
        self._run_pair(4, [5, 3, 1], [7, 3, 1], 2, 6)

    def test_runtime_empty_chip(self) -> None:
        # Traced height 0: the chip stays live in the zones; its windows are
        # all dead rows and its lay-in writes only masked zeros.
        self._run_pair(4, [5, 0], [8, 4], 2, 0)

    def test_statically_empty_chip_among_live(self) -> None:
        self._run_pair(4, [5, None, 3], [8, 0, 4], 2, 0)

    def test_constraint_free_chip(self) -> None:
        self._run_pair(4, [5, 8, 3], [8, 8, 4], 2, 7, constraint_free=frozenset({1}))

    def test_full_depth_chunks_every_round(self) -> None:
        # depth >= num_vars: the remainder zone runs zero rounds and only the
        # finals come off the materialized state.
        self._run_pair(3, [5, 2], [8, 4], 3, 0)

    def test_extension_field_challenges(self) -> None:
        # The production dtype mix: base-field arrival, EF challenges — the
        # chunk fold's level-0 base subtract must embed exactly.
        self._run_pair(4, [5, 2], [8, 4], 2, 7, ef_challenges=True)

    def test_keccak_class_shaped(self) -> None:
        # Structurally keccak-class-shaped: many chips, mixed column counts
        # (wide + narrow), constraint-free chips among constrained ones,
        # odd / power / height-1 / runtime-empty heights, caps above heights
        # (class union), several rounds of chunked prefix with window tails,
        # then a monolithic remainder + tail scan.
        self._run_pair(
            7,
            [100, 37, 64, 5, 0, 1, 90, 33],
            [128, 40, 64, 8, 4, 2, 96, 48],
            2,
            64,
            num_cols=[3, 3, 3, 3, 3, 3, 8, 1],
            constraint_free=frozenset({6, 7}),
            ef_challenges=True,
        )

    def test_under_bounding_cap_fails_loud(self) -> None:
        nchips = 2
        heights = [5, 2]
        cols = [_NUM_COLS] * nchips
        traces = [_witness_trace(11 + i, heights[i]) for i in range(nchips)]
        beta = _rand(77, ())
        zeta = _rand(7, (4,))
        summand = JaggedZerocheckSummand(
            eval_fns=[_eval_fn] * nchips,
            alphas=[rlc_coeffs(_rand(99 + i, ()), _K) for i in range(nchips)],
            lambdas=_rand(55, (nchips,)),
            beta=beta,
            public_values=_PV,
        )
        cap_class = TotalCapClass.from_heights(heights, cols)
        flat = pack_flat_arrival(traces, heights, cap_class, 4)
        with self.assertRaisesRegex(ValueError, "does not bound chip 0"):
            prove_jagged_zerocheck(
                summand,
                [],
                [fnp.asarray(h, fnp.int32) for h in heights],
                zeta,
                _ScriptedTranscript.replaying([_rand(1000 + r, ()) for r in range(4)]),
                claims=_claims(beta, traces, zeta),
                total_cap_class=cap_class,
                flat_arrival=flat,
                num_cols=cols,
                chunk_class=TotalCapChunkClass(
                    depth=2, chip_height_caps=(4, 4), chunk_cap=0
                ),
            )

    def test_zones_shared_across_shards_of_one_class(self) -> None:
        # Two shards of one chunk class: the second prove must add ZERO new
        # chunk-zone compiles (heights, challenges, and window starts ride as
        # traced values — the #334 shard-invariance contract).
        nchips = 2
        caps = (8, 4)
        cols = [_NUM_COLS] * nchips
        beta = _rand(77, ())
        zeta = _rand(7, (4,))
        challenges = [_rand(1000 + r, ()) for r in range(4)]
        summand = JaggedZerocheckSummand(
            eval_fns=[_eval_fn] * nchips,
            alphas=[rlc_coeffs(_rand(99 + i, ()), _K) for i in range(nchips)],
            lambdas=_rand(55, (nchips,)),
            beta=beta,
            public_values=_PV,
        )
        cap_class = TotalCapClass(area_cap=24)
        chunk = TotalCapChunkClass(depth=2, chip_height_caps=caps, chunk_cap=0)

        def prove(heights: list[int]):  # type: ignore[no-untyped-def]
            traces = [_witness_trace(11 + i, h) for i, h in enumerate(heights)]
            flat = pack_flat_arrival(traces, heights, cap_class, 4)
            return prove_jagged_zerocheck(
                summand,
                [],
                [fnp.asarray(h, fnp.int32) for h in heights],
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=_claims(beta, traces, zeta),
                total_cap_class=cap_class,
                flat_arrival=flat,
                num_cols=cols,
                chunk_class=chunk,
            )

        prove([5, 2])
        sizes = (
            jagged._chunk_round_partials._cache_size(),
            jagged._chunk_state_window._cache_size(),
            jagged._totalcap_tail._cache_size(),
        )
        prove([3, 4])
        self.assertEqual(
            sizes,
            (
                jagged._chunk_round_partials._cache_size(),
                jagged._chunk_state_window._cache_size(),
                jagged._totalcap_tail._cache_size(),
            ),
        )


if __name__ == "__main__":
    absltest.main()
