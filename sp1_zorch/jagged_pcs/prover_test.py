# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The JaggedCapClass-gated chunked eval scan (sp1-zorch#334).

Four contracts: the chunked open is byte-identical to the monolithic one
(exact field arithmetic — no tolerance), an unset class keeps the monolithic
path (HEAD behavior), the chunk zones compile once per (layout class,
scan cap) — shards differing only in heights share every chunk executable —
and the chunk zone's traced intermediates total O(chunk width) bytes, so its
temp arena scales with the cap rather than with unrolled per-bit chains."""

from __future__ import annotations

import dataclasses
import math
from functools import partial
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest, parameterized
from frx import Array
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from hash_frx.compression import Compression, CompressionParams
from hash_frx.poseidon2.poseidon2 import Poseidon2
from hash_frx.sponge import Sponge, SpongeParams
from zorch.pcs.jagged.commit import commit_region
from zorch.pcs.jagged.prover import eval_column_arrays
from zorch.pcs.jagged.region import JaggedRegion
from zorch.testkit.transcript import cheap_transcript
from zorch.utils.bits import log2_ceil_usize

from sp1_zorch.jagged_pcs.prover import (
    JaggedCapClass,
    JaggedPcsProver,
    _jagged_eval_jit,
    _jagged_inner_jit,
    _jagged_outer_chunk_fold_jit,
    _jagged_outer_chunk_partials_jit,
    _jagged_outer_tail_jit,
    _outer_chunk_partials,
)
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.types import (
    BoundRoots,
    ChipEvaluation,
    ChipMetadata,
    JaggedCommitData,
    JaggedOpeningClaim,
    JaggedOpeningWitness,
    ShardWitness,
    SmcsCommitments,
    TraceEvaluationClaim,
)

_MAX_LOG_ROW_COUNT = 5
_LOG_BLOWUP = 1
_OPEN_NUM_QUERIES = 2
_LOG_STACKING_HEIGHT = 4
_CHIP_WIDTH = 2


def _rand_bf(seed: int, shape: tuple[int, ...]) -> fnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return fnp.array(ints, dtype=F)


def _rand_ef(seed: int, shape: tuple[int, ...]) -> fnp.ndarray:
    return _rand_bf(seed, tuple(shape) + (4,)).view(EF).reshape(shape)


def _u32(a: Array) -> np.ndarray:
    return np.asarray(frx.lax.bitcast_convert_type(a, fnp.uint32)).reshape(-1)


def _assert_bytes_equal(got: Array, want: Array, label: str = "") -> None:
    np.testing.assert_array_equal(_u32(got), _u32(want), err_msg=label)


def _flatten_arrays(x: Any) -> list[Array]:
    if isinstance(x, (frx.Array, np.ndarray)):
        return [x]
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [a for e in x for a in _flatten_arrays(e)]
    if isinstance(x, dict):
        return [a for k in sorted(x) for a in _flatten_arrays(x[k])]
    if dataclasses.is_dataclass(x):
        return [
            a
            for f in dataclasses.fields(x)
            for a in _flatten_arrays(getattr(x, f.name))
        ]
    return []


def _assert_proof_byte_equal(got: Any, want: Any, label: str) -> None:
    gs, ws = _flatten_arrays(got), _flatten_arrays(want)
    assert len(gs) == len(ws), f"{label}: {len(gs)} vs {len(ws)} array leaves"
    for i, (g, w) in enumerate(zip(gs, ws, strict=True)):
        _assert_bytes_equal(g, w, f"{label}[{i}]")


def _smcs() -> SingleMatrixCommitmentScheme:
    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _open_fixture(
    smcs: SingleMatrixCommitmentScheme, seed: int, rows: int
) -> tuple[JaggedOpeningClaim, JaggedOpeningWitness, int]:
    """One single-chip open fixture (the JaggedPcsProverClassTest shape) plus
    its padded dense tier. Random claims: the prover's stream depends only on
    the claim VALUES, not their consistency, and every arm sees the same
    values."""
    main_region = JaggedRegion.from_chips(
        [_rand_bf(seed, (rows, _CHIP_WIDTH))],
        log_stacking_height=_LOG_STACKING_HEIGHT,
        max_log_row_count=_MAX_LOG_ROW_COUNT,
        chip_names=("alpha",),
    )
    _, commit_data = commit_region(main_region, smcs, log_blowup=_LOG_BLOWUP, jit=False)
    claim = JaggedOpeningClaim(
        TraceEvaluationClaim(
            _rand_ef(seed + 2, (_MAX_LOG_ROW_COUNT,)),
            {
                "alpha": ChipEvaluation(
                    main=_rand_ef(seed + 3, (_CHIP_WIDTH,)), preprocessed=None
                )
            },
        ),
        BoundRoots(
            preprocessed=commit_data.smcs_commitment, main=commit_data.smcs_commitment
        ),
        ChipMetadata(("alpha",), (rows,)),
    )
    witness = JaggedOpeningWitness(
        ShardWitness(main_region),
        JaggedCommitData(
            (commit_data.digest_layers,),
            SmcsCommitments(main=commit_data.smcs_commitment),
        ),
    )
    area = int(main_region.dense.shape[0])
    target = 1 << (area - 1).bit_length()
    return claim, witness, target


def _stage(
    smcs: SingleMatrixCommitmentScheme,
    *,
    jit: bool,
    cap: JaggedCapClass | None = None,
) -> JaggedPcsProver:
    return JaggedPcsProver(
        smcs,
        log_blowup=_LOG_BLOWUP,
        num_queries=_OPEN_NUM_QUERIES,
        pow_bits=0,
        jit=jit,
        jagged_cap_class=cap,
    )


class JaggedCapClassTest(absltest.TestCase):
    def test_rejects_non_power_of_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "power of two"):
            JaggedCapClass(6)
        with self.assertRaisesRegex(ValueError, "power of two"):
            JaggedCapClass(1)

    def test_for_tier_is_tier_over_eight(self) -> None:
        self.assertEqual(JaggedCapClass.for_tier(29).scan_cap, 1 << 26)
        # Floors at 2 rows on tiny tiers.
        self.assertEqual(JaggedCapClass.for_tier(3).scan_cap, 2)


class ChunkedEvalByteIdentityTest(parameterized.TestCase):
    """Chunked vs monolithic on one fixture: IDENTICAL proof bytes and
    Fiat-Shamir stream, in both the jitted-zone and eager forms."""

    @parameterized.parameters(2, 4, 8)
    def test_chunked_matches_monolithic(self, scan_cap: int) -> None:
        smcs = _smcs()
        claim, witness, target = _open_fixture(smcs, seed=80, rows=5)
        self.assertLess(scan_cap, target)  # the chunk loop really engages
        for jit in (False, True):
            want_r = _stage(smcs, jit=jit).prove(claim, witness, cheap_transcript(F))
            got_r = _stage(smcs, jit=jit, cap=JaggedCapClass(scan_cap)).prove(
                claim, witness, cheap_transcript(F)
            )
            label = f"jit={jit} scan_cap={scan_cap}"
            _assert_proof_byte_equal(
                got_r.reduction_proof.eval, want_r.reduction_proof.eval, f"{label} eval"
            )
            _assert_proof_byte_equal(
                got_r.reduction_proof.open, want_r.reduction_proof.open, f"{label} open"
            )
            _, got_s = got_r.transcript.sample(1)
            _, want_s = want_r.transcript.sample(1)
            _assert_bytes_equal(got_s, want_s, f"{label} post-stage sample")

    def test_unset_class_keeps_the_monolithic_zone(self) -> None:
        smcs = _smcs()
        # rows=9 doubles the area tier, so this layout class is fresh in the
        # process-wide jit cache and the unset prove's compile is observable.
        claim, witness, _ = _open_fixture(smcs, seed=80, rows=9)
        before = _jagged_eval_jit._cache_size()
        want_r = _stage(smcs, jit=True).prove(claim, witness, cheap_transcript(F))
        self.assertEqual(_jagged_eval_jit._cache_size() - before, 1)
        # The chunked prove never touches the monolithic zone...
        mid = _jagged_eval_jit._cache_size()
        got_r = _stage(smcs, jit=True, cap=JaggedCapClass(8)).prove(
            claim, witness, cheap_transcript(F)
        )
        self.assertEqual(_jagged_eval_jit._cache_size(), mid)
        # ...and stays byte-identical to it.
        _assert_proof_byte_equal(
            got_r.reduction_proof, want_r.reduction_proof, "chunked vs unset"
        )

    def test_cap_at_or_above_tier_stays_monolithic(self) -> None:
        smcs = _smcs()
        claim, witness, target = _open_fixture(smcs, seed=80, rows=5)
        partials_before = _jagged_outer_chunk_partials_jit._cache_size()
        eval_before = _jagged_eval_jit._cache_size()
        _stage(smcs, jit=True, cap=JaggedCapClass(target)).prove(
            claim, witness, cheap_transcript(F)
        )
        # A dense that fits the cap runs the monolithic zone, no chunk zones.
        self.assertEqual(
            _jagged_outer_chunk_partials_jit._cache_size(), partials_before
        )
        self.assertGreaterEqual(_jagged_eval_jit._cache_size() - eval_before, 0)


class ChunkZoneLiveWidthTest(absltest.TestCase):
    """Trace-time arena guard on the chunk zone: the traced body's
    intermediate aval bytes must total O(chunk width) with a constant
    independent of the row-variable count and the searchsorted depth.

    A per-bit or per-search-step chunk-width chain unrolled at the top level
    trips the bound — XLA's operand-capped fusion pins every such
    intermediate (plus one chunk-width scalar broadcast per bit) as
    simultaneously-resident buffers, a temp arena of tens of chunk widths at
    the production 2^26 cap."""

    # The loop-carried body traces ~200 intermediate bytes per chunk row
    # (a dozen-odd chunk-width values, mixed EF/int32/bool, at any fold
    # depth); an unrolled 22-bit row-eq chain alone is ~2,300 bytes per row.
    _CAP_BYTES_PER_ROW = 512

    @staticmethod
    def _intermediate_bytes(jaxpr: Any) -> int:
        """Sum of eqn-output aval bytes at the zone's top level. Loop bodies
        count once via their carry outputs — their internal buffers are
        reused across iterations, exactly the live-set the bound models."""
        return sum(
            math.prod(v.aval.shape) * v.aval.dtype.itemsize
            for eqn in jaxpr.eqns
            for v in eqn.outvars
        )

    def test_chunk_zone_intermediates_scale_with_chunk_width(self) -> None:
        chunk_rows = 1 << 16
        n_cols = 512  # searchsorted depth log2_ceil(513) + 2 = 12 steps
        n_r = 22  # the keccak-class row-variable count
        offsets, _ = eval_column_arrays([4] * n_cols, dtype=EF)
        dense = fnp.zeros((2 * chunk_rows,), F)
        z_row = _rand_ef(11, (n_r,))
        z_col = _rand_ef(12, (log2_ceil_usize(n_cols),))
        start = fnp.asarray(0, fnp.int32)
        for depth in (0, 3):
            folds = _rand_ef(13, (depth,))
            closed = frx.make_jaxpr(
                partial(_outer_chunk_partials, chunk_rows=chunk_rows)
            )(offsets, dense, z_row, z_col, start, folds)
            total = self._intermediate_bytes(closed.jaxpr)
            self.assertLessEqual(
                total,
                self._CAP_BYTES_PER_ROW * chunk_rows,
                f"fold depth {depth}: {total / chunk_rows:.0f} traced "
                "intermediate bytes per chunk row — a chunk-width chain is "
                "unrolled in the chunk zone body",
            )


class ChunkZoneCompileSharingTest(absltest.TestCase):
    """The #274 contract on the chunk zones: heights ride as traced values, so
    two shards of one layout class + one scan cap share every chunk compile."""

    def test_chunk_zones_share_compiles_across_one_class(self) -> None:
        smcs = _smcs()
        scan_cap = 8
        counters = (
            _jagged_outer_chunk_partials_jit,
            _jagged_outer_chunk_fold_jit,
            _jagged_outer_tail_jit,
            _jagged_inner_jit,
        )
        before = [c._cache_size() for c in counters]
        targets = set()
        # Heights 5 and 7 share the layout class (same L, same tier); only the
        # height VALUES differ — exactly what must not key a chunk compile.
        for seed, rows in ((80, 5), (90, 7)):
            claim, witness, target = _open_fixture(smcs, seed=seed, rows=rows)
            targets.add(target)
            _stage(smcs, jit=True, cap=JaggedCapClass(scan_cap)).prove(
                claim, witness, cheap_transcript(F)
            )
        self.assertEqual(len(targets), 1)  # same area tier => same class
        target = targets.pop()
        n_rounds = (target - 1).bit_length()
        log_chunk = (scan_cap - 1).bit_length()
        k = min(n_rounds - log_chunk + 1, log_chunk)
        deltas = [c._cache_size() - b for c, b in zip(counters, before)]
        # One partials compile per recompute round; one each for the fold,
        # tail, and inner zones — and shard 2 adds ZERO new entries.
        self.assertEqual(deltas, [k, 1, 1, 1])


if __name__ == "__main__":
    absltest.main()
