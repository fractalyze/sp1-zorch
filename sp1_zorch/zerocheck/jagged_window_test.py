# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-identity of PER-CHIP round windows vs the round-uniform ones.

`TotalCapClass.chip_height_caps` narrows each chip's round window from the
round-uniform machine half-width (`eq_widths`) to that chip's own live pairs
(`chip_eq_widths`). The narrowing is only legal because the rows it drops are
exact field zeros:

- `constraint_eval`'s ``live_width`` mask is applied LAST and covers the whole
  per-row value (constraint RLC + column dot), so every row at index >= the
  chip's live pairs is the field's zero — a property the round-uniform path
  ALREADY depends on, since it sums those rows in today.
- The reduce is ``sum(v * eq)`` over a finite field, where addition is exact
  and associative, so dropping zero terms cannot move a single bit.
- ``eq[threshold_half]`` (the padding correction's last-live-row Lagrange) is
  read at ``max(live_pair - 1, 0)``, strictly inside the narrowed window.

So this suite asserts uint32 equality — round polys, challenges, AND finals —
between a class carrying per-chip caps and the same class without them, at the
real `prove_jagged_zerocheck` call site, across: tight caps, loose (class
union) caps, odd heights, height 1, runtime-empty and statically-empty chips,
constraint-free chips, mixed column counts, extension-field challenges over a
base-field arrival, ``num_vars`` past the unrolled prefix (so the TAIL SCAN's
single-shape body is exercised), and the chunked prefix's monolithic
remainder. One case re-anchors to the exact static path.

Two structural guards ride along: the narrowed run must add NO distinct
constraint-kernel key (`_round_constraint_eval_cached`'s statics), and two
shards of one class must share every compile.

Mirrors `jagged_chunk_test`: no tolerances, Montgomery-form u32 equality only.
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
    chip_eq_widths,
    eq_widths,
    pack_flat_arrival,
    prove_jagged_zerocheck,
)

# Witness chip: columns [a, b, c] with a == 1 on every real row, so both
# constraints vanish there while C(0_row) != 0 keeps the padded-row correction
# live (the same fixture shape as jagged_test / jagged_chunk_test).
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
    ``beta**(j+1)``-weighted column MLE openings at zeta."""
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


class _KeyRecorder:
    """Records every constraint-kernel key the round bodies emit.

    The key is exactly `_round_constraint_eval_cached`'s static triple
    ``(eval_fn, window_rows, num_cols)`` — what a compiled constraint kernel is
    identified by. Counting DISTINCT triples before and after narrowing is the
    direct test of "no new kernels": a narrowed window that split a key two
    chips shared would show up as a higher count.
    """

    def __init__(self) -> None:
        self.keys: list[tuple[Any, int, int]] = []

    def __enter__(self) -> "_KeyRecorder":
        self._orig = jagged._round_constraint_eval_cached

        def spy(*args: Any, window_rows: int, num_cols: int, **kw: Any) -> Any:
            self.keys.append((args[0], window_rows, num_cols))
            return self._orig(*args, window_rows=window_rows, num_cols=num_cols, **kw)

        jagged._round_constraint_eval_cached = spy
        return self

    def __exit__(self, *exc: Any) -> None:
        jagged._round_constraint_eval_cached = self._orig

    @property
    def distinct(self) -> int:
        return len({(id(f), w, c) for f, w, c in self.keys})

    @property
    def window_rows_total(self) -> int:
        """Total window rows the round bodies asked constraint_eval for —
        the quantity per-chip windows exist to cut."""
        return sum(w for _, w, _ in self.keys)

    def windows_by_body(self, n_live: int) -> list[list[int]]:
        """The per-live-chip window each traced round body used, in body order.

        A round body emits ``n_live * 3`` calls — chips in order, 3 t-points
        each — and the driver traces one body per unrolled prefix round plus
        ONE for the tail scan, so this recovers the emitted window schedule
        without reaching into the driver."""
        per_body = n_live * 3
        assert len(self.keys) % per_body == 0, (len(self.keys), per_body)
        out = []
        for b in range(len(self.keys) // per_body):
            chunk = self.keys[b * per_body : (b + 1) * per_body]
            for i in range(n_live):
                assert {k[1] for k in chunk[i * 3 : i * 3 + 3]} == {chunk[i * 3][1]}
            out.append([chunk[i * 3][1] for i in range(n_live)])
        return out


class PerChipWindowByteIdentityTest(parameterized.TestCase):
    """uint32-exact equality of per-chip windows vs the round-uniform ones."""

    def _run_pair(
        self,
        num_vars: int,
        heights: Sequence[int | None],
        caps: Sequence[int],
        *,
        num_cols: Sequence[int] | None = None,
        constraint_free: frozenset[int] = frozenset(),
        ef_challenges: bool = False,
        chunk_depth: int = 0,
        also_static_oracle: bool = False,
    ) -> tuple[_KeyRecorder, _KeyRecorder]:
        """``heights`` entries are ints (traced at the driver seam) or None for
        a STATICALLY empty chip (host 0). Runs the same shard twice on the same
        arrival — once with ``chip_height_caps`` on the class, once without —
        and demands byte equality. Returns the (uniform, narrowed) key
        recorders so callers can assert on the emitted kernel keys."""
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
        beta = rand_c(77, ())
        zeta = rand_c(7, (num_vars,))
        challenges = [rand_c(1000 + r, ()) for r in range(num_vars)]
        summand = JaggedZerocheckSummand(
            eval_fns=eval_fns,
            alphas=alphas,
            lambdas=rand_c(55, (nchips,)),
            beta=beta,
            public_values=_PV,
        )
        area = TotalCapClass.from_heights(h_ints, cols).area_cap
        uniform = TotalCapClass(area_cap=area)
        narrowed = TotalCapClass(area_cap=area, chip_height_caps=tuple(caps))
        flat = pack_flat_arrival(traces, h_ints, uniform, num_vars)
        # None stays a HOST 0 (statically empty chip); ints ride traced.
        num_reals = [0 if h is None else fnp.asarray(h, fnp.int32) for h in heights]
        chunk = (
            TotalCapChunkClass(depth=chunk_depth, chip_height_caps=tuple(caps))
            if chunk_depth
            else None
        )

        def run(cap_class: TotalCapClass) -> Any:
            return prove_jagged_zerocheck(
                summand,
                [],
                num_reals,
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=_claims(beta, traces, zeta),
                total_cap_class=cap_class,
                flat_arrival=flat,
                num_cols=cols,
                chunk_class=chunk,
            )

        with _KeyRecorder() as rec_w:
            finals_w, _, want = run(uniform)
        with _KeyRecorder() as rec_g:
            finals_g, _, got = run(narrowed)

        # The byte comparison alone cannot see a driver-level window bug: a
        # mis-sized window in `make_round_step`'s call sites narrows BOTH arms
        # identically and cancels out. So assert the emitted windows directly
        # against the soundness condition — every window must still cover its
        # chip's live pairs at the round it serves (the SCAN body serves rounds
        # `unroll..num_vars-1`, and round `unroll` is its binding one).
        # Only on the monolithic path: the chunked prefix's zones and its
        # remainder are their own jit programs, so which of them RE-TRACES on
        # the second arm depends on process-wide cache state and the recorded
        # body list is not a fixed schedule.
        if not chunk_depth:
            live_idx = [i for i in range(nchips) if heights[i] is not None]
            unroll = min(jagged._SHRINK_ROUNDS, num_vars)
            for rec, name in ((rec_w, "uniform"), (rec_g, "narrowed")):
                bodies = rec.windows_by_body(len(live_idx))
                self.assertLen(bodies, unroll + (unroll < num_vars))
                for rnd, widths in enumerate(bodies):
                    rnd = min(rnd, unroll)
                    for j, w in zip(live_idx, widths, strict=True):
                        live_pairs = -(-h_ints[j] // (1 << (rnd + 1)))
                        self.assertGreaterEqual(
                            w, live_pairs, f"{name} round {rnd} chip {j}"
                        )
                        self.assertLessEqual(w, eq_widths(num_vars, rnd)[rnd])

        label = f"nv={num_vars} heights={h_ints} caps={list(caps)}"
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
                claims=_claims(beta, traces, zeta),
            )
            _assert_bytes_equal(got.round_poly, want_s.round_poly, "static polys")
            for i, (fg, fs) in enumerate(zip(finals_g, finals_s, strict=True)):
                _assert_bytes_equal(fg, fs, f"static finals[{i}]")
        return rec_w, rec_g

    def test_tight_caps(self) -> None:
        # Caps == heights: the most aggressive narrowing a legal class allows.
        self._run_pair(4, [5, 2], [5, 2])

    def test_loose_class_union_caps(self) -> None:
        # The production shape: caps are a union over shards, above every
        # height here.
        self._run_pair(4, [5, 2], [8, 4])

    def test_static_oracle_anchor(self) -> None:
        self._run_pair(4, [5, 3], [8, 4], also_static_oracle=True)

    def test_odd_heights_and_height_one(self) -> None:
        self._run_pair(4, [5, 3, 1], [7, 3, 1])

    def test_runtime_empty_chip(self) -> None:
        # Traced height 0 with a nonzero cap: the chip stays live, its window
        # is all dead rows, and the `nr_live == 0` clamp reads eq[0].
        self._run_pair(4, [5, 0], [8, 4])

    def test_runtime_empty_chip_zero_cap(self) -> None:
        # Cap 0 too: the window floors at 2 (the eq_widths XLA workaround).
        self._run_pair(4, [5, 0], [8, 0])

    def test_statically_empty_chip_among_live(self) -> None:
        self._run_pair(4, [5, None, 3], [8, 0, 4])

    def test_constraint_free_chip(self) -> None:
        self._run_pair(4, [5, 8, 3], [8, 8, 4], constraint_free=frozenset({1}))

    def test_tail_scan_runs(self) -> None:
        # num_vars > _SHRINK_ROUNDS: rounds 5.. run in the fixed-shape SCAN,
        # whose single body takes each chip's round-5 window for every tail
        # round. Live pairs only halve from there, so that one window bounds
        # them all — this is the case the prefix cannot cover.
        self.assertGreater(7, jagged._SHRINK_ROUNDS)
        self._run_pair(7, [100, 9, 33], [128, 16, 48])

    def test_tail_scan_bound_is_tight(self) -> None:
        # The tail scan's bound, with NO slack — the case that actually pins
        # it. At num_vars=7 the tail window floors at 2 and every live-pair
        # count is 1 or 2, so a mis-sized tail window hides. Here num_vars=9
        # and heights sit at the cap, so round 5's live pairs EQUAL the window
        # (8 and 5): narrowing the tail body by even one row drops a live row.
        nv, rounds = 9, jagged._SHRINK_ROUNDS
        for cap, h in ((512, 500), (320, 300)):
            self.assertEqual(
                chip_eq_widths(cap, nv, rounds)[rounds], -(-h // (1 << (rounds + 1)))
            )
        self._run_pair(nv, [500, 300], [512, 320])

    def test_extension_field_challenges(self) -> None:
        # The production dtype mix: base-field arrival, EF challenges.
        self._run_pair(4, [5, 2], [8, 4], ef_challenges=True)

    def test_chunked_prefix_remainder(self) -> None:
        # The chunked prefix owns rounds 0..d-1 with its OWN per-chip windows;
        # the class caps must narrow the monolithic remainder it hands off to
        # (`_totalcap_tail`) without moving a byte.
        self._run_pair(7, [100, 9, 33], [128, 16, 48], chunk_depth=2)

    def test_keccak_class_shaped(self) -> None:
        # Structurally keccak-class-shaped: many chips, mixed column counts
        # (wide + narrow), constraint-free chips among constrained ones, odd /
        # power / height-1 / runtime-empty / statically-empty heights, caps
        # above heights (class union), a prefix + tail scan, EF challenges.
        rec_w, rec_g = self._run_pair(
            7,
            [100, 37, 64, 5, 0, 1, 90, 33, None],
            [128, 40, 64, 8, 4, 2, 96, 48, 0],
            num_cols=[3, 3, 3, 3, 3, 3, 8, 1, 3],
            constraint_free=frozenset({6, 7}),
            ef_challenges=True,
        )
        # The point of the unit, in the emitted numbers.
        self.assertLess(rec_g.window_rows_total, rec_w.window_rows_total)

    def test_no_new_kernel_keys(self) -> None:
        # Constraint 3, measured: narrowing must not raise the number of
        # distinct (eval_fn, window_rows, num_cols) triples the round bodies
        # emit — that triple IS what a compiled constraint kernel is keyed on.
        for kwargs in (
            {"num_vars": 7, "heights": [100, 9, 33], "caps": [128, 16, 48]},
            {
                "num_vars": 7,
                "heights": [100, 37, 5, 90, 33],
                "caps": [128, 40, 8, 96, 48],
                "constraint_free": frozenset({3, 4}),
                "num_cols": [3, 3, 3, 8, 1],
            },
        ):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                rec_w, rec_g = self._run_pair(**kwargs)  # type: ignore[arg-type]
                self.assertEqual(len(rec_g.keys), len(rec_w.keys))
                self.assertLessEqual(rec_g.distinct, rec_w.distinct)

    def test_shared_kernel_key_keeps_one_window(self) -> None:
        # The adversarial case for constraint 3: two constraint-free chips of
        # EQUAL width share one kernel key today (eval_fn is None for both).
        # Different caps must NOT split it — `_grouped_chip_windows` gives the
        # group one window, the max of its members.
        rec_w, rec_g = self._run_pair(
            7,
            [100, 9],
            [128, 16],
            constraint_free=frozenset({0, 1}),
        )
        self.assertEqual(rec_g.distinct, rec_w.distinct)
        # Both chips took the wider (cap=128) schedule, so each round body
        # emits ONE window width, not two.
        by_round: dict[int, set[int]] = {}
        for idx, (_, w, _) in enumerate(rec_g.keys):
            by_round.setdefault(idx // 6, set()).add(w)  # 2 chips * 3 t-points
        self.assertLen(by_round, 6)  # 5 unrolled prefix rounds + the scan body
        for widths_in_round in by_round.values():
            self.assertLen(widths_in_round, 1)
        self.assertEqual(
            [next(iter(by_round[r])) for r in range(6)],
            chip_eq_widths(128, 7, jagged._SHRINK_ROUNDS),
        )

    def test_under_bounding_cap_fails_loud(self) -> None:
        # An under-bounding cap would narrow a chip's window past its live
        # pairs and silently drop live rows: it must fail at dispatch instead
        # (the #351 clamped-DUS lesson). Static heights: checked in the driver.
        heights = [5, 2]
        cols = [_NUM_COLS] * 2
        traces = [_witness_trace(11 + i, heights[i]) for i in range(2)]
        beta = _rand(77, ())
        zeta = _rand(7, (4,))
        summand = JaggedZerocheckSummand(
            eval_fns=[_eval_fn] * 2,
            alphas=[rlc_coeffs(_rand(99 + i, ()), _K) for i in range(2)],
            lambdas=_rand(55, (2,)),
            beta=beta,
            public_values=_PV,
        )
        cls = TotalCapClass(
            area_cap=TotalCapClass.from_heights(heights, cols).area_cap,
            chip_height_caps=(4, 4),
        )
        with self.assertRaisesRegex(ValueError, "does not bound chip 0"):
            prove_jagged_zerocheck(
                summand,
                traces,
                heights,
                zeta,
                _ScriptedTranscript.replaying([_rand(1000 + r, ()) for r in range(4)]),
                claims=_claims(beta, traces, zeta),
                total_cap_class=cls,
                num_cols=None,
            )

    def test_cap_count_mismatch_fails_loud(self) -> None:
        heights = [5, 2]
        cols = [_NUM_COLS] * 2
        traces = [_witness_trace(11 + i, heights[i]) for i in range(2)]
        beta = _rand(77, ())
        zeta = _rand(7, (4,))
        summand = JaggedZerocheckSummand(
            eval_fns=[_eval_fn] * 2,
            alphas=[rlc_coeffs(_rand(99 + i, ()), _K) for i in range(2)],
            lambdas=_rand(55, (2,)),
            beta=beta,
            public_values=_PV,
        )
        cls = TotalCapClass(
            area_cap=TotalCapClass.from_heights(heights, cols).area_cap,
            chip_height_caps=(8,),
        )
        with self.assertRaisesRegex(ValueError, "1 chip height caps for 2 chips"):
            prove_jagged_zerocheck(
                summand,
                traces,
                heights,
                zeta,
                _ScriptedTranscript.replaying([_rand(1000 + r, ()) for r in range(4)]),
                claims=_claims(beta, traces, zeta),
                total_cap_class=cls,
            )

    def test_shards_of_one_class_share_compiles(self) -> None:
        # Constraint 2, measured: the windows come from the CLASS, so a second
        # shard of the same class (different heights) must add ZERO compiles.
        nchips = 2
        cols = [_NUM_COLS] * nchips
        beta = _rand(77, ())
        zeta = _rand(7, (7,))
        challenges = [_rand(1000 + r, ()) for r in range(7)]
        summand = JaggedZerocheckSummand(
            eval_fns=[_eval_fn] * nchips,
            alphas=[rlc_coeffs(_rand(99 + i, ()), _K) for i in range(nchips)],
            lambdas=_rand(55, (nchips,)),
            beta=beta,
            public_values=_PV,
        )
        cls = TotalCapClass(area_cap=1024, chip_height_caps=(128, 16))

        def prove(heights: list[int]) -> Any:
            traces = [_witness_trace(11 + i, h) for i, h in enumerate(heights)]
            flat = pack_flat_arrival(traces, heights, cls, 7)
            return prove_jagged_zerocheck(
                summand,
                [],
                [fnp.asarray(h, fnp.int32) for h in heights],
                zeta,
                _ScriptedTranscript.replaying(challenges),
                claims=_claims(beta, traces, zeta),
                total_cap_class=cls,
                flat_arrival=flat,
                num_cols=cols,
            )

        prove([100, 9])
        before = jagged._round_constraint_eval_cached._cache_size()
        prove([64, 15])
        self.assertEqual(before, jagged._round_constraint_eval_cached._cache_size())


class ChipEqWidthsTest(absltest.TestCase):
    """The window schedule itself — host arithmetic, production scale."""

    def test_bounds_live_pairs_every_round(self) -> None:
        # The soundness condition, exhaustively: for every height at or below
        # the cap and every round, the window must still cover the chip's live
        # pair count ceil(h / 2**(r+1)) — otherwise a live row would leave the
        # reduce.
        for cap in (0, 1, 2, 3, 5, 8, 33, 64, 127, 1024):
            wins = chip_eq_widths(cap, 12, jagged._SHRINK_ROUNDS)
            for h in range(0, cap + 1):
                for r, w in enumerate(wins):
                    live_pairs = -(-h // (1 << (r + 1)))
                    self.assertGreaterEqual(w, live_pairs, f"cap={cap} h={h} r={r}")
                    # And the padding correction's eq index stays in range.
                    self.assertLess(max(live_pairs - 1, 0), w)

    def test_never_wider_than_uniform_and_floored_at_two(self) -> None:
        uniform = eq_widths(12, jagged._SHRINK_ROUNDS)
        for cap in (0, 1, 7, 64, 1 << 20):
            wins = chip_eq_widths(cap, 12, jagged._SHRINK_ROUNDS)
            self.assertLen(wins, len(uniform))
            for w, u in zip(wins, uniform, strict=True):
                self.assertLessEqual(w, u)
                self.assertGreaterEqual(w, 2)

    def test_production_scale_window_rows(self) -> None:
        # MAX_LOG_ROW_COUNT = 22 (shard_prover/replay.py): the round-uniform
        # schedule charges every live chip 5,177,344 window rows per shard —
        # 2**21..2**17 for the 5 unrolled prefix rounds plus 17 tail rounds at
        # 2**16. That total is what per-chip windows cut.
        nv, rounds = 22, jagged._SHRINK_ROUNDS
        uniform = eq_widths(nv, rounds)
        total = sum(uniform[:rounds]) + (nv - rounds) * uniform[rounds]
        self.assertEqual(total, 5_177_344)
        # A chip capped at 2**k pays 2**(22-k) times less: the schedule is the
        # uniform one shifted down, so the saving is the whole ratio of the
        # chip's cap to the machine width.
        for cap, want in ((1 << 21, 2_588_672), (1 << 16, 80_896), (1024, 1_264)):
            w = chip_eq_widths(cap, nv, rounds)
            self.assertEqual(sum(w[:rounds]) + (nv - rounds) * w[rounds], want)
            self.assertEqual(total // want, (1 << nv) // cap)


if __name__ == "__main__":
    absltest.main()
