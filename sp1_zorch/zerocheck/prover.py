# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1's zerocheck stage: the shard-prover glue around the jagged round engine.

Everything here is derivation, not proving: the three stage challenges in
SP1's order (alpha constraint batching -> beta GKR opening batch -> chip-RLC
lambda; the mapping lives on ``sample_stage_challenges``),
zeta as the row tail of the GKR evaluation point, each chip's GKR opening
claim as the beta-power weighting of its ``[main | prep]`` column openings,
the per-chip column-major traces sliced out of the committed regions, and the
stage's transcript tail — the per-chip opened values absorbed via
``OpenedValuesRound`` before any evaluation-stage sampling. The round engine
(`prove_jagged_zerocheck`) owns the sumcheck itself.

Reference: whir-zorch ``sp1/shard_prover/prover.py``, its zerocheck (SP1
"phase 3") block, mirroring SP1's schedule —
https://github.com/fractalyze/sp1/blob/640d8b80c/crates/hypercube/src/prover/shard.rs
Stage / dump vocabulary: ``docs/architecture.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import frx
import frx.numpy as fnp
from frx import Array
from rw_constraints import Chip
from zk_dtypes import efinfo
from zorch.pcs.jagged.region import JaggedRegion
from zorch.round import ProverRound
from zorch.stage import ProveResult, ProverStage
from zorch.transcript import Transcript, sample_challenge

from sp1_zorch.logup_gkr.prover import (
    flat_openings_absorb,
    select_openings,
)
from sp1_zorch.types import (
    ChipEvaluation,
    ShardWitness,
    TraceEvaluationClaim,
    ZerocheckClaim,
    ZerocheckProof,
)
from sp1_zorch.zerocheck.coeffs import gkr_powers, rlc_coeffs
from sp1_zorch.zerocheck.jagged import (
    JaggedZerocheckSummand,
    TotalCapChunkClass,
    TotalCapClass,
    pack_flat_arrival,
    prove_jagged_zerocheck,
)


def chip_traces(
    chip_names: Sequence[str],
    num_reals: Sequence[int],
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
) -> list[Array]:
    """Per-chip column-major ``[main | prep]`` traces, exactly ``nr`` rows each.

    Main-first matches the GKR claim's beta-weighting (the claims batch
    ``concat([main_eval, prep_eval])``); prep is height-padded / truncated to
    the chip's ``num_real``. The round driver owns all further padding.
    """
    bf = main_region.dense.dtype
    prep_idx = (
        {n: k for k, n in enumerate(prep_region.chip_names)} if prep_region else {}
    )
    traces = []
    for i, name in enumerate(chip_names):
        nr = int(num_reals[i])
        mw = int(main_region.chip_widths[i])
        start = main_region.chip_starts[i]
        if mw > 0 and nr > 0:
            cols = main_region.dense[start : start + nr * mw].reshape(mw, nr)
        else:
            cols = fnp.zeros((mw, nr), dtype=bf)
        if prep_region is not None and name in prep_idx:
            k = prep_idx[name]
            pw = int(prep_region.chip_widths[k])
            p_h = int(prep_region.chip_heights[k])
            p_start = prep_region.chip_starts[k]
            if pw > 0 and p_h > 0:
                prep = prep_region.dense[p_start : p_start + p_h * pw].reshape(pw, p_h)
                if p_h > nr:
                    prep = prep[:, :nr]
                else:
                    prep = fnp.pad(prep, ((0, 0), (0, nr - p_h)))
            else:
                prep = fnp.zeros((pw, nr), dtype=bf)
            if pw > 0:
                cols = fnp.concatenate([cols, prep], axis=0)
        traces.append(cols)
    return traces


# Value-keyed memo so a chip's 2-ary eval_fn has a STABLE identity across
# prove calls: the identity is the jit-zone static key of the round body's
# cached `constraint_eval` trace (`jagged._round_constraint_eval_cached`) and
# the probe memo key below — a fresh closure per prove call would bust both
# caches process-wide. Chips live for the process (the loader registry), so
# identity-keying is sound; strong refs match that lifetime.
_EVAL_FN_MEMO: dict[
    tuple[Chip, int, int, Optional[int]], Callable[[Array, Array], Array]
] = {}


def export_order_eval_fn(
    chip: Chip, main_width: int, num_cols: int, block_idx: Optional[int] = None
) -> Callable[[Array, Array], Array]:
    """The chip's 2-ary ``eval_constraints`` accepting ``[main | prep]`` rows.

    rw-constraints exports evaluate a flat trace in the exporter's
    ``[preprocessed | main]`` column order (the recursion Select constraint
    reads its 5 main value columns at flat indices 8..12 of 13), while the
    zerocheck's traces, opened values, and beta column batch all follow SP1's
    wire order ``[main | prep]``. This seam rotates each row into export
    order before evaluating — the single place the two conventions meet, for
    the prover's summand and the verifier dual alike. A main-only chip
    (``num_cols == main_width``) needs no rotation; the closure carries only
    static widths, so it is legal under the jitted stage bodies.

    ``block_idx`` selects one SP1 ``BlockAir`` block instead of the whole AIR
    (``Chip.num_blocks`` > 1 only for KeccakPermute and the Weierstrass
    add/double chips). Block ``b``'s constraint columns are the contiguous
    slice of the whole chip's, in order, so the blocks' α-folded values sum to
    the unsplit one bit for bit — see ``block_split_test.py``. Each block gets
    its own memo entry, which is what gives it a STABLE identity for the
    round-body jit-zone key.
    """
    key = (chip, main_width, num_cols, block_idx)
    hit = _EVAL_FN_MEMO.get(key)
    if hit is not None:
        return hit

    fn: Callable[[Array, Array], Array]
    if block_idx is None:
        fn = chip.eval_constraints
    else:
        b = block_idx

        def block_fn(rows: Array, public_values: Array) -> Array:
            return chip.eval_constraints(rows, public_values, block_idx=b)

        fn = block_fn

    if num_cols == main_width:
        _EVAL_FN_MEMO[key] = fn
        return fn

    def eval_fn(rows: Array, public_values: Array) -> Array:
        export_rows = fnp.concatenate(
            [rows[..., main_width:], rows[..., :main_width]], axis=-1
        )
        return fn(export_rows, public_values)

    _EVAL_FN_MEMO[key] = eval_fn
    return eval_fn


def bind_pv(chip: Chip, public_values: Array) -> Callable[[Array], Array]:
    """Bind the public-values vector; ``eval_constraints`` ignores it for
    constraints that declare no ``pv_arg``. Shared by the stage and its
    verifier dual — the one definition of how a chip's constraint circuit
    sees the statement."""
    return lambda trace: chip.eval_constraints(trace, public_values)


# Host-side memo: the count depends only on the circuit and the probe SHAPES,
# never the statement's values, and the un-memoized probe re-traced the whole
# constraint circuit into the ENCLOSING jit trace (dead equations — only
# .shape[-1] is read) on every stage trace. Keyed on the eval_fn identity
# (stable via _EVAL_FN_MEMO / bound-method equality) + the probe avals.
_PROBE_MEMO: dict[tuple[Any, int, Any, tuple[int, ...], Any], int] = {}


def probe_num_constraints(
    eval_fn: Callable[[Array, Array], Array],
    width: int,
    ef: Any,
    public_values: Array,
) -> int:
    """A chip's constraint count, from a one-row zero probe — the constraint
    functions may emit several columns each, so the count is not readable
    off the manifest. One definition: it sizes the constraint-RLC fold on
    both the prover and the verifier dual. ``eval_fn`` is the chip's 2-ary
    ``eval_constraints``; the statement is threaded, not closed over.

    The probe runs under ``frx.eval_shape`` on abstract inputs (memoized on
    the eval_fn + probe avals), so the circuit never enters an enclosing
    trace — the count is shape metadata, not a computation."""
    key = (
        eval_fn,
        width,
        ef,
        tuple(public_values.shape),
        public_values.dtype,
    )
    hit = _PROBE_MEMO.get(key)
    if hit is None:
        out = frx.eval_shape(
            eval_fn,
            frx.ShapeDtypeStruct((1, width), ef),
            frx.ShapeDtypeStruct(tuple(public_values.shape), public_values.dtype),
        )
        hit = int(out.shape[-1])
        _PROBE_MEMO[key] = hit
    return hit


def sample_stage_challenges(
    transcript: Transcript, ef: Any
) -> tuple[Transcript, Array, Array, Array]:
    """The three zerocheck stage challenges in SP1's order, one per batching
    dimension: ``batching`` is alpha — one RLC across a chip's K constraints
    (``coeffs.rlc_coeffs``); ``gkr_batch`` is beta — across a chip's columns
    (``coeffs.gkr_powers``); ``lambda_`` batches across chips (the jagged
    engine re-applies it every round; a chip index is a batch axis, not a
    sumcheck variable). Sampled inside zerocheck, after the GKR stage. One
    definition driven by the prover and the verifier dual, so the sampling
    schedule cannot drift between their Fiat-Shamir streams."""
    limbs = efinfo(ef).degree
    transcript, batching = sample_challenge(transcript, ef, limbs)
    transcript, gkr_batch = sample_challenge(transcript, ef, limbs)
    transcript, lambda_ = sample_challenge(transcript, ef, limbs)
    return transcript, batching, gkr_batch, lambda_


def gkr_opening_claims(openings: Sequence[ChipEvaluation], gkr_batch: Array) -> Array:
    """Each chip's GKR opening claim: its ``[main | prep]`` evaluations
    weighted by the shared beta powers — the seed of the round engine's
    ``p(1) = claim - p(0)`` identity. One definition: the prover seeds the
    sumcheck with these, the verifier dual re-derives its claimed sum from
    them."""
    evals = [opening.all_evals() for opening in openings]
    max_cols = max(e.shape[0] for e in evals)
    gkr_all = (
        gkr_powers(gkr_batch, max_cols) if max_cols else fnp.zeros(0, gkr_batch.dtype)
    )
    return fnp.stack([fnp.sum(gkr_all[: e.shape[0]] * e) for e in evals])


def split_opened_values(
    finals: Sequence[Array],
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
) -> dict[str, ChipEvaluation]:
    """Split the stage's final folded traces into per-chip opened values.

    ``finals[c]`` stacks chip ``c``'s ``[main | prep]`` columns (the
    ``chip_traces`` order) with each column's evaluation at the sumcheck
    point in position 0. The split is the shared view of the openings: the
    stage's transcript absorbs, the jagged-eval stage's per-column claims,
    and the wire's ``ShardOpenedValues`` all read it."""
    prep_widths = (
        dict(zip(prep_region.chip_names, prep_region.chip_widths, strict=True))
        if prep_region
        else {}
    )
    opened = {}
    for i, name in enumerate(main_region.chip_names):
        final = finals[i]
        # A zero-variable run folds nothing; position 0 only exists when the
        # buffer kept its live pair.
        evals = (
            final[:, 0]
            if final.shape[1] > 0
            else fnp.zeros((final.shape[0],), dtype=final.dtype)
        )
        mw = int(main_region.chip_widths[i])
        pw = prep_widths.get(name, 0)
        opened[name] = ChipEvaluation(
            main=evals[:mw],
            preprocessed=evals[mw : mw + pw] if pw else None,
        )
    return opened


class OpenedValuesRound:
    """SP1's post-zerocheck opened-values absorb stream: the chip count, then
    per chip the length-prefixed ``[preprocessed | main]`` evaluations at the
    sumcheck point. Every evaluation-stage challenge is sampled after these
    absorbs, so the schedule lives here once (the same single-source rule as
    the shard preamble and the GKR head): ``prove_shard_zerocheck`` drives it
    for every stage consumer, and the verifier dual will absorb the proof's
    opened values through the same Round. A chip with no preprocessed trace
    absorbs a bare zero length, matching SP1's empty-Vec framing — the one
    knob on the shared ``flat_openings_absorb`` (the GKR chip-openings
    framing absorbs nothing there). ``chip_names`` fixes the absorb order —
    the caller's statement, never the mapping's own iteration order, which
    is proof-controlled once the verifier dual drives this Round.
    Carry-agnostic; the message is the opened values, the wire's
    structure-bound payload."""

    def __init__(
        self, opened_values: Mapping[str, ChipEvaluation], chip_names: Sequence[str]
    ) -> None:
        self._opened_values = opened_values
        self._chip_names = chip_names

    def __call__(
        self, carry: Any, transcript: Transcript
    ) -> tuple[Any, Transcript, Mapping[str, ChipEvaluation]]:
        flat = flat_openings_absorb(
            select_openings(self._opened_values, self._chip_names),
            empty_prep_absorbs_zero=True,
        )
        return carry, transcript.observe(flat), self._opened_values


@dataclass(frozen=True)
class ZerocheckChunkSpec:
    """Name-keyed form of the opt-in chunked total-cap prefix knob
    (``TotalCapChunkClass``): per-chip round-0 height caps by CHIP NAME (the
    class union, e.g. the group manifest's per-chip GKR bounds), aligned to a
    shard's chip order at prove time — the spec never assumes an order.
    ``depth`` and ``chunk_cap`` pass through; a chip missing a cap fails
    loud."""

    depth: int
    chip_height_caps: tuple[tuple[str, int], ...]
    chunk_cap: int = 0

    def chunk_class_for(self, chip_names: Sequence[str]) -> TotalCapChunkClass:
        caps = dict(self.chip_height_caps)
        missing = [n for n in chip_names if n not in caps]
        if missing:
            raise ValueError(
                f"zerocheck chunk spec carries no height cap for {missing}"
            )
        return TotalCapChunkClass(
            depth=self.depth,
            chip_height_caps=tuple(int(caps[n]) for n in chip_names),
            chunk_cap=self.chunk_cap,
        )


def prove_shard_zerocheck(
    chips: Mapping[str, Chip],
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    public_values: Array,
    eval_point: Array,
    chip_openings: Mapping[str, ChipEvaluation],
    transcript: Transcript,
    *,
    max_log_row_count: int,
    num_reals: Sequence[Array] | None = None,
    total_cap_class: TotalCapClass | None = None,
    flat_arrival: Array | None = None,
    num_cols: Sequence[int] | None = None,
    main_widths: Sequence[int] | None = None,
    prep_widths: Sequence[int] | None = None,
    chip_names: Sequence[str] | None = None,
    chunk_class: TotalCapChunkClass | None = None,
) -> tuple[Transcript, ZerocheckProof]:
    """Reduce every chip's constraint zero-sum and GKR opening claim to one
    point claim via the jagged sumcheck.

    ``eval_point`` and ``chip_openings`` are the LogUp-GKR stage's outputs:
    zeta is the point's last ``max_log_row_count`` coordinates (the row
    variables), and each chip's claim is its openings RLC'd under the GKR
    opening-batch challenge — computed here from the same ``gkr_powers``
    weights the round engine applies, bit-for-bit.

    ``num_reals`` (optional, traced int32 scalars) switches to the
    shard-invariant jit path where the shard's real heights only bound the
    live rows at run time, so the whole stage body's compile keys on the
    ``total_cap_class`` + chip set, never a shard's exact heights
    (byte-identical to the exact-heights path): the single shared
    total-Σ-heights-cap buffer (fractalyze/sp1-zorch#242) — ``main_region``
    arrives repacked to a caller-chosen shard-invariant per-chip row cap and
    each trace is that wide.
    """
    ef = eval_point.dtype

    transcript, batching_challenge, gkr_batch, lambda_ = sample_stage_challenges(
        transcript, ef
    )

    zeta = eval_point[-max_log_row_count:]

    chip_names = list(chip_names) if chip_names is not None else main_region.chip_names
    if flat_arrival is not None:
        # Flat jagged arrival (pack_flat_arrival): no per-chip trace buffers
        # exist — the constraint seams read column counts / main widths from
        # the statics the caller threads through.
        if num_reals is None or total_cap_class is None:
            raise ValueError("flat_arrival rides the traced total_cap_class path")
        if num_cols is None or main_widths is None:
            raise ValueError("flat_arrival needs num_cols and main_widths")
        eval_fns = [
            export_order_eval_fn(chips[name], int(main_widths[i]), int(num_cols[i]))
            for i, name in enumerate(chip_names)
        ]
        claims = gkr_opening_claims(
            [chip_openings[name] for name in chip_names], gkr_batch
        )
        alphas = [
            rlc_coeffs(
                batching_challenge,
                probe_num_constraints(fn, int(nc), ef, public_values),
            )
            for fn, nc in zip(eval_fns, num_cols, strict=True)
        ]
        lambdas = rlc_coeffs(lambda_, len(chip_names))
        finals, transcript, msgs = prove_jagged_zerocheck(
            JaggedZerocheckSummand(
                eval_fns=eval_fns,
                alphas=alphas,
                lambdas=lambdas,
                beta=gkr_batch,
                public_values=public_values,
            ),
            [],
            list(num_reals),
            zeta,
            transcript,
            claims=claims,
            total_cap_class=total_cap_class,
            flat_arrival=flat_arrival,
            num_cols=num_cols,
            chunk_class=chunk_class,
        )
        # The opened-values split needs only per-chip widths — statics on the
        # flat path (no region object enters the jit body: a per-shard region
        # shape would poison the class-keyed compile cache).
        pw_list = (
            [int(w) for w in prep_widths]
            if prep_widths is not None
            else [0] * len(chip_names)
        )
        opened_values = {}
        for i, name in enumerate(chip_names):
            final = finals[i]
            evals = (
                final[:, 0]
                if final.shape[1] > 0
                else fnp.zeros((final.shape[0],), dtype=final.dtype)
            )
            mw = int(main_widths[i])
            pw = pw_list[i]
            opened_values[name] = ChipEvaluation(
                main=evals[:mw],
                preprocessed=evals[mw : mw + pw] if pw else None,
            )
        _, transcript, _ = OpenedValuesRound(opened_values, chip_names)(
            None, transcript
        )
        claimed_sum = fnp.sum(claims * lambdas)
        return transcript, ZerocheckProof(
            batching_challenge=batching_challenge,
            gkr_opening_batch_challenge=gkr_batch,
            lambda_=lambda_,
            zeta=zeta,
            claimed_sum=claimed_sum,
            finals=finals,
            opened_values=opened_values,
            msgs=msgs,
        )
    if chunk_class is not None:
        raise ValueError("chunk_class rides the flat_arrival total-cap path")
    if num_reals is None:
        num_reals = list(main_region.chip_heights)
        traces = chip_traces(chip_names, num_reals, main_region, prep_region)
    else:
        if total_cap_class is None:
            raise ValueError(
                "runtime (traced) num_reals require total_cap_class: the "
                "trace slicing cannot derive from a traced height"
            )
        # The region arrives repacked to a caller-chosen shard-invariant
        # per-chip cap (its own chip_heights): trace shapes are static, so the
        # compile keys on the caller's repack choice. The class carries no
        # height-derived arrival cap to check against — bounding every live
        # height is the caller's contract.
        caps = [int(h) for h in main_region.chip_heights]
        traces = chip_traces(chip_names, caps, main_region, prep_region)
        # The cap slice keeps real preprocessed rows past a shard's live
        # height; zero them — the round driver's zero-tail contract is
        # load-bearing (the fold touches the full buffer width).
        traces = [
            fnp.where(fnp.arange(t.shape[1]) < nr, t, fnp.zeros((), t.dtype))
            for t, nr in zip(traces, num_reals, strict=True)
        ]
    # The chip's 2-ary ``eval_constraints`` is the eval_fn; the statement is
    # threaded through ``constraint_eval``'s ``aux_operands`` at the fold sites,
    # not closed over — a closure would carry a tracer into the composite under
    # the jitted stage body.
    eval_fns = [
        export_order_eval_fn(
            chips[name], int(main_region.chip_widths[i]), int(t.shape[0])
        )
        for i, (name, t) in enumerate(zip(chip_names, traces))
    ]

    claims = gkr_opening_claims([chip_openings[name] for name in chip_names], gkr_batch)

    alphas = [
        rlc_coeffs(
            batching_challenge,
            probe_num_constraints(fn, t.shape[0], ef, public_values),
        )
        for fn, t in zip(eval_fns, traces)
    ]
    lambdas = rlc_coeffs(lambda_, len(chip_names))

    finals, transcript, msgs = prove_jagged_zerocheck(
        JaggedZerocheckSummand(
            eval_fns=eval_fns,
            alphas=alphas,
            lambdas=lambdas,
            beta=gkr_batch,
            public_values=public_values,
        ),
        traces,
        num_reals,
        zeta,
        transcript,
        claims=claims,
        total_cap_class=total_cap_class,
    )

    # The stage's transcript tail: absorb the opened values so every stage
    # consumer samples the evaluation-stage challenges from SP1's stream.
    opened_values = split_opened_values(finals, main_region, prep_region)
    _, transcript, _ = OpenedValuesRound(opened_values, chip_names)(None, transcript)

    # The wire's claimed_sum: the per-chip claims under the same chip RLC
    # weights the round engine applies.
    claimed_sum = fnp.sum(claims * lambdas)

    return transcript, ZerocheckProof(
        batching_challenge=batching_challenge,
        gkr_opening_batch_challenge=gkr_batch,
        lambda_=lambda_,
        zeta=zeta,
        claimed_sum=claimed_sum,
        finals=finals,
        opened_values=opened_values,
        msgs=msgs,
    )


if TYPE_CHECKING:
    # mypy-enforced seam conformance -- driven by `prove_rounds`.
    _: type[ProverRound] = OpenedValuesRound


class ZerocheckProver(
    ProverStage[ZerocheckClaim, ShardWitness, TraceEvaluationClaim, ZerocheckProof]
):
    """Zerocheck stage over ``prove_shard_zerocheck``, consuming the GKR
    point and openings off its source claim. The stage absorbs the per-chip opened
    values itself (``OpenedValuesRound`` in ``zerocheck.prover``); this Stage
    surfaces them in its reduced claim for the jagged opening and the
    wire's ShardOpenedValues.

    The stage body runs under one cached outer ``@jit`` on the total-cap
    contract (fractalyze/sp1-zorch#242): a ``TotalCapClass`` bounds the one
    flat jagged round buffer, the arrival is packed to the class shape in an
    eager prologue, and the shard's real heights ride as one traced int32
    vector, so the body's compile keys on the class and the chip set alone --
    shards that differ only in row counts share one executable (exact heights
    bust the cache: 22 distinct shape signatures across the 25-shard rsp
    block). With no class pinned, the shard's own a-priori-tight class is
    derived (per-shard compile, same body). pv-reading constraint circuits
    are legal because the statement rides ``constraint_eval``'s declared
    ``aux_operands`` operand, not a closure the composite would reject.
    Byte-identical to an eager exact-heights prove, and CPU-executable (the
    former eager-only fallback was a stale fractalyze/frx#168 workaround)."""

    def __init__(
        self,
        chips: Mapping[str, Chip],
        *,
        max_log_row_count: int,
        total_cap_class: TotalCapClass | None = None,
        chunk_spec: ZerocheckChunkSpec | None = None,
    ) -> None:
        self._chips = chips
        self._max_log_row_count = max_log_row_count
        self._total_cap_class = total_cap_class
        # Opt-in chunked total-cap prefix (TotalCapChunkClass): when set, the
        # stage runs as eager orchestration over chunk zones instead of the
        # single `_jit_body_totalcap_traced` program, so no XLA arena holds
        # the round-0 buffer. When unset (the default), the monolithic body —
        # its jaxpr, compile-cache entries, and AOT call keys — is untouched.
        self._chunk_spec = chunk_spec

    @staticmethod
    @partial(
        frx.jit,
        static_argnames=(
            "chips",
            "max_log_row_count",
            "total_cap_class",
            "chip_names",
            "num_cols",
            "main_widths",
            "prep_widths",
        ),
    )
    def _jit_body_totalcap_traced(
        flat_arrival: Array,
        public_values: Array,
        eval_point: Array,
        chip_openings: Mapping[str, ChipEvaluation],
        num_reals: Array,
        transcript: Transcript,
        *,
        chips: tuple[tuple[str, Chip], ...],
        max_log_row_count: int,
        total_cap_class: TotalCapClass,
        chip_names: tuple[str, ...],
        num_cols: tuple[int, ...],
        main_widths: tuple[int, ...],
        prep_widths: tuple[int, ...],
    ) -> tuple[Transcript, tuple[Any, ...]]:
        # The shard-invariant total-cap body (sp1-zorch#242): the arrival is
        # the ONE class-shaped flat jagged buffer (`pack_flat_arrival`) and
        # the shard's real heights ride in `num_reals` (one traced int32
        # vector); every other per-chip datum is a class-level static. The
        # compile keys on (chips, total_cap_class, the static tuples) alone —
        # shards of one class share the executable, and no per-shard region
        # shape enters the cache key.
        transcript, proof = prove_shard_zerocheck(
            dict(chips),
            None,
            None,
            public_values,
            eval_point,
            chip_openings,
            transcript,
            max_log_row_count=max_log_row_count,
            num_reals=[num_reals[i] for i in range(len(chip_names))],
            total_cap_class=total_cap_class,
            flat_arrival=flat_arrival,
            num_cols=num_cols,
            main_widths=main_widths,
            prep_widths=prep_widths,
            chip_names=chip_names,
        )
        return transcript, (
            proof.batching_challenge,
            proof.gkr_opening_batch_challenge,
            proof.lambda_,
            proof.zeta,
            proof.claimed_sum,
            proof.finals,
            proof.opened_values,
            proof.msgs,
        )

    def prove(
        self,
        claim: ZerocheckClaim,
        witness: ShardWitness,
        transcript: Transcript,
    ) -> ProveResult[TraceEvaluationClaim, ZerocheckProof]:
        # Shard-invariant flat prologue (sp1-zorch#242): pack the
        # class-shaped flat jagged arrival EAGERLY from the exact-height
        # traces — heights are host ints here, and the pack mirrors the
        # cols*evenpad(h) cumsum the traced body derives, so the layouts
        # agree. No chip pads to the class window (a wide class made that
        # uniform 2W padding overflow int32 element indexing and dwarf the
        # live area); the arrival is live rows + zeros, in the base field.
        names = witness.main_region.chip_names
        heights_host = [int(h) for h in witness.main_region.chip_heights]
        traces = chip_traces(
            names, heights_host, witness.main_region, witness.prep_region
        )
        # No pinned class: derive this shard's own a-priori-tight class
        # (per-shard compile, same traced body).
        total_cap_class = self._total_cap_class or TotalCapClass.from_heights(
            heights_host, [int(t.shape[0]) for t in traces]
        )
        flat = pack_flat_arrival(
            traces, heights_host, total_cap_class, self._max_log_row_count
        )
        prep_w = (
            {
                n: int(w)
                for n, w in zip(
                    witness.prep_region.chip_names,
                    witness.prep_region.chip_widths,
                )
            }
            if witness.prep_region is not None
            else {}
        )
        if self._chunk_spec is not None:
            # Chunked prefix (opt-in): EAGER stage orchestration — the chunk
            # zones and the monolithic remainder are their own jit programs
            # (`jagged._prove_total_cap_chunked`), so no single arena spans
            # the round-0 buffer. Same ops, same transcript stream —
            # byte-identical to the monolithic body below.
            transcript, proof = prove_shard_zerocheck(
                self._chips,
                None,
                None,
                claim.public_values,
                claim.gkr.eval_point,
                claim.gkr.chip_openings,
                transcript,
                max_log_row_count=self._max_log_row_count,
                num_reals=[fnp.asarray(h, fnp.int32) for h in heights_host],
                total_cap_class=total_cap_class,
                flat_arrival=flat,
                num_cols=tuple(int(t.shape[0]) for t in traces),
                main_widths=tuple(int(w) for w in witness.main_region.chip_widths),
                prep_widths=tuple(prep_w.get(n, 0) for n in names),
                chip_names=tuple(names),
                chunk_class=self._chunk_spec.chunk_class_for(names),
            )
            return ProveResult(
                TraceEvaluationClaim(proof.msgs.challenge, proof.opened_values),
                proof,
                transcript,
            )
        transcript, fields = self._jit_body_totalcap_traced(
            flat,
            claim.public_values,
            claim.gkr.eval_point,
            claim.gkr.chip_openings,
            fnp.asarray(heights_host, fnp.int32),
            transcript,
            chips=tuple(self._chips.items()),
            max_log_row_count=self._max_log_row_count,
            total_cap_class=total_cap_class,
            chip_names=tuple(names),
            num_cols=tuple(int(t.shape[0]) for t in traces),
            main_widths=tuple(int(w) for w in witness.main_region.chip_widths),
            prep_widths=tuple(prep_w.get(n, 0) for n in names),
        )
        (
            batching_challenge,
            gkr_batch,
            lambda_,
            zeta,
            claimed_sum,
            finals,
            opened_values,
            msgs,
        ) = fields
        proof = ZerocheckProof(
            batching_challenge=batching_challenge,
            gkr_opening_batch_challenge=gkr_batch,
            lambda_=lambda_,
            zeta=zeta,
            claimed_sum=claimed_sum,
            finals=finals,
            opened_values=opened_values,
            msgs=msgs,
        )
        return ProveResult(
            TraceEvaluationClaim(proof.msgs.challenge, proof.opened_values),
            proof,
            transcript,
        )
