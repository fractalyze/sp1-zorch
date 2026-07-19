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

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import frx.numpy as fnp
from frx import Array
from rw_constraints import Chip

from zk_dtypes import efinfo

from zorch.pcs.jagged.region import JaggedRegion
from sp1_zorch.logup_gkr.prover import (
    ChipEvaluation,
    flat_openings_absorb,
    select_openings,
)
from sp1_zorch.zerocheck.jagged import (
    JaggedZerocheckSummand,
    TotalCapClass,
    prove_jagged_zerocheck,
)
from sp1_zorch.zerocheck.coeffs import gkr_powers, rlc_coeffs
from zorch.round import Round
from zorch.sumcheck.prover import RoundMsg
from zorch.transcript import Transcript, sample_challenge


@dataclass(frozen=True)
class ZerocheckProof:
    """The zerocheck stage's proof: the three stage challenges and the eq
    point (the byte-match harness and the jagged-opening stage consume them,
    and neither holds the pre-stage transcript to re-sample), the wire's
    claimed sum (the lambda-Horner fold of the per-chip GKR opening claims,
    SP1's zerocheck RLC — retained because only this stage holds the claims),
    the per-chip final folded traces with their split ``opened_values`` view
    (the evaluation stage's per-column claims and the wire's
    ShardOpenedValues), and the stacked round messages whose ``challenge``
    is the sumcheck point."""

    batching_challenge: Array
    gkr_opening_batch_challenge: Array
    lambda_: Array
    zeta: Array
    claimed_sum: Array
    finals: list[Array]
    opened_values: dict[str, ChipEvaluation]
    msgs: RoundMsg


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


def export_order_eval_fn(
    chip: Chip, main_width: int, num_cols: int
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
    """
    fn = chip.eval_constraints
    if num_cols == main_width:
        return fn

    def eval_fn(rows: Array, public_values: Array) -> Array:
        export_rows = fnp.concatenate(
            [rows[..., main_width:], rows[..., :main_width]], axis=-1
        )
        return fn(export_rows, public_values)

    return eval_fn


def bind_pv(chip: Chip, public_values: Array) -> Callable[[Array], Array]:
    """Bind the public-values vector; ``eval_constraints`` ignores it for
    constraints that declare no ``pv_arg``. Shared by the stage and its
    verifier dual — the one definition of how a chip's constraint circuit
    sees the statement."""
    return lambda trace: chip.eval_constraints(trace, public_values)


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
    ``eval_constraints``; the statement is threaded, not closed over."""
    return eval_fn(fnp.zeros((1, width), dtype=ef), public_values).shape[-1]


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


def gkr_opening_claims(
    openings: Sequence[ChipEvaluation], gkr_batch: Array
) -> Array:
    """Each chip's GKR opening claim: its ``[main | prep]`` evaluations
    weighted by the shared beta powers — the seed of the round engine's
    ``p(1) = claim - p(0)`` identity. One definition: the prover seeds the
    sumcheck with these, the verifier dual re-derives its claimed sum from
    them."""
    evals = [opening.all_evals() for opening in openings]
    max_cols = max(e.shape[0] for e in evals)
    gkr_all = (
        gkr_powers(gkr_batch, max_cols)
        if max_cols
        else fnp.zeros(0, gkr_batch.dtype)
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


class OpenedValuesRound(Round):
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
    arrives repacked to ``2*window`` rows per chip and each trace is that
    wide.
    """
    ef = eval_point.dtype

    transcript, batching_challenge, gkr_batch, lambda_ = sample_stage_challenges(
        transcript, ef
    )

    zeta = eval_point[-max_log_row_count:]

    chip_names = (
        list(chip_names) if chip_names is not None else main_region.chip_names
    )
    if flat_arrival is not None:
        # Flat jagged arrival (pack_flat_arrival): no per-chip trace buffers
        # exist — the constraint seams read column counts / main widths from
        # the statics the caller threads through.
        if num_reals is None or total_cap_class is None:
            raise ValueError(
                "flat_arrival rides the traced total_cap_class path"
            )
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
    if num_reals is None:
        num_reals = list(main_region.chip_heights)
        traces = chip_traces(chip_names, num_reals, main_region, prep_region)
    else:
        if total_cap_class is None:
            raise ValueError(
                "runtime (traced) num_reals require total_cap_class: the "
                "trace slicing cannot derive from a traced height"
            )
        # The total-cap shared buffer presents one shard-invariant per-chip
        # cap for the trace slice: `2*window`. The region must already be
        # repacked to that cap.
        caps = [2 * total_cap_class.window] * len(chip_names)
        if tuple(main_region.chip_heights) != tuple(int(c) for c in caps):
            raise ValueError(
                "runtime num_reals expect a main region repacked to "
                "2*total_cap_class.window rows per chip: chip heights "
                f"{main_region.chip_heights} != caps {tuple(caps)}"
            )
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
        export_order_eval_fn(chips[name], int(main_region.chip_widths[i]), int(t.shape[0]))
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
