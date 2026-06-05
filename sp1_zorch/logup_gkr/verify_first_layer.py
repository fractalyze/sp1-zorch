# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""rsp byte-match harness for the LogUp-GKR first layer — a runnable.

Rebuilds the first GKR layer from the dumped traces plus the dump's own
``alpha`` / ``beta_seed`` challenges (``gpu_gkr_state.txt``), then compares
against ``gpu_first_layer.txt``; also anchors the transition schedule against
the per-round layer heights in ``gkr_sumcheck_rounds.txt``. No transcript
involved — that stream is byte-matched separately against
``gpu_gkr_state.txt``. Exits non-zero on any mismatch.

SP1 stores the same four stride-2 planes (n0/n1/d0/d1) we do, so buffer
heads compare directly; only the diag's row-count bookkeeping (height,
interaction_row_counts, start_indices) is in SP1's internal ``col_h`` units
(a quarter of the real height — half of one plane's per-interaction slots),
so those comparisons convert units rather than the layer.

    bazel run //sp1_zorch/logup_gkr:verify_first_layer -- \\
        --shard_dir=/path/to/rsp_dump/shardN

``--accounting_only`` skips all field math and checks just the static slot
accounting (fast iteration on chip-set / unit / schedule questions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
from absl import app, flags
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import (
    build_gkr_chips,
    generate_first_layer,
    sp1_col_h,
    sp1_next_row_counts,
)
from sp1_zorch.shard_prover.fixture_loader import (
    _parse_ef_list,
    _parse_int_list,
    _parse_kv_lines,
    check_match,
    load_fixture_shard,
)
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.utils.bits import log2_ceil_usize

_SHARD_DIR = flags.DEFINE_string(
    "shard_dir", None, "rsp shard dump directory (e.g. .../rsp_dump/shard1)."
)
_ACCOUNTING_ONLY = flags.DEFINE_bool(
    "accounting_only",
    False,
    "Check only the static slot accounting (no field math).",
)

# SP1 core machine parameters (same as verify_trace_commit).
_LOG_STACKING_HEIGHT = 21
_MAX_LOG_ROW_COUNT = 22


def _round_schedule_check(gkr_chips, traces, shard_dir: Path) -> bool:
    """Per-round layer heights in ``gkr_sumcheck_rounds.txt`` vs the schedule.

    SP1 bookkeeps the schedule in col_h units, where ``ceil(rc / 4) * 2``
    saturates at 2 col_h units per interaction. Our materialized planes run
    the same step in slot units (2x col_h) and so saturate at 2 slots = 1
    col_h unit — same dense MLE, more implicit padding — which is why this
    check re-runs the schedule in SP1's units instead of converting the
    materialized row counts. Static accounting only, no field math.
    """
    blocks = (shard_dir / "gkr_sumcheck_rounds.txt").read_text().split(
        "--- round ---"
    )
    rounds = [_parse_kv_lines(b) for b in blocks if b.strip()]

    counts = tuple(
        sp1_col_h(traces.per_chip[c.name].array.shape[0])
        for c in gkr_chips
        for _ in c.interactions
    )
    heights: dict[int, int] = {}
    nrv = _MAX_LOG_ROW_COUNT - 1
    while nrv > 1:
        counts = sp1_next_row_counts(counts)
        nrv -= 1
        heights[nrv] = sum(counts)

    by_nrv = {int(r["nrv"]): int(r["height"]) for r in rounds}
    levels = sorted(by_nrv)
    ok = check_match(
        f"per-round heights, nrv {levels[0]}..{levels[-1]} (col_h units)",
        [by_nrv[n] for n in levels],
        [heights[n] for n in levels],
    )
    niv = log2_ceil_usize(len(counts))
    ok &= check_match(
        "per-round niv",
        sorted({int(r["niv"]) for r in rounds}),
        [niv],
    )
    return ok


def _accounting(shard, ref: dict[str, str], shard_dir: Path) -> bool:
    """Static slot accounting vs the dump, in the dump's col_h units.

    Field-math-free, so chip-set / unit / schedule questions iterate in
    seconds. SP1's height covers every real interaction (zero-height chips
    included) and gives the power-of-two interaction padding no rows.
    """
    traces = shard.main_trace_data.traces
    gkr_chips = build_gkr_chips(shard.main_trace_data.chips, traces.chip_order)
    units = sum(
        sp1_col_h(traces.per_chip[c.name].array.shape[0]) * len(c.interactions)
        for c in gkr_chips
    )
    ok = check_match("height (col_h units)", units, int(ref["height"]))
    ok &= _round_schedule_check(gkr_chips, traces, shard_dir)
    return ok


def main(argv) -> None:
    del argv
    shard_dir = Path(_SHARD_DIR.value)
    shard = load_fixture_shard(shard_dir)
    ref = _parse_kv_lines((shard_dir / "gpu_first_layer.txt").read_text())

    if _ACCOUNTING_ONLY.value:
        sys.exit(0 if _accounting(shard, ref, shard_dir) else 1)

    traces = shard.main_trace_data.traces
    order = traces.chip_order

    main_region = JaggedRegion.from_chips(
        [traces.per_chip[n].array for n in order],
        log_stacking_height=_LOG_STACKING_HEIGHT,
        max_log_row_count=_MAX_LOG_ROW_COUNT,
        chip_names=order,
    )
    prep = shard.preprocessed_traces
    prep_names = tuple(sorted(prep))
    prep_region = (
        JaggedRegion.from_chips(
            [prep[n] for n in prep_names],
            log_stacking_height=_LOG_STACKING_HEIGHT,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=prep_names,
        )
        if prep
        else None
    )

    state = _parse_kv_lines(
        (shard_dir / "gpu_gkr_state.txt").read_text(), skip_unkeyed=True
    )
    alpha = _parse_ef_list(state["alpha"])[0]
    seeds = []
    while f"beta_seed[{len(seeds)}]" in state:
        seeds.append(_parse_ef_list(state[f"beta_seed[{len(seeds)}]"]))
    betas = expand_eq_to_hypercube(
        jnp.concatenate(seeds), jnp.array(1, dtype=EF)
    )

    gkr_chips = build_gkr_chips(shard.main_trace_data.chips, order)
    layer = generate_first_layer(gkr_chips, main_region, prep_region, alpha, betas)

    ok = True
    # Unit conversion: dump counts in col_h (= our slot_count / 2). SP1's
    # buffer gives the power-of-two interaction padding NO rows (their
    # contribution is the sumcheck's eq-sum adjustment); we materialize them
    # whir-style as trailing neutral segments, so the dump height covers only
    # the real-interaction prefix.
    n_real = sum(len(c.interactions) for c in gkr_chips)
    starts = layer.start_indices
    ok &= check_match("height (col_h units)", starts[n_real] // 2, int(ref["height"]))
    ok &= check_match(
        "num_row_variables (SP1 fixed depth)",
        _MAX_LOG_ROW_COUNT - 1,
        int(ref["num_row_variables"]),
    )
    rc_head = _parse_int_list(ref["interaction_row_counts_head"])
    ok &= check_match(
        "row_counts head (col_h units)",
        [rc // 2 for rc in layer.row_counts[: len(rc_head)]],
        rc_head,
    )
    si_head = _parse_int_list(ref["start_indices_head"])
    ok &= check_match(
        "start_indices head (col_h units)",
        [si // 2 for si in starts[: len(si_head)]],
        si_head,
    )

    # SP1 stores the same four stride-2 planes we do (n0 = even rows, ...);
    # the dump heads print the n0 / d0 planes directly. Only the diag's
    # row-count bookkeeping above is in col_h units.
    num_head = jnp.array(_parse_int_list(ref["num_buf_head"]), dtype=F)
    ok &= check_match(
        "num_buf head (n0 plane)",
        layer.numerator_0[: len(num_head)],
        num_head,
    )
    den_head = _parse_ef_list(ref["den_buf_head"])
    ok &= check_match(
        "den_buf head (d0 plane)",
        layer.denominator_0[: len(den_head)],
        den_head,
    )

    ok &= _round_schedule_check(gkr_chips, traces, shard_dir)

    if not ok:
        sys.exit(1)
    print("first layer byte-match: ALL OK")


if __name__ == "__main__":
    flags.mark_flag_as_required("shard_dir")
    app.run(main)
