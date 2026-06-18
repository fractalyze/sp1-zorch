# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Emit SP1 reference-prover bench inputs from a real rsp shard dump.

The SP1-vs-zkx wall-clock comparison must run BOTH provers on the SAME shard.
zkx's side is measured directly (``bench_sp1_logup_gkr`` / ``bench_trace_commit``
on ``--shard-dir``). SP1's native benches (``sp1-gpu`` crate) take *synthetic*
inputs, so to make them prove the same shape this tool extracts the shard's real
dimensions and writes them in each SP1 bench's input format:

  * ``--stage logup_gkr`` -> a ``layer_workloads.json`` for SP1's
    ``logup_gkr_bench``: the first-layer per-interaction column heights in SP1's
    ``col_h`` units (``ceil(max(h,8)/4)`` per interaction, padded to a power-of-
    two interaction count), plus ``num_row_variables``. SP1 fills these heights
    with random ``Felt`` numerators / ``Ext`` denominators (base-field faithful),
    so the timing reflects the real shard's structure.

  * ``--stage commit`` -> the ``merkle_bench`` dimensions for the SMCS commit:
    ``num_leaves = stacking_height << log_blowup`` (committed codeword rows) and
    ``width = K`` (stacked columns). Run as
    ``merkle_bench --sizes=<log2(num_leaves)> --width=<K>``.

CPU-only (no prove); reads the trace metas + manifest, not the GPU path. See
``docs/sp1-baseline.md`` for the full comparison procedure and recorded numbers.

    JAX_PLATFORMS=cpu PYTHONPATH="$PWD:/abs/path/to/zorch" \\
        python sp1_zorch/tools/sp1_baseline.py \\
        --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard17 --stage logup_gkr
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from sp1_zorch.commit.region import JaggedRegion  # noqa: E402
from sp1_zorch.logup_gkr.circuit import build_gkr_chips, sp1_col_h  # noqa: E402
from sp1_zorch.shard_prover.fixture_loader import load_fixture_shard  # noqa: E402
from sp1_zorch.shard_prover.replay import MAX_LOG_ROW_COUNT, shard_regions  # noqa: E402

# SP1's trace commit uses a rate-1/4 RS code (matches sp1_zorch.commit
# verify_trace_commit / bench_trace_commit _LOG_BLOWUP).
_LOG_BLOWUP = 2


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _dump_height(shard_dir: Path) -> int | None:
    """The reference ``height`` (col_h units) SP1 recorded in the dump, for a
    cross-check that the extracted counts match the real shard."""
    fl = shard_dir / "gpu_first_layer.txt"
    if not fl.exists():
        return None
    m = re.search(r"height=(\d+)", fl.read_text())
    return int(m.group(1)) if m else None


def logup_gkr_workload(shard_dir: Path) -> dict:
    """First-layer ``interaction_row_counts`` (col_h units) + num_row_variables,
    in SP1 ``logup_gkr_bench`` Workload shape."""
    shard = load_fixture_shard(shard_dir)
    traces = shard.main_trace_data.traces
    gkr_chips = build_gkr_chips(shard.main_trace_data.chips, traces.chip_order)
    counts: list[int] = []
    for chip in gkr_chips:
        col_h = sp1_col_h(int(traces.per_chip[chip.name].array.shape[0]))
        counts.extend([col_h] * len(chip.interactions))
    n_pad = _next_pow2(len(counts)) - len(counts)
    counts.extend([sp1_col_h(0)] * n_pad)
    return {
        "interaction_row_counts": counts,
        "num_row_variables": MAX_LOG_ROW_COUNT - 1,
    }


def commit_dims(shard_dir: Path) -> dict:
    """SMCS-commit Merkle dimensions for SP1 ``merkle_bench``."""
    shard = load_fixture_shard(shard_dir)
    region: JaggedRegion = shard_regions(shard)[0]
    stacking = 1 << region.log_stacking_height
    width = region.dense.shape[0] // stacking
    num_leaves = stacking << _LOG_BLOWUP
    return {
        "num_leaves": num_leaves,
        "log_num_leaves": num_leaves.bit_length() - 1,
        "width": width,
        "log_blowup": _LOG_BLOWUP,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", required=True, type=Path)
    ap.add_argument("--stage", choices=["logup_gkr", "commit"], required=True)
    ap.add_argument(
        "--out", type=Path, default=None, help="logup_gkr: layer_workloads.json path"
    )
    args = ap.parse_args()

    if args.stage == "logup_gkr":
        w = logup_gkr_workload(args.shard_dir)
        total = sum(w["interaction_row_counts"])
        ref = _dump_height(args.shard_dir)
        tag = "OK" if ref is None or abs(total - ref) <= len(w["interaction_row_counts"]) else "MISMATCH"
        print(
            f"{args.shard_dir.name}: n_interactions={len(w['interaction_row_counts'])} "
            f"total_col_h={total} num_row_variables={w['num_row_variables']} "
            f"(dump height={ref}, {tag})"
        )
        out = args.out or Path(f"layer_workloads_{args.shard_dir.name}.json")
        json.dump([w], out.open("w"))
        print(f"  -> {out}  (copy to sp1-gpu/crates/logup_gkr/layer_workloads.json, run logup_gkr_bench)")
    else:
        d = commit_dims(args.shard_dir)
        print(
            f"{args.shard_dir.name}: merkle_bench --sizes={d['log_num_leaves']} "
            f"--width={d['width']}   (num_leaves={d['num_leaves']}, log_blowup={d['log_blowup']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
