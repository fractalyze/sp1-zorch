# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Multi-shard zero-recompile check for the traced total-cap zerocheck route.

Proves every ``--shard-dirs`` shard's zerocheck stage through ONE
``ZerocheckStage(total_cap_class=...)`` instance in one process and reports
the ``_jit_body_totalcap_traced`` compile count. Shards of one
``TotalCapClass`` (same chip set; the class bounding every shard in the
group) must share a single stage executable — the multi-shard acceptance
criterion of the traced total-cap jit path (fractalyze/sp1-zorch#242). The
chips mapping is loaded once and reused across shards, mirroring a
process-held machine (``chips`` is an identity-keyed static jit arg; fresh
Chip objects per shard would bust the cache and mask the result).

``--zc_class_json`` pins the class, exactly as ``verify_prove_shard`` /
``verify_zerocheck`` take it: assemble it as the per-field max of the
``ZC_CLASS`` lines ``verify_prove_shard`` prints for each shard.

GKR outputs are seeded per shard via the rolled jit prove and cached as the
replay module's npz (``save_gkr_cache``/``load_gkr_cache``), so reruns skip
straight to the stage under test.

    bazel run //sp1_zorch/zerocheck:verify_zerocheck_recompile -- \
        --shard-dirs="$PWD/dump/shard17,$PWD/dump/shard18" \
        --gkr-cache-dir="$PWD/gkr_caches" \
        --zc_class_json="$PWD/zc_class.json"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import frx

from sp1_zorch.shard_prover.fixture_loader import load_fixture_shard
from sp1_zorch.shard_prover.prove_shard import (
    ShardBridge,
    ZerocheckStage,
)
from sp1_zorch.shard_prover.replay import (
    MAX_LOG_ROW_COUNT,
    load_gkr_cache,
    save_gkr_cache,
    seed_gkr_outputs_rolled,
    shard_regions,
)
from sp1_zorch.zerocheck.jagged import TotalCapClass


def _gkr_inputs(shard, shard_dir: Path, main_region, prep_region, cache_dir: Path):
    cache = cache_dir / f"gkr_cache_{shard_dir.name}.npz"
    if cache.exists():
        print(f"[{shard_dir.name}] loading GKR outputs from {cache}", flush=True)
        return load_gkr_cache(cache)
    print(f"[{shard_dir.name}] seeding GKR via the rolled jit prove...", flush=True)
    transcript, eval_point, openings = seed_gkr_outputs_rolled(
        shard, shard_dir, main_region, prep_region
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_gkr_cache(cache, eval_point, openings, transcript)
    return eval_point, openings, transcript


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dirs",
        required=True,
        help="comma-separated shard dump dirs of ONE total-cap class (same "
        "chip set).",
    )
    parser.add_argument(
        "--gkr-cache-dir",
        required=True,
        help="directory for per-shard GKR-output caches (the replay npz).",
    )
    parser.add_argument(
        "--zc_class_json",
        required=True,
        help='JSON {"area_cap"} pinning the shard-invariant '
        "zerocheck TotalCapClass; it must bound every --shard-dirs shard. "
        "Assemble it as the per-field max of the ZC_CLASS lines "
        "verify_prove_shard prints.",
    )
    args = parser.parse_args()

    shard_dirs = [Path(p) for p in args.shard_dirs.split(",")]
    if len(shard_dirs) < 2:
        sys.exit("need at least two shard dirs to demonstrate executable sharing")
    cache_dir = Path(args.gkr_cache_dir)
    with open(args.zc_class_json) as f:
        c = {k: int(v) for k, v in json.load(f).items()}
    cap_class = TotalCapClass(area_cap=c["area_cap"])

    stage = None
    chip_set = None
    before = ZerocheckStage._jit_body_totalcap_traced._cache_size()
    for shard_dir in shard_dirs:
        shard = load_fixture_shard(shard_dir)
        main_region, prep_region = shard_regions(shard)
        if stage is None:
            # One chips mapping + one Stage for the whole group (identity-keyed
            # static jit arg), like a process-held machine.
            chip_set = tuple(main_region.chip_names)
            stage = ZerocheckStage(
                shard.main_trace_data.chips,
                max_log_row_count=MAX_LOG_ROW_COUNT,
                total_cap_class=cap_class,
            )
        elif tuple(main_region.chip_names) != chip_set:
            sys.exit(
                f"{shard_dir.name} chip set {main_region.chip_names} != "
                f"{chip_set}: not one total-cap class"
            )
        eval_point, openings, transcript = _gkr_inputs(
            shard, shard_dir, main_region, prep_region, cache_dir
        )
        bridge = replace(
            ShardBridge(main_region, prep_region, shard.main_trace_data.public_values),
            gkr_eval_point=eval_point,
            gkr_chip_openings=openings,
        )
        start = time.perf_counter()
        _, _, proof = stage(bridge, transcript)
        frx.block_until_ready(proof.finals)
        wall = time.perf_counter() - start
        compiles = ZerocheckStage._jit_body_totalcap_traced._cache_size() - before
        print(
            f"[{shard_dir.name}] zerocheck stage {wall * 1e3:.1f} ms "
            f"(incl. eager flat pack + any compile); cumulative stage compiles: "
            f"{compiles}",
            flush=True,
        )

    compiles = ZerocheckStage._jit_body_totalcap_traced._cache_size() - before
    print(f"total shards: {len(shard_dirs)}, stage compiles: {compiles}", flush=True)
    if compiles != 1:
        print("FAIL: shards of one total-cap class did not share one executable")
        return 1
    print("OK: one stage executable shared across the total-cap class")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
