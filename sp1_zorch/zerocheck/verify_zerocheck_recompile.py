# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Multi-shard zero-recompile check for the cap-class zerocheck path.

Proves every ``--shard-dirs`` shard's zerocheck stage through ONE
``ZerocheckStage(width_caps=...)`` instance in one process and reports
the ``_jit_body_capped`` compile count. Shards of one cap class (same chip
set; per-chip caps = max heights across the group) must share a single stage
executable — the multi-shard acceptance criterion of the cap-class jit path.
The chips mapping is loaded once and reused across shards, mirroring a
process-held machine (``chips`` is an identity-keyed static jit arg; fresh
Chip objects per shard would bust the cache and mask the result).

GKR outputs are seeded per shard via the rolled jit prove and cached as the
replay module's npz (``save_gkr_cache``/``load_gkr_cache``), so reruns skip
straight to the stage under test.

    bazel run //sp1_zorch/zerocheck:verify_zerocheck_recompile -- \
        --shard-dirs="$PWD/dump/shard17,$PWD/dump/shard18" \
        --gkr-cache-dir="$PWD/gkr_caches"
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import frx

from sp1_zorch.shard_prover.fixture_loader import _parse_kv_lines, load_fixture_shard
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


def _family_caps(shard_dirs: list[Path]) -> dict[str, int]:
    """Per-chip caps = next-pow2 of the max ``num_real`` across the group's
    ``gpu_traces`` metas — the family-max policy, keyed by chip name. Floored
    to a valid buffer width (>=4, multiple of 4): a raw max can be 0 (a chip
    uniformly empty across the group) or non-mult-of-4, both illegal caps."""
    raw: dict[str, int] = {}
    for d in shard_dirs:
        for meta in sorted((d / "gpu_traces").glob("*.meta")):
            kv = _parse_kv_lines(meta.read_text())
            raw[meta.stem] = max(raw.get(meta.stem, 0), int(kv["num_real"]))
    return {
        n: min(max(4, 1 << ((h - 1).bit_length())) if h > 0 else 4, 1 << MAX_LOG_ROW_COUNT)
        for n, h in raw.items()
    }


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
        help="comma-separated shard dump dirs of ONE cap class (same chip set).",
    )
    parser.add_argument(
        "--gkr-cache-dir",
        required=True,
        help="directory for per-shard GKR-output caches (the replay npz).",
    )
    args = parser.parse_args()

    shard_dirs = [Path(p) for p in args.shard_dirs.split(",")]
    if len(shard_dirs) < 2:
        sys.exit("need at least two shard dirs to demonstrate executable sharing")
    cache_dir = Path(args.gkr_cache_dir)
    caps = _family_caps(shard_dirs)

    stage = None
    chip_set = None
    before = ZerocheckStage._jit_body_capped._cache_size()
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
                width_caps=caps,
            )
        elif tuple(main_region.chip_names) != chip_set:
            sys.exit(
                f"{shard_dir.name} chip set {main_region.chip_names} != "
                f"{chip_set}: not one cap class"
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
        compiles = ZerocheckStage._jit_body_capped._cache_size() - before
        print(
            f"[{shard_dir.name}] zerocheck stage {wall * 1e3:.1f} ms "
            f"(incl. eager repack + any compile); cumulative stage compiles: "
            f"{compiles}",
            flush=True,
        )

    compiles = ZerocheckStage._jit_body_capped._cache_size() - before
    print(f"total shards: {len(shard_dirs)}, stage compiles: {compiles}", flush=True)
    if compiles != 1:
        print("FAIL: shards of one cap class did not share one executable")
        return 1
    print("OK: one stage executable shared across the cap class")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
