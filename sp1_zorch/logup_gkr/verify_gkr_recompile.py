# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Multi-shard zero-recompile check for the class-shaped LogUp-GKR route.

Proves every ``--shard-dirs`` shard through ONE ``GkrCapClass`` + ONE
``gkr_chips`` tuple in one process; shards after the first must add ZERO
compiles in every class-keyed zone (first-layer/open in sp1-zorch,
transition/round zones in zorch). Assemble ``--gkr_class_json`` as the per-chip
max of the ``GKR_CLASS`` lines ``verify_gkr_prove``/``verify_prove_shard``
print. Byte-match stays ``verify_gkr_prove``'s job; this checks compile
sharing and prints per-shard walls.

    bazel run //sp1_zorch/logup_gkr:verify_gkr_recompile -- \
        --shard-dirs="$PWD/dump/shard17,$PWD/dump/shard18" \
        --gkr_class_json="$PWD/gkr_class.json"
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import frx

from zorch.logup_gkr.circuit import _jagged_transition_core
from zorch.logup_gkr.jagged_prover import _jagged_round_zone
from sp1_zorch.logup_gkr.circuit import (
    GkrCapClass,
    _chip_first_layer_capped,
    build_gkr_chips,
)
from sp1_zorch.logup_gkr.prover import _head_zone, open_traces_capped
from sp1_zorch.shard_prover.fixture_loader import load_fixture_shard
from sp1_zorch.shard_prover.replay import replay_gkr, shard_regions

# Every class-keyed zone the stage dispatches through, with where it lives.
_ZONES = {
    "head(sp1)": _head_zone,
    "first_layer(sp1)": _chip_first_layer_capped,
    "open(sp1)": open_traces_capped,
    "transition(zorch)": _jagged_transition_core,
    "round_zone(zorch)": _jagged_round_zone,
}


def _zone_sizes() -> dict[str, int]:
    return {name: fn._cache_size() for name, fn in _ZONES.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dirs",
        required=True,
        help="comma-separated shard dump dirs of ONE GkrCapClass (same chip "
        "set).",
    )
    parser.add_argument(
        "--gkr_class_json",
        required=True,
        help='JSON {"chip_heights": {name: bound}} pinning the shard-'
        "invariant GkrCapClass; it must bound every --shard-dirs shard. "
        "Assemble it as the per-chip max of the GKR_CLASS lines "
        "verify_gkr_prove / verify_prove_shard print.",
    )
    parser.add_argument(
        "--gkr_pow_bits",
        type=int,
        default=12,
        help="GKR grind bits (SP1 hardcodes GKR_GRINDING_BITS = 12).",
    )
    args = parser.parse_args()

    shard_dirs = [Path(p) for p in args.shard_dirs.split(",")]
    if len(shard_dirs) < 2:
        sys.exit("need at least two shard dirs to demonstrate executable sharing")
    with open(args.gkr_class_json) as f:
        bounds = json.load(f)["chip_heights"]

    gkr_chips = None
    cap_class = None
    chip_set = None
    after_first: dict[str, int] | None = None
    for shard_dir in shard_dirs:
        shard = load_fixture_shard(shard_dir)
        main_region, prep_region = shard_regions(shard)
        if gkr_chips is None:
            # One gkr_chips tuple + one class for the whole group (the inner
            # zones key statically on them), like a process-held machine.
            chip_set = tuple(main_region.chip_names)
            gkr_chips = build_gkr_chips(shard.main_trace_data.chips, chip_set)
            cap_class = GkrCapClass(
                tuple(int(bounds[name]) for name in chip_set)
            )
        elif tuple(main_region.chip_names) != chip_set:
            sys.exit(
                f"{shard_dir.name} chip set {main_region.chip_names} != "
                f"{chip_set}: not one GkrCapClass"
            )
        start = time.perf_counter()
        _, proof = replay_gkr(
            shard,
            shard_dir,
            main_region,
            prep_region,
            pow_bits=args.gkr_pow_bits,
            cap_class=cap_class,
            gkr_chips=gkr_chips,
        )
        frx.block_until_ready(proof.eval_point)
        wall = time.perf_counter() - start
        sizes = _zone_sizes()
        if after_first is None:
            after_first = sizes
            deltas = "baseline"
        else:
            deltas = ", ".join(
                f"{name}:+{sizes[name] - after_first[name]}" for name in _ZONES
            )
        print(
            f"[{shard_dir.name}] gkr stage {wall * 1e3:.1f} ms "
            f"(incl. eager pack + any compile); zone compiles vs shard 1: "
            f"{deltas}",
            flush=True,
        )
        # Release this shard's device buffers before the next iteration
        # allocates — holding a spent proof + regions while the next shard
        # proves stacks two shards' residency and OOMs a wide core-shard
        # pair (verify_prove_shard releases the same way between passes).
        del shard, main_region, prep_region, proof
        gc.collect()

    final = _zone_sizes()
    new_compiles = {n: final[n] - after_first[n] for n in _ZONES}
    print(
        f"total shards: {len(shard_dirs)}, compiles added after shard 1: "
        f"{new_compiles}",
        flush=True,
    )
    if any(new_compiles.values()):
        print("FAIL: shards of one GkrCapClass did not share every zone executable")
        return 1
    print("OK: every class-keyed zone executable shared across the class")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
