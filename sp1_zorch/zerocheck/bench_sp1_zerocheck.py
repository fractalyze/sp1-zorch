# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Benchmark the zerocheck stage as the full shard prover drives it.

Sibling of ``logup_gkr/bench_sp1_logup_gkr.py``. The shard prover runs
zerocheck (``ShardZerocheckRound``) on the LogUp-GKR stage's outputs -- the
real transcript, GKR eval point, and per-chip openings. This harness replays
the pipeline up to those outputs, then times ONLY ``prove_shard_zerocheck``,
so the number is the zerocheck slice in shard-prover context, not a synthetic
micro-bench.

The GKR replay dominates wall time (hours, eager), so ``--gkr-cache`` persists
its outputs (eval point, chip openings, post-GKR sponge state); reruns load the
cache and jump straight to the stage under test. The cache format mirrors
``verify_zerocheck``'s, so a cache seeded by either tool is interchangeable.

    # seed the cache once (hours of eager GKR replay), then bench warm:
    bazel run //sp1_zorch/zerocheck:bench_sp1_zerocheck -- \
        --shard-dir=/path/to/rsp_dump/shardN --gkr-cache=/path/to/gkr.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from sp1_zorch.shard_prover.fixture_loader import (
    _parse_ef_list,
    _parse_kv_lines,
    check_match,
    load_fixture_shard,
)
from sp1_zorch.shard_prover.replay import (
    MAX_LOG_ROW_COUNT,
    clone_diag,
    load_gkr_cache,
    save_gkr_cache,
    seed_gkr_outputs_rolled,
    shard_regions,
)
from sp1_zorch.zerocheck.stage import prove_shard_zerocheck


def _seed_gkr_sealed(shard, shard_dir: Path, main_region, prep_region):
    """Seed the GKR outputs via the fast rolled jit prove (minutes, not the
    eager path's hours) and seal them against the dump's post-GKR diag before
    they're trusted as zerocheck inputs. The seal is byte-exact, so it also
    catches any transcript drift in the rolled seed (mirrors verify_zerocheck)."""
    transcript, eval_point, chip_openings = seed_gkr_outputs_rolled(
        shard, shard_dir, main_region, prep_region
    )
    post = _parse_kv_lines((shard_dir / "gpu_post_gkr_diag.txt").read_text())
    if not check_match(
        "post_gkr_diag (GKR seal)", clone_diag(transcript), int(post["post_gkr_diag"])
    ):
        sys.exit("GKR replay diverged from the dump; zerocheck inputs are invalid.")
    return eval_point, chip_openings, transcript


def _anchor(batching_challenge, round_poly, shard_dir: Path) -> None:
    """Cheap pre-timing guard: a drifted setup would otherwise time the wrong
    computation. Check the stage's batching challenge and the joint claimed sum
    against the dump (a subset of verify_zerocheck's gating checks)."""
    state = _parse_kv_lines(
        (shard_dir / "gpu_zerocheck_state.txt").read_text().split("\nchip ")[0]
    )
    ok = check_match(
        "batching_challenge",
        batching_challenge,
        _parse_ef_list(state["batching_challenge"])[0],
    )
    # The joint claim seeds round 0's p(0)+p(1) identity: claimed_sum = c0 + sum(c).
    p0 = round_poly[0]
    ok &= check_match(
        "claimed_sum", p0[0] + jnp.sum(p0), _parse_ef_list(state["claimed_sum"])[0]
    )
    if not ok:
        sys.exit("zerocheck anchor diverged from the dump; aborting before timing.")


class Sp1ZerocheckBenchmark(JaxBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="sp1-zorch",
            version="0.1.0",
            default_iterations=5,
            # One warmup absorbs the cold compile; subsequent iterations are warm.
            default_warmup=1,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--shard-dir", required=True, help="rsp shard dump directory."
        )
        parser.add_argument(
            "--gkr-cache",
            default=None,
            help="npz path for the GKR replay outputs; loaded when present, "
            "seeded (rolled jit prove, minutes) otherwise. Recommended so reruns "
            "skip straight to the zerocheck stage under test.",
        )

    def get_ops(self, args: argparse.Namespace):
        shard_dir = Path(args.shard_dir)
        shard = load_fixture_shard(shard_dir)
        main_region, prep_region = shard_regions(shard)

        cache = Path(args.gkr_cache) if args.gkr_cache else None
        if cache is not None and cache.suffix != ".npz":
            # np.savez appends .npz; normalize so exists() and the write target
            # name the same file.
            cache = cache.with_name(cache.name + ".npz")
        if cache is not None and cache.exists():
            print(f"loading GKR outputs from {cache}", flush=True)
            eval_point, openings, transcript = load_gkr_cache(cache)
        else:
            print("no GKR cache; seeding via the rolled jit prove...", flush=True)
            eval_point, openings, transcript = _seed_gkr_sealed(
                shard, shard_dir, main_region, prep_region
            )
            if cache is not None:
                save_gkr_cache(cache, eval_point, openings, transcript)
                print(f"saved GKR outputs to {cache}", flush=True)

        chips = shard.main_trace_data.chips
        public_values = shard.main_trace_data.public_values
        total_rows = int(sum(main_region.chip_heights))

        # jit the stage the way prove_shard does -- chips + max_log_row_count
        # static via closure (constraint structure is host-known), regions +
        # arrays traced -- so the zorch.sumcheck marker survives to the vendor
        # emitter and warm iterations cache-hit.
        @jax.jit
        def _prove(mr, pr, pv, ep, ops, tr):
            _, proof = prove_shard_zerocheck(
                chips, mr, pr, pv, ep, ops, tr, max_log_row_count=MAX_LOG_ROW_COUNT
            )
            # Return JAX-typed pieces only -- ZerocheckProof is not a
            # jit-returnable pytree. The anchor reads batching_challenge +
            # round_poly[0]; the timed run forces the whole stage via the round
            # polys, the bound challenge chain, and the per-chip finals.
            return (
                proof.batching_challenge,
                proof.msgs.round_poly,
                proof.msgs.challenge,
                proof.finals,
            )

        stage_args = (main_region, prep_region, public_values, eval_point, openings, transcript)

        # Anchor once (this also pays the cold compile); a drifted replay aborts
        # here rather than producing a fast wrong number.
        batching0, round_poly0, _, _ = _prove(*stage_args)
        jax.block_until_ready((batching0, round_poly0))
        _anchor(batching0, round_poly0, shard_dir)

        def _run():
            _, round_poly, challenge, finals = _prove(*stage_args)
            # The round polys, bound challenge chain, and per-chip finals force
            # the whole stage.
            return round_poly, challenge, finals

        yield BenchmarkOp(
            # The real-shard zerocheck stage time, joinable on the dashboard with
            # bench_sp1_logup_gkr.
            name="zerocheck_total",
            fn=_run,
            metadata={
                "shard": shard_dir.name,
                "field": "koalabear",
                "num_chips": str(len(main_region.chip_names)),
            },
            throughput_unit="rows/s",
            throughput_count=total_rows,
        )


def main() -> int:
    return Sp1ZerocheckBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
