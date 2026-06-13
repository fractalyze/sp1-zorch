# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""LogUp-GKR prove benchmark over a real rsp shard (GPU).

Drives the production jagged prover (``prove_logup_gkr``) on a captured SP1
shard and reports the warm wall-clock -- the number the LogUp-GKR A/B is
judged on. The scope matches whir-zorch's ``_phase("logup_gkr")`` (its
``prove_logup_gkr``): the full stage including the per-chip trace evaluation
at the final GKR point and its Fiat-Shamir absorb, not the bare sumcheck
chain. The batched PCS *opening proof* is a separate post-zerocheck stage
(``JaggedEvalRound``) and is out of scope here.

This replaces an earlier dense/synthetic sweep that drove zorch's uniform
prover at SP1 dimensions: that was not a valid A/B -- the dense envelope is
~14x the jagged work and byte-matches nothing. The bench now drives the same
jagged path the shard prover ships, on real interaction traces.

Byte-anchor: before timing, the head challenges (alpha, beta seeds) are
re-derived through the prover's own glue Rounds
(``sp1_zorch.logup_gkr.head``) and checked against the dump, so a drifted
preamble/witness aborts before timing the wrong computation. The definitive
full-stream byte-match is the separate ``verify_gkr_prove`` runnable (its
``post_gkr_diag`` scalar seals every round poly, opening, and trace eval).

The cold prove is a multi-minute XLA compile cascade; zkbench's warmup
absorbs it and the timed iterations are warm. The staged prove has no single
``lowered.compile()``, so the bench carries no compile metric -- observe
compile out of band.

A ``py_binary`` (manual ``bazel run``), GPU-only: the blocks segfault under
``@jit`` on the ZKX CPU backend.

    bazel run //sp1_zorch/logup_gkr:bench_sp1_logup_gkr -- \\
        --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard17
"""

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from sp1_zorch.logup_gkr.circuit import build_gkr_chips
from sp1_zorch.logup_gkr.head import GrindRound, HeadChallengesRound
from sp1_zorch.logup_gkr.prover import num_beta_values, prove_logup_gkr
from sp1_zorch.shard_prover.fixture_loader import (
    _parse_ef_list,
    _parse_kv_lines,
    check_match,
    load_fixture_shard,
)
from sp1_zorch.shard_prover.replay import (
    MAX_LOG_ROW_COUNT,
    preamble_transcript,
    shard_regions,
)

# SP1 hardcodes GKR_GRINDING_BITS = 12; the witness is replayed from the dump,
# so the grind search itself is never timed here.
_GKR_POW_BITS = 12


class Sp1LogupGkrBenchmark(JaxBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="sp1-zorch",
            version="0.1.0",
            default_iterations=5,
            # One warmup absorbs the multi-minute cold compile; the round zone
            # (zorch#249) then makes every subsequent iteration warm.
            default_warmup=1,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--shard-dir",
            type=str,
            required=True,
            help="rsp shard dump directory (e.g. .../rsp_dump/shard17).",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        # The harness runs each op as it is yielded, so file IO, region
        # assembly, and the anchor gate below -- everything up to the yield --
        # happen before any timing starts.
        shard_dir = Path(args.shard_dir)
        shard = load_fixture_shard(shard_dir)
        state = _parse_kv_lines(
            (shard_dir / "gpu_gkr_state.txt").read_text(), skip_unkeyed=True
        )

        traces = shard.main_trace_data.traces
        order = traces.chip_order
        main_region, prep_region = shard_regions(shard)
        num_betas = num_beta_values(shard.main_trace_data.chips)
        gkr_chips = build_gkr_chips(shard.main_trace_data.chips, order)
        num_row_variables = MAX_LOG_ROW_COUNT - 1
        preamble = preamble_transcript(shard, shard_dir)
        witness = jnp.array(int(state["witness"]), F)

        # Byte-anchor the head (alpha + beta seeds) through the prover's own
        # glue Rounds so a drifted preamble/witness aborts before timing the
        # wrong computation. This is the cheap head leg, not the full prove --
        # the definitive full-stream match is verify_gkr_prove's job.
        ok = True
        _, transcript, _ = GrindRound(witness, pow_bits=_GKR_POW_BITS)(None, preamble)
        _, _, head = HeadChallengesRound(num_betas)(None, transcript)
        ok &= check_match("alpha", head.alpha, _parse_ef_list(state["alpha"])[0])
        for i in range(head.beta_seeds.shape[0]):
            ok &= check_match(
                f"beta_seed[{i}]",
                head.beta_seeds[i],
                _parse_ef_list(state[f"beta_seed[{i}]"])[0],
            )
        if not ok:
            sys.exit("head anchors diverged from the dump; aborting")

        def _prove():
            # jit=True keeps the `zorch.sumcheck` composite intact for the
            # vendor's register-resident emitter; with the module-level
            # shape-keyed round zone (zorch#249) warm iterations cache-hit
            # instead of re-tracing each layer. The chain is rebuilt per call
            # so the lazy one-live-layer release holds -- peak residency is a
            # single pyramid layer, not the whole pyramid.
            _, proof = prove_logup_gkr(
                gkr_chips,
                main_region,
                prep_region,
                preamble,
                num_betas=num_betas,
                num_row_variables=num_row_variables,
                pow_bits=_GKR_POW_BITS,
                witness=witness,
                jit=True,
            )
            # Block on arrays that transitively force the whole stage: the
            # final eval point carries the sequential round chain, and the
            # per-chip openings carry the trace evaluation at that point.
            return proof.eval_point, [
                ev.all_evals() for ev in proof.chip_openings.values()
            ]

        total_rows = sum(traces.per_chip[n].num_real for n in order)
        yield BenchmarkOp(
            # The real-shard logup-gkr stage time, joinable on the dashboard
            # with whir-zorch's `_phase("logup_gkr")`.
            name="logup_gkr_total",
            fn=_prove,
            metadata={
                "shard": shard_dir.name,
                "field": "koalabear",
                "num_chips": str(len(order)),
                "num_gkr_chips": str(len(gkr_chips)),
                "num_betas": str(num_betas),
                "num_row_variables": str(num_row_variables),
            },
            throughput_unit="rows/s",
            throughput_count=total_rows,
        )


def main() -> int:
    return Sp1LogupGkrBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
