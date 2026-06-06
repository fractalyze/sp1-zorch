# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""LogUp-GKR stage phase ablation over an rsp capture (GPU benchmark).

Phases mirror the eventual ``prove_shard`` per-phase report: ``first_layer``
(interaction fingerprinting), ``circuit_layers`` (the transition pyramid),
``gkr_rounds`` (the per-layer sumcheck chain), ``trace_open`` (all-chip
openings at the final point), plus ``total`` (the whole
``prove_logup_gkr``). ``total`` is the op that joins the SP1 reference's
logup-gkr stage time; the four sub-phases are this prover's own ablation
(SP1's bench does not split the stage the same way). These op ids name the
real-shard phase spans -- a distinct family from the synthetic
``logup_gkr_r{rv}_i{ni}`` sweep in ``bench_sp1_logup_gkr.py``, so a
dashboard must not union the two ``*_total`` ids.
SP1 reference:
https://github.com/fractalyze/sp1/blob/e2c02f376/sp1-gpu/crates/logup_gkr/bin/logup_gkr_zkbench.rs

File loading, region assembly, and every phase's inputs stay outside the
timers.

The phase inputs are re-derived here rather than captured from inside
``prove_logup_gkr`` (its lazy layer release is load-bearing; threading
capture hooks through it would change what it benchmarks). The re-derived
head is byte-anchored against the dump's recorded challenges (alpha, beta
seeds, z1, round-0 lambda/claim) and the run aborts before timing anything
on any drift, so the phase inputs cannot silently diverge from what the
real prove sees.

Unlike the real prove, the ablation ops hold their inputs live across
iterations: the setup pyramid stays resident whenever ``gkr_rounds`` or
``trace_open`` is selected, so co-selecting those with ``circuit_layers``
or ``total`` doubles peak pyramid residency. Run ``--ops total`` in its
own invocation for the headline number; subset the rest as memory allows.

    bazel run //sp1_zorch/logup_gkr:bench_logup_gkr_phases -- \\
        --shard-dir=/path/to/rsp_dump/shardN
"""

import argparse
import functools
import sys
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from sp1_zorch.logup_gkr.circuit import (
    build_gkr_chips,
    generate_circuit_layers,
    generate_first_layer,
)
from sp1_zorch.logup_gkr.prover import (
    extract_sp1_outputs,
    num_beta_values,
    open_traces,
    prove_logup_gkr,
)
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
from zorch.logup_gkr.jagged_prover import JaggedGkrLayerRound
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.round import ProveChain
from zorch.transcript import sample_challenge
from zorch.utils.bits import log2_ceil_usize, log2_strict_usize

_EF_LIMBS = 4
# SP1 hardcodes GKR_GRINDING_BITS = 12; the witness is replayed from the
# dump, so the grind search itself is outside every phase here.
_GKR_POW_BITS = 12

_OPS = ("first_layer", "circuit_layers", "gkr_rounds", "trace_open", "total")


def _layer_planes(layer):
    """The four MLE planes — returned from the layer ops so the harness's
    ``block_until_ready`` reaches them (``JaggedGkrLayer`` is not a pytree)."""
    return (
        layer.numerator_0,
        layer.numerator_1,
        layer.denominator_0,
        layer.denominator_1,
    )


class LogupGkrPhasesBenchmark(JaxBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="sp1-zorch",
            version="0.1.0",
            default_iterations=5,
            default_warmup=1,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--shard-dir",
            type=str,
            required=True,
            help="rsp shard dump directory (e.g. .../rsp_dump/shard1).",
        )
        parser.add_argument(
            "--ops",
            nargs="+",
            choices=_OPS,
            default=list(_OPS),
            help="Op subset, for bounding resident memory per invocation.",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        # The harness runs each op as it is yielded, so everything below up
        # to the first yield — file IO, regions, phase-entry states, and the
        # anchor gate — happens before any timing starts.
        ops = set(args.ops)
        need_first = bool({"circuit_layers", "gkr_rounds", "trace_open"} & ops)
        need_layers = bool({"gkr_rounds", "trace_open"} & ops)
        need_chain = "trace_open" in ops

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

        # The head replicated from prove_logup_gkr, anchored to the dump so
        # this copy cannot drift from the prover (see module docstring).
        ok = True
        transcript = preamble.observe(witness)
        transcript, pow_sample = transcript.sample(1)
        if int(pow_sample[0]) & ((1 << _GKR_POW_BITS) - 1):
            raise ValueError("recorded witness fails the proof of work")
        transcript, alpha = sample_challenge(transcript, EF, _EF_LIMBS)
        ok &= check_match("alpha", alpha, _parse_ef_list(state["alpha"])[0])
        seeds = []
        for i in range(log2_ceil_usize(num_betas)):
            transcript, seed = sample_challenge(transcript, EF, _EF_LIMBS)
            ok &= check_match(
                f"beta_seed[{i}]", seed, _parse_ef_list(state[f"beta_seed[{i}]"])[0]
            )
            seeds.append(seed)
        # SP1 samples one extra public-values challenge here and discards it.
        transcript, _ = sample_challenge(transcript, EF, _EF_LIMBS)
        one = jnp.ones((), dtype=EF)
        betas = (
            one[None] if not seeds else expand_eq_to_hypercube(jnp.stack(seeds), one)
        )

        if need_first:
            first_layer = generate_first_layer(
                gkr_chips, main_region, prep_region, alpha, betas
            )

        if need_layers:
            layers = generate_circuit_layers(first_layer, num_row_variables)
            output = extract_sp1_outputs(layers[-1])
            bf_dtype = main_region.dense.dtype
            transcript = transcript.observe(
                jnp.array(output.numerator.shape[0], bf_dtype)
            )
            transcript = transcript.observe(output.numerator)
            transcript = transcript.observe(
                jnp.array(output.denominator.shape[0], bf_dtype)
            )
            transcript = transcript.observe(output.denominator)
            coords = []
            for i in range(log2_strict_usize(output.numerator.shape[0])):
                transcript, c = sample_challenge(transcript, EF, _EF_LIMBS)
                ok &= check_match(f"z1[{i}]", c, _parse_ef_list(state[f"z1[{i}]"])[0])
                coords.append(c)
            z1 = jnp.stack(coords)
            carry = (eval_mle(output.numerator, z1), eval_mle(output.denominator, z1))
            chain_entry = transcript

            def _rounds_fn():
                stack = list(layers)
                chain = ProveChain(
                    JaggedGkrLayerRound(stack.pop(), _EF_LIMBS)
                    for _ in range(len(stack))
                )
                return chain((*carry, z1), chain_entry)

        if need_chain:
            (_, _, eval_point), open_entry, round_proofs = _rounds_fn()
            rounds_ref = _parse_kv_lines(
                (shard_dir / "gkr_sumcheck_rounds.txt")
                .read_text()
                .split("--- round ---")[1],
                skip_unkeyed=True,
            )
            ok &= check_match(
                "round[0].lambda",
                round_proofs[0].lam,
                _parse_ef_list(rounds_ref["lambda"])[0],
            )
            ok &= check_match(
                "round[0].claim",
                round_proofs[0].claim,
                _parse_ef_list(rounds_ref["claim"])[0],
            )

        if not ok:
            # Drifted phase inputs would time the wrong computation.
            sys.exit("phase-entry anchors diverged from the dump; aborting")

        meta = {
            "shard": shard_dir.name,
            "field": "koalabear",
            "num_chips": str(len(order)),
            "num_gkr_chips": str(len(gkr_chips)),
            "num_betas": str(num_betas),
            "num_row_variables": str(num_row_variables),
        }
        total_rows = sum(traces.per_chip[n].num_real for n in order)

        def _op(name, fn) -> BenchmarkOp:
            return BenchmarkOp(
                name=name,
                fn=fn,
                metadata=meta,
                throughput_unit="rows/s",
                throughput_count=total_rows,
            )

        if "first_layer" in ops:
            yield _op(
                "logup_gkr_first_layer",
                lambda: _layer_planes(
                    generate_first_layer(
                        gkr_chips, main_region, prep_region, alpha, betas
                    )
                ),
            )
        if "circuit_layers" in ops:
            # The floor depends on every transition, so blocking its planes
            # waits for the whole pyramid.
            yield _op(
                "logup_gkr_circuit_layers",
                lambda: _layer_planes(
                    generate_circuit_layers(first_layer, num_row_variables)[-1]
                ),
            )
        if "gkr_rounds" in ops:
            yield _op("logup_gkr_rounds", _rounds_fn)
        if "trace_open" in ops:
            yield _op(
                "logup_gkr_trace_open",
                functools.partial(
                    open_traces,
                    main_region,
                    prep_region,
                    eval_point,
                    open_entry,
                    trace_dimension=num_row_variables + 1,
                ),
            )
        if "total" in ops:
            yield _op(
                "logup_gkr_total",
                functools.partial(
                    prove_logup_gkr,
                    gkr_chips,
                    main_region,
                    prep_region,
                    preamble,
                    num_betas=num_betas,
                    num_row_variables=num_row_variables,
                    pow_bits=_GKR_POW_BITS,
                    witness=witness,
                ),
            )


def main() -> int:
    return LogupGkrPhasesBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
