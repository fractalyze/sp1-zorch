# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Trace-commit phase ablation over an rsp capture (GPU benchmark).

Phases mirror the eventual ``prove_shard`` per-phase report so these numbers
slot into the e2e ablation unchanged: ``pack`` (region assembly from
already-loaded traces), ``rs_encode`` (stacked encode + bit-reverse), and
``smcs`` (Merkle commit + shape/structure binds). File loading stays outside
every timer.

    bazel run //sp1_zorch/commit:bench_trace_commit -- \\
        --shard-dir=/path/to/rsp_dump/shardN --region=prep
"""

import argparse
import functools
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
from jax import lax
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from sp1_zorch.commit.region import JaggedRegion, _pack_chip_data
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.fixture_loader import read_dump
from zorch.coding.reed_solomon import ReedSolomon
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

# SP1 core machine parameters (sp1-hypercube core config).
_LOG_STACKING_HEIGHT = 21
_MAX_LOG_ROW_COUNT = 22
_LOG_BLOWUP = 2


class TraceCommitBenchmark(JaxBenchmark):
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
            "--region",
            choices=["prep", "main"],
            default="prep",
            help="Which trace region to commit.",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        # Everything below up to the op definitions is untimed setup: file
        # IO, chip selection, and the phase inputs each op consumes.
        dump = read_dump(Path(args.shard_dir))
        traces = dump.preprocessed if args.region == "prep" else dump.traces
        chips = [traces[name] for name in sorted(traces)]
        region = JaggedRegion.from_chips(
            chips,
            log_stacking_height=_LOG_STACKING_HEIGHT,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=tuple(sorted(traces)),
        )
        S = 1 << _LOG_STACKING_HEIGHT
        K = region.dense.shape[0] // S
        dtype = region.dense.dtype

        perm = Poseidon2(koalabear16_params())
        smcs = SingleMatrixCommitmentScheme(
            Sponge(perm, SpongeParams(rate=8, out=8)),
            Compression(perm, CompressionParams(arity=2, chunk=8)),
        )
        rs = ReedSolomon(message_len=S, blowup=1 << _LOG_BLOWUP, dtype=dtype)

        nonempty = tuple(c for c in chips if c.shape[0] > 0 and c.shape[1] > 0)
        num_added_vals = int(region.dense.shape[0]) - region.raw_size

        @jax.jit
        def _rs_phase(dense):
            # SP1's bit-reversed codeword order rides the encode phase.
            return lax.bit_reverse(rs.encode(dense.reshape(K, S)), dimensions=(1,))

        @jax.jit
        def _smcs_phase(codeword, row_counts, column_counts):
            commitment, digest_layers = smcs.commit(codeword.T)
            return smcs.bind_structure(commitment, row_counts, column_counts), (
                digest_layers
            )

        # Phase inputs, materialized outside the timers.
        dense = region.dense
        codeword = jax.block_until_ready(_rs_phase(dense))
        row_counts = jnp.array(region.row_counts, dtype=dtype)
        column_counts = jnp.array(region.column_counts, dtype=dtype)

        meta = {
            "region": args.region,
            "num_chips": str(region.num_chips),
            "stacked_columns": str(K),
            "log_stacking_height": str(_LOG_STACKING_HEIGHT),
            "log_blowup": str(_LOG_BLOWUP),
        }
        dense_elems = int(dense.shape[0])

        yield BenchmarkOp(
            name=f"trace_commit_{args.region}_pack",
            fn=functools.partial(
                _pack_chip_data,
                nonempty,
                num_added_vals=num_added_vals,
                pad_dtype=dtype,
            ),
            lower=functools.partial(
                _pack_chip_data.lower,
                nonempty,
                num_added_vals=num_added_vals,
                pad_dtype=dtype,
            ),
            metadata=meta,
            throughput_unit="elems/s",
            throughput_count=dense_elems,
        )
        yield BenchmarkOp(
            name=f"trace_commit_{args.region}_rs_encode",
            fn=functools.partial(_rs_phase, dense),
            lower=functools.partial(_rs_phase.lower, dense),
            metadata=meta,
            throughput_unit="elems/s",
            throughput_count=dense_elems,
        )
        yield BenchmarkOp(
            name=f"trace_commit_{args.region}_smcs",
            fn=functools.partial(_smcs_phase, codeword, row_counts, column_counts),
            lower=functools.partial(
                _smcs_phase.lower, codeword, row_counts, column_counts
            ),
            metadata=meta,
            throughput_unit="rows/s",
            throughput_count=S << _LOG_BLOWUP,
        )


def main() -> int:
    return TraceCommitBenchmark().run()


if __name__ == "__main__":
    main()
