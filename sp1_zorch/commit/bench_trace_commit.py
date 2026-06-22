# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""GPU benchmark — trace-commit compile vs runtime on the rsp shard data.

``verify_trace_commit`` wraps one commit in ``time.monotonic()``, so its number
is region-build + trace + lower + compile + execute lumped together and cannot
tell compile from steady-state runtime. This subclasses ``zkbench.JaxBenchmark``
(like the sibling ``bench_smcs_open_verify``) so each phase reports the
compile/runtime split via the ``lower`` thunk (run ``--phase compile`` and
``--phase runtime`` as separate invocations) and the GPU memory high-water mark.

  - commit_region : the production entry — full @jit commit (encode+Merkle+bind)
  - rs_encode     : the NTT alone
  - smcs_commit   : the bit-reversed codeword Merkle commit (poseidon2) alone
  - codeword_transpose : the [K, block_len] -> [block_len, K] relayout alone —
    the transpose the OLD row-major commit paid in-@jit. The column-major
    commit_region now reads the [K, N] codeword by column and skips it, so this
    op is the reference for the cost the column-major fix removed (#140).

Pass ``--drop_codeword`` to time commit_region in production drop_ldes mode,
where the retained-codeword transpose is dropped entirely.

The region is built once (eager ``from_chips``) as setup. ``commit_region`` is
yielded first so it runs before the breakdown materializes a ~6 GB codeword
(zkbench consumes ops lazily, one finished before the next is pulled).

GPU-only; needs a composite-capable plugin via ``ZKX_GPU_PLUGIN_PATH`` (the
byte-match plugin), since the pinned wheel cannot legalize the poseidon2 / NTT
composites — same as ``bench_smcs_open_verify``.

    ZKX_GPU_PLUGIN_PATH=/path/to/pjrt_c_api_gpu_plugin.so JAX_PLATFORMS=cuda \\
        bazel run //sp1_zorch/commit:bench_trace_commit -- \\
        --shard_dir=/path/to/rsp_dump/shard1
"""

import argparse
import functools
from collections.abc import Iterable
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.commit.trace_commit import _commit_jit, commit_region
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.fixture_loader import read_dump
from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

# SP1 core machine parameters (sp1-hypercube core config): 2^21 stacking
# height, 2^22 max rows, 4x blowup — the params the byte-match harness pins.
_LOG_STACKING_HEIGHT = 21
_MAX_LOG_ROW_COUNT = 22
_LOG_BLOWUP = 2


def _smcs() -> SingleMatrixCommitmentScheme:
    """SP1's koalabear16 SMCS (rate/out=8, arity=2, chunk=8)."""
    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _commit_bound(
    region: JaggedRegion,
    smcs: SingleMatrixCommitmentScheme,
    drop_codeword: bool = False,
) -> Array:
    """Run the full commit but return only the small bound commitment.

    The benchmark harness holds the last ``fn()`` result while the next call
    runs, so returning the full ``TraceCommitData`` (~10 GB of codeword + mle +
    digests at rsp scale) would keep two copies live and OOM a 32 GB card.
    Returning ``bound`` blocks on the whole pipeline yet retains nothing big.

    ``drop_codeword`` (SP1's drop_ldes) is the production mode: the codeword is
    not retained as an output, so XLA frees it right after the Merkle commit.
    """
    bound, _ = commit_region(
        region, smcs, log_blowup=_LOG_BLOWUP, jit=True, drop_codeword=drop_codeword
    )
    return bound


class TraceCommitBenchmark(JaxBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="sp1-zorch",
            version="0.1.0",
            default_iterations=5,
            default_warmup=2,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--shard_dir", required=True, help="rsp shard dump directory."
        )
        parser.add_argument(
            "--drop_codeword",
            action="store_true",
            help="commit_region in production drop_ldes mode (codeword not "
            "retained as an output).",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        dump = read_dump(Path(args.shard_dir))
        smcs = _smcs()
        region = JaggedRegion.from_chips(
            [dump.traces[name] for name in sorted(dump.traces)],
            log_stacking_height=_LOG_STACKING_HEIGHT,
            max_log_row_count=_MAX_LOG_ROW_COUNT,
            chip_names=tuple(sorted(dump.traces)),
        )
        S = 1 << region.log_stacking_height
        K = region.dense.shape[0] // S
        message = region.dense.reshape(K, S)
        row_counts = jnp.array(region.row_counts, dtype=region.dense.dtype)
        column_counts = jnp.array(region.column_counts, dtype=region.dense.dtype)
        leaves = S << _LOG_BLOWUP  # committed codeword rows (Merkle leaves)
        meta = {
            "field": "koalabear",
            "chips": str(len(dump.traces)),
            "log_stacking_height": str(_LOG_STACKING_HEIGHT),
            "log_blowup": str(_LOG_BLOWUP),
        }

        # Production entry. fn times the full commit_region (the host-side
        # TraceCommitData wrap is ~free); lower times the @jit zone's compile.
        commit_kw = {
            "smcs": smcs,
            "log_blowup": _LOG_BLOWUP,
            "drop_codeword": args.drop_codeword,
        }
        yield BenchmarkOp(
            name="commit_region",
            fn=functools.partial(
                _commit_bound, region, smcs, drop_codeword=args.drop_codeword
            ),
            lower=functools.partial(
                _commit_jit.lower, message, row_counts, column_counts, **commit_kw
            ),
            metadata=meta,
            throughput_unit="leaves/s",
            throughput_count=leaves,
            measure_memory=True,
        )

        # Supplementary breakdown: the NTT encode and the Merkle commit alone.
        # Created after the commit_region op so its 6 GB codeword is not live
        # while commit_region runs.
        code = BitReversedReedSolomon(
            message_len=S, blowup=1 << _LOG_BLOWUP, dtype=message.dtype
        )
        encode = jax.jit(code.encode)
        yield BenchmarkOp(
            name="rs_encode",
            fn=functools.partial(encode, message),
            lower=functools.partial(encode.lower, message),
            metadata=meta,
            throughput_unit="leaves/s",
            throughput_count=leaves,
        )
        # The column-major SMCS commits the codeword in its native [K, N] encode
        # layout (a leaf is a column) — no transpose, matching what commit_region
        # feeds it.
        codeword = jax.block_until_ready(encode(message))
        commit = jax.jit(smcs.commit, static_argnames=("column_major",))
        yield BenchmarkOp(
            name="smcs_commit",
            fn=functools.partial(commit, codeword, column_major=True),
            lower=functools.partial(commit.lower, codeword, column_major=True),
            metadata=meta,
            throughput_unit="leaves/s",
            throughput_count=leaves,
        )
        # Time the codeword [K, block_len] -> [block_len, K] relayout alone: the
        # transpose the OLD row-major commit paid in-@jit. The column-major
        # commit_region now skips it (it reads the [K, N] codeword by column), so
        # this op is the reference for the cost that fix removed.
        transpose = jax.jit(lambda c: c.T)
        yield BenchmarkOp(
            name="codeword_transpose",
            fn=functools.partial(transpose, codeword),
            lower=functools.partial(transpose.lower, codeword),
            metadata=meta,
            throughput_unit="leaves/s",
            throughput_count=leaves,
        )


def main() -> int:
    return TraceCommitBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
