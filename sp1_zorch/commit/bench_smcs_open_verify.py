# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""GPU benchmark — SP1 SMCS open / verify throughput on zorch's Merkle blocks.

open/verify has no standalone SP1 FFI entry (the vendored ``libsp1_gpu_jax_ffi``
exports only ``sp1_merkle_commit_kb31`` + ``sp1_verify_shard``), so a direct
vs-SP1 A/B is impossible — this is an **absolute-throughput** regression gate,
the open/verify companion to the commit-side numbers on fractalyze/sp1-zorch#2.

  - open   = ``SMCS.open_batch``  : vmap of ``MerkleTree.open`` over Q queries
             (sibling gather).
  - verify = ``SMCS.verify_batch``: vmap over Q of the ``reconstruct_root`` fold
             (log_height compress permutes) + the SP1 domain-separator rebind.

Each is one fused ``@jit`` program; the ``lower`` thunk lets zkbench split the
phase grid into compile vs runtime. A ``py_binary`` (manual ``bazel run``), not a
``py_test``: the blocks segfault under ``@jit`` on the ZKX CPU backend, so the
bench is GPU-only — same as ``bench_sp1_logup_gkr``.

``commit`` emits the ``zorch.merkle_commit`` composite, so this needs a
composite-capable plugin. The pinned ``jax_cuda12_pjrt`` wheel errors with
``stablehlo.composite is unknown`` (the stablehlo#83 + zkx#497 legalization isn't
in the pin yet), so run against a composite-capable plugin via
``ZKX_GPU_PLUGIN_PATH`` until it is the pinned default. The #2 baseline (RTX 5090,
koalabear poseidon2-16) is recorded on the issue.

    ZKX_GPU_PLUGIN_PATH=/path/to/pjrt_c_api_gpu_plugin.so \\
        bazel run //sp1_zorch/commit:bench_smcs_open_verify -- --log-heights 16 18 20 22
"""

import argparse
import functools
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import koalabear_mont as F
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams


def _smcs() -> SingleMatrixCommitmentScheme:
    """SP1's koalabear16 SMCS (rate/out=8, arity=2, chunk=8) — the params the
    byte-match tests pin."""
    perm = Poseidon2(koalabear16_params())
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    return SingleMatrixCommitmentScheme(sponge, comp)


def _query_indices(num_queries: int, height: int) -> Array:
    """Deterministic spread of query rows; 7919 is coprime to every power-of-two
    height, so the stride never collapses onto a single row."""
    return jnp.asarray((np.arange(num_queries) * 7919) % height, dtype=jnp.int32)


def _make_open(smcs: SingleMatrixCommitmentScheme):
    @jax.jit
    def open_fn(matrix: Array, indices: Array, layers: list[Array]):
        return smcs.open_batch(indices, matrix, layers)

    return open_fn


def _make_verify(smcs: SingleMatrixCommitmentScheme, height: int, width: int):
    @jax.jit
    def verify_fn(commitment: Array, indices: Array, rows: Array, proofs: list[Array]):
        def one(commitment: Array, idx: Array, row: Array, proof: list[Array]) -> Array:
            return smcs.verify_batch(commitment, (height, width), idx, row, proof)

        # commitment is broadcast (in_axes=None); only the per-query leaves map.
        return jax.vmap(one, in_axes=(None, 0, 0, 0))(commitment, indices, rows, proofs)

    return verify_fn


class SmcsOpenVerifyBenchmark(JaxBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="sp1-zorch",
            version="0.1.0",
            # 10 (vs the logup bench's 5): open/verify are sub-ms, so more
            # samples tighten the median.
            default_iterations=10,
            default_warmup=3,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--log-heights",
            type=int,
            nargs="+",
            default=[20],
            help="log2 committed-matrix rows to sweep (#2 baseline: 16 18 20 22)",
        )
        parser.add_argument(
            "--num-queries",
            type=int,
            default=100,
            help="opened rows per commitment (FRI query count; default 100)",
        )
        parser.add_argument(
            "--width",
            type=int,
            default=8,
            help="committed-matrix width (default 8)",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        smcs = _smcs()
        open_fn = _make_open(smcs)
        commit_fn = jax.jit(smcs.commit)
        width = args.width
        for logh in args.log_heights:
            height = 1 << logh
            matrix = jnp.arange(height * width, dtype=F).reshape(height, width)
            commitment, layers = commit_fn(matrix)
            indices = _query_indices(args.num_queries, height)
            # Setup (not timed): produce the rows + sibling paths verify consumes.
            rows, proofs = open_fn(matrix, indices, layers)
            verify_fn = _make_verify(smcs, height, width)

            meta = {
                "field": "koalabear",
                "log_height": str(logh),
                "width": str(width),
                "num_queries": str(args.num_queries),
            }
            open_args = (matrix, indices, layers)
            yield BenchmarkOp(
                name=f"smcs_open_h{logh}_q{args.num_queries}",
                fn=functools.partial(open_fn, *open_args),
                lower=functools.partial(open_fn.lower, *open_args),
                metadata=meta,
                throughput_unit="queries/s",
                throughput_count=args.num_queries,
            )
            verify_args = (commitment, indices, rows, proofs)
            yield BenchmarkOp(
                name=f"smcs_verify_h{logh}_q{args.num_queries}",
                fn=functools.partial(verify_fn, *verify_args),
                lower=functools.partial(verify_fn.lower, *verify_args),
                metadata=meta,
                throughput_unit="queries/s",
                throughput_count=args.num_queries,
            )


def main() -> int:
    return SmcsOpenVerifyBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
