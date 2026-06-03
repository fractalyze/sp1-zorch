"""GPU benchmark — SP1-shaped dense LogUp-GKR prove.

The SP1-specific consumer of zorch's agnostic dense prover
(``@zorch//zorch/logup_gkr``): only SP1 sizing/config lives here, the prover
stays in zorch (no scheme-agnostic fork). Knobs and defaults mirror SP1's
reference bench, pinned so future readers can diff against it:
https://github.com/fractalyze/sp1/blob/e2c02f376/sp1-gpu/crates/logup_gkr/bin/logup_gkr_zkbench.rs

SP1 sizes the first layer by ``(num_interactions, row_variables)``; zorch's
prover by ``(num_interaction_variables, num_row_variables)``. The bridge:
``iv = log2(next_pow2(num_interactions))`` (SP1 pads interactions to a power of
two) and ``rv = row_variables``.

Times the whole dense prove (trace gen + the per-layer sumcheck chain), like
zorch's ``bench_logup_gkr`` — not a single layer reduction.

A ``py_binary`` (manual ``bazel run``), not a ``py_test``: the blocks segfault
under ``@jit`` on the ZKX CPU backend, so the bench is GPU-only.

    bazel run //sp1_zorch/logup_gkr:bench_sp1_logup_gkr -- --row-variables 16 18 20
"""

import argparse
from collections.abc import Iterable, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import koalabear_mont as F
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from zorch.logup_gkr.circuit import GkrLayer
from zorch.logup_gkr.testing import prove_gkr


def _rand_field(seed: int, shape: Sequence[int], dtype: Any) -> Array:
    """Inlined copy of zorch's ``testkit.random_field.rand_field`` — that target
    is visible only to ``//zorch:__subpackages__``, so an external consumer can't
    dep it. Draw canonical ints in ``[0, 2**30)`` (< every supported prime); the
    field dtype Montgomery-encodes them on cast."""
    ints = np.random.default_rng(seed).integers(0, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=dtype)


def _first_layer(seed: int, iv: int, rv: int) -> GkrLayer:
    """Four distinct seeds so numerators/denominators don't alias; Montgomery
    field to match zorch's ``bench_logup_gkr``."""
    width = 1 << (iv + rv)
    return GkrLayer(
        numerator_0=_rand_field(seed, (width,), F),
        numerator_1=_rand_field(seed + 1, (width,), F),
        denominator_0=_rand_field(seed + 2, (width,), F),
        denominator_1=_rand_field(seed + 3, (width,), F),
        num_interaction_variables=iv,
    )


def _num_challenges(iv: int, rv: int) -> int:
    """Upper bound on Fiat-Shamir draws: ``iv + 1`` for the output point, then per
    proved layer at most ``lam + (iv + rv) sumcheck rounds + 1`` reduction.
    StubTranscript reads only the prefix it needs, so over-estimating is free."""
    return (iv + 1) + rv * (iv + rv + 2)


def _prove(first: GkrLayer, challenges: Array) -> Array:
    """Return the last-proved layer's round polys — the tail of the sequential
    carry, so awaiting it awaits the whole prove (the value worth timing)."""
    _, _, proofs, _ = prove_gkr(first, challenges)
    return proofs[-1].round_polys


class Sp1LogupGkrBenchmark(JaxBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="sp1-zorch",
            version="0.1.0",
            # 5/3 mirrors SP1's logup_gkr_zkbench so the harness config matches.
            default_iterations=5,
            default_warmup=3,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--row-variables",
            type=int,
            nargs="+",
            default=[18],
            help="log2 rows folded by the GKR pyramid (SP1 default: 18)",
        )
        parser.add_argument(
            "--num-interactions",
            type=int,
            default=64,
            help="interactions at the floor, padded to a power of two (SP1 default: 64)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="PRNG seed (SP1 default: 42)",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        # SP1 pads interactions to a power of two before counting interaction vars.
        iv = (args.num_interactions - 1).bit_length()
        for rv in args.row_variables:
            first = _first_layer(args.seed, iv, rv)
            challenges = _rand_field(args.seed + 99, (_num_challenges(iv, rv),), F)
            yield BenchmarkOp(
                # Op id matches SP1's logup_gkr_zkbench so the dashboard joins them.
                name=f"logup_gkr_r{rv}_i{args.num_interactions}_total",
                fn=lambda f=first, c=challenges: _prove(f, c),
                metadata={
                    "field": "koalabear",
                    "interaction_variables": str(iv),
                    "row_variables": str(rv),
                    "num_interactions": str(args.num_interactions),
                    "seed": str(args.seed),
                },
                throughput_unit="evals/s",
                throughput_count=1 << (iv + rv),
            )


def main() -> int:
    return Sp1LogupGkrBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
