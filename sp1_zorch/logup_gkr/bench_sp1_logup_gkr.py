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

Times the whole dense prove (trace gen + the per-layer sumcheck chain) traced
into one fused ``@jit`` program, so the launch-bound eager dispatch wall
collapses to a single compilation. Not eager op-by-op, not a single layer
reduction. The ``lower`` thunk lets zkbench split the phase grid: compile
(``compile_time`` / ``compile_memory``) vs runtime (``latency`` / ``memory``).

The Fiat-Shamir / MLE field is koalabearx4 — the production workload: real
logup denominators are extension-field, which is where the "XLA materializes
EF intermediates to HBM" register pressure this bench exists to measure
actually lives. The fused prove is built here rather than reusing ``zorch``'s
``prove_gkr_jitted``, which hardcodes a base-field transcript; parameterizing
that upstream (and moving this bench onto the marked poseidon2 transcript) is
the follow-up.

A ``py_binary`` (manual ``bazel run``), not a ``py_test``: the blocks segfault
under ``@jit`` on the ZKX CPU backend, so the bench is GPU-only. It also needs
a ZKX plugin that round-trips >int64 field constants through HLO text
(fractalyze/zkx#569).

    bazel run //sp1_zorch/logup_gkr:bench_sp1_logup_gkr -- --row-variables 16 18 20
"""

import argparse
import functools
from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import zk_dtypes
from jax import Array
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from zorch.logup_gkr.circuit import GkrLayer
from zorch.logup_gkr.testing import prove_gkr_with_transcript
from zorch.testkit.transcript import cheap_transcript

# The production Fiat-Shamir / MLE field: real logup denominators are
# extension-field (koalabearx4).
EF = zk_dtypes.koalabearx4_mont


def _rand_field(seed: int, shape: Sequence[int]) -> Array:
    """Inlined copy of zorch's ``testkit.random_field.rand_field`` — that target
    is visible only to ``//zorch:__subpackages__``, so an external consumer can't
    dep it. Draw canonical ints in ``[0, 2**30)`` (< every supported prime); the
    field dtype Montgomery-encodes them on cast (extension fields fill the
    leading limb, the rest stay zero — enough to exercise EF arithmetic)."""
    ints = np.random.default_rng(seed).integers(0, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=EF)


def _first_layer_mles(seed: int, iv: int, rv: int) -> tuple[Array, Array, Array, Array]:
    """The four random dense first-layer MLEs (n0, n1, d0, d1), ``2**(iv + rv)``
    wide; four distinct seeds so they don't alias. Fixed-length so the arity
    flowing into the prove is checkable, not a star-tuple."""
    width = 1 << (iv + rv)
    return (
        _rand_field(seed, (width,)),
        _rand_field(seed + 1, (width,)),
        _rand_field(seed + 2, (width,)),
        _rand_field(seed + 3, (width,)),
    )


@functools.partial(jax.jit, static_argnums=(4,))
def _prove(n0: Array, n1: Array, d0: Array, d1: Array, iv: int) -> Array:
    """The whole dense GKR prove fused into one program. The four first-layer
    MLEs are the traced inputs; `iv` (static) fixes the pyramid height, hence
    the unrolled layer count. Returns the last proved layer's round
    polynomials — the tail of the sequential carry, so it transitively forces
    the whole prove."""
    first = GkrLayer(n0, n1, d0, d1, num_interaction_variables=iv)
    proofs = prove_gkr_with_transcript(first, cheap_transcript(EF))[2]
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
            # iv is static (fixes the pyramid height); the four MLEs are the
            # traced inputs.
            op_args = (*_first_layer_mles(args.seed, iv, rv), iv)
            yield BenchmarkOp(
                # Op id matches SP1's logup_gkr_zkbench so the dashboard joins them.
                name=f"logup_gkr_r{rv}_i{args.num_interactions}_total",
                fn=functools.partial(_prove, *op_args),
                lower=functools.partial(_prove.lower, *op_args),
                metadata={
                    "field": "koalabearx4",
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
