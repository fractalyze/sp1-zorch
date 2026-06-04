# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""GPU smoke: fold a real SP1 chip's constraints through zorch.constraint_eval.

Loads a chip from ``rw_constraints`` (riscv-witness's exported per-row
constraint evaluation) and folds its K constraints under one challenge via
``constraint_rlc``, asserting byte-equality with the plain
``eval_fn(rows) @ rlc_coeffs(alpha, K)`` dot the SP1 reference computes. This
is the consumer half of the convergence: ``chip.eval_constraints(trace) ->
[N, K]`` is exactly the ``eval_fn`` ``constraint_rlc`` expects, so
riscv-witness's eval drops in with no glue.

A ``py_binary`` (manual ``bazel run``), not a ``py_test``: the chip eval runs
on the ZKX GPU backend, where eager evaluation of a real constraint crashes
the plugin — it must go through ``@jax.jit``. The trace is built with
``.view(F)`` (the canonical int IS the raw Montgomery bitpattern) rather than
the ``dtype=F`` Montgomery-encode kernel, which destabilizes the plugin when
mixed with the constraint body's own ``.view`` bitcasts.

    bazel run //sp1_zorch/zerocheck:rw_fold_smoke -- --chip jalr
"""

from __future__ import annotations

import argparse
import atexit
import shutil
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from zk_dtypes import koalabear_mont as F

from rw_constraints import ConstraintRegistry, bundled_constraints_dir

from sp1_zorch.zerocheck.prover import constraint_rlc, rlc_coeffs


def _registry() -> ConstraintRegistry:
    """A registry over the wheel's bundled constraint data.

    ``ConstraintRegistry.load`` rejects any chip file whose resolved path
    escapes its version dir. Under Bazel runfiles each chip ``.py`` is a
    symlink into ``execroot/.../external``, so it always "escapes". Copy the
    bundled tree into a real directory (symlinks dereferenced) and load from
    there. A source checkout with data in place needs no copy."""
    src = bundled_constraints_dir()
    if src is None:
        return ConstraintRegistry()
    tmp = tempfile.mkdtemp(prefix="rw_constraints_")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    dst = Path(tmp) / "constraints"
    shutil.copytree(src, dst, symlinks=False)
    return ConstraintRegistry(base_dir=dst)


def _random_trace(seed: int, rows: int, num_cols: int) -> jax.Array:
    """A random ``[rows, num_cols]`` field trace. Draw canonical ints in
    ``[0, 2**30)`` (below the modulus) and ``.view`` them as field elements:
    their raw bitpattern is already a valid Montgomery encoding, so this skips
    the encode kernel."""
    ints = np.random.default_rng(seed).integers(
        0, 1 << 30, size=(rows, num_cols), dtype=np.uint32
    )
    return jnp.asarray(ints).view(F)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", default="jalr", help="chip name (default: jalr)")
    parser.add_argument("--rows", type=int, default=16, help="trace rows (default: 16)")
    parser.add_argument("--alpha", type=int, default=7, help="challenge (default: 7)")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed (default: 0)")
    args = parser.parse_args()
    # Reject degenerate inputs that pass byte-equality without exercising the
    # fold: an empty trace, or alpha=0 which zeroes all but the last coefficient.
    if args.rows < 1:
        parser.error("--rows must be >= 1")
    if args.alpha == 0:
        parser.error("--alpha must be non-zero")

    chip = _registry().load("sp1", "v1", constraint_field_dtype=F)[args.chip]
    trace = _random_trace(args.seed, args.rows, chip.num_cols)
    eval_fn = chip.eval_constraints
    alpha = jnp.full((), args.alpha, dtype=jnp.uint32).view(F)

    # K is static; read it from an abstract trace — make_jaxpr does not execute,
    # so it sidesteps both the eager-eval plugin crash and an extra GPU compile.
    # Keep the fold and the dot as separate jits so each is lowered
    # independently: the byte-equality is a cross-check, and once CompositeOp is
    # enabled one fused jit could CSE the marked path into the plain dot.
    num_constraints = jax.make_jaxpr(eval_fn)(trace).out_avals[0].shape[-1]
    fold = jax.jit(lambda t: constraint_rlc(eval_fn, t, alpha, num_constraints))
    dot = jax.jit(lambda t: eval_fn(t) @ rlc_coeffs(alpha, num_constraints))
    folded = jax.block_until_ready(fold(trace))
    dotted = jax.block_until_ready(dot(trace))
    byte_equal = bool(jnp.array_equal(folded.view(jnp.uint32), dotted.view(jnp.uint32)))

    print(
        f"chip={args.chip} num_cols={chip.num_cols} K={num_constraints} "
        f"rows={args.rows} byte_equal={byte_equal}"
    )
    if not byte_equal:
        print("FAIL: constraint_rlc diverged from the dot it marks", file=sys.stderr)
        return 1
    print("OK: rw_constraints eval folded via constraint_rlc, byte-equal to the dot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
