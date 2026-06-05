# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 zerocheck: a chip's K constraints folded under one challenge as a single
`zorch.constraint_eval` composite.

Zerocheck = constraint evaluation + sumcheck reduction. This module owns the
SP1-specific coefficient layouts — the constraint-fold RLC and the GKR
opening-batch weights — and hands the eval + fold to zorch's agnostic
`constraint_eval` combinator so the K constraints never materialize and the
region lowers toward one kernel. The chip's per-row constraint evaluation
(`eval_fn`) is SP1 glue's to supply; its body is riscv-witness's, not
sp1-zorch's. Running the sumcheck reduction over the folded result is out of
this module's scope.

The RLC coefficients are descending powers of the challenge because that is the
order SP1's reference prover folds constraints in (`shifted_powers(1, alpha,
K)[::-1]`); the layout is SP1's, so it lives here rather than in zorch.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from zorch.constraint_eval import constraint_eval


def _shifted_powers(shift: Array, generator: Array, size: int) -> Array:
    """``[shift, shift*generator, ..., shift*generator**(size-1)]`` — SP1's
    ``shifted_powers``. One cumprod rather than a per-power traced multiply:
    wide chips put hundreds of columns through this, and field multiplication
    reassociates exactly."""
    return jnp.cumprod(jnp.full((size,), generator).at[0].set(shift))


# Reference: SP1's `chip_powers_of_alpha` constraint-fold coefficients —
# https://github.com/fractalyze/sp1/blob/640d8b80c/crates/hypercube/src/prover/shard.rs#L520-L524
def rlc_coeffs(alpha: Array, num_constraints: int) -> Array:
    """SP1's RLC coefficients for `num_constraints` constraints: descending
    powers of `alpha`, ``[alpha**(K-1), ..., alpha, 1]``."""
    if num_constraints < 1:
        raise ValueError("num_constraints must be >= 1")
    return _shifted_powers(jnp.ones_like(alpha), alpha, num_constraints)[::-1]


# Reference: SP1's GKR opening-batch weights, column j carrying beta**(j+1)
# (`gkr_opening_batch_randomness.powers().skip(1)`) —
# https://github.com/fractalyze/sp1/blob/640d8b80c/crates/hypercube/src/prover/shard.rs#L541-L549
def gkr_powers(beta: Array, num_cols: int) -> Array:
    """SP1's GKR column-batching weights ``[beta, beta**2, ..., beta**num_cols]``."""
    if num_cols < 1:
        raise ValueError("num_cols must be >= 1")
    return _shifted_powers(beta, beta, num_cols)


def constraint_rlc(
    eval_fn: Callable[[Array], Array],
    rows: Array,
    alpha: Array,
    num_constraints: int,
) -> Array:
    """Fold a chip's K constraints over `rows` under `alpha` as one
    `zorch.constraint_eval`. `eval_fn(rows)` produces `[..., K]`; the result
    drops the K axis. Equivalent to ``eval_fn(rows) @ rlc_coeffs(alpha, K)``,
    but marked so the recognizing emitter lowers it to one kernel."""
    return constraint_eval(eval_fn, rows, rlc_coeffs(alpha, num_constraints))
