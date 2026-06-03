# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 zerocheck: a chip's K constraints folded under one challenge as a single
`zorch.constraint_eval` composite.

Zerocheck = constraint evaluation + sumcheck reduction. This module owns the
SP1-specific half of the constraint-eval step — the random-linear-combination
coefficient layout — and hands the eval + fold to zorch's agnostic
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


def rlc_coeffs(alpha: Array, num_constraints: int) -> Array:
    """SP1's RLC coefficients for `num_constraints` constraints: descending
    powers of `alpha`, ``[alpha**(K-1), ..., alpha, 1]``."""
    if num_constraints < 1:
        raise ValueError("num_constraints must be >= 1")
    powers = [jnp.ones_like(alpha)]
    for _ in range(num_constraints - 1):
        powers.append(powers[-1] * alpha)
    return jnp.stack(powers[::-1])


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
