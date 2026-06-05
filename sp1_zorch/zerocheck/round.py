# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 zerocheck sumcheck round: prove ``sum_x eq(zeta, x) * C_alpha(trace(x))``.

A chip's K constraints are folded under ``alpha`` into one value per row by
zorch's ``constraint_eval`` (the SP1 RLC layout lives in ``prover.rlc_coeffs``);
the zerocheck weights that by the equality polynomial ``eq(zeta, .)`` and sums
over the boolean hypercube. ``ZerocheckRound`` supplies only that summand to
zorch's summand-generic per-variable driver (``zorch.prove``), so the round
reduction, the Fiat-Shamir thread, and the ``zorch.sumcheck`` fusion are reused
unchanged — only ``eq * constraint-fold`` is SP1's.

State threaded through the driver is ``[eq, col_0, ..., col_{nc-1}]``: the eq
MLE followed by the chip's column-major trace.

``MultiChipZerocheckRound`` is the joint sumcheck over several chips of equal
height: the round polynomial is the ``lambda``-RLC of the per-chip
``eq * constraint-fold`` summands, and the driver's single per-round challenge
folds every chip at once. Its state concatenates the chips' columns,
``[eq, *chip_0_cols, *chip_1_cols, ...]``, regrouped by the static per-chip
column counts. Equal height only — chips that retire at different heights
(jagged) need a height-aware schedule and are a separate round.
"""
from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial, reduce

import jax
import jax.numpy as jnp
from jax import Array

from zorch.constraint_eval import constraint_eval
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.prove import RoundMsg, prove
from zorch.round import Round
from zorch.transcript import Transcript


def _fold_chip(
    eval_fn: Callable[[Array], Array], cols: tuple[Array, ...], alpha: Array
) -> Array:
    """``eval_fn(trace) @ alpha`` over the driver's lifted factors.

    ``eval_fn`` wants a 2-D ``[rows, nc]`` block, so the lifted
    ``[degree+1, live]`` columns are flattened into the row axis for the fold
    and reshaped back. Shared by the single- and multi-chip rounds so the
    per-chip fold has a single source."""
    trace = jnp.stack(cols, axis=-1)
    lead = trace.shape[:-1]
    folded = constraint_eval(eval_fn, trace.reshape(-1, trace.shape[-1]), alpha)
    return folded.reshape(lead)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["alpha"],
    meta_fields=["eval_fn", "degree"],
)
@dataclass(frozen=True)
class ZerocheckRound(Round):
    """Per-variable sumcheck round whose summand is ``eq * C_alpha(trace)``.

    ``alpha`` is the descending-power RLC vector (``prover.rlc_coeffs``);
    ``eval_fn`` maps a ``[rows, nc]`` trace block to its ``[rows, K]``
    constraints; ``degree`` is the round-polynomial degree — the chip's
    constraint degree plus one for the linear eq factor."""

    alpha: Array
    eval_fn: Callable[[Array], Array]
    degree: int

    def _combine(self, eq: Array, *cols: Array) -> Array:
        return eq * _fold_chip(self.eval_fn, cols, self.alpha)


def prove_zerocheck(
    eval_fn: Callable[[Array], Array],
    columns: Sequence[Array],
    alpha: Array,
    zeta: Array,
    transcript: Transcript,
    *,
    degree: int,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """Run the single-chip zerocheck sumcheck.

    ``columns`` are the chip's column-major trace MLEs (each ``2**num_vars``
    wide); ``zeta`` is the eq point (``num_vars`` coordinates). Returns the
    folded state, the advanced transcript, and the stacked per-round
    ``RoundMsg`` (``.round_poly`` is the proof, ``.challenge`` the point)."""
    eq = expand_eq_to_hypercube(zeta, jnp.ones((), zeta.dtype))
    state = [eq, *columns]
    return prove(
        ZerocheckRound(alpha=alpha, eval_fn=eval_fn, degree=degree), state, transcript
    )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["alphas", "lambdas"],
    meta_fields=["eval_fns", "col_counts", "degree"],
)
@dataclass(frozen=True)
class MultiChipZerocheckRound(Round):
    """Joint sumcheck round over equal-height chips: ``eq * sum_c lambda_c *
    C_{alpha_c}(trace_c)``.

    Per chip ``c``, ``eval_fns[c]`` maps a ``[rows, col_counts[c]]`` block to its
    ``[rows, K_c]`` constraints, folded under ``alphas[c]`` by ``constraint_eval``;
    the per-chip folds combine under ``lambdas`` (one coefficient per chip). The
    driver's lifted factors arrive flat (``eq`` then every chip's columns
    concatenated), so ``_combine`` re-slices them with ``col_counts``. ``degree``
    is the round-poly degree — the **max** chip constraint degree plus one for the
    eq factor; lifting a lower-degree chip to that shared domain is exact."""

    alphas: tuple[Array, ...]
    lambdas: Array
    eval_fns: tuple[Callable[[Array], Array], ...]
    col_counts: tuple[int, ...]
    degree: int

    def _combine(self, eq: Array, *cols: Array) -> Array:
        terms = []
        offset = 0
        for eval_fn, nc, alpha, lam in zip(
            self.eval_fns, self.col_counts, self.alphas, self.lambdas, strict=True
        ):
            folded = _fold_chip(eval_fn, cols[offset : offset + nc], alpha)
            offset += nc
            terms.append(lam * folded)
        return eq * reduce(operator.add, terms)


def prove_multi_chip_zerocheck(
    eval_fns: Sequence[Callable[[Array], Array]],
    columns_per_chip: Sequence[Sequence[Array]],
    alphas: Sequence[Array],
    lambdas: Array,
    zeta: Array,
    transcript: Transcript,
    *,
    degree: int,
) -> tuple[list[Array], Transcript, RoundMsg]:
    """Run the equal-height multi-chip joint zerocheck sumcheck.

    ``columns_per_chip[c]`` are chip ``c``'s column-major trace MLEs (every chip
    the same ``2**num_vars`` width); ``alphas[c]`` is its constraint-RLC vector
    and ``lambdas[c]`` its cross-chip coefficient. ``zeta`` is the shared eq point.
    Returns the folded state, the advanced transcript, and the stacked per-round
    ``RoundMsg``."""
    nc = len(columns_per_chip)
    if nc == 0:
        raise ValueError("at least one chip is required")
    if lambdas.ndim != 1:
        raise ValueError(f"lambdas must be 1-D, got shape {lambdas.shape}")
    if not (nc == len(eval_fns) == len(alphas) == lambdas.shape[0]):
        raise ValueError(
            "eval_fns, columns_per_chip, alphas, and lambdas must agree on the "
            f"chip count: {len(eval_fns)}, {nc}, {len(alphas)}, {lambdas.shape[0]}"
        )
    eq = expand_eq_to_hypercube(zeta, jnp.ones((), zeta.dtype))
    # Fail closed on the equal-height contract: a jagged or empty chip would
    # otherwise surface as an opaque shape error inside the driver's scan.
    width = eq.shape[0]
    for i, chip in enumerate(columns_per_chip):
        if not chip:
            raise ValueError(f"chip {i} must have at least one column")
        for col in chip:
            if col.shape != (width,):
                raise ValueError(
                    f"chip {i} has a column of shape {col.shape}; every column "
                    f"must be ({width},) for the equal-height joint zerocheck"
                )
    state = [eq, *(col for chip in columns_per_chip for col in chip)]
    round = MultiChipZerocheckRound(
        alphas=tuple(alphas),
        lambdas=lambdas,
        eval_fns=tuple(eval_fns),
        col_counts=tuple(len(chip) for chip in columns_per_chip),
        degree=degree,
    )
    return prove(round, state, transcript)
