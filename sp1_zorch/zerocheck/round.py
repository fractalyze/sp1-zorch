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
MLE followed by the chip's column-major trace. Single chip; the jagged
multi-chip RLC across chips is a separate round.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from zorch.constraint_eval import constraint_eval
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.sumcheck.prover import RoundMsg, prove
from zorch.round import Round
from zorch.transcript import Transcript


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
        """``eq * (eval_fn(trace) @ alpha)`` over the driver's lifted factors.
        ``eval_fn`` wants a 2-D ``[rows, nc]`` block, so the lifted
        ``[degree+1, live]`` columns are flattened into the row axis for the
        fold and reshaped back."""
        trace = jnp.stack(cols, axis=-1)
        lead = trace.shape[:-1]
        folded = constraint_eval(
            self.eval_fn, trace.reshape(-1, trace.shape[-1]), self.alpha
        )
        return eq * folded.reshape(lead)


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
