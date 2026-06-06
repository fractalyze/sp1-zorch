# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1's zerocheck stage: the shard-prover glue around the jagged round engine.

Everything here is derivation, not proving: the three stage challenges in
SP1's order (constraint batching -> GKR opening batch -> chip-RLC lambda),
zeta as the row tail of the GKR evaluation point, each chip's GKR opening
claim as the beta-power weighting of its ``[main | prep]`` column openings,
and the per-chip column-major traces sliced out of the committed regions. The
round engine (`prove_jagged_zerocheck`) owns the sumcheck itself.

Reference: whir-zorch ``sp1/shard_prover/prover.py`` (Phase 3) mirroring SP1's
zerocheck schedule —
https://github.com/fractalyze/sp1/blob/640d8b80c/crates/hypercube/src/prover/shard.rs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import jax.numpy as jnp
from jax import Array
from rw_constraints import Chip

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.logup_gkr.prover import ChipEvaluation
from sp1_zorch.zerocheck.jagged import prove_jagged_zerocheck
from sp1_zorch.zerocheck.prover import gkr_powers, rlc_coeffs
from zorch.sumcheck.prover import RoundMsg
from zorch.transcript import Transcript, sample_challenge

# An SP1 extension-field challenge is four base-field squeezes.
_EF_LIMBS = 4


@dataclass(frozen=True)
class ZerocheckProof:
    """The zerocheck stage's proof: the three stage challenges and the eq
    point (the byte-match harness and the jagged-opening stage consume them,
    and neither holds the pre-stage transcript to re-sample), the per-chip
    final folded traces, and the stacked round messages whose ``challenge``
    is the sumcheck point."""

    batching_challenge: Array
    gkr_opening_batch_challenge: Array
    lambda_: Array
    zeta: Array
    finals: list[Array]
    msgs: RoundMsg


def chip_traces(
    chip_names: Sequence[str],
    num_reals: Sequence[int],
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
) -> list[Array]:
    """Per-chip column-major ``[main | prep]`` traces, exactly ``nr`` rows each.

    Main-first matches the GKR claim's beta-weighting (the claims batch
    ``concat([main_eval, prep_eval])``); prep is height-padded / truncated to
    the chip's ``num_real``. The round driver owns all further padding.
    """
    bf = main_region.dense.dtype
    prep_idx = (
        {n: k for k, n in enumerate(prep_region.chip_names)} if prep_region else {}
    )
    traces = []
    for i, name in enumerate(chip_names):
        nr = int(num_reals[i])
        mw = int(main_region.chip_widths[i])
        start = main_region.chip_starts[i]
        if mw > 0 and nr > 0:
            cols = main_region.dense[start : start + nr * mw].reshape(mw, nr)
        else:
            cols = jnp.zeros((mw, nr), dtype=bf)
        if prep_region is not None and name in prep_idx:
            k = prep_idx[name]
            pw = int(prep_region.chip_widths[k])
            p_h = int(prep_region.chip_heights[k])
            p_start = prep_region.chip_starts[k]
            if pw > 0 and p_h > 0:
                prep = prep_region.dense[p_start : p_start + p_h * pw].reshape(pw, p_h)
                if p_h < nr:
                    prep = jnp.concatenate(
                        [prep, jnp.zeros((pw, nr - p_h), dtype=prep.dtype)], axis=1
                    )
                elif p_h > nr:
                    prep = prep[:, :nr]
            else:
                prep = jnp.zeros((pw, nr), dtype=bf)
            if pw > 0:
                cols = jnp.concatenate([cols, prep], axis=0)
        traces.append(cols)
    return traces


def _bind_pv(chip: Chip, public_values: Array) -> Callable[[Array], Array]:
    """Bind the public-values vector; ``eval_constraints`` ignores it for
    constraints that declare no ``pv_arg``."""
    return lambda trace: chip.eval_constraints(trace, public_values)


def prove_shard_zerocheck(
    chips: Mapping[str, Chip],
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    public_values: Array,
    eval_point: Array,
    chip_openings: Mapping[str, ChipEvaluation],
    transcript: Transcript,
    *,
    max_log_row_count: int,
) -> tuple[Transcript, ZerocheckProof]:
    """Reduce every chip's constraint zero-sum and GKR opening claim to one
    point claim via the jagged sumcheck.

    ``eval_point`` and ``chip_openings`` are the LogUp-GKR stage's outputs:
    zeta is the point's last ``max_log_row_count`` coordinates (the row
    variables), and each chip's claim is its openings RLC'd under the GKR
    opening-batch challenge — computed here from the same ``gkr_powers``
    weights the round engine applies, bit-for-bit.
    """
    ef = eval_point.dtype

    # SP1 samples lambda inside zerocheck, after the two batch challenges.
    transcript, batching_challenge = sample_challenge(transcript, ef, _EF_LIMBS)
    transcript, gkr_batch = sample_challenge(transcript, ef, _EF_LIMBS)
    transcript, lambda_ = sample_challenge(transcript, ef, _EF_LIMBS)

    zeta = eval_point[-max_log_row_count:]

    chip_names = main_region.chip_names
    num_reals = list(main_region.chip_heights)
    traces = chip_traces(chip_names, num_reals, main_region, prep_region)
    eval_fns = [_bind_pv(chips[name], public_values) for name in chip_names]

    max_cols = max(t.shape[0] for t in traces)
    gkr_all = gkr_powers(gkr_batch, max_cols) if max_cols else jnp.zeros(0, ef)
    claims = []
    for name in chip_names:
        opening = chip_openings[name]
        if opening.preprocessed is not None:
            all_evals = jnp.concatenate([opening.main, opening.preprocessed])
        else:
            all_evals = opening.main
        claims.append(jnp.sum(gkr_all[: all_evals.shape[0]] * all_evals))

    # Constraint counts come from a one-row probe — a chip's constraint
    # functions may emit several columns each, so the count is not readable
    # off the manifest.
    alphas = [
        rlc_coeffs(
            batching_challenge, fn(jnp.zeros((1, t.shape[0]), dtype=ef)).shape[-1]
        )
        for fn, t in zip(eval_fns, traces)
    ]
    lambdas = rlc_coeffs(lambda_, len(chip_names))

    finals, transcript, msgs = prove_jagged_zerocheck(
        eval_fns,
        traces,
        num_reals,
        alphas,
        lambdas,
        zeta,
        transcript,
        beta=gkr_batch,
        claims=claims,
    )
    return transcript, ZerocheckProof(
        batching_challenge=batching_challenge,
        gkr_opening_batch_challenge=gkr_batch,
        lambda_=lambda_,
        zeta=zeta,
        finals=finals,
        msgs=msgs,
    )
