# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 zerocheck verifier: the zorch-native dual of ``prove_shard_zerocheck``.

Mirrors SP1's reference verifier, pinned for diffing:
https://github.com/fractalyze/sp1/blob/640d8b80c/crates/hypercube/src/verifier/shard.rs
(``verify_zerocheck``). The three stage challenges are the dual's own
samples; the wire's copies ride for non-verifier consumers and are pinned so
a stale serialization cannot drift past the dual. The claimed sum is
re-derived from the LogUp-GKR stage's leaf-checked chip openings, the round
replay runs under the agnostic ``zorch.verify`` scan, and the per-chip
opened values are absorbed through the same shared ``OpenedValuesRound`` the
prover drives.

The per-round rule is the plain coefficient-form sumcheck check — SP1's
``partially_verify_sumcheck_proof`` applies no per-round eq adjustment; the
eq factor enters once, in the final oracle check. What keeps the round local
to sp1-zorch is the challenge squeeze: SP1's zerocheck binds each variable
with ONE base-field squeeze (the prover's ``observe_and_sample(rlc, 1)``),
where zorch's ``CoeffsSumcheckRound`` derives the challenge in the claim's
extension field.

The final oracle check closes the reduction: the lambda-RLC over chips of
``constraint RLC on the opened row - padded-row adjustment * geq + the
beta-weighted column batch``, scaled by ``eq(zeta, z_row)``, must equal the
replay's final claim. Trace heights are statement inputs (the same source as
the GKR dual's); SP1 reads them off the proof's opened-values degrees
instead. SP1's per-chip opening-width check has no statement-side source
here yet — a mis-shaped opening row fails the constraint evaluation or the
jagged-eval stage's column manifest instead of a dedicated shape error.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from rw_constraints import Chip

from sp1_zorch.logup_gkr.head import EF_LIMBS
from sp1_zorch.logup_gkr.prover import ChipEvaluation, select_openings
from sp1_zorch.logup_gkr.verifier import virtual_padding_geq
from sp1_zorch.zerocheck.jagged import DEGREE
from sp1_zorch.zerocheck.prover import constraint_rlc, gkr_powers, rlc_coeffs
from sp1_zorch.zerocheck.stage import OpenedValuesRound, ZerocheckProof, _bind_pv
from zorch.poly.univariate import eval_coeffs
from zorch.round import Round
from zorch.transcript import Transcript, sample_challenge
from zorch.verify import verify


@partial(jax.tree_util.register_dataclass, data_fields=[], meta_fields=[])
@dataclass(frozen=True)
class ZerocheckSumcheckRound(Round):
    """One zerocheck sumcheck round: the coefficient-form claim identity
    (``s(0) = c_0``, ``s(1) = sum(c)``) with SP1's challenge rule — one
    base-field squeeze per variable, via the same fused
    ``observe_and_sample`` primitive the prover binds with."""

    def __call__(
        self, claim: Array, msg: Array, transcript: Transcript
    ) -> tuple[Array, Transcript, Array, Array]:
        ok = claim == msg[0] + jnp.sum(msg)
        transcript, r = transcript.observe_and_sample(msg, 1)
        return eval_coeffs(msg, r[0]), transcript, r[0], ok


def _all_evals(opening: ChipEvaluation) -> Array:
    """A chip's ``[main | prep]`` evaluation vector — the beta-power batching
    order shared by the GKR claims and the zerocheck column batch."""
    if opening.preprocessed is not None:
        return jnp.concatenate([opening.main, opening.preprocessed])
    return opening.main


def verify_shard_zerocheck(
    chips: Mapping[str, Chip],
    chip_names: Sequence[str],
    chip_heights: Mapping[str, int],
    public_values: Array,
    eval_point: Array,
    chip_openings: Mapping[str, ChipEvaluation],
    proof: ZerocheckProof,
    transcript: Transcript,
    *,
    max_log_row_count: int,
) -> tuple[Transcript, Array, Array]:
    """Verify a zerocheck stage proof on a transcript positioned after the
    LogUp-GKR stage; returns ``(transcript, zc_sumcheck_point, ok)``.

    ``eval_point`` and ``chip_openings`` are the LogUp-GKR dual's outputs —
    already leaf-checked, so the claims derived from them are trusted
    inputs, mirroring the prover's reads of ``ShardCarry``.
    ``zc_sumcheck_point`` is the dual's own sampled challenge list (the
    prover's ``msgs.challenge`` order); the caller threads it to the
    jagged-eval dual. ``ok`` is a traced scalar AND of every acceptance leg;
    malformed proof *structure* raises instead — shapes are host decisions,
    the same split SP1 makes between its shape errors and field checks.
    """
    ef = eval_point.dtype
    if eval_point.shape[0] < max_log_row_count:
        raise ValueError(
            f"the GKR evaluation point must carry at least the "
            f"{max_log_row_count} row variables, got {eval_point.shape[0]}"
        )
    if proof.msgs.round_poly.shape != (max_log_row_count, DEGREE + 1):
        raise ValueError(
            f"need one degree-{DEGREE} coefficient poly per row variable "
            f"({max_log_row_count}, {DEGREE + 1}), got "
            f"{proof.msgs.round_poly.shape}"
        )
    opened = select_openings(proof.opened_values, chip_names)
    opened_rows = [_all_evals(ev) for ev in opened]

    # SP1 samples lambda inside zerocheck, after the two batch challenges.
    transcript, batching = sample_challenge(transcript, ef, EF_LIMBS)
    transcript, gkr_batch = sample_challenge(transcript, ef, EF_LIMBS)
    transcript, lambda_ = sample_challenge(transcript, ef, EF_LIMBS)

    zeta = eval_point[-max_log_row_count:]

    gkr_evals = [_all_evals(chip_openings[name]) for name in chip_names]
    max_cols = max(e.shape[0] for e in gkr_evals + opened_rows)
    gkr_all = gkr_powers(gkr_batch, max_cols) if max_cols else jnp.zeros(0, ef)
    claims = jnp.stack([jnp.sum(gkr_all[: e.shape[0]] * e) for e in gkr_evals])
    lambdas = rlc_coeffs(lambda_, len(chip_names))
    claimed_sum = jnp.sum(claims * lambdas)

    # Pin the wire's challenge and claim copies. Shape-strict: a
    # broadcastable wrong-shape copy must reject, not broadcast.
    ok_wire = (
        jnp.array_equal(batching, proof.batching_challenge)
        & jnp.array_equal(gkr_batch, proof.gkr_opening_batch_challenge)
        & jnp.array_equal(lambda_, proof.lambda_)
        & jnp.array_equal(zeta, proof.zeta)
        & jnp.array_equal(claimed_sum, proof.claimed_sum)
    )

    point, final_claim, transcript, ok_rounds = verify(
        ZerocheckSumcheckRound(), claimed_sum, proof.msgs.round_poly, transcript
    )
    ok_point = jnp.array_equal(point, proof.msgs.challenge)

    # The final oracle check, SP1's trace-openings consistency block: at the
    # bound point, each chip's constraint RLC on its opened row — corrected
    # for the virtual padding rows, whose constant constraint value the
    # prover's summand subtracts — plus its beta-weighted column batch,
    # scaled by the bound eq factor, must reproduce the replay's final claim.
    one = jnp.ones((), ef)
    z_row = point[::-1].astype(ef)
    eq_val = jnp.prod(zeta * z_row + (one - zeta) * (one - z_row))
    # The threshold domain is one variable wider than the point so a
    # full-height chip stays representable (SP1's ``point_extended``).
    point_extended = jnp.pad(z_row, (1, 0))
    geq_by_height: dict[int, Array] = {}
    terms = []
    for name, opened_row in zip(chip_names, opened_rows):
        height = chip_heights[name]
        if height not in geq_by_height:
            geq_by_height[height] = virtual_padding_geq(height, point_extended)
        geq = geq_by_height[height]
        eval_fn = _bind_pv(chips[name], public_values)
        # Constraint counts come from a one-row probe, as in the prover — a
        # chip's constraint functions may emit several columns each.
        zero_row = jnp.zeros((1, opened_row.shape[0]), dtype=ef)
        num_constraints = eval_fn(zero_row).shape[-1]
        if num_constraints:
            constraint_term = constraint_rlc(
                eval_fn, opened_row[None, :], batching, num_constraints
            )[0]
            padded_row_adj = constraint_rlc(
                eval_fn, zero_row, batching, num_constraints
            )[0]
        else:
            constraint_term = jnp.zeros((), ef)
            padded_row_adj = jnp.zeros((), ef)
        batch_term = jnp.sum(gkr_all[: opened_row.shape[0]] * opened_row)
        terms.append(constraint_term - padded_row_adj * geq + batch_term)
    rlc_eval = eq_val * jnp.sum(jnp.stack(terms) * lambdas)
    ok_eval = jnp.array_equal(final_claim, rlc_eval)

    # The stage's transcript tail, through the same shared Round the prover
    # drives: every evaluation-stage challenge samples after these absorbs.
    _, transcript, _ = OpenedValuesRound(proof.opened_values, chip_names)(
        None, transcript
    )

    ok = ok_wire & ok_rounds & ok_point & ok_eval
    return transcript, point, ok
