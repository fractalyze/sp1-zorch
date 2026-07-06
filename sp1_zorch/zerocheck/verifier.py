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
eq factor enters once, in the final oracle check. The challenge squeeze is
SP1's ``sample_ext_element`` (one extension element per variable, degree
base squeezes reinterpreted) — exactly the rule zorch's
``CoeffsSumcheckRound`` owns via ``challenge_limbs``, so the round replay
is the stock zorch round under the agnostic ``zorch.verify`` scan
(fractalyze/sp1-zorch#88).

The final oracle check closes the reduction: the lambda-RLC over chips of
``constraint RLC on the opened row - padded-row adjustment * geq + the
beta-weighted column batch``, scaled by ``eq(zeta, z_row)``, must equal the
replay's final claim. Trace heights are statement inputs (the same source as
the GKR dual's); SP1 reads them off the proof's opened-values degrees
instead. SP1's per-chip opening-width check is the chain round's:
``ShardZerocheckVerifierRound`` checks every opening against its statement
shape before delegating to this module, so the opened values reaching the
replay and the oracle check here are statement-shaped already.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import jax.numpy as jnp
from jax import Array
from rw_constraints import Chip
from zk_dtypes import efinfo

from sp1_zorch.logup_gkr.prover import ChipEvaluation, select_openings
from sp1_zorch.logup_gkr.verifier import padding_geqs
from sp1_zorch.zerocheck.jagged import DEGREE
from sp1_zorch.zerocheck.coeffs import constraint_rlc, rlc_coeffs
from sp1_zorch.zerocheck.stage import (
    OpenedValuesRound,
    ZerocheckProof,
    bind_pv,
    gkr_opening_claims,
    probe_num_constraints,
    sample_stage_challenges,
)
from zorch.poly.eq import eval_eq
from zorch.sumcheck.verifier import CoeffsSumcheckRound
from zorch.transcript import Transcript
from zorch.verify import verify


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
    opened_rows = [ev.all_evals() for ev in opened]

    transcript, batching, gkr_batch, lambda_ = sample_stage_challenges(
        transcript, ef
    )

    zeta = eval_point[-max_log_row_count:]

    claims = gkr_opening_claims(
        [chip_openings[name] for name in chip_names], gkr_batch
    )
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
        CoeffsSumcheckRound(DEGREE, efinfo(ef).degree),
        claimed_sum,
        proof.msgs.round_poly,
        transcript,
    )
    ok_point = jnp.array_equal(point, proof.msgs.challenge)

    # The final oracle check, SP1's trace-openings consistency block: at the
    # bound point, each chip's constraint RLC on its opened row — corrected
    # for the virtual padding rows, whose constant constraint value the
    # prover's summand subtracts — plus its beta-weighted column batch,
    # scaled by the bound eq factor, must reproduce the replay's final claim.
    z_row = point[::-1]
    eq_val = eval_eq(zeta, z_row)
    geq_by_height = padding_geqs((chip_heights[n] for n in chip_names), z_row)
    # The same beta-power column batch as the claims derivation, on the
    # zerocheck opened values instead of the GKR openings.
    batch_terms = gkr_opening_claims(opened, gkr_batch)
    terms = []
    for name, opened_row in zip(chip_names, opened_rows):
        geq = geq_by_height[chip_heights[name]]
        eval_fn = bind_pv(chips[name], public_values)
        num_constraints = probe_num_constraints(eval_fn, opened_row.shape[0], ef)
        if num_constraints:
            # Row 0 is the opening, row 1 a zero row: one batched fold yields
            # the opened evaluation and the padded-row adjustment together
            # (the same move as the GKR leaf check), instead of tracing the
            # circuit twice more. Indexed, not unpacked — iterating an
            # extension-field array dispatches lax.sign (the expand_eq
            # gotcha).
            both = constraint_rlc(
                eval_fn,
                jnp.stack([opened_row, jnp.zeros_like(opened_row)]),
                batching,
                num_constraints,
            )
            constraint_term, padded_row_adj = both[0], both[1]
        else:
            constraint_term = jnp.zeros((), ef)
            padded_row_adj = jnp.zeros((), ef)
        terms.append(constraint_term - padded_row_adj * geq)
    rlc_eval = eq_val * jnp.sum((jnp.stack(terms) + batch_terms) * lambdas)
    ok_eval = jnp.array_equal(final_claim, rlc_eval)

    # The stage's transcript tail, through the same shared Round the prover
    # drives: every evaluation-stage challenge samples after these absorbs.
    _, transcript, _ = OpenedValuesRound(proof.opened_values, chip_names)(
        None, transcript
    )

    ok = ok_wire & ok_rounds & ok_point & ok_eval
    return transcript, point, ok
