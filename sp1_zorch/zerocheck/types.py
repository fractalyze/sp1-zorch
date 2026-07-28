# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The zerocheck reduction's wire type.

Separate from `prover.py` so a verifier — this block's or a composite's — reads
the reduction proof without importing the prover that produced it. The two roles
of a claim reduction are separately deployable (`zorch.stage`), which a shared
type module is what makes possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from frx import Array
from zorch.sumcheck.prover import RoundMsg

from sp1_zorch.logup_gkr.types import ChipEvaluation


@dataclass(frozen=True)
class ZerocheckProof:
    """Reduces "every chip's constraints vanish" to a `TraceEvaluationClaim`.

    A verifier replays the multi-chip sumcheck in ``msgs`` — whose
    ``challenge`` accumulates into the point the next Stage opens at — and is
    left owing only that the values folded along the way are the committed
    trace's.

    Several fields are retained rather than re-derived, because their only
    other source is state the consumer does not hold: the three challenges and
    the eq point, because neither the byte-match harness nor the jagged
    opening keeps the pre-stage transcript to re-sample them; the claimed sum
    (the lambda-Horner fold of the per-chip GKR opening claims, SP1's
    zerocheck RLC), because only this Stage sees those claims; and the
    per-chip final folded traces, whose split ``opened_values`` view is both
    the evaluation Stage's per-column claims and the wire's
    ShardOpenedValues.
    """

    batching_challenge: Array
    gkr_opening_batch_challenge: Array
    lambda_: Array
    zeta: Array
    claimed_sum: Array
    finals: list[Array]
    opened_values: dict[str, ChipEvaluation]
    msgs: RoundMsg
