# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The jagged-PCS opening's wire type.

Separate from `prover.py` so a verifier — this block's or a composite's — reads
the reduction proof without importing the prover that produced it. The two roles
of a claim reduction are separately deployable (`zorch.stage`), which a shared
type module is what makes possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from zorch.pcs.jagged.open import StackedOpenProof
from zorch.pcs.jagged.prover import JaggedEvalMsg

from sp1_zorch.shard_prover.types import SmcsCommitments


@dataclass(frozen=True)
class JaggedPcsProof:
    """Discharges a `JaggedOpeningClaim`, leaving nothing to prove.

    Two legs: the outer/inner sumcheck reducing the committed trace to a
    single value ``D(z_final)``, then the stacked BaseFold open showing that
    value really is the commitment's, at that point.
    """

    eval: JaggedEvalMsg
    open: StackedOpenProof
    smcs_commitments: SmcsCommitments
