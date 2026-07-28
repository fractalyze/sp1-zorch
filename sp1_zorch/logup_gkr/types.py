# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The LogUp-GKR reduction's wire types.

Separate from `prover.py` so a verifier — this block's or a composite's — reads
the reduction proof without importing the prover that produced it. The two roles
of a claim reduction are separately deployable (`zorch.stage`), which a shared
type module is what makes possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
from frx import Array
from zorch.logup_gkr.circuit import LogUpGkrOutput
from zorch.logup_gkr.jagged_prover import JaggedLayerProof


# Pytree: both evals are array leaves (preprocessed is None for prep-less
# chips), so a carry holding these openings stays an arrays-only pytree.
@partial(
    frx.tree_util.register_dataclass,
    data_fields=["main", "preprocessed"],
    meta_fields=[],
)
@dataclass(frozen=True)
class ChipEvaluation:
    """One chip's trace openings at the final GKR point."""

    main: Array  # (width,) EF, one eval per main column
    preprocessed: Array | None  # (prep width,) EF, when the chip has prep

    def all_evals(self) -> Array:
        """The ``[main | prep]`` evaluation vector — the column order of the
        beta-power batching shared by the GKR opening claims and the
        zerocheck column batch."""
        if self.preprocessed is not None:
            return fnp.concatenate([self.main, self.preprocessed])
        return self.main


@dataclass(frozen=True)
class LogupGkrProof:
    """Reduces the shard's LogUp bus-balance statement to a `GkrOutputClaim`.

    A verifier replays the layer chain from output to input — grind witness,
    circuit output, one round proof per layer — and arrives at the evaluation
    point and per-chip openings the next Stage takes as its hypothesis. What
    it proves is that those openings are the trace's; what it leaves open is
    everything about the constraints.

    Each layer's sumcheck point rides on its ``JaggedLayerProof.point``
    (zorch retains it at prove time); the shard wire serializes it per layer
    (``point_and_eval``).
    """

    pow_witness: Array
    circuit_output: LogUpGkrOutput
    round_proofs: list[JaggedLayerProof]
    eval_point: Array
    chip_openings: dict[str, ChipEvaluation]
