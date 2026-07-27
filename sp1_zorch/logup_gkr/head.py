# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The SP1 LogUp-GKR head schedule as consumer glue Rounds.

SP1's challenger enters the GKR layers through a fixed head: grind, alpha,
beta seeds, one discarded public-values challenge, the length-prefixed
output-MLE observes, z1. Per zorch's stage-composition design
(``docs/stage-composition.md`` in zorch), scheme glue like this composes as
small consumer-local Rounds so that every consumer of the schedule — the
prover, the byte-match harness, the shard benchmark — runs ONE definition
instead of a hand-mirrored copy. The discarded challenge lives inside
``HeadChallengesRound``: no caller can forget it, and a schedule edit lands
everywhere at once.

Each round's message carries the values it sampled, so a harness checks
fixture anchors off the message while threading the same transcript the
prover threads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import frx.numpy as fnp
from frx import Array
from zk_dtypes import efinfo
from zk_dtypes import koalabearx4_mont as EF

from zorch.challenge import ChallengePolicy
from zorch.logup_gkr.circuit import LogUpGkrOutput
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.round import ProverRound
from zorch.transcript import GrindingTranscript, Transcript, sample_challenge
from zorch.utils.bits import log2_ceil_usize, log2_strict_usize

# An SP1 extension-field challenge is one base-field squeeze per coefficient.
EF_LIMBS = efinfo(EF).degree

# Every SP1 GKR and jagged-sumcheck challenge is drawn in the extension
# field; one policy so no round can be constructed against a narrower one.
EF_CHALLENGES = ChallengePolicy(EF)


def _sample_ef_point(
    transcript: Transcript, num_coords: int
) -> tuple[Transcript, Array]:
    """Sample ``num_coords`` extension-field challenges as one stacked point.

    The head samples a point twice — the beta seeds and the output-binding z1 —
    and SP1 draws each coordinate as its own `sample_ext_element`, so the loop
    is part of the transcript schedule rather than an implementation detail
    that could be replaced by a single wider squeeze.
    """
    coords = []
    for _ in range(num_coords):
        transcript, c = sample_challenge(transcript, EF, EF_LIMBS)
        coords.append(c)
    return transcript, fnp.stack(coords) if coords else fnp.zeros((0,), EF)


class GrindRound:
    """Observe the grind witness and judge the proof-of-work gate.

    Delegates to the transcript's ``check_witness`` so the gate predicate is
    zorch's one definition (shared with the ``grind`` search). The verdict is
    host-read only when ``pow_bits > 0`` — a zero-bit gate always passes, and
    skipping the read keeps the stage traceable inside one ``@jit``; replay
    callers (harness diagnostics) use that to advance a recorded witness's
    stream without re-judging it. The message is the witness, the one value
    of this round the proof records.
    """

    def __init__(self, pow_witness: Array, *, pow_bits: int = 0) -> None:
        self._pow_witness = pow_witness
        self._pow_bits = pow_bits

    def __call__(
        self, carry: Any, transcript: GrindingTranscript
    ) -> tuple[Any, GrindingTranscript, Array]:
        transcript, ok = transcript.check_witness(self._pow_bits, self._pow_witness)
        if self._pow_bits > 0 and not bool(ok):
            raise ValueError(f"witness fails the {self._pow_bits}-bit proof of work")
        return carry, transcript, self._pow_witness


@dataclass(frozen=True)
class HeadChallenges:
    """The head's sampled challenges. ``betas`` is the eq-expansion of the
    seeds (a single one when there are none) — derived here once so every
    consumer fingerprints interactions with the same tensor. ``pv_challenge``
    is SP1's public-values constraint-folding seed: the prover samples it only
    to advance the stream, but the verifier's output cumulative-sum leg folds
    the public-values constraints under it (``eval_public_values``)."""

    alpha: Array  # () EF
    beta_seeds: Array  # (num_seeds,) EF; empty when num_betas == 1
    betas: Array  # (2^num_seeds,) EF
    pv_challenge: Array  # () EF


class HeadChallengesRound:
    """Sample alpha, the beta seeds, and SP1's public-values challenge.

    The public-values challenge advances the stream on every consumer (it is
    sampled in the schedule's fixed slot); the prover discards its value, the
    verifier keeps it to fold the public-values constraints in the output
    cumulative-sum leg."""

    def __init__(self, num_betas: int) -> None:
        self._num_betas = num_betas

    def __call__(
        self, carry: Any, transcript: Transcript
    ) -> tuple[Any, Transcript, HeadChallenges]:
        transcript, alpha = sample_challenge(transcript, EF, EF_LIMBS)
        transcript, beta_seeds = _sample_ef_point(
            transcript, log2_ceil_usize(self._num_betas)
        )
        transcript, pv_challenge = sample_challenge(transcript, EF, EF_LIMBS)
        one = fnp.ones((), dtype=EF)
        betas = (
            expand_eq_to_hypercube(beta_seeds, one)
            if beta_seeds.shape[0]
            else one[None]
        )
        return carry, transcript, HeadChallenges(alpha, beta_seeds, betas, pv_challenge)


class OutputBindRound:
    """Bind the circuit output: SP1's length-prefixed MLE observes, then z1
    and the output evaluations.

    The carry-out ``(num_eval, den_eval, z1)`` is the GKR layer chain's
    entry carry — this round is the seam between the head and the layers.
    The length prefixes absorb as elements of the MLEs' base field, matching
    SP1's serialization of the extension-field MLEs.
    """

    def __init__(self, output: LogUpGkrOutput) -> None:
        self._output = output

    def __call__(
        self, carry: Any, transcript: Transcript
    ) -> tuple[tuple[Array, Array, Array], Transcript, Array]:
        del carry
        num = self._output.numerator
        den = self._output.denominator
        prefix_dtype = efinfo(num.dtype).base_field_dtype
        transcript = transcript.observe(fnp.array(num.shape[0], prefix_dtype))
        transcript = transcript.observe(num)
        transcript = transcript.observe(fnp.array(den.shape[0], prefix_dtype))
        transcript = transcript.observe(den)
        transcript, z1 = _sample_ef_point(transcript, log2_strict_usize(num.shape[0]))
        return (eval_mle(num, z1), eval_mle(den, z1), z1), transcript, z1


if TYPE_CHECKING:
    # mypy-enforced seam conformance -- the head rounds feed `prove_rounds`.
    _grind: type[ProverRound] = GrindRound
    _head: type[ProverRound] = HeadChallengesRound
    _bind: type[ProverRound] = OutputBindRound
