# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The SP1 LogUp-GKR head schedule, as functions both roles call.

SP1's challenger enters the GKR layers through a fixed head: grind, alpha,
beta seeds, one discarded public-values challenge, the length-prefixed
output-MLE observes, z1. Each step runs exactly once, so these are plain
shared functions — the same treatment `absorb_preamble` and `bind_commitment`
get in `shard_prover.prove_shard`, and for the same reason: every consumer of
the schedule (the prover, the verifier dual, the byte-match harnesses) calls
ONE definition, so a schedule edit cannot land in one Fiat-Shamir stream and
not another. The discarded public-values challenge lives inside
`sample_head_challenges` where no caller can forget it.

They are deliberately not Rounds. A Round is one step of a *recurrence*,
driven by `prove_rounds` over a threaded carry; the head is a fixed prologue
of three unrelated steps, and dressing them as Rounds only meant every call
site passed `None` for a carry it then discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
from frx import Array
from zk_dtypes import efinfo
from zk_dtypes import koalabearx4_mont as EF
from zorch.challenge import ChallengePolicy
from zorch.logup_gkr.circuit import LogUpGkrOutput
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.multilinear import eval_mle
from zorch.transcript import Transcript, sample_challenge
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


def absorb_grind(
    transcript: Transcript, pow_witness: Array, *, pow_bits: int = 0
) -> Transcript:
    """Observe the grind witness and judge the proof-of-work gate.

    Delegates to the transcript's ``check_witness`` so the gate predicate is
    zorch's one definition (shared with the ``grind`` search). The verdict is
    host-read only when ``pow_bits > 0`` — a zero-bit gate always passes, and
    skipping the read keeps the stage traceable inside one ``@jit``; replay
    callers (harness diagnostics) use that to advance a recorded witness's
    stream without re-judging it. The message is the witness, the one value
    of this step the proof records.

    The verifier dual does not call this: it needs the verdict as a traced leg
    of its `ok`, so it calls the same `check_witness` predicate directly.
    """
    transcript, ok = transcript.check_witness(pow_witness, pow_bits=pow_bits)
    if pow_bits > 0 and not bool(ok):
        raise ValueError(f"witness fails the {pow_bits}-bit proof of work")
    return transcript


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


def sample_head_challenges(
    transcript: Transcript, num_betas: int
) -> tuple[Transcript, HeadChallenges]:
    """Sample alpha, the beta seeds, and SP1's public-values challenge.

    The public-values challenge advances the stream on every consumer (it is
    sampled in the schedule's fixed slot); the prover discards its value, the
    verifier keeps it to fold the public-values constraints in the output
    cumulative-sum leg."""
    transcript, alpha = sample_challenge(transcript, EF, EF_LIMBS)
    transcript, beta_seeds = _sample_ef_point(transcript, log2_ceil_usize(num_betas))
    transcript, pv_challenge = sample_challenge(transcript, EF, EF_LIMBS)
    one = fnp.ones((), dtype=EF)
    betas = (
        expand_eq_to_hypercube(beta_seeds, one) if beta_seeds.shape[0] else one[None]
    )
    return transcript, HeadChallenges(alpha, beta_seeds, betas, pv_challenge)


def bind_circuit_output(
    transcript: Transcript, output: LogUpGkrOutput
) -> tuple[Transcript, tuple[Array, Array, Array]]:
    """Bind the circuit output: SP1's length-prefixed MLE observes, then z1
    and the output evaluations.

    Returns the GKR layer chain's entry carry ``(num_eval, den_eval, z1)`` —
    this is the seam between the head and the layers. The length prefixes
    absorb as elements of the MLEs' base field, matching SP1's serialization
    of the extension-field MLEs.
    """
    num = output.numerator
    den = output.denominator
    prefix_dtype = efinfo(num.dtype).base_field_dtype
    transcript = transcript.observe(fnp.array(num.shape[0], prefix_dtype))
    transcript = transcript.observe(num)
    transcript = transcript.observe(fnp.array(den.shape[0], prefix_dtype))
    transcript = transcript.observe(den)
    transcript, z1 = _sample_ef_point(transcript, log2_strict_usize(num.shape[0]))
    return transcript, (eval_mle(num, z1), eval_mle(den, z1), z1)
