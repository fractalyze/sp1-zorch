# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared full-shard fixture for the prover/verifier test suites.

One tiny single-chip shard — witness-shaped (column ``a == 1`` on real rows
with ``C(0_row) != 0``) so the zerocheck statement holds and the padded-row
correction stays live — proven through the full ``ShardProver`` and paired
with the matching ``ShardVerifier``. The stacking height matches
the chip area so the committed stack and the eval stage's packed dense agree
(a real shard's area is a multiple of the stacking height). Consumed by the
chain-level mirror test (``shard_prover/verify_shard_test``) and the stage-4
per-leg tamper test (``jagged/verifier_test``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
import numpy as np
from frx import Array
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont as F
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.pcs.jagged.region import JaggedRegion
from zorch.testkit.transcript import cheap_transcript

from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.prove_shard import ShardProver
from sp1_zorch.types import (
    ChipMetadata,
    ChipWidths,
    MachineVerifyingKey,
    ShardClaim,
    ShardProof,
    ShardWitness,
)
from sp1_zorch.shard_prover.verify_shard import ShardVerifier

MAX_LOG_ROW_COUNT = 5
CHIP_HEIGHT = 4
CHIP_WIDTH = 2
LOG_STACKING_HEIGHT = 3
_NUM_BETAS = 3


def rand_bf(seed: int, shape: tuple[int, ...]) -> fnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return fnp.array(ints, dtype=F)


class _WitnessChip:
    """Witness-shaped stub: column ``a == 1`` on real rows, so the constraint
    vanishes there while ``C(0_row) != 0`` keeps the padded-row correction
    live in the zerocheck dual's oracle check."""

    def eval_constraints(self, trace: Array, public_values: Array) -> Array:
        a, b = trace[:, 0], trace[:, 1]
        one = fnp.ones((), trace.dtype)
        return fnp.stack([(a - one) * (b - one)], axis=-1)


@dataclass(frozen=True)
class ShardFixture:
    """An honest prover run plus the matching dual chain.

    ``proof`` is the assembled ``ShardProof`` — the commitment plus one named
    section per Stage,
    which is what the verifier role consumes. ``prover_transcript`` is the
    prover's post-opening transcript, for byte-matching the dual's stream."""

    smcs: SingleMatrixCommitmentScheme
    vk: MachineVerifyingKey
    public_values: fnp.ndarray
    chips: dict[str, Any]
    prover: ShardProver
    verifier: ShardVerifier
    claim: ShardClaim
    witness: ShardWitness
    proof: ShardProof
    prover_transcript: Any


def small_shard_fixture() -> ShardFixture:
    """Build the fixture and run the four-stage prover once."""
    main_region = JaggedRegion.from_chips(
        [
            fnp.concatenate(
                [
                    fnp.ones((CHIP_HEIGHT, 1), dtype=F),
                    rand_bf(1, (CHIP_HEIGHT, 1)),
                ],
                axis=1,
            )
        ],
        log_stacking_height=LOG_STACKING_HEIGHT,
        max_log_row_count=MAX_LOG_ROW_COUNT,
        chip_names=("alpha",),
    )
    public_values = rand_bf(30, (8,))
    vk = MachineVerifyingKey(
        preprocessed_commit=rand_bf(31, (8,)),
        pc_start=rand_bf(32, (3,)),
        cum_sum_x=rand_bf(33, (7,)),
        cum_sum_y=rand_bf(34, (7,)),
        enable_untrusted=0,
    )
    chip_metadata = ChipMetadata(("alpha",), (CHIP_HEIGHT,))
    gkr_chips = (
        GkrChip(
            "alpha",
            (
                Interaction(
                    values=(VirtualPairCol.single_main(1),),
                    multiplicity=VirtualPairCol.single_main(0),
                    kind=3,
                    is_send=True,
                ),
            ),
        ),
    )
    perm = Poseidon2(koalabear16_params())
    smcs = SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )
    chips = {"alpha": _WitnessChip()}
    shared = dict(
        smcs=smcs,
        log_blowup=1,
        gkr_chips=gkr_chips,
        chips=chips,
        num_betas=_NUM_BETAS,
        num_row_variables=MAX_LOG_ROW_COUNT - 1,
        max_log_row_count=MAX_LOG_ROW_COUNT,
    )
    prover = ShardProver(
        open_num_queries=2,
        **shared,
    )
    # Synthetic shard, 8-element random public values, no real public-values
    # bus: the structural / stage-dual mirror these suites pin is orthogonal
    # to the output-layer balance leg, which is covered on a real shard in
    # logup_gkr/public_values_test.
    verifier = ShardVerifier(
        chip_names=("alpha",),
        chip_widths={"alpha": ChipWidths(CHIP_WIDTH)},
        log_stacking_height=LOG_STACKING_HEIGHT,
        open_num_queries=2,
        verify_public_values=False,
        **shared,
    )

    claim = ShardClaim(vk, public_values, chip_metadata)
    witness = ShardWitness(main_region, None)
    proved = prover.prove(claim, witness, cheap_transcript(F))

    return ShardFixture(
        smcs=smcs,
        vk=vk,
        public_values=public_values,
        chips=chips,
        prover=prover,
        verifier=verifier,
        claim=claim,
        witness=witness,
        proof=proved.reduction_proof,
        prover_transcript=proved.transcript,
    )
