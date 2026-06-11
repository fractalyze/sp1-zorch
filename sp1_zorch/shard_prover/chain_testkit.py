# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared four-stage chain fixture for the dual-chain test suites.

One tiny single-chip shard — witness-shaped (column ``a == 1`` on real rows
with ``C(0_row) != 0``) so the zerocheck statement holds and the padded-row
correction stays live — proven through the full ``prove_shard_chain`` and
paired with the matching ``verify_shard_chain``. The stacking height matches
the chip area so the committed stack and the eval stage's packed dense agree
(a real shard's area is a multiple of the stacking height). Consumed by the
chain-level mirror test (``shard_prover/verify_shard_test``) and the stage-4
per-leg tamper test (``jagged/verifier_test``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont as BF

from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.round import ProveChain, VerifyChain
from zorch.testkit.transcript import cheap_transcript

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.prove_shard import (
    ShardCarry,
    preamble_chip_metadata,
    prove_shard_chain,
)
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from sp1_zorch.shard_prover.verify_shard import verify_shard_chain

MAX_LOG_ROW_COUNT = 5
CHIP_HEIGHT = 4
_LOG_STACKING_HEIGHT = 3
_NUM_BETAS = 3


def rand_bf(seed: int, shape) -> jnp.ndarray:
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=shape, dtype=np.int64)
    return jnp.array(ints, dtype=BF)


class _WitnessChip:
    """Witness-shaped stub: column ``a == 1`` on real rows, so the constraint
    vanishes there while ``C(0_row) != 0`` keeps the padded-row correction
    live in the zerocheck dual's oracle check."""

    def eval_constraints(self, trace, public_values):
        a, b = trace[:, 0], trace[:, 1]
        one = jnp.ones((), trace.dtype)
        return jnp.stack([(a - one) * (b - one)], axis=-1)


@dataclass(frozen=True)
class ShardChainFixture:
    """An honest prover run plus the matching dual chain.

    ``messages`` is the prover chain's message list in stage order
    (commitment, LogUp-GKR proof, zerocheck proof, jagged-eval proof) —
    the dual chain's proof object. ``prover_transcript`` is the prover's
    post-stage-4 transcript, for byte-matching the dual's output stream."""

    smcs: SingleMatrixCommitmentScheme
    vk: MachineVerifyingKey
    public_values: jnp.ndarray
    prove_chain: ProveChain
    dual: VerifyChain
    messages: list[Any]
    prover_transcript: Any


def small_shard_chain_fixture() -> ShardChainFixture:
    """Build the fixture and run the four-stage prover once."""
    main_region = JaggedRegion.from_chips(
        [
            jnp.concatenate(
                [
                    jnp.ones((CHIP_HEIGHT, 1), dtype=BF),
                    rand_bf(1, (CHIP_HEIGHT, 1)),
                ],
                axis=1,
            )
        ],
        log_stacking_height=_LOG_STACKING_HEIGHT,
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
    metadata = preamble_chip_metadata(("alpha",), [CHIP_HEIGHT], dtype=BF)
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
        vk=vk,
        chip_metadata=metadata,
        gkr_chips=gkr_chips,
        chips=chips,
        num_betas=_NUM_BETAS,
        num_row_variables=MAX_LOG_ROW_COUNT - 1,
        max_log_row_count=MAX_LOG_ROW_COUNT,
    )
    prove_chain = prove_shard_chain(open_num_queries=2, **shared)
    # Synthetic shard, 8-element random public values, no real public-values
    # bus: the structural / stage-dual mirror these suites pin is orthogonal
    # to the output-layer balance leg, which is covered on a real shard in
    # logup_gkr/public_values_test.
    dual = verify_shard_chain(
        chip_names=("alpha",),
        chip_heights={"alpha": CHIP_HEIGHT},
        log_stacking_height=_LOG_STACKING_HEIGHT,
        open_num_queries=2,
        verify_public_values=False,
        **shared,
    )

    carry = ShardCarry(main_region, None, public_values)
    transcript = cheap_transcript(BF)
    messages = []
    for round_ in prove_chain.rounds:
        carry, transcript, msg = round_(carry, transcript)
        messages.append(msg)

    return ShardChainFixture(
        smcs=smcs,
        vk=vk,
        public_values=public_values,
        prove_chain=prove_chain,
        dual=dual,
        messages=messages,
        prover_transcript=transcript,
    )
