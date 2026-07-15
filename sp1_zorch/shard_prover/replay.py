# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared scaffolding for the rsp byte-match runnables.

The transcript-driven ``verify_*`` runnables replay the same shard entry:
SP1's duplex challenger over poseidon2 koalabear16, the preamble observation
stream, the jagged regions at SP1's core machine parameters, and the layered
GKR leg those streams feed. One definition keeps the runnables' Fiat-Shamir
streams from drifting apart — a preamble divergence shows up as a stage
mismatch two tools later, which is exactly the kind of hunt this module
exists to prevent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.jagged.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import build_gkr_chips
from sp1_zorch.logup_gkr.prover import (
    ChipEvaluation,
    LogupGkrProof,
    num_beta_values,
    prove_logup_gkr,
)
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.fixture_loader import _parse_int_list, _parse_kv_lines
from sp1_zorch.shard_prover.prove_shard import PreambleStage, preamble_chip_metadata
from sp1_zorch.shard_prover.types import ShardData
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.transcript import DuplexState, DuplexTranscript, Transcript

# SP1 core machine parameters.
LOG_STACKING_HEIGHT = 21
MAX_LOG_ROW_COUNT = 22

# SP1's challenger: poseidon2 koalabear16 duplex sponge at rate 8.
RATE = 8


def to_u32(a: Array) -> np.ndarray:
    """Montgomery-form u32 bitpatterns — the byte-match comparison unit."""
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32))


def from_u32(u32: np.ndarray, dtype: Any) -> Array:
    """Raw u32 Mont bitpatterns -> field array (EF collapses a trailing 4)."""
    return jax.lax.bitcast_convert_type(jnp.asarray(u32, dtype=jnp.uint32), dtype)


class JitPermutation:
    """`Permutation` wrapper with a jitted `permute`.

    The transcript drives the permutation from eager host loops where each
    un-jitted permute re-dispatches its few hundred field ops per call; one
    compile here collapses that to a single dispatch."""

    def __init__(self, inner: Poseidon2) -> None:
        self._inner = inner
        self.width: int = inner.width
        self.dtype: Any = inner.dtype
        self.has_dedicated_fusion: bool = inner.has_dedicated_fusion
        self._permute = jax.jit(inner.permute)

    def permute(self, state: Array) -> Array:
        return self._permute(state)

    # Value identity from the wrapped permutation. JitPermutation rides as a
    # static meta_field in DuplexTranscript, so it keys the jit cache whenever a
    # transcript is a jit argument (the jitted stage bodies: zerocheck,
    # jagged-eval, and the GKR inner zones). Without value-equality every fresh_transcript()
    # is a fresh wrapper -> a new cache key -> the whole-stage @jit recompiles on
    # every prove. Poseidon2 already carries value-equality for this exact
    # reason (zorch#214); the wrapper must forward it.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JitPermutation):
            return NotImplemented
        return self._inner == other._inner

    def __hash__(self) -> int:
        return hash(self._inner)


def fresh_transcript() -> DuplexTranscript:
    """SP1's challenger at its initial state."""
    return DuplexTranscript.new(JitPermutation(Poseidon2(koalabear16_params())), RATE)


def preamble_transcript(shard: ShardData, shard_dir: Path) -> Transcript:
    """The challenger state SP1 enters GKR with: a fresh duplex sponge run
    through ``PreambleStage`` — the prover's own absorb schedule — with the
    dump's commitment. The commitment is the dump's value -- our own
    main-commit byte-match is the trace-commit stage's concern."""
    commit_kv = _parse_kv_lines((shard_dir / "gpu_commitment.txt").read_text())
    # gpu_commitment.txt carries canonical integers, so encode rather than view.
    commitment = jnp.array(_parse_int_list(commit_kv["main_commit"]), F)

    traces = shard.main_trace_data.traces
    names = traces.chip_order
    num_reals = [traces.per_chip[name].num_real for name in names]
    preamble = PreambleStage(
        vk=shard.vk,
        public_values=shard.main_trace_data.public_values,
        commitment=commitment,
        chip_metadata=preamble_chip_metadata(names, num_reals, dtype=F),
    )
    _, transcript, _ = preamble(None, fresh_transcript())
    return transcript


def clone_diag(transcript: Transcript) -> int:
    """SP1's challenger diags are one squeeze off a clone; the functional
    transcript makes the clone free."""
    _, sample = transcript.sample(1)
    return int(sample[0])


def save_gkr_cache(
    path: Path,
    eval_point: Array,
    openings: dict[str, ChipEvaluation],
    transcript: DuplexTranscript,
) -> None:
    """Persist the post-GKR zerocheck inputs (eval point, per-chip openings,
    live sponge state) as an npz. Shared by the zerocheck bench and
    ``verify_zerocheck`` so a cache seeded by either tool loads in the other."""
    st = transcript.state
    data: dict[str, np.ndarray] = {
        "eval_point": to_u32(eval_point),
        "chips": np.array(sorted(openings)),
        "t_input": to_u32(st.input_buffer),
        "t_output": to_u32(st.output_buffer),
        "t_sponge": to_u32(st.sponge_state),
        "t_in_pos": np.int32(int(st.in_pos)),
        "t_out_pos": np.int32(int(st.out_pos)),
    }
    for name, ev in openings.items():
        data[f"main:{name}"] = to_u32(ev.main)
        if ev.preprocessed is not None:
            data[f"prep:{name}"] = to_u32(ev.preprocessed)
    np.savez(path, **data)


def load_gkr_cache(
    path: Path,
) -> tuple[Array, dict[str, ChipEvaluation], Transcript]:
    """Inverse of ``save_gkr_cache``: the eval point, per-chip openings, and a
    transcript restored to the saved sponge state."""
    with np.load(path) as z:
        eval_point = from_u32(z["eval_point"], EF)
        openings = {
            str(name): ChipEvaluation(
                main=from_u32(z[f"main:{name}"], EF),
                preprocessed=(
                    from_u32(z[f"prep:{name}"], EF) if f"prep:{name}" in z else None
                ),
            )
            for name in z["chips"]
        }
        state = DuplexState(
            input_buffer=from_u32(z["t_input"], F),
            output_buffer=from_u32(z["t_output"], F),
            sponge_state=from_u32(z["t_sponge"], F),
            in_pos=jnp.int32(int(z["t_in_pos"])),
            out_pos=jnp.int32(int(z["t_out_pos"])),
        )
    base = fresh_transcript()
    return eval_point, openings, DuplexTranscript(base.permutation, base.rate, state)


def replay_gkr(
    shard: ShardData,
    shard_dir: Path,
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
    *,
    pow_bits: int,
) -> tuple[Transcript, LogupGkrProof]:
    """The pipeline through the layered GKR prove, grind skipped via the
    dump's witness. One call site for the stage invocation so the GKR and
    zerocheck runnables cannot drift on its wiring; each caller owns its own
    checks against the dump."""
    state = _parse_kv_lines(
        (shard_dir / "gpu_gkr_state.txt").read_text(), skip_unkeyed=True
    )
    preamble = preamble_transcript(shard, shard_dir)
    order = shard.main_trace_data.traces.chip_order
    gkr_chips = build_gkr_chips(shard.main_trace_data.chips, order)
    return prove_logup_gkr(
        gkr_chips,
        main_region,
        prep_region,
        preamble,
        num_betas=num_beta_values(shard.main_trace_data.chips),
        num_row_variables=MAX_LOG_ROW_COUNT - 1,
        pow_bits=pow_bits,
        witness=jnp.array(int(state["witness"]), F),
    )


def seed_gkr_outputs_rolled(
    shard: ShardData,
    shard_dir: Path,
    main_region: JaggedRegion,
    prep_region: JaggedRegion | None,
) -> tuple[Transcript, Array, dict[str, ChipEvaluation]]:
    """Fast GKR-output seed: the marked rolled prove compiled as one jit
    (sp1-zorch#55) -- minutes vs ``replay_gkr``'s eager hours, byte-identical.

    Returns only the zerocheck inputs (post-GKR transcript, eval point, per-chip
    openings), NOT a ``LogupGkrProof`` -- the proof isn't a jit-returnable pytree
    and the round proofs aren't needed downstream. The recorded witness replays
    at ``pow_bits=0`` (the zero-bit GrindRound observes it without the host-read
    that would break the single jit, so the transcript still matches the judged
    path); the caller seals against the dump's post-GKR diag. Used to seed a
    cache for the zerocheck bench (sp1-zorch#115)."""
    state = _parse_kv_lines(
        (shard_dir / "gpu_gkr_state.txt").read_text(), skip_unkeyed=True
    )
    preamble = preamble_transcript(shard, shard_dir)
    order = shard.main_trace_data.traces.chip_order
    gkr_chips = build_gkr_chips(shard.main_trace_data.chips, order)
    num_betas = num_beta_values(shard.main_trace_data.chips)
    num_row_variables = MAX_LOG_ROW_COUNT - 1
    witness = jnp.array(int(state["witness"]), F)

    # chips static via closure; regions/transcript/witness traced -- mirrors how
    # the rolled bench jits the stage, keeping the zorch.sumcheck composite
    # intact for the vendor emitter. Return arrays/pytrees, never the proof.
    def _rolled(mr, pr, tr, w):
        t, proof = prove_logup_gkr(
            gkr_chips,
            mr,
            pr,
            tr,
            num_betas=num_betas,
            num_row_variables=num_row_variables,
            pow_bits=0,
            witness=w,
        )
        return t, proof.eval_point, proof.chip_openings

    return jax.jit(_rolled)(main_region, prep_region, preamble, witness)


def shard_regions(shard: ShardData) -> tuple[JaggedRegion, JaggedRegion | None]:
    """The shard's main/prep jagged regions at SP1's core machine parameters
    (prep chips in sorted-name order, SP1's preprocessed enumeration)."""
    traces = shard.main_trace_data.traces
    order = traces.chip_order
    main_region = JaggedRegion.from_chips(
        [traces.per_chip[n].array for n in order],
        log_stacking_height=LOG_STACKING_HEIGHT,
        max_log_row_count=MAX_LOG_ROW_COUNT,
        chip_names=order,
    )
    prep = shard.preprocessed_traces
    prep_names = tuple(sorted(prep))
    prep_region = (
        JaggedRegion.from_chips(
            [prep[n] for n in prep_names],
            log_stacking_height=LOG_STACKING_HEIGHT,
            max_log_row_count=MAX_LOG_ROW_COUNT,
            chip_names=prep_names,
        )
        if prep
        else None
    )
    return main_region, prep_region
