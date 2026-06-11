# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 shard-prover input types, mirroring sp1-hypercube's ``ShardData`` /
``MainTraceData``
(https://github.com/fractalyze/sp1/blob/e2c02f376/crates/hypercube/src/prover/shard.rs)
and ``MachineVerifyingKey``
(https://github.com/fractalyze/sp1/blob/e2c02f376/crates/hypercube/src/verifier/config.rs).

Field-element arrays carry raw Montgomery u32 (``koalabear_mont`` views) so
downstream byte-match stages compare bytes directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from rw_constraints import Chip
    from zorch.transcript import Transcript

# SP1 v1: every shard's public-values vector is padded to this length on both
# the prover and verifier side; PV-aware chips index fixed slots in the padded
# layout (sp1-hypercube ``PROOF_MAX_NUM_PVS``).
PROOF_MAX_NUM_PVS = 187


@dataclass(frozen=True)
class MachineVerifyingKey:
    """SP1 mirror: ``MachineVerifyingKey<C>``."""

    preprocessed_commit: Array  # [8] digest
    pc_start: Array  # [3]
    cum_sum_x: Array  # [7] SepticDigest x-coordinate
    cum_sum_y: Array  # [7] SepticDigest y-coordinate
    enable_untrusted: int  # 0 or 1

    def observe_into(self, transcript: Transcript) -> Transcript:
        """Absorb the vk in SP1's order (``config.rs::observe_into``):
        commit, pc_start, cum sums, enable_untrusted, six zero pads."""
        dtype = self.preprocessed_commit.dtype
        transcript = transcript.observe(self.preprocessed_commit)
        transcript = transcript.observe(self.pc_start)
        transcript = transcript.observe(self.cum_sum_x)
        transcript = transcript.observe(self.cum_sum_y)
        transcript = transcript.observe(jnp.array(self.enable_untrusted, dtype))
        return transcript.observe(jnp.zeros(6, dtype))


@dataclass(frozen=True)
class TraceShape:
    """One trace matrix's statement shape."""

    height: int
    width: int


@dataclass(frozen=True)
class ChipShape:
    """One chip's statement trace shapes: the main trace, plus the
    preprocessed trace when the chip carries one — the verifier-side
    counterpart of SP1's ``chip.width()`` / ``chip.preprocessed_width()``
    (``crates/hypercube/src/verifier/shard.rs``). One record per chip keeps
    the height/width/prep statement atomic: a half-stated preprocessed
    trace is unrepresentable."""

    main: TraceShape
    prep: TraceShape | None = None


@dataclass(frozen=True)
class ChipOpenedValues:
    """SP1 mirror: ``ChipOpenedValues<F, EF>`` — one chip's zerocheck
    openings as the shard-proof wire carries them. ``degree`` is the chip's
    padded height; the wire stores its bits MSB-first over
    ``max_log_row_count + 1`` positions."""

    preprocessed_evals: Array | None
    main_evals: Array
    degree: int


@dataclass(frozen=True)
class ChipTrace:
    """SP1 mirror: ``Trace<F, B>`` — (trace matrix, live row count)."""

    array: Array
    num_real: int


@dataclass(frozen=True)
class Traces:
    """SP1 mirror: ``Traces<F, B>`` = ordered ``chip name -> ChipTrace``.

    ``chip_order`` is the canonical iteration order for every downstream
    stage (commit packing, GKR circuit, zerocheck batching). It is the
    insertion order of what ``from_arrays`` receives — producers fix the
    order (the dump reader walks name-sorted files) so independently-built
    shards agree on layout.
    """

    per_chip: dict[str, ChipTrace]
    chip_order: tuple[str, ...]

    @classmethod
    def from_arrays(
        cls, arrays: dict[str, Array], num_reals: dict[str, int]
    ) -> "Traces":
        names = tuple(arrays.keys())
        return cls(
            per_chip={
                n: ChipTrace(array=arrays[n], num_real=num_reals[n]) for n in names
            },
            chip_order=names,
        )


@dataclass(frozen=True)
class MainTraceData:
    """SP1 mirror: ``MainTraceData`` — main traces + shard public values +
    the chip definitions (constraints/interactions) evaluating them."""

    traces: Traces
    public_values: Array
    chips: dict[str, "Chip"]


@dataclass(frozen=True)
class ShardData:
    """One shard's prover input.

    ``preprocessed_traces`` stays raw here; committing it into SP1's
    ``ProvingKey.preprocessed_data`` form belongs to the trace-commit
    stage, which owns the jagged packing that commitment runs on.
    """

    vk: MachineVerifyingKey
    preprocessed_traces: dict[str, Array]
    main_trace_data: MainTraceData
