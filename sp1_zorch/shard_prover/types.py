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
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
from frx import Array

if TYPE_CHECKING:
    from rw_constraints import Chip
    from zorch.pcs.jagged.region import JaggedRegion
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
        transcript = transcript.observe(fnp.array(self.enable_untrusted, dtype))
        return transcript.observe(fnp.zeros(6, dtype))


@dataclass(frozen=True)
class ChipWidths:
    """One chip's column counts — SP1's ``chip.width()`` /
    ``chip.preprocessed_width()`` (``crates/hypercube/src/verifier/shard.rs``).

    A static property of the AIR, identical on every shard, so it is role
    configuration rather than claim data. `prep` is None when the chip carries
    no preprocessed trace, which keeps a half-stated preprocessed trace
    unrepresentable. The other axis — how many rows each chip holds — varies
    shard to shard and rides `ChipMetadata` on the claim.
    """

    main: int
    prep: int | None = None


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


@dataclass(frozen=True)
class ChipMetadata:
    """Which chips this shard holds and how many real rows each one has, in
    SP1's chip order.

    The claim-side half of the trace dimensions: row counts change shard to
    shard, so the statement has to give them, while column counts are fixed by
    each chip's AIR and stay role configuration (`ChipWidths`). Held as values
    rather than as the absorb stream they encode — `preamble_stream` derives
    that — so both roles read the same statement instead of a blob only the
    transcript can consume.
    """

    chip_names: tuple[str, ...]
    num_reals: tuple[int, ...]

    def __post_init__(self) -> None:
        # Two parallel tuples, so the pairing is an invariant rather than a
        # shape. Checked here because a mismatch is otherwise inert until
        # something zips them, and the likeliest way to get one is passing a
        # region's `row_counts` (its `chip_heights` plus two stacking entries)
        # where its `chip_heights` belong.
        if len(self.chip_names) != len(self.num_reals):
            raise ValueError(
                f"{len(self.chip_names)} chip names but "
                f"{len(self.num_reals)} row counts"
            )

    def by_chip(self) -> dict[str, int]:
        return dict(zip(self.chip_names, self.num_reals, strict=True))

    def preamble_stream(self, *, dtype: Any) -> Array:
        """The preamble's chip-metadata stream as one flat array: chip count,
        then per chip (num_real, name length, name bytes). One flat absorb
        matches SP1's per-value observes byte-for-byte while skipping hundreds
        of single-element transcript calls."""
        metadata: list[int] = [len(self.chip_names)]
        for name, num_real in zip(self.chip_names, self.num_reals, strict=True):
            metadata.append(int(num_real))
            metadata.append(len(name))
            metadata.extend(name.encode("ascii"))
        return fnp.array(metadata, dtype)


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["main_region", "prep_region"],
    meta_fields=[],
)
@dataclass(frozen=True)
class ShardWitness:
    """The trace that makes a `ShardClaim` true: the shard's own rows, plus
    the preprocessed rows when the shard has them.

    A pytree, so the whole witness crosses a ``@jit`` boundary as one donated
    argument. Its leaves are exactly the regions' dense buffers — a `None`
    prep region is an empty subtree and contributes none.
    """

    main_region: JaggedRegion
    prep_region: JaggedRegion | None = None
