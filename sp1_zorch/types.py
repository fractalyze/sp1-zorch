# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard's claims, witnesses and wire types, mirroring sp1-hypercube's
``ShardData`` / ``MainTraceData``
(https://github.com/fractalyze/sp1/blob/e2c02f376/crates/hypercube/src/prover/shard.rs)
and ``MachineVerifyingKey``
(https://github.com/fractalyze/sp1/blob/e2c02f376/crates/hypercube/src/verifier/config.rs).

Both roles of a reduction read these, so neither imports the other to do it;
the zorch imports stay TYPE_CHECKING-only for the same reason `shard_prover`
takes no zorch dep (#60).

Field-element arrays carry raw Montgomery u32 (``koalabear_mont`` views) so
downstream byte-match stages compare bytes directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import frx
import frx.numpy as fnp
from frx import Array

if TYPE_CHECKING:
    from rw_constraints import Chip
    from zorch.logup_gkr.circuit import LogUpGkrOutput
    from zorch.logup_gkr.jagged_prover import JaggedLayerProof
    from zorch.pcs.jagged.open import StackedOpenProof
    from zorch.pcs.jagged.prover import JaggedEvalMsg
    from zorch.sumcheck.prover import RoundMsg
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
    def from_arrays(cls, arrays: dict[str, Array], num_reals: dict[str, int]) -> Traces:
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
    chips: dict[str, Chip]


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
class ChipOpenedValues:
    """SP1 mirror: ``ChipOpenedValues<F, EF>`` — one chip's zerocheck
    openings as the shard-proof wire carries them, converted from the
    in-flight `ChipEvaluation`. ``degree`` is the chip's padded height, the
    field the in-flight form has no need of; the wire stores its bits
    MSB-first over ``max_log_row_count + 1`` positions."""

    preprocessed_evals: Array | None
    main_evals: Array
    degree: int


# Pytree: both evals are array leaves (preprocessed is None for prep-less
# chips), so a carry holding these openings stays an arrays-only pytree.
@partial(
    frx.tree_util.register_dataclass,
    data_fields=["main", "preprocessed"],
    meta_fields=[],
)
@dataclass(frozen=True)
class ChipEvaluation:
    """One chip's trace openings at the final GKR point.

    The in-flight form: a pytree, so it threads the jitted opens as a carry.
    `ChipOpenedValues` is the same openings in the shard wire's shape, which
    additionally spells out the chip's row count; `serialize.chip_opened_values`
    is the one conversion between them.
    """

    main: Array  # (width,) EF, one eval per main column
    preprocessed: Array | None  # (prep width,) EF, when the chip has prep

    def all_evals(self) -> Array:
        """The ``[main | prep]`` evaluation vector — the column order of the
        beta-power batching shared by the GKR opening claims and the
        zerocheck column batch."""
        if self.preprocessed is not None:
            return fnp.concatenate([self.main, self.preprocessed])
        return self.main


@dataclass(frozen=True, kw_only=True)
class BoundRoots:
    """The structure-bound Merkle roots an opening is checked against.

    Both roles derive these — the verifier from the vk and the proof's
    commitment — so they are claim data. The raw `SmcsCommitments` the prover
    also holds are the same two arrays before structure binding, so only a
    distinct type stops the two being passed to each other's slot; the fields
    are keyword-only for the same reason.
    """

    preprocessed: Array
    main: Array

    def in_round_order(self, has_preprocessed: bool) -> list[Array]:
        """The roots the opening binds against, in SP1's round order."""
        return [self.preprocessed, self.main] if has_preprocessed else [self.main]


@dataclass(frozen=True, kw_only=True)
class SmcsCommitments:
    """Per-round SMCS commitments before structure binding — the wire's
    ``original_commitments``.

    Prover output the verifier cannot derive, so this is proof data, not claim
    data. `preprocessed` is absent when the shard commits no preprocessed
    region, which is why the arity varies where `BoundRoots` is fixed: SP1's
    verifier always carries the vk's preprocessed commitment even when the
    prover has no preprocessed region to commit.
    """

    preprocessed: Array | None = None
    main: Array

    def in_round_order(self) -> list[Array]:
        return (
            [self.preprocessed, self.main]
            if self.preprocessed is not None
            else [self.main]
        )


@dataclass(frozen=True)
class ShardClaim:
    """Some trace of this shape is a valid execution of the shard.

    Spelled out: there exists a trace holding `chip_metadata`'s chips at its
    row counts, whose preprocessed part is the one `vk` commits to, on which
    every chip's AIR constraints vanish and whose LogUp bus balances against
    `public_values`. Nothing here names that trace — it is existentially
    quantified, and the prover exhibits one by committing to it, which is why
    the commitment is proof data rather than a field of the statement.
    """

    vk: MachineVerifyingKey
    public_values: Array
    chip_metadata: ChipMetadata


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


@dataclass(frozen=True)
class GkrOutputClaim:
    """The trace's LogUp columns take `chip_openings` at `eval_point`.

    What LogUp-GKR reduces the bus-balance statement to: checking a whole
    logarithmic-derivative argument becomes checking a handful of column
    values at one point. Both roles derive it — the prover from its own layer
    chain, the verifier by replaying the proof — so neither has to be trusted
    for it.
    """

    eval_point: Array
    chip_openings: Mapping[str, ChipEvaluation]


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


@dataclass(frozen=True)
class ZerocheckClaim:
    """Every chip's AIR constraints vanish on the trace — conditionally on
    `gkr`, whose column openings the constraint sum folds in.

    Conditional because zerocheck never re-proves the LogUp leg: it inherits
    `gkr` as a hypothesis and discharges only the constraint half, so the two
    together are what pin the trace. `public_values` supplies the operands the
    PV-reading constraint circuits index.
    """

    public_values: Array
    gkr: GkrOutputClaim
    chip_metadata: ChipMetadata


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


@dataclass(frozen=True)
class TraceEvaluationClaim:
    """The trace evaluates to `opened_values` at `point`.

    Zerocheck reduces to this and the jagged opening discharges it: once the
    constraint sum is checked, all that remains is that the values it was
    computed over really are the committed trace's.
    """

    point: Array
    opened_values: Mapping[str, ChipEvaluation]


@dataclass(frozen=True)
class JaggedOpeningClaim:
    """The trace committed under `roots` evaluates to
    `evaluation.opened_values` at `evaluation.point`.

    `TraceEvaluationClaim` asserts this of *the* trace; binding it to `roots`
    is what ties the assertion to the one the prover actually committed to, so
    discharging this claim leaves nothing to prove.
    """

    evaluation: TraceEvaluationClaim
    roots: BoundRoots
    chip_metadata: ChipMetadata


@dataclass(frozen=True)
class JaggedOpeningWitness:
    """What discharging a `JaggedOpeningClaim` takes: the trace itself, and
    the prover data the PCS kept from committing it."""

    trace: ShardWitness
    commit_data: JaggedCommitData


@dataclass(frozen=True)
class JaggedCommitData:
    """What the jagged PCS's commit half hands to its open half, per round in
    [prep, main] order.

    Fiat-Shamir splits the halves across the whole shard proof — the
    commitment must bind the transcript before LogUp-GKR draws a challenge,
    and the open cannot run until zerocheck produces the point to open at — so
    this bridges that gap. It is not prover-only state: the open draws the
    wire's ``original_commitments`` straight from `commitments`, and each
    tree's top layer is serialized as that round's raw root. Only the lower
    layers stay prover-side, to answer the query openings.

    Committing binds in three steps, and which level goes where is why two of
    them are kept here and the third is not:

    - raw Merkle root, ``digest_layers[-1][0]`` — on the wire as that round's
      raw root;
    - shape-bound, `commitments` — the wire's ``original_commitments``;
    - structure-bound — what the transcript absorbs and `BoundRoots.main` is
      checked against, so it rides `ShardProof` rather than this type.
    """

    digest_layers: tuple[list[Array], ...]
    commitments: SmcsCommitments


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


@dataclass(frozen=True)
class ShardProof:
    """What a verifier needs to check a `ShardClaim` without the trace.

    The commitment fixes which trace is being talked about; the three
    reduction proofs then carry the verifier along the same chain the prover
    walked, one section per Stage, ending at the trivial claim.
    """

    commitment: Array  # structure-bound main root; see JaggedCommitData
    gkr: LogupGkrProof
    zerocheck: ZerocheckProof
    jagged: JaggedPcsProof
