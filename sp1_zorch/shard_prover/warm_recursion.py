# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Recursion-machine shard synthesis + the staged chain the recursion warm drives.

The SP1 recursion stages (normalize / combine / shrink) prove through the SAME
four-phase chain as a core shard — trace commit, LogUp-GKR, zerocheck, jagged
opening — but at the recursion machines' statics: the combine machine
(normalize + combine) and the shrink machine, over the 8 ``sp1/recursion/v1``
chips, with NO cap-class pins (the production recursion prove passes
``tc_class=None`` / ``gkr_class=None``, so every zone keys on the shard's
exact array shapes). Those shapes are fixed circuit shapes per program /
arity — block-INDEPENDENT — so the whole recursion compile family is seedable
from a small shape spec, without any shard dump.

The shapes themselves are not derivable inside this repo (they are properties
of riscv-witness's recursion programs and setup bundles), so they arrive as a
per-program ``recursion shape spec`` JSON — a list of entries::

    [{"name": "normalize", "stage": "normalize", "public_values_len": 168,
      "chips": {"BaseAlu": {"rows": 32, "cols": 4,
                            "prep_rows": 64, "prep_cols": 7}, ...}}, ...]

``rows`` is the chip's emitted main height (its ``num_reals``), ``prep_*``
the keygen-height preprocessed matrix (0/absent = no prep for that chip).
Every value is a constant of the (program, SP1 rev) pair — capture once from
a real run's recursion shard (any block).

:func:`build_recursion_shard` rebuilds the production recursion ``ShardData``
shape-for-shape: zero-filled arrays constructed with the same calls the
production builder uses (host ``uint32`` cells -> ``jnp.array(...).astype``
for main traces + public values, ``.view`` for prep, the fixed-width VK
fields), chips resolved with the production width rule (rw chip when manifest
``num_cols`` equals main + prep width, constraint-less stub otherwise), chip
order sorted by SP1 name. All arrays are uncommitted, as production's
host-decoded inputs are — the chain's own eager ``device_put``\\ s originate
every committed lineage identically here and in a real prove, so the
per-parameter commitment component of the cache key matches by construction.

:func:`prove_stage_chain` then runs the four phases in the production staged
order (commit -> bind -> GKR -> zerocheck -> open) with the stage machine's
parameters. Driven under ``warm_worker``'s ``frx.jit`` intercept (see
``warm_recursion_worker``) it lowers + compiles every recursion zone into the
persistent cache without executing a kernel; driven bare it is a real
(executing) recursion prove of the synthetic shard — the two request
identical cache keys.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

# frx's import hook redirects the `jax` name to frx for every later import,
# but only for names not already in sys.modules — whichever package loads
# first owns the name for the process. This module defers its frx imports
# into the builder/chain functions, so without an eager anchor a caller can
# reach chip/replay code that imports the transition-era `jax` wheel first
# (via rw_constraints), and the deferred `import frx.numpy` then resolves
# jax internals against that wheel and fails.
import frx  # noqa: F401  (isort: skip)

if TYPE_CHECKING:
    from sp1_zorch.types import MachineVerifyingKey, ShardData

# SP1 hardcodes the LogUp-GKR grind at 12 bits for every stage
# (crates/hypercube/src/verifier/shard.rs); the basefold grind is the
# per-machine ``fri_pow_bits``.
GKR_GRINDING_BITS = 12


@dataclass(frozen=True)
class RecursionMachine:
    """One recursion machine's prover statics.

    Mirrors sp1 ``crates/prover/src/components.rs`` (per-machine config) and
    ``crates/primitives/src/fri_params.rs`` (the two FRI configs); the same
    five values the production recursion prove threads into the shard chain.
    Every field is a jit static, so a value drift against production computes
    disjoint cache keys.
    """

    log_stacking_height: int
    log_blowup: int
    num_queries: int
    fri_pow_bits: int
    max_log_row_count: int


# The combine machine serves Normalize AND Combine (SP1 reuses one
# RecursionAir; only the prover parameters split combine vs shrink).
COMBINE_MACHINE = RecursionMachine(
    log_stacking_height=20,
    log_blowup=2,
    num_queries=124,
    fri_pow_bits=16,
    max_log_row_count=21,
)
SHRINK_MACHINE = RecursionMachine(
    log_stacking_height=18,
    log_blowup=3,
    num_queries=94,
    fri_pow_bits=22,
    max_log_row_count=19,
)

STAGE_MACHINES: dict[str, RecursionMachine] = {
    "normalize": COMBINE_MACHINE,
    "combine": COMBINE_MACHINE,
    "shrink": SHRINK_MACHINE,
}

# CamelCase SP1 recursion driver chip names -> snake_case rw manifest names —
# the 8 canonical ``sp1/recursion/v1`` chips. The drivers' 3 wrap-machine
# chips (Poseidon2SBox / Poseidon2LinearLayer / ExtFeltConvert) have no
# recursion/v1 constraints and never reach a recursion ``ShardData``.
RECURSION_NAME_MAP: dict[str, str] = {
    "BaseAlu": "base_alu",
    "ExtAlu": "ext_alu",
    "MemoryConst": "memory_const",
    "MemoryVar": "memory_var",
    "Poseidon2WideDeg3": "poseidon2_wide_deg3",
    "PrefixSumChecks": "prefix_sum_checks",
    "PublicValues": "public_values",
    "Select": "select",
}

# Bincode ``MachineVerifyingKey`` field widths (in field elements) — the fixed
# recursion VK layout the production builder decodes.
_VK_PC_START = 3
_VK_CUM_SUM = 7  # SepticDigest<F> half (x and y each)
_VK_PREPROCESSED_COMMIT = 8


@dataclass(frozen=True)
class ChipShape:
    """One chip's fixed recursion trace shape: main at the emitted height
    (``num_reals``), prep at the keygen height (0 = no prep)."""

    rows: int
    cols: int
    prep_rows: int = 0
    prep_cols: int = 0


@dataclass(frozen=True)
class StageShapes:
    """One warm entry: a stage machine + its fixed shard shapes."""

    name: str
    stage: str
    chips: dict[str, ChipShape] = field(default_factory=dict)
    public_values_len: int = 0

    def machine(self) -> RecursionMachine:
        return STAGE_MACHINES[self.stage]


def load_spec(path: str | Path) -> list[StageShapes]:
    """Parse a recursion shape spec JSON into validated entries.

    Raises ``ValueError`` on an unknown stage, a chip outside the 8 canonical
    recursion chips, a non-positive main shape, or duplicate entry names —
    a malformed spec must fail the seed loudly, not fill wrong keys.
    """
    raw = json.loads(Path(path).read_text())
    entries: list[StageShapes] = []
    for e in raw:
        name = e.get("name", e["stage"])
        stage = e["stage"]
        if stage not in STAGE_MACHINES:
            raise ValueError(
                f"spec entry {name!r}: unknown stage {stage!r} "
                f"(expected one of {sorted(STAGE_MACHINES)})"
            )
        chips: dict[str, ChipShape] = {}
        for chip_name, s in e["chips"].items():
            if chip_name not in RECURSION_NAME_MAP:
                raise ValueError(
                    f"spec entry {name!r}: {chip_name!r} is not a canonical "
                    f"recursion chip (expected one of {sorted(RECURSION_NAME_MAP)})"
                )
            shape = ChipShape(
                rows=int(s["rows"]),
                cols=int(s["cols"]),
                prep_rows=int(s.get("prep_rows", 0)),
                prep_cols=int(s.get("prep_cols", 0)),
            )
            if shape.rows <= 0 or shape.cols <= 0:
                raise ValueError(
                    f"spec entry {name!r}: {chip_name} main shape must be "
                    f"positive, got {shape.rows}x{shape.cols}"
                )
            chips[chip_name] = shape
        if not chips:
            raise ValueError(f"spec entry {name!r}: no chips")
        pv_len = int(e["public_values_len"])
        if pv_len <= 0:
            raise ValueError(f"spec entry {name!r}: public_values_len must be > 0")
        entries.append(
            StageShapes(name=name, stage=stage, chips=chips, public_values_len=pv_len)
        )
    names = [e.name for e in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate spec entry names: {names}")
    if not entries:
        raise ValueError("empty recursion shape spec")
    return entries


def load_recursion_chips() -> dict[str, Any]:
    """The 8 ``sp1/recursion/v1`` rw chip definitions, snake_case-keyed."""
    from sp1_zorch.shard_prover.chip_loader import load_sp1_chips

    return load_sp1_chips("sp1", "recursion/v1")


def _synthetic_vk() -> MachineVerifyingKey:
    """A zero-valued ``MachineVerifyingKey`` at the fixed recursion VK widths.

    Field values never enter a compile key — only the four array shapes and
    dtypes do — so zeros stand in for any program's VK. Constructed with the
    production decode's array calls (host ``<u4`` words ->
    ``jnp.array(...).astype``) so dtype/weak-type match exactly.
    """
    import frx.numpy as jnp
    import numpy as np
    from zk_dtypes import koalabear_mont as BF

    from sp1_zorch.types import MachineVerifyingKey

    def zeros(count: int) -> Any:
        return jnp.array(np.zeros(count, dtype="<u4")).astype(BF)

    return MachineVerifyingKey(
        preprocessed_commit=zeros(_VK_PREPROCESSED_COMMIT),
        pc_start=zeros(_VK_PC_START),
        cum_sum_x=zeros(_VK_CUM_SUM),
        cum_sum_y=zeros(_VK_CUM_SUM),
        enable_untrusted=0,
    )


def _resolve_chips(
    shapes: dict[str, ChipShape], rw_chips: dict[str, Any]
) -> dict[str, Any]:
    """The production chip-resolution rule over spec shapes.

    A chip gets its rw ``recursion/v1`` definition only when the manifest
    ``num_cols`` (prep + main columns) equals the spec's main + prep width;
    otherwise a constraint-less stub keeps indexing/layout. Same policy as
    the production recursion assembly and this repo's dump loader.
    """
    from sp1_zorch.shard_prover.chip_loader import make_chip_stub

    chips: dict[str, Any] = {}
    for name in sorted(shapes):
        s = shapes[name]
        rw_chip = rw_chips.get(RECURSION_NAME_MAP.get(name, name.lower()))
        total = s.cols + s.prep_cols
        if rw_chip is not None and getattr(rw_chip, "num_cols", None) == total:
            chips[name] = rw_chip
        else:
            chips[name] = make_chip_stub(name, s.cols)
    return chips


def build_recursion_shard(entry: StageShapes, rw_chips: dict[str, Any]) -> ShardData:
    """A zero-filled recursion ``ShardData`` at ``entry``'s exact shapes.

    Arrays are built with the production builder's calls — main traces and
    public values as host ``<u4`` cells Mont-encoded via ``.astype``, prep
    reinterpreted via ``.view`` — so dtypes, weak types, and (un)committed
    placement all match a real recursion assembly; only the values are zero,
    and values never enter a compile key.
    """
    import frx.numpy as jnp
    import numpy as np
    from zk_dtypes import koalabear_mont as BF

    from sp1_zorch.types import MainTraceData, ShardData, Traces

    traces = {}
    num_reals = {}
    preprocessed = {}
    for name in sorted(entry.chips):
        s = entry.chips[name]
        cells = np.zeros((s.rows, s.cols), dtype="<u4")
        traces[name] = jnp.array(cells).astype(BF)
        num_reals[name] = s.rows
        if s.prep_rows and s.prep_cols:
            prep_cells = np.zeros((s.prep_rows, s.prep_cols), dtype="<u4")
            preprocessed[name] = jnp.array(prep_cells).view(BF)
    if not preprocessed:
        raise ValueError(
            f"entry {entry.name!r} has no preprocessed chips; SP1 requires "
            "non-empty prep"
        )
    public_values = jnp.array(np.zeros(entry.public_values_len, dtype="<u4")).astype(BF)
    return ShardData(
        vk=_synthetic_vk(),
        preprocessed_traces=preprocessed,
        main_trace_data=MainTraceData(
            traces=Traces.from_arrays(traces, num_reals),
            public_values=public_values,
            chips=_resolve_chips(entry.chips, rw_chips),
        ),
    )


@functools.cache
def _smcs() -> Any:
    """SP1's single-matrix commitment — identical for every stage, one
    instance per process (an identity-keyed jit static; a fresh one per
    chain re-traces every zone)."""
    from hash_frx.compression import Compression, CompressionParams
    from hash_frx.poseidon2.poseidon2 import Poseidon2
    from hash_frx.sponge import Sponge, SpongeParams
    from zorch.commit.smcs import SingleMatrixCommitmentScheme

    from sp1_zorch.poseidon2.koalabear16 import koalabear16_params

    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def prove_stage_chain(
    machine: RecursionMachine,
    shard: ShardData,
    *,
    gkr_pow_bits: int = GKR_GRINDING_BITS,
) -> Any:
    """Run the four-phase shard chain at ``machine``'s statics.

    The production recursion prove's staged order — commit ->
    ``bind_commitment`` -> GKR -> zerocheck -> jagged open — with
    ``tc_class`` / ``gkr_class`` unset (recursion pins no caps; every zone
    keys on the shard's own shapes). Proof encoding is not run: bincode
    encoding is host work over the sections and contributes no cache entry.

    Under the ``warm_worker`` intercept this compiles every zone without
    executing; bare, it is a real prove of the shard. Returns the jagged
    opening result (the last phase's output) so callers can anchor on chain
    completion.
    """
    from zorch.pcs.jagged.region import JaggedRegion

    from sp1_zorch.logup_gkr.circuit import build_gkr_chips
    from sp1_zorch.logup_gkr.prover import num_beta_values
    from sp1_zorch.shard_prover.prove_shard import ShardProver, bind_commitment
    from sp1_zorch.shard_prover.replay import fresh_transcript
    from sp1_zorch.types import (
        ChipMetadata,
        JaggedOpeningClaim,
        JaggedOpeningWitness,
        ShardClaim,
        ShardWitness,
        ZerocheckClaim,
    )

    traces = shard.main_trace_data.traces
    order = traces.chip_order
    num_reals = [traces.per_chip[n].num_real for n in order]
    main_region = JaggedRegion.from_chips(
        [traces.per_chip[n].array for n in order],
        log_stacking_height=machine.log_stacking_height,
        max_log_row_count=machine.max_log_row_count,
        chip_names=order,
    )
    prep = shard.preprocessed_traces
    prep_names = tuple(sorted(prep))
    prep_region = (
        JaggedRegion.from_chips(
            [prep[n] for n in prep_names],
            log_stacking_height=machine.log_stacking_height,
            max_log_row_count=machine.max_log_row_count,
            chip_names=prep_names,
        )
        if prep
        else None
    )

    chips = shard.main_trace_data.chips
    gkr_chips = build_gkr_chips(chips, order)
    stage = ShardProver(
        smcs=_smcs(),
        log_blowup=machine.log_blowup,
        gkr_chips=gkr_chips,
        chips=chips,
        num_betas=num_beta_values(chips),
        num_row_variables=machine.max_log_row_count - 1,
        max_log_row_count=machine.max_log_row_count,
        open_num_queries=machine.num_queries,
        open_pow_bits=machine.fri_pow_bits,
        pow_bits=gkr_pow_bits,
        zerocheck_total_cap_class=None,
        gkr_cap_class=None,
    )
    claim = ShardClaim(
        shard.vk,
        shard.main_trace_data.public_values,
        ChipMetadata(tuple(order), tuple(int(r) for r in num_reals)),
    )
    witness = ShardWitness(main_region, prep_region)

    transcript = fresh_transcript()
    commitment, commit_data = stage.opening.commit(witness)
    transcript, roots = bind_commitment(transcript, claim, commitment)
    gkr = stage.gkr.prove(claim, witness, transcript)
    zerocheck = stage.zerocheck.prove(
        ZerocheckClaim(claim.public_values, gkr.reduced_claim, claim.chip_metadata),
        witness,
        gkr.transcript,
    )
    return stage.opening.prove(
        JaggedOpeningClaim(zerocheck.reduced_claim, roots, claim.chip_metadata),
        JaggedOpeningWitness(witness, commit_data),
        zerocheck.transcript,
    )
