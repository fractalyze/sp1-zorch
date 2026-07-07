# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shard ingest from a co-located trace-gen producer (no dump files).

A co-located riscv-witness GPU trace-gen (fractalyze/riscv-witness#2251)
hands per-chip device ``jax.Array``s — ``riscv_witness.chips_to_jax``: raw
Montgomery ``uint32`` cells, rw-named, fired chips only — plus the bundle's
``num_reals`` and a device public-values vector
(``riscv_witness.public_values_to_jax``). This module is
``fixture_loader``'s dump-free sibling: the same ``ShardData`` / region
assembly with the trace payload arriving by device handoff instead of
``np.fromfile``, so it never leaves the GPU between trace-gen and prove
(fractalyze/sp1-zorch#200).

Raw-Montgomery ``uint32`` is bit-identical to ``koalabear_mont``, so ingest
is a dtype *view* on every array — never a conversion or copy, and never a
device-to-host transfer.
"""

from __future__ import annotations

from collections.abc import Mapping

from jax import Array
from zk_dtypes import koalabear_mont
from zorch.pcs.jagged.region import JaggedRegion

from sp1_zorch.shard_prover.chip_loader import rw_name_to_sp1
from sp1_zorch.shard_prover.fixture_loader import resolve_chips
from sp1_zorch.shard_prover.replay import shard_regions
from sp1_zorch.shard_prover.types import (
    MachineVerifyingKey,
    MainTraceData,
    ShardData,
    Traces,
)


def regions_from_producer(
    chips: dict[str, Array],
    *,
    num_reals: Mapping[str, int],
    public_values: Array,
    vk: MachineVerifyingKey,
    preprocessed: Mapping[str, Array] | None = None,
) -> tuple[JaggedRegion, JaggedRegion | None, ShardData]:
    """Assemble a producer-fed shard: the jagged regions plus its
    :class:`ShardData`.

    ``chips`` / ``num_reals`` are the producer bundle's rw-named outputs
    as-is; ``public_values`` is its device ``uint32`` vector. ``vk`` and
    ``preprocessed`` stay per-program host loads, exactly as the fixture
    path produces them — the producer exports neither.

    Returns ``(main_region, prep_region, shard)``, so the prove call is the
    fixture path's, unchanged::

        chain(
            ShardCarry(main_region, prep_region,
                       shard.main_trace_data.public_values),
            fresh_transcript(),
        )

    Structural trace validation (ndim, one dtype, height caps) is
    ``JaggedRegion.from_chips``'s, via :func:`shard_regions` — not
    duplicated here. Ingest checks only what the region cannot see: the
    rw->SP1 naming and the producer's ``num_reals`` against the array
    heights (equal by construction on the device arm, where rows arrive
    unpadded — so a mismatch means a torn bundle, not padding).
    """
    prep = dict(preprocessed) if preprocessed else {}

    # Iterate in sorted SP1-name order: the fixture path's chip_order comes
    # from a sorted walk of SP1-named dump files, and the rw / SP1 collations
    # disagree ("uint256_mul" < "utype" but "UType" < "Uint256MulMod"), so the
    # producer's sorted-rw-name dict order is re-sorted after mapping. This
    # keeps chip_order — and with it every downstream layout and the
    # preamble's Fiat-Shamir chip metadata — identical to a fixture-loaded
    # shard of the same traces.
    traces: dict[str, Array] = {}
    reals: dict[str, int] = {}
    for sp1_name, rw_name in sorted((rw_name_to_sp1(n), n) for n in chips):
        trace = chips[rw_name].view(koalabear_mont)
        num_real = int(num_reals[rw_name])
        if num_real != trace.shape[0]:
            raise ValueError(
                f"producer num_reals[{rw_name!r}] = {num_real} != trace "
                f"height {trace.shape[0]}"
            )
        traces[sp1_name] = trace
        reals[sp1_name] = num_real

    shard = ShardData(
        vk=vk,
        preprocessed_traces=prep,
        main_trace_data=MainTraceData(
            traces=Traces.from_arrays(traces, reals),
            public_values=public_values.view(koalabear_mont),
            chips=resolve_chips(traces, prep),
        ),
    )
    return (*shard_regions(shard), shard)
