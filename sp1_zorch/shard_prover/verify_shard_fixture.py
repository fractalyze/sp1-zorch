# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""rsp dump fixture check — a runnable, not a unittest.

Real-block data (~1.5 GB/shard) makes this an iteration tool, not part of the
test suite: the loader's output must match the dump's own metadata (chip set,
trace dims, public values, VK) and resolve real AIR constraints for the bulk
of the chips. Exits non-zero on any mismatch so it still gates scripts/CI.

    bazel run //sp1_zorch/shard_prover:verify_shard_fixture -- \\
        --shard_dir=/path/to/rsp_dump/shardN
"""

from __future__ import annotations

import sys
from pathlib import Path

from absl import app, flags
from zk_dtypes import koalabear_mont as F

from sp1_zorch.shard_prover.fixture_loader import _parse_kv_lines, load_fixture_shard

_SHARD_DIR = flags.DEFINE_string(
    "shard_dir", None, "rsp shard dump directory (e.g. .../rsp_dump/shard1)."
)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(
        f"[{label}] {'OK' if ok else 'MISMATCH'}{' — ' + detail if detail and not ok else ''}"
    )
    return ok


def main(argv: list[str]) -> None:
    del argv
    if not _SHARD_DIR.value:
        raise app.UsageError("--shard_dir is required")
    shard_dir = Path(_SHARD_DIR.value)
    shard = load_fixture_shard(shard_dir)
    traces = shard.main_trace_data.traces

    ok = True
    metas = {p.stem for p in (shard_dir / "gpu_traces").glob("*.meta")}
    ok &= _check("chip set", set(traces.chip_order) == metas)

    dims_ok = True
    for name in traces.chip_order:
        meta = _parse_kv_lines((shard_dir / "gpu_traces" / f"{name}.meta").read_text())
        chip = traces.per_chip[name]
        if chip.array.shape != (int(meta["num_real"]), int(meta["width"])):
            dims_ok = _check(f"dims:{name}", False, f"{chip.array.shape} vs meta")
        if chip.num_real != int(meta["num_real"]) or chip.array.dtype != F:
            dims_ok = _check(f"meta:{name}", False, "num_real/dtype")
    ok &= _check("trace dims vs metas", dims_ok)

    ok &= _check(
        "public values (SP1 max PVs)",
        shard.main_trace_data.public_values.shape == (187,),
    )
    ok &= _check(
        "vk + prep loaded",
        shard.vk.preprocessed_commit.shape == (8,) and bool(shard.preprocessed_traces),
    )
    chips = shard.main_trace_data.chips
    with_constraints = [n for n, c in chips.items() if c.constraint_names]
    # A real shard must attach actual AIR constraints for the bulk of its
    # chips — an all-stub shard means the name/width mapping broke.
    ok &= _check(
        "constraints attached",
        len(with_constraints) > len(chips) // 2,
        f"{len(with_constraints)}/{len(chips)}",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    app.run(main)
