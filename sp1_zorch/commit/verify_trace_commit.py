# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""rsp byte-match harness for the trace commit — a runnable, not a unittest.

Real-block data (~1.5 GB/shard) makes this an iteration tool, not part of the
test suite: the prep region must reproduce the dump VK's
``preprocessed_commit`` and the main region the dump's ``main_commit``.
Exits non-zero on any mismatch so it still gates scripts/CI.

    bazel run //sp1_zorch/commit:verify_trace_commit -- \\
        --shard_dir=/path/to/rsp_dump/shardN --stage=prep

``--stage=main`` needs a CUDA GPU.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax.numpy as jnp
from absl import app, flags
from zk_dtypes import koalabear_mont as F

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.commit.trace_commit import commit_region
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.fixture_loader import _parse_kv_lines, read_dump
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

_SHARD_DIR = flags.DEFINE_string(
    "shard_dir", None, "rsp shard dump directory (e.g. .../rsp_dump/shard1)."
)
_STAGE = flags.DEFINE_enum(
    "stage", "prep", ["prep", "main", "all"], "Which region(s) to byte-match."
)

# SP1 core machine parameters (whir-zorch prove_shard_benchmark /
# sp1-hypercube core config): 2^21 stacking height, 2^22 max rows, 4x blowup.
_LOG_STACKING_HEIGHT = 21
_MAX_LOG_ROW_COUNT = 22
_LOG_BLOWUP = 2


def _commit(
    traces: dict, smcs: SingleMatrixCommitmentScheme, *, jit: bool
) -> jnp.ndarray:
    region = JaggedRegion.from_chips(
        [traces[name] for name in sorted(traces)],
        log_stacking_height=_LOG_STACKING_HEIGHT,
        max_log_row_count=_MAX_LOG_ROW_COUNT,
        chip_names=tuple(sorted(traces)),
    )
    commitment, _ = commit_region(region, smcs, log_blowup=_LOG_BLOWUP, jit=jit)
    return commitment


def _check(label: str, got: jnp.ndarray, want: jnp.ndarray) -> bool:
    ok = bool(jnp.all(got == want))
    print(f"[{label}] {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print(f"  got : {got}")
        print(f"  want: {want}")
    return ok


def main(argv: list[str]) -> None:
    del argv
    if not _SHARD_DIR.value:
        raise app.UsageError("--shard_dir is required")
    shard_dir = Path(_SHARD_DIR.value)
    dump = read_dump(shard_dir)
    perm = Poseidon2(koalabear16_params())
    smcs = SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )

    ok = True
    if _STAGE.value in ("prep", "all"):
        t0 = time.monotonic()
        # Small region: eager avoids paying the fused-pipeline compile.
        got = _commit(dump.preprocessed, smcs, jit=False)
        ok &= _check("prep vs vk.preprocessed_commit", got, dump.vk.preprocessed_commit)
        print(f"  ({time.monotonic() - t0:.1f}s)")
    if _STAGE.value in ("main", "all"):
        kv = _parse_kv_lines((shard_dir / "gpu_commitment.txt").read_text())
        want = jnp.array(
            [int(x.strip()) for x in kv["main_commit"].strip("[]").split(",")],
            dtype=jnp.uint32,
        ).astype(F)
        if int(kv["num_chips"]) != len(dump.traces):
            print(
                f"[main] chip count mismatch: {kv['num_chips']} != {len(dump.traces)}"
            )
            ok = False
        t0 = time.monotonic()
        # The @jit zone is what fits the rsp-scale main region on 32 GB
        # (see sp1_zorch.commit.trace_commit).
        got = _commit(dump.traces, smcs, jit=True)
        ok &= _check("main vs gpu_commitment.main_commit", got, want)
        print(f"  ({time.monotonic() - t0:.1f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    app.run(main)
