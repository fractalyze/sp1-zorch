# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""rsp byte-match harness for the assembled prove_shard chain -- a runnable.

Runs ``prove_shard_chain`` (the ``ProveChain`` of trace commit -> LogUp-GKR
-> zerocheck -> jagged evaluation proof) over a real rsp dump and seals the
composition against the reference:

- the commitment the chain's ``TraceCommitRound`` computes must equal the
  dump's ``main_commit`` (``gpu_commitment.txt``);
- the GKR evaluation point's row tail (SP1's ``zeta``) must equal
  ``gpu_z_row.txt``. ``zeta`` is a sponge image of every byte the chain
  observed through the LogUp-GKR leg, so this one match transitively pins the
  preamble (vk, public values, commitment, chip metadata) and the GKR leg,
  proving the Round wiring reproduces SP1's transcript. The zerocheck sumcheck
  point is not dumped to a txt (it lives only in the proof JSON); ``final_eval``
  below pins the zerocheck rounds instead. ``gpu_z_row.txt`` is SP1's ``zeta``,
  not the zerocheck point -- see ``zerocheck/verify_zerocheck.py``, the
  authoritative per-stage check.
- the jagged eval's outer sumcheck claim must equal
  ``phase4_sumcheck_claim``, sealing the eval stage's z_col sampling and
  per-column claim assembly.

With ``--ffi_verify`` the tool additionally assembles the bincode wire
(``encode_vk`` + ``encode_shard_proof``) and runs SP1's own verifier over it
via the ``sp1_verify_shard`` FFI (``SP1_JAX_FFI_LIB`` must point at
``libsp1_gpu_jax_ffi.so``) — the end-to-end acceptance gate of
fractalyze/sp1-zorch#21.

Each stage's internals are gated by its own runnable
(``commit:verify_trace_commit``, ``logup_gkr:verify_gkr_prove``,
``zerocheck:verify_zerocheck``, plus the eval stage's ``jagged:prover_test``
/ ``jagged:open_test``); this tool checks the composition, not each stage's
math. The chain wiring itself is unit-tested against a synthetic reference in
``prove_shard_test``.

Real-block data (~1.5 GB/shard) plus the GPU trace commit keep this a
runnable, not a unit test. Needs a CUDA GPU.

    bazel run //sp1_zorch/shard_prover:verify_prove_shard -- \\
        --shard_dir=/path/to/rsp_dump/shardN

Wall-clock is dominated by XLA/zkx GPU compiles, not kernel runtime — the
per-stage timings printed during the run show the split. Pass ``--runs=N``
to prove the chain N times in one process: run 1 is cold (compiles), runs
2+ are warm (executables reused), so the warm per-stage ``[stage X] Yms``
lines are the ones to compare against SP1's native prover. Across separate
processes, set ``JAX_COMPILATION_CACHE_DIR`` to a per-toolchain directory so
every run after the first skips the compiles; leave it unset for byte-match
gates (a cache shared across toolchains has served wrong executables).

``--max_stage=N`` runs + byte-checks only the first N stages (1=trace-commit ..
4=full), a cheaper loop that skips the downstream stages' compile.
``--drop_main_codeword`` (default on) drops the main codeword at commit and
re-encodes it at the open (SP1's drop_ldes) so it never pins ~6 GB through GKR +
zerocheck; pass ``=false`` to keep it resident (small shards / >32 GB cards).

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from absl import app, flags
from zk_dtypes import koalabear_mont as F

from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.logup_gkr.circuit import build_gkr_chips
from sp1_zorch.logup_gkr.prover import num_beta_values
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.fixture_loader import (
    _parse_ef_list,
    _parse_int_list,
    _parse_kv_lines,
    check_match,
    load_fixture_shard,
)
from sp1_zorch.shard_prover.prove_shard import (
    ShardCarry,
    preamble_chip_metadata,
    prove_shard_chain,
)
from sp1_zorch.shard_prover.replay import (
    MAX_LOG_ROW_COUNT,
    fresh_transcript,
    shard_regions,
)
from sp1_zorch.shard_prover.serialize import encode_shard_proof, encode_vk
from sp1_zorch.shard_prover.sp1_ffi import sp1_verify_shard
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.poly.univariate import eval_coeffs
from zorch.round import Round

# SP1 core machine parameters (whir-zorch prove_shard_benchmark): 4x blowup.
_LOG_BLOWUP = 2

_SHARD_DIR = flags.DEFINE_string(
    "shard_dir", None, "rsp shard dump directory (e.g. .../rsp_dump/shardN)."
)
_GKR_POW_BITS = flags.DEFINE_integer(
    "gkr_pow_bits", 12, "GKR grind bits (SP1 hardcodes GKR_GRINDING_BITS = 12)."
)
_OPEN_NUM_QUERIES = flags.DEFINE_integer(
    "open_num_queries", 100, "BaseFold FRI query count (open phase)."
)
_OPEN_POW_BITS = flags.DEFINE_integer(
    "open_pow_bits", 0, "BaseFold FRI query-phase grind bits (open phase)."
)
_FFI_VERIFY = flags.DEFINE_bool(
    "ffi_verify",
    False,
    "Assemble the bincode wire and verify it with SP1's sp1_verify_shard FFI.",
)
_RUNS = flags.DEFINE_integer(
    "runs",
    1,
    "Prove the chain this many times. Run 1 is cold (pays the XLA/zkx "
    "compiles); runs 2+ are warm (the compiled executables are reused), so the "
    "warm per-stage times are the ones to compare against SP1's native prover. "
    "Golden checks run on the final pass.",
)
_MAX_STAGE = flags.DEFINE_integer(
    "max_stage",
    4,
    "Run + byte-check only the first N stages, then stop: 1=trace-commit, "
    "2=+LogUp-GKR, 3=+zerocheck, 4=full chain (default). Cuts the downstream "
    "stages' multi-minute compile for a cheaper iteration loop; golden checks "
    "for stages beyond N are skipped.",
)
_DROP_MAIN_CODEWORD = flags.DEFINE_bool(
    "drop_main_codeword",
    True,
    "Drop the main region's ~6 GB codeword at commit and re-encode it from the "
    "message MLE at the open (SP1's drop_ldes), so it never stays device-"
    "resident through GKR + zerocheck -- clears the rsp full-chain >32 GB OOM "
    "(#55, #124). True matches the production rsp full chain. Set false to keep "
    "the codeword resident (the small-shard / >32 GB-card path); false on rsp "
    "shard17 OOMs.",
)


class _TimedRound(Round):
    """Print each stage's wall-clock so the compile-vs-runtime split is
    visible on every run (async dispatch makes unblocked timings lie, so
    block on the stage's output first). Proof messages that are plain
    dataclasses are opaque to ``block_until_ready``; work that only feeds
    such a message (the jagged open's query gathers) attributes to the
    next timed section instead."""

    def __init__(self, inner: Round) -> None:
        self._inner = inner

    def __call__(self, carry, transcript):
        t0 = time.monotonic()
        out = self._inner(carry, transcript)
        jax.block_until_ready(out)
        print(
            f"[stage {type(self._inner).__name__}] "
            f"{(time.monotonic() - t0) * 1e3:.1f}ms",
            flush=True,
        )
        return out


def main(argv) -> None:
    del argv
    shard_dir = Path(_SHARD_DIR.value)
    shard = load_fixture_shard(shard_dir)
    main_region, prep_region = shard_regions(shard)

    main = shard.main_trace_data
    order = main.traces.chip_order
    num_reals = [main.traces.per_chip[name].num_real for name in order]

    perm = Poseidon2(koalabear16_params())
    smcs = SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )
    # The GKR witness is consumed only by LogUp-GKR; a trace-commit-only run
    # (--max_stage=1) slices that stage off, so don't require the gkr fixture.
    n = max(1, min(4, _MAX_STAGE.value))
    witness = None
    if n >= 2:
        gkr_state = _parse_kv_lines(
            (shard_dir / "gpu_gkr_state.txt").read_text(), skip_unkeyed=True
        )
        witness = jnp.array(int(gkr_state["witness"]), F)
    chain = prove_shard_chain(
        smcs=smcs,
        log_blowup=_LOG_BLOWUP,
        vk=shard.vk,
        chip_metadata=preamble_chip_metadata(order, num_reals, dtype=F),
        gkr_chips=build_gkr_chips(main.chips, order),
        chips=main.chips,
        num_betas=num_beta_values(main.chips),
        num_row_variables=MAX_LOG_ROW_COUNT - 1,
        max_log_row_count=MAX_LOG_ROW_COUNT,
        pow_bits=_GKR_POW_BITS.value,
        open_num_queries=_OPEN_NUM_QUERIES.value,
        open_pow_bits=_OPEN_POW_BITS.value,
        witness=witness,
        # Required at rsp scale for the commit (see sp1_zorch.commit
        # .trace_commit). The GKR stage keeps its `zorch.sumcheck` composite via
        # the rolled marker, independent of this flag.
        jit=True,
        # Drop the main codeword at commit and re-encode it at the open (SP1's
        # drop_ldes), so its ~6 GB never stays device-resident through GKR +
        # zerocheck -- clears the rsp shard17 full-chain >32 GB OOM (see
        # prove_shard_chain's docstring / #55, #124). Byte-identical; defaults on
        # for the rsp path, --drop_main_codeword=false keeps it resident.
        drop_main_codeword=_DROP_MAIN_CODEWORD.value,
    )
    # Slice to the first N stages (--max_stage) so the downstream stages' compile
    # is skipped for a cheaper loop. ProveChain collects one message per round,
    # so msgs[:n] are exactly the stages that ran.
    rounds = chain.rounds[:n]

    # Prove ``--runs`` times: run 1 pays the XLA/zkx compile, runs 2+ reuse it.
    # Each Round is jitted (prove_shard_chain(jit=True)); _TimedRound blocks after
    # each stage to print its wall-clock, so the per-stage split is visible. That
    # wall is host-dispatch-bound, not GPU compute -- for an honest per-stage GPU
    # number use nsys kernel-active time on the warm pass (#124).
    runs = _RUNS.value
    chain.rounds = [_TimedRound(rnd) for rnd in rounds]
    for i in range(runs):
        kind = "cold" if i == 0 else "warm"
        print(
            f"=== prove pass {i + 1}/{runs} ({kind}, stages 1..{n}) ===",
            flush=True,
        )
        t0 = time.monotonic()
        carry, _, msgs = chain(
            ShardCarry(main_region, prep_region, main.public_values),
            fresh_transcript(),
        )
        print(f"chain run: {(time.monotonic() - t0) * 1e3:.1f}ms", flush=True)
    # Golden checks for the stages that ran (msgs has one entry per stage).
    # The trace commit must equal SP1's dumped commitment; gpu_commitment.txt
    # carries canonical integers, so encode to compare.
    commitment = msgs[0]
    commit_kv = _parse_kv_lines((shard_dir / "gpu_commitment.txt").read_text())
    want_commit = jnp.array(_parse_int_list(commit_kv["main_commit"]), F)
    ok = check_match(
        "commitment vs gpu_commitment.main_commit", commitment, want_commit
    )

    gkr = zc = jagged = None
    if n >= 2:
        # gpu_z_row.txt is SP1's `zeta` -- the LogUp-GKR evaluation point's row
        # tail (eval_point[-MAX_LOG_ROW_COUNT:], the GKR logup_evaluations.point),
        # NOT the zerocheck sumcheck point. zeta is a sponge image of every byte
        # observed through the GKR leg, so matching it seals the preamble + GKR
        # leg; the zerocheck rounds are sealed by final_eval below. Mirrors
        # zerocheck/verify_zerocheck.py's `zeta (z_row)` check.
        gkr = msgs[1]
        z_row = _parse_ef_list((shard_dir / "gpu_z_row.txt").read_text())
        ok &= check_match(
            "zeta (gkr eval-point row tail) vs gpu_z_row",
            gkr.eval_point[-MAX_LOG_ROW_COUNT:],
            z_row,
        )
    if n >= 3:
        zc = msgs[2]
        state = _parse_kv_lines(
            (shard_dir / "gpu_zerocheck_state.txt").read_text().split("\nchip ")[0]
        )
        ok &= check_match(
            "final_eval",
            eval_coeffs(zc.msgs.round_poly[-1], zc.msgs.challenge[-1]),
            _parse_ef_list(state["final_eval"])[0],
        )
    if n >= 4:
        # The jagged eval's outer sumcheck claim seals z_col + the column-claim
        # assembly: claim = Sum_c eq(z_col, c) * column_claim[c].
        jagged = msgs[3]
        phase4_claim = _parse_ef_list(
            (shard_dir / "phase4_sumcheck_claim.txt").read_text()
        )[0]
        ok &= check_match(
            "phase4 outer sumcheck claim",
            jagged.eval.outer_sumcheck_claim,
            phase4_claim,
        )

    if not ok:
        sys.exit(1)
    print(f"prove_shard chain (stages 1..{n}) byte-match: ALL OK")

    if n >= 4 and _FFI_VERIFY.value:
        t0 = time.monotonic()
        vk_bytes = encode_vk(shard.vk)
        proof_bytes = encode_shard_proof(
            carry,
            commitment,
            gkr,
            zc,
            jagged,
            max_log_row_count=MAX_LOG_ROW_COUNT,
        )
        print(
            f"bincode: vk {len(vk_bytes)} B, proof {len(proof_bytes)} B "
            f"({time.monotonic() - t0:.1f}s)"
        )
        sp1_verify_shard(
            vk_bytes,
            proof_bytes,
            log_blowup=_LOG_BLOWUP,
            num_queries=_OPEN_NUM_QUERIES.value,
            pow_bits=_OPEN_POW_BITS.value,
            gkr_pow_bits=_GKR_POW_BITS.value,
        )
        print("sp1_verify_shard: ACCEPTED")


if __name__ == "__main__":
    flags.mark_flag_as_required("shard_dir")
    app.run(main)
