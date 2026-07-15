# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""rsp byte-match harness for the assembled prove_shard chain -- a runnable.

Runs ``prove_shard_chain`` (the ``ProveChain`` of trace commit -> LogUp-GKR
-> zerocheck -> jagged evaluation proof) over a real rsp dump and seals the
composition against the reference:

- the commitment the chain's ``TraceCommitStage`` computes must equal the
  dump's ``main_commit`` (``gpu_commitment.txt``);
- the GKR evaluation point's row tail (SP1's ``zeta``) must equal
  ``gpu_z_row.txt``. ``zeta`` is a sponge image of every byte the chain
  observed through the LogUp-GKR leg, so this one match transitively pins the
  preamble (vk, public values, commitment, chip metadata) and the GKR leg,
  proving the Stage wiring reproduces SP1's transcript. The zerocheck sumcheck
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

Each downstream stage's internals are gated by its own runnable
(``logup_gkr:verify_gkr_prove``, ``zerocheck:verify_zerocheck``, plus the eval
stage's ``jagged:prover_test`` / ``jagged:open_test``); the trace-commit stage's
byte-match is this tool at ``--max_stage=1`` and its structure is unit-tested in
``commit:trace_commit_test``. This tool checks the composition, not each stage's
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

Each stage's golden check runs the instant that stage finishes, so a mismatch
exits non-zero *before* the later stages pay their compile -- a stage-1 commit
mismatch aborts in ~one trace-commit, not after the whole chain.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from absl import app, flags
from zk_dtypes import koalabear_mont as F

from zorch.commit.smcs import SingleMatrixCommitmentScheme
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
    ShardBridge,
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
    "Each stage's golden check runs as that stage finishes (every pass), "
    "aborting on a mismatch.",
)
_MAX_STAGE = flags.DEFINE_integer(
    "max_stage",
    4,
    "Run + byte-check only the first N stages, then stop: 1=trace-commit, "
    "2=+LogUp-GKR, 3=+zerocheck, 4=full chain (default). Cuts the downstream "
    "stages' multi-minute compile for a cheaper iteration loop; golden checks "
    "for stages beyond N are skipped.",
)
class _TimedRound(Round):
    """Print each stage's wall-clock so the compile-vs-runtime split is
    visible on every run (async dispatch makes unblocked timings lie, so
    block on the stage's output first). Proof messages that are plain
    dataclasses are opaque to ``block_until_ready``; work that only feeds
    such a message (the jagged open's query gathers) attributes to the
    next timed section instead.

    With ``check`` set, run that stage's golden byte-check the instant the
    stage finishes and ``sys.exit(1)`` on a mismatch -- so a stage-k mismatch
    aborts before stage k+1 pays its (multi-minute) compile, instead of the
    chain running to completion and the checks firing only at the end.
    ``check`` takes the stage's output message and returns ``True`` on match
    (it prints its own OK / MISMATCH line)."""

    def __init__(self, inner: Round, check=None) -> None:
        self._inner = inner
        self._check = check

    def __call__(self, carry, transcript):
        t0 = time.monotonic()
        out = self._inner(carry, transcript)
        jax.block_until_ready(out)
        label = type(self._inner).__name__
        print(
            f"[stage {label}] {(time.monotonic() - t0) * 1e3:.1f}ms",
            flush=True,
        )
        # out is (carry, transcript, msg); check the stage's message and abort
        # now so the later stages' compile is never paid on a mismatch.
        if self._check is not None and not self._check(out[2]):
            print(
                f"[stage {label}] fail-fast: byte-mismatch -- skipping the "
                f"remaining stages' compile",
                flush=True,
            )
            sys.exit(1)
        return out


def main(argv) -> None:
    del argv
    shard_dir = Path(_SHARD_DIR.value)
    shard = load_fixture_shard(shard_dir)
    main_region, prep_region = shard_regions(shard)

    main = shard.main_trace_data
    order = main.traces.chip_order
    num_reals = [main.traces.per_chip[name].num_real for name in order]

    # Drop the shard once its raw trace arrays are copied into the region dense:
    # the duplicate would otherwise stay resident through the GKR pyramid and
    # overflow the memory budget on wide shards. vk/chips/public_values are
    # metadata and pin no trace data.
    vk = shard.vk
    chips = main.chips
    public_values = main.public_values
    gkr_chips = build_gkr_chips(chips, order)
    chip_metadata = preamble_chip_metadata(order, num_reals, dtype=F)
    num_betas = num_beta_values(chips)
    del shard, main
    gc.collect()

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
        vk=vk,
        chip_metadata=chip_metadata,
        gkr_chips=gkr_chips,
        chips=chips,
        num_betas=num_betas,
        num_row_variables=MAX_LOG_ROW_COUNT - 1,
        max_log_row_count=MAX_LOG_ROW_COUNT,
        pow_bits=_GKR_POW_BITS.value,
        open_num_queries=_OPEN_NUM_QUERIES.value,
        open_pow_bits=_OPEN_POW_BITS.value,
        witness=witness,
        jit=True,
    )
    # Slice to the first N stages (--max_stage) so the downstream stages' compile
    # is skipped for a cheaper loop. ProveChain collects one message per round,
    # so msgs[:n] are exactly the stages that ran.
    rounds = chain.rounds[:n]

    # Parse the golden references up front: a missing/malformed fixture then
    # fails at startup rather than after stage 1's ~2-3 min cold compile, and
    # each file is read once instead of per warm pass. Only stages 1..n are
    # parsed -- a --max_stage prefix never needs a later stage's fixture.
    # The trace commit must equal SP1's dumped commitment; gpu_commitment.txt
    # carries canonical integers, so encode to compare.
    commit_kv = _parse_kv_lines((shard_dir / "gpu_commitment.txt").read_text())
    want_commit = jnp.array(_parse_int_list(commit_kv["main_commit"]), F)
    # gpu_z_row.txt is SP1's `zeta` -- the LogUp-GKR eval point's row tail
    # (eval_point[-MAX_LOG_ROW_COUNT:]), NOT the zerocheck point. zeta is a sponge
    # image of every byte observed through the GKR leg, so matching it seals the
    # preamble + GKR leg; final_eval seals the zerocheck rounds.
    want_z_row = (
        _parse_ef_list((shard_dir / "gpu_z_row.txt").read_text()) if n >= 2 else None
    )
    if n >= 3:
        zc_state = _parse_kv_lines(
            (shard_dir / "gpu_zerocheck_state.txt").read_text().split("\nchip ")[0]
        )
        want_final_eval = _parse_ef_list(zc_state["final_eval"])[0]
    else:
        want_final_eval = None
    # The jagged eval's outer sumcheck claim seals z_col + the column-claim
    # assembly: claim = Sum_c eq(z_col, c) * column_claim[c].
    want_phase4 = (
        _parse_ef_list((shard_dir / "phase4_sumcheck_claim.txt").read_text())[0]
        if n >= 4
        else None
    )

    # Per-stage golden byte-checks, wired into the timed round wrapper (below) to
    # fire the instant their stage finishes and abort on a mismatch -- so a
    # stage-k mismatch never pays stage k+1's (multi-minute) compile, instead of
    # every check firing after the whole chain runs.
    def _check_commit(msg):
        return check_match("commitment vs gpu_commitment.main_commit", msg, want_commit)

    def _check_gkr(msg):
        return check_match(
            "zeta (gkr eval-point row tail) vs gpu_z_row",
            msg.eval_point[-MAX_LOG_ROW_COUNT:],
            want_z_row,
        )

    def _check_zerocheck(msg):
        return check_match(
            "final_eval",
            eval_coeffs(msg.msgs.round_poly[-1], msg.msgs.challenge[-1]),
            want_final_eval,
        )

    def _check_jagged(msg):
        return check_match(
            "phase4 outer sumcheck claim",
            msg.eval.outer_sumcheck_claim,
            want_phase4,
        )

    stage_checks = [_check_commit, _check_gkr, _check_zerocheck, _check_jagged][:n]

    # Prove ``--runs`` times: run 1 pays the XLA/zkx compile, runs 2+ reuse it.
    # Each Round is jitted (prove_shard_chain(jit=True)); _TimedRound blocks after
    # each stage to print its wall-clock + run its golden check, so the per-stage
    # split is visible and a mismatch aborts before the next stage compiles. That
    # wall is host-dispatch-bound, not GPU compute -- for an honest per-stage GPU
    # number use nsys kernel-active time on the warm pass (#124).
    runs = _RUNS.value
    chain.rounds = [_TimedRound(rnd, check) for rnd, check in zip(rounds, stage_checks)]
    for i in range(runs):
        kind = "cold" if i == 0 else "warm"
        print(
            f"=== prove pass {i + 1}/{runs} ({kind}, stages 1..{n}) ===",
            flush=True,
        )
        t0 = time.monotonic()
        # Release the prior pass's device buffers before this pass allocates. The
        # bridge pins the shard's trace regions plus the GKR openings; holding a
        # spent pass resident while the next re-allocates the pyramid intermediate
        # is what tips a wide shard over the card on --runs>=2.
        bridge = msgs = None
        bridge, _, msgs = chain(
            ShardBridge(main_region, prep_region, public_values),
            fresh_transcript(),
        )
        print(f"chain run: {(time.monotonic() - t0) * 1e3:.1f}ms", flush=True)
    # Each stage's golden check already ran inside the round wrapper and exits
    # on a mismatch, so reaching here means stages 1..n all byte-matched.
    print(f"prove_shard chain (stages 1..{n}) byte-match: ALL OK")

    if n >= 4 and _FFI_VERIFY.value:
        # n is capped at 4, so n >= 4 means the full chain ran: msgs is exactly
        # the four stage messages the bincode wire needs, in order.
        commitment, gkr, zc, jagged = msgs
        t0 = time.monotonic()
        vk_bytes = encode_vk(vk)
        proof_bytes = encode_shard_proof(
            bridge,
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
