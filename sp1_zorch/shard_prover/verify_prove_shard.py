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
processes, set ``FRX_COMPILATION_CACHE_DIR`` to a per-toolchain directory so
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
import json
import sys
import time
from pathlib import Path

import frx
import frx.numpy as fnp
from absl import app, flags
from zk_dtypes import koalabear_mont as F

from zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.logup_gkr.circuit import GkrCapClass, build_gkr_chips
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
from sp1_zorch.zerocheck.jagged import TotalCapClass
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.poly.univariate import eval_coeffs
from zorch.round import Round

# SP1 core machine parameters (whir-zorch prove_shard_benchmark): 4x blowup.
_LOG_BLOWUP = 2

_SHARD_DIR = flags.DEFINE_string(
    "shard_dir",
    None,
    "rsp shard dump directory (e.g. .../rsp_dump/shardN). Comma-separate "
    "several to prove them sequentially in ONE process: jitted stage bodies "
    "whose static keys match are then compiled once and reused — with "
    "--zc_class_json this is the shard-invariance check (the second "
    "same-class shard's zerocheck must skip the cold compile).",
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
_ZC_CLASS_JSON = flags.DEFINE_string(
    "zc_class_json",
    None,
    'JSON {"area_cap"} pinning the shard-invariant zerocheck '
    "TotalCapClass; every shard of one class shares ONE zerocheck compile. "
    "Default: each shard's own a-priori-tight class (per-shard compile). "
    "Assemble a cross-shard class as the per-field max of the printed "
    "ZC_CLASS lines. The jagged-packed round buffer costs area_cap extension-"
    "field elements — a class bounding a much larger shard prices every "
    "shard at that area.",
)
_GKR_CLASS_JSON = flags.DEFINE_string(
    "gkr_class_json",
    None,
    'JSON {"chip_heights": {name: bound}} pinning the shard-invariant '
    "GkrCapClass; shards of one class share every LogUp-GKR "
    "zone compile. Default: each shard's own a-priori-tight class (per-shard "
    "compile). Assemble a cross-shard class as the per-chip max of the "
    "printed GKR_CLASS lines.",
)
_JAXPROF_DIR = flags.DEFINE_string(
    "jaxprof_dir",
    None,
    "Write an frx profiler trace of the last (warm) prove pass here.",
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
        frx.block_until_ready(out)
        label = type(self._inner).__name__
        elapsed_ms = (time.monotonic() - t0) * 1e3
        # Device-pool telemetry per stage boundary: `mem` is resident after the
        # stage, `peak` the pool high-water so far. On a mid-stage OOM the
        # PREVIOUS stage's line is the resident set the failing alloc fought.
        stats = frx.local_devices()[0].memory_stats() or {}
        print(
            f"[stage {label}] {elapsed_ms:.1f}ms"
            f" mem={stats.get('bytes_in_use', 0) / 2**30:.2f}GiB"
            f" peak={stats.get('peak_bytes_in_use', 0) / 2**30:.2f}GiB",
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
    shard_dirs = [Path(p) for p in _SHARD_DIR.value.split(",")]
    # One chips mapping per chip set, reused across shards: the zerocheck jit
    # keys statically on the chips tuple, so two fixture loads must present the
    # SAME objects for the second shard to hit the first's compile. The SMCS is
    # a static key of the commit/eval bodies for the same reason — one instance
    # for the whole run.
    shared_chips: dict[tuple[str, ...], object] = {}
    perm = Poseidon2(koalabear16_params())
    smcs = SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )
    failed: list[str] = []
    for shard_dir in shard_dirs:
        if len(shard_dirs) > 1:
            print(f"===== shard {shard_dir.name} =====", flush=True)
        try:
            _verify_shard(shard_dir, smcs, shared_chips)
        except SystemExit:
            # A stage byte-mismatch fail-fasts the SHARD; keep sweeping the
            # rest — later shards share the compile cache either way.
            failed.append(shard_dir.name)
            print(f"===== {shard_dir.name} FAILED: byte-mismatch =====", flush=True)
        except Exception as e:  # OOM / lowering limits: report, keep sweeping
            failed.append(shard_dir.name)
            print(
                f"===== {shard_dir.name} FAILED: {type(e).__name__}: {e} =====",
                flush=True,
            )
    if failed:
        sys.exit(f"failed shards: {', '.join(failed)}")


def _verify_shard(
    shard_dir: Path, smcs: SingleMatrixCommitmentScheme, shared_chips: dict
) -> None:
    shard = load_fixture_shard(shard_dir)
    main_region, prep_region = shard_regions(shard)

    main = shard.main_trace_data
    order = main.traces.chip_order
    num_reals = [main.traces.per_chip[name].num_real for name in order]

    # Drop the shard once its raw trace arrays are copied into the region
    # dense: the duplicate would otherwise stay resident through the GKR
    # pyramid and overflow the memory budget on wide shards. vk / chips /
    # public_values are metadata and pin no trace data. The chips/gkr_chips
    # pair is shared per chip set across shards (the stage jits key statically
    # on object identity).
    vk = shard.vk
    chips, gkr_chips = shared_chips.setdefault(
        tuple(order), (main.chips, build_gkr_chips(main.chips, order))
    )
    public_values = main.public_values
    chip_metadata = preamble_chip_metadata(order, num_reals, dtype=F)
    num_betas = num_beta_values(chips)
    del shard, main
    gc.collect()

    # zerocheck rides the traced total-Σheights-cap round (sp1-zorch#242):
    # buffer bounds come from a TotalCapClass, the shard's real heights ride as
    # one traced int32 vector, and the compile keys on the class + chip set —
    # shards of one class share the executable. Default class: this shard's
    # own a-priori-tight bounds; --zc_class_json pins a cross-shard class
    # (assemble it as the per-field max of the ZC_CLASS lines printed here).
    print("CHIP_HEIGHTS " + " ".join(f"{n}:{int(r)}" for n, r in zip(order, num_reals)), flush=True)
    prep_widths = (
        {n: int(prep_region.chip_widths[k]) for k, n in enumerate(prep_region.chip_names)}
        if prep_region is not None
        else {}
    )
    chip_cols = [
        int(main_region.chip_widths[i]) + prep_widths.get(name, 0)
        for i, name in enumerate(order)
    ]
    own_class = TotalCapClass.from_heights([int(r) for r in num_reals], chip_cols)
    print(
        "ZC_CLASS "
        + json.dumps(
            {"area_cap": own_class.area_cap}
        ),
        flush=True,
    )
    tc_class = own_class
    if _ZC_CLASS_JSON.value:
        with open(_ZC_CLASS_JSON.value) as f:
            c = {k: int(v) for k, v in json.load(f).items()}
        tc_class = TotalCapClass(area_cap=c["area_cap"])

    # LogUp-GKR rides the same shard-invariant contract on per-chip height
    # bounds; --gkr_class_json pins a cross-shard class (per-chip max of the
    # GKR_CLASS lines printed here).
    # From the region heights (what the stage packs), not num_reals — the
    # two agree on real rows but the pack's bound check runs on the region.
    own_gkr = GkrCapClass.from_heights(
        [int(h) for h in main_region.chip_heights]
    )
    print(
        "GKR_CLASS "
        + json.dumps({"chip_heights": dict(zip(order, own_gkr.chip_heights))}),
        flush=True,
    )
    gkr_class = own_gkr
    if _GKR_CLASS_JSON.value:
        with open(_GKR_CLASS_JSON.value) as f:
            bounds = json.load(f)["chip_heights"]
        gkr_class = GkrCapClass(tuple(int(bounds[name]) for name in order))

    # The jagged class is fully derived — no pin flag. Same (L, n_d) ⇒
    # eval-zone cache hit; same K ⇒ open prologue/query hit; the fold zone is
    # K-independent and always shared (sp1-zorch#274).
    regions_jc = [r for r in (prep_region, main_region) if r is not None]
    jagged_l = sum(sum(int(c) for c in r.column_counts) for r in regions_jc)
    jagged_ks = [
        int(r.dense.shape[0]) >> int(r.log_stacking_height) for r in regions_jc
    ]
    total_area = sum(int(r.dense.shape[0]) for r in regions_jc)
    print(
        "JAGGED_CLASS "
        + json.dumps(
            {
                "L": jagged_l,
                "n_d": (total_area - 1).bit_length() + 1,
                "K": jagged_ks,
                "rlc_bits": max(sum(jagged_ks) - 1, 0).bit_length(),
            }
        ),
        flush=True,
    )

    # The GKR witness is consumed only by LogUp-GKR; a trace-commit-only run
    # (--max_stage=1) slices that stage off, so don't require the gkr fixture.
    n = max(1, min(4, _MAX_STAGE.value))
    witness = None
    if n >= 2:
        gkr_state = _parse_kv_lines(
            (shard_dir / "gpu_gkr_state.txt").read_text(), skip_unkeyed=True
        )
        witness = fnp.array(int(gkr_state["witness"]), F)
    # The zerocheck and GKR jits key statically on the chips / gkr_chips
    # tuples, so a multi-shard run must present the SAME objects to every
    # same-chip-set shard — a fresh fixture load's chips would miss the
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
        zerocheck_total_cap_class=tc_class,
        gkr_cap_class=gkr_class,
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
    want_commit = fnp.array(_parse_int_list(commit_kv["main_commit"]), F)
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
    _prof_dir = _JAXPROF_DIR.value
    for i in range(runs):
        kind = "cold" if i == 0 else "warm"
        print(
            f"=== prove pass {i + 1}/{runs} ({kind}, stages 1..{n}) ===",
            flush=True,
        )
        _prof = _prof_dir and i == runs - 1  # profile the last (warm) pass only
        if _prof:
            frx.profiler.start_trace(_prof_dir)
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
        frx.block_until_ready((bridge, msgs))
        print(f"chain run: {(time.monotonic() - t0) * 1e3:.1f}ms", flush=True)
        if _prof:
            frx.profiler.stop_trace()
            print(f"jaxprof written to {_prof_dir}", flush=True)
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
