# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Staged shard-prove runner over rsp GPU-trace dumps — THE local GPU harness.

Drives the shard proof STAGE-BY-STAGE (trace commit -> LogUp-GKR -> zerocheck
-> jagged evaluation proof) with a memory-release point between stages: each
stage's spent result object is dropped (only the proof section and the
transcript threading survive) and gc runs before the next stage allocates, so
the card holds one stage's working set at a time. That release discipline is
what keeps a full rsp shard provable on a 32 GB card; the staging order is the
production staged driver's (same call order, same transcript threading, so the
Fiat-Shamir stream is the composite ``ShardProver.prove``'s byte for byte).

Per-stage golden byte-checks run the instant each stage finishes against the
dump's references (``gpu_commitment.txt``, ``gpu_z_row.txt``,
``gpu_zerocheck_state.txt``, ``phase4_sumcheck_claim.txt``) and fail-fast — a
phase-k mismatch aborts before phase k+1 pays its (multi-minute) compile. A
full run additionally bincode-encodes the proof and prints ``PROOF_SHA256``,
the cross-run byte-golden line; ``--ffi_verify`` runs SP1's own verifier over
the wire bytes (``SP1_JAX_FFI_LIB`` must point at ``libsp1_gpu_jax_ffi.so``).

    bazel run //tools:staged_prove_shard -- \\
        --shard_dir=/path/to/rsp_dump/shardN

Wall-clock is dominated by XLA/zkx GPU compiles, not kernel runtime — the
per-phase timings printed during the run show the split. Pass ``--runs=N`` to
prove the chain N times in one process: run 1 is cold (compiles), runs 2+ are
warm (executables reused), so the warm per-phase ``[phase X] Yms`` lines are
the ones to compare against SP1's native prover. Across separate processes,
set ``FRX_COMPILATION_CACHE_DIR`` to a per-toolchain directory so every run
after the first skips the compiles; leave it unset for byte-match gates (a
cache shared across toolchains has served wrong executables).

``--max_phase=N`` runs + byte-checks only the first N phases (1=trace-commit
.. 4=full), a cheaper loop that skips the downstream compiles. The numbering
is SP1's own tracing spans -- see ``docs/architecture.md`` for how they map
onto this repo's Stages.

Class census logging: every shard prints its ``CHIP_HEIGHTS`` / ``ZC_CLASS``
/ ``GKR_CLASS`` / ``JAGGED_CLASS`` lines (the compile-class keys); assemble
cross-shard pin files as the per-field max of those lines, or let
``warm_shard_cache`` emit the group manifest. Class resolution (manifest >
pin flags > the shard's own tight class) is
``sp1_zorch.shard_prover.compile_classes.resolve_classes`` — the same
definition the warm filler keys its cache cover on.

Real-block data (~1.5 GB/shard) plus the GPU trace commit keep this a
runnable, not a unit test. Needs a CUDA GPU.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import frx
import frx.numpy as fnp
from absl import app, flags
from frx import Array
from zk_dtypes import koalabear_mont as F
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.poly.univariate import eval_coeffs
from zorch.transcript import DuplexTranscript

from sp1_zorch.logup_gkr.circuit import build_gkr_chips
from sp1_zorch.logup_gkr.prover import num_beta_values
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from sp1_zorch.shard_prover.compile_classes import (
    jagged_class,
    resolve_classes,
    tight_classes,
)
from sp1_zorch.shard_prover.fixture_loader import (
    _parse_ef_list,
    _parse_int_list,
    _parse_kv_lines,
    check_match,
    load_fixture_shard,
)
from sp1_zorch.shard_prover.prove_shard import (
    ShardProver,
    bind_commitment,
)
from sp1_zorch.shard_prover.replay import (
    MAX_LOG_ROW_COUNT,
    fresh_transcript,
    shard_regions,
)
from sp1_zorch.shard_prover.serialize import encode_shard_proof, encode_vk
from sp1_zorch.shard_prover.sp1_ffi import sp1_verify_shard
from sp1_zorch.types import (
    ChipMetadata,
    JaggedCommitData,
    JaggedOpeningClaim,
    JaggedOpeningWitness,
    ShardClaim,
    ShardProof,
    ShardWitness,
    ZerocheckClaim,
)

# A phase's golden check: takes that phase's proof section, prints OK/MISMATCH.
PhaseCheck = Callable[[Any], bool]

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
_PROOF_SHA256 = flags.DEFINE_bool(
    "proof_sha256",
    True,
    "On a full run (--max_phase=4), bincode-encode the proof and print its "
    "sha256 — the cross-run/cross-stack byte-golden line. Encoding is host "
    "work over the final proof; disable for compile-only drivers.",
)
_RUNS = flags.DEFINE_integer(
    "runs",
    1,
    "Prove the chain this many times. Run 1 is cold (pays the XLA/zkx "
    "compiles); runs 2+ are warm (the compiled executables are reused), so the "
    "warm per-phase times are the ones to compare against SP1's native prover. "
    "Each phase's golden check runs as that phase finishes (every pass), "
    "aborting on a mismatch.",
)
_MAX_PHASE = flags.DEFINE_integer(
    "max_phase",
    4,
    "Run + byte-check only the first N SP1 phases, then stop: 1=trace-commit, "
    "2=+LogUp-GKR, 3=+zerocheck, 4=full chain (default). Cuts the downstream "
    "multi-minute compile for a cheaper iteration loop; golden checks for "
    "phases beyond N are skipped.",
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
_GROUP_MANIFEST_JSON = flags.DEFINE_string(
    "group_manifest_json",
    None,
    'JSON {shard_name: {"area_cap": N, "gkr": {chip: bound}}} pinning a '
    "per-shard zerocheck + GKR class, so a single multi-shard process can "
    "prove several chip-set groups at once and still share one compile within "
    "each group (the group-max class per chip set). Overrides "
    "--zc_class_json / --gkr_class_json field-by-field for any shard it "
    "names; shards absent from the manifest fall back to those flags or "
    "their own tight class.",
)
_JAXPROF_DIR = flags.DEFINE_string(
    "jaxprof_dir",
    None,
    "Write an frx profiler trace of the last (warm) prove pass here.",
)


def _device_arrays(value: Any, _seen: dict[int, Any] | None = None) -> list[Array]:
    """Every `frx.Array` reachable from `value`, for blocking on a phase.

    A phase returns plain frozen dataclasses — `ProveResult`, and proof
    sections like `LogupGkrProof` — none of which are registered pytrees, so
    `block_until_ready` cannot see inside them and returns before the device
    work lands. Timing what it returns measures host dispatch, not the phase.
    Walking the fields instead makes the wait real regardless of registration.
    """
    # Keyed by id, so keep each visited object alive: a freed intermediate can
    # have its id reused by a later one, which would silently skip it.
    _seen = _seen if _seen is not None else {}
    if id(value) in _seen:
        return []
    _seen[id(value)] = value
    if isinstance(value, frx.Array):
        return [value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = [getattr(value, f.name) for f in dataclasses.fields(value)]
    if isinstance(value, Mapping):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_device_arrays(v, _seen))
        return out
    return []


def _timed(
    label: str, checks: Sequence[PhaseCheck], index: int, run: Callable[[], Any]
) -> Any:
    """Run one phase, print its wall-clock, and byte-check it immediately.

    Async dispatch makes unblocked timings lie, so block on the phase's output
    first. Proof sections that are plain dataclasses are opaque to
    ``block_until_ready``; work that only feeds such a section (the jagged
    open's query gathers) attributes to the next timed phase instead.

    A phase's golden check runs the instant that phase finishes and exits on a
    mismatch, so a phase-k mismatch aborts before phase k+1 pays its
    (multi-minute) compile rather than every check firing at the end.
    """
    t0 = time.monotonic()
    out = run()
    frx.block_until_ready(_device_arrays(out))
    elapsed_ms = (time.monotonic() - t0) * 1e3
    # Device-pool telemetry per phase boundary: `mem` is resident after the
    # phase, `peak` the pool high-water so far. On a mid-phase OOM the
    # PREVIOUS phase's line is the resident set the failing alloc fought.
    stats = frx.local_devices()[0].memory_stats() or {}
    print(
        f"[phase {label}] {elapsed_ms:.1f}ms"
        f" mem={stats.get('bytes_in_use', 0) / 2**30:.2f}GiB"
        f" peak={stats.get('peak_bytes_in_use', 0) / 2**30:.2f}GiB",
        flush=True,
    )
    check = checks[index] if index < len(checks) else None
    if check is not None:
        # The PCS commit half returns (commitment, prover data); every Stage
        # role returns a ProveResult.
        section = out[0] if isinstance(out, tuple) else out.reduction_proof
        if not check(section):
            print(
                f"[phase {label}] fail-fast: byte-mismatch -- skipping the "
                f"remaining phases' compile",
                flush=True,
            )
            sys.exit(1)
    return out


def _release_stage() -> None:
    """The between-stage release point: collect the spent stage's dropped
    result object (and everything it pinned) before the next stage allocates.
    One stage's working set at a time is what fits a full rsp shard on a
    32 GB card."""
    gc.collect()


def _prove_staged(
    prover: ShardProver,
    claim: ShardClaim,
    witness: ShardWitness,
    transcript: DuplexTranscript,
    n: int,
    checks: Sequence[PhaseCheck],
) -> tuple[ShardProof | None, list[Any], Any, JaggedCommitData | None]:
    """Run the first ``n`` SP1 phases stage-by-stage, releasing between them.

    Same call order and transcript threading as the composite
    ``ShardProver.prove``, so the Fiat-Shamir stream is the composite's byte
    for byte. Between stages only the proof section and the threading values
    (transcript, reduced claim, commit data) survive — the spent stage result
    is dropped and :func:`_release_stage` runs, so device buffers pinned by a
    finished stage are freed before the next stage's peak.
    """
    evaluation = None
    gkr_claim = gkr_proof = zc_proof = open_proof = None

    commitment, commit_data = _timed(
        "TraceCommit",
        checks,
        0,
        lambda: prover.opening.commit(witness),
    )
    transcript, roots = bind_commitment(transcript, claim, commitment)

    if n >= 2:
        _release_stage()
        gkr = _timed(
            "LogupGkrProver",
            checks,
            1,
            lambda: prover.gkr.prove(claim, witness, transcript),
        )
        transcript = gkr.transcript
        gkr_claim = gkr.reduced_claim
        gkr_proof = gkr.reduction_proof
        del gkr
    if n >= 3:
        # n >= 3 ran the GKR branch above, so its reduced claim exists. Bound
        # to a local because the lambda defers the read past any narrowing.
        assert gkr_claim is not None
        source_claim = gkr_claim
        _release_stage()
        zerocheck = _timed(
            "ZerocheckProver",
            checks,
            2,
            lambda: prover.zerocheck.prove(
                ZerocheckClaim(claim.public_values, source_claim, claim.chip_metadata),
                witness,
                transcript,
            ),
        )
        transcript = zerocheck.transcript
        evaluation = zerocheck.reduced_claim
        zc_proof = zerocheck.reduction_proof
        del zerocheck
    if n >= 4:
        assert evaluation is not None
        evaluation_claim = evaluation
        _release_stage()
        opening = _timed(
            "JaggedPcsProver",
            checks,
            3,
            lambda: prover.opening.prove(
                JaggedOpeningClaim(evaluation_claim, roots, claim.chip_metadata),
                JaggedOpeningWitness(witness, commit_data),
                transcript,
            ),
        )
        open_proof = opening.reduction_proof
        del opening
    # Only a full run has a shard proof. A --max_phase prefix ran some phases
    # and has their sections; it does not have a ShardProof, so it does not
    # claim one -- the wire assembly below is gated on the full run anyway.
    sections = [
        s for s in (commitment, gkr_proof, zc_proof, open_proof) if s is not None
    ]
    proof = ShardProof(*sections) if len(sections) == 4 else None
    return proof, sections, evaluation, commit_data


def main(argv: Sequence[str]) -> None:
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
            _prove_shard_dir(shard_dir, smcs, shared_chips)
        except SystemExit:
            # A phase byte-mismatch fail-fasts the SHARD; keep sweeping the
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


def _prove_shard_dir(
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
    shard_chip_metadata = ChipMetadata(tuple(order), tuple(num_reals))
    num_betas = num_beta_values(chips)
    del shard, main
    gc.collect()

    # zerocheck rides the traced total-Σheights-cap round (sp1-zorch#242):
    # buffer bounds come from a TotalCapClass, the shard's real heights ride as
    # one traced int32 vector, and the compile keys on the class + chip set —
    # shards of one class share the executable. LogUp-GKR rides the same
    # shard-invariant contract on per-chip height bounds (GkrCapClass).
    print(
        "CHIP_HEIGHTS " + " ".join(f"{n}:{int(r)}" for n, r in zip(order, num_reals)),
        flush=True,
    )
    own_tc, own_gkr, own_slot = tight_classes(
        main_region, prep_region, order, num_reals, gkr_chips
    )
    print("ZC_CLASS " + json.dumps({"area_cap": own_tc.area_cap}), flush=True)
    print(
        "GKR_CLASS "
        + json.dumps(
            {
                "chip_heights": dict(zip(order, own_gkr.chip_heights)),
                "slot_cap": own_slot,
            }
        ),
        flush=True,
    )
    print("JAGGED_CLASS " + json.dumps(jagged_class(main_region, prep_region)))

    # Pin resolution: the shard's group-manifest entry > the global class
    # flags > the shard's own tight class (compile_classes.resolve_classes,
    # the same definition warm_shard_cache keys its cache cover on).
    zc_spec = gkr_spec = manifest_entry = None
    if _ZC_CLASS_JSON.value:
        with open(_ZC_CLASS_JSON.value) as f:
            zc_spec = {k: int(v) for k, v in json.load(f).items()}
    if _GKR_CLASS_JSON.value:
        with open(_GKR_CLASS_JSON.value) as f:
            gkr_spec = json.load(f)
    if _GROUP_MANIFEST_JSON.value:
        with open(_GROUP_MANIFEST_JSON.value) as f:
            manifest_entry = json.load(f).get(shard_dir.name)
    tc_class, gkr_class = resolve_classes(
        order,
        own_tc,
        own_gkr,
        manifest_entry=manifest_entry,
        zc_spec=zc_spec,
        gkr_spec=gkr_spec,
    )

    # The GKR witness is consumed only by LogUp-GKR; a trace-commit-only run
    # (--max_phase=1) slices that off, so don't require the gkr fixture.
    n = max(1, min(4, _MAX_PHASE.value))
    witness = None
    if n >= 2:
        gkr_state = _parse_kv_lines(
            (shard_dir / "gpu_gkr_state.txt").read_text(), skip_unkeyed=True
        )
        witness = fnp.array(int(gkr_state["witness"]), F)
    shard_claim = ShardClaim(vk, public_values, shard_chip_metadata)
    shard_witness = ShardWitness(main_region, prep_region)
    prover = ShardProver(
        smcs=smcs,
        log_blowup=_LOG_BLOWUP,
        gkr_chips=gkr_chips,
        chips=chips,
        num_betas=num_betas,
        num_row_variables=MAX_LOG_ROW_COUNT - 1,
        max_log_row_count=MAX_LOG_ROW_COUNT,
        pow_bits=_GKR_POW_BITS.value,
        open_num_queries=_OPEN_NUM_QUERIES.value,
        open_pow_bits=_OPEN_POW_BITS.value,
        pow_witness=witness,
        jit=True,
        zerocheck_total_cap_class=tc_class,
        gkr_cap_class=gkr_class,
    )

    # Parse the golden references up front: a missing/malformed fixture then
    # fails at startup rather than after phase 1's ~2-3 min cold compile, and
    # each file is read once instead of per warm pass. Only phases 1..n are
    # parsed -- a --max_phase prefix never needs a later phase's fixture.
    # The trace commit must equal SP1's dumped commitment; gpu_commitment.txt
    # carries canonical integers, so encode to compare.
    commit_kv = _parse_kv_lines((shard_dir / "gpu_commitment.txt").read_text())
    want_commit = fnp.array(_parse_int_list(commit_kv["main_commit"]), F)
    # gpu_z_row.txt is SP1's `zeta` -- the LogUp-GKR eval point's row tail
    # (eval_point[-MAX_LOG_ROW_COUNT:]), NOT the zerocheck point. zeta is a
    # sponge image of every byte observed through the GKR leg, so matching it
    # seals the preamble + GKR leg; final_eval seals the zerocheck rounds.
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

    # Per-phase golden byte-checks, wired into the timed round wrapper to fire
    # the instant their phase finishes and abort on a mismatch -- so a phase-k
    # mismatch never pays phase k+1's (multi-minute) compile, instead of every
    # check firing after the whole chain runs.
    def _check_commit(msg: Any) -> bool:
        return check_match("commitment vs gpu_commitment.main_commit", msg, want_commit)

    def _check_gkr(msg: Any) -> bool:
        return check_match(
            "zeta (gkr eval-point row tail) vs gpu_z_row",
            msg.eval_point[-MAX_LOG_ROW_COUNT:],
            want_z_row,
        )

    def _check_zerocheck(msg: Any) -> bool:
        return check_match(
            "final_eval",
            eval_coeffs(msg.msgs.round_poly[-1], msg.msgs.challenge[-1]),
            want_final_eval,
        )

    def _check_jagged(msg: Any) -> bool:
        return check_match(
            "phase4 outer sumcheck claim",
            msg.eval.outer_sumcheck_claim,
            want_phase4,
        )

    phase_checks = [_check_commit, _check_gkr, _check_zerocheck, _check_jagged][:n]

    # Prove ``--runs`` times: run 1 pays the XLA/zkx compile, runs 2+ reuse it.
    # Each phase is jitted (ShardProver(jit=True)); `_timed` blocks after each
    # phase to print its wall-clock + run its golden check, so the per-phase
    # split is visible and a mismatch aborts before the next phase compiles.
    # That wall is host-dispatch-bound, not GPU compute -- for an honest
    # per-phase GPU number use nsys kernel-active time on the warm pass (#124).
    runs = _RUNS.value
    _prof_dir = _JAXPROF_DIR.value
    for i in range(runs):
        kind = "cold" if i == 0 else "warm"
        print(
            f"=== prove pass {i + 1}/{runs} ({kind}, phases 1..{n}) ===",
            flush=True,
        )
        _prof = _prof_dir and i == runs - 1  # profile the last (warm) pass only
        if _prof:
            frx.profiler.start_trace(_prof_dir)
        t0 = time.monotonic()
        # Release the prior pass's device buffers before this pass allocates:
        # holding a spent pass resident while the next re-allocates the pyramid
        # intermediate is what tips a wide shard over the card on --runs>=2.
        proof = sections = commit_data = evaluation = None
        proof, sections, evaluation, commit_data = _prove_staged(
            prover, shard_claim, shard_witness, fresh_transcript(), n, phase_checks
        )
        frx.block_until_ready(sections)
        print(f"chain run: {(time.monotonic() - t0) * 1e3:.1f}ms", flush=True)
        if _prof:
            frx.profiler.stop_trace()
            print(f"jaxprof written to {_prof_dir}", flush=True)
    # Each phase's golden check already ran inside the round wrapper and exits
    # on a mismatch, so reaching here means phases 1..n all byte-matched.
    print(f"prove_shard chain (phases 1..{n}) byte-match: ALL OK")

    if n >= 4 and (_PROOF_SHA256.value or _FFI_VERIFY.value):
        # n is capped at 4, so n >= 4 means every phase ran and `proof` is a
        # real ShardProof -- a shorter prefix leaves it None and never gets
        # here.
        assert proof is not None
        assert evaluation is not None
        assert commit_data is not None
        t0 = time.monotonic()
        vk_bytes = encode_vk(vk)
        # The SMCS commitments are not passed here: they reached the opening as
        # part of its witness, and the jagged proof inside `proof` carries them.
        proof_bytes = encode_shard_proof(
            shard_claim,
            shard_witness,
            proof,
            evaluation,
            commit_data.digest_layers,
            max_log_row_count=MAX_LOG_ROW_COUNT,
        )
        print(
            f"bincode: vk {len(vk_bytes)} B, proof {len(proof_bytes)} B "
            f"({time.monotonic() - t0:.1f}s)"
        )
        print(
            f"PROOF_SHA256 {hashlib.sha256(proof_bytes).hexdigest()}",
            flush=True,
        )
        if _FFI_VERIFY.value:
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
