# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""rsp byte-match harness for the zerocheck stage -- a runnable.

Replays the pipeline up to zerocheck (preamble transcript -> layered GKR),
runs ``prove_shard_zerocheck`` on the real transcript, and compares every
stage value against the reference dump (stage / dump-file vocabulary:
``docs/shard-pipeline.md``):

- ``gpu_zerocheck_state.txt`` -- batching + GKR opening-batch challenges,
  the joint claimed sum, round count, final eval;
- ``phase3_lambda.txt`` -- the chip-RLC lambda;
- ``gpu_z_row.txt`` -- the full sumcheck point. Each challenge is a sponge
  image of every byte observed before it, so 22 matches transitively pin the
  whole round-poly stream the prover emitted;
- ``phase3_chip_opened_values_full.txt`` -- per-chip main/prep opened values;
- ``gpu_univariate.txt`` -- per-round poly values (rounds 1..21; SP1 does not
  log round 0 here). The dump's 4-value-per-round encoding is not pinned by
  any in-repo consumer yet, so this check reports which candidate
  representation matched rather than gating.

The GKR replay dominates the wall time (hours, eager) while zerocheck
iterations are the part under development -- ``--gkr_cache`` persists the
GKR outputs (eval point, chip openings, post-GKR sponge state) so reruns
skip straight to the stage under test.

    bazel run //sp1_zorch/zerocheck:verify_zerocheck -- \\
        --shard_dir=/path/to/rsp_dump/shardN \\
        --gkr_cache=/path/to/cache.npz

Exits non-zero on any gating mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from absl import app, flags
from jax import Array
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.logup_gkr.prover import ChipEvaluation
from sp1_zorch.shard_prover.fixture_loader import (
    _parse_ef_list,
    _parse_kv_lines,
    check_match,
    load_fixture_shard,
)
from sp1_zorch.shard_prover.replay import (
    MAX_LOG_ROW_COUNT,
    clone_diag,
    fresh_transcript,
    from_u32,
    replay_gkr,
    shard_regions,
    to_u32,
)
from sp1_zorch.zerocheck.stage import ZerocheckProof, prove_shard_zerocheck
from zorch.poly.univariate import eval_coeffs
from zorch.transcript import DuplexState, DuplexTranscript, Transcript

_SHARD_DIR = flags.DEFINE_string(
    "shard_dir", None, "rsp shard dump directory (e.g. .../rsp_dump/shard1)."
)
_GKR_CACHE = flags.DEFINE_string(
    "gkr_cache",
    None,
    "npz path for the GKR replay outputs; loaded when present, written after "
    "a fresh GKR run otherwise.",
)
_GKR_POW_BITS = flags.DEFINE_integer(
    "gkr_pow_bits",
    12,
    "GKR grind bits (SP1 hardcodes GKR_GRINDING_BITS = 12).",
)


def _save_gkr_cache(
    path: Path,
    eval_point: Array,
    openings: dict[str, ChipEvaluation],
    transcript: DuplexTranscript,
) -> None:
    st = transcript.state
    data: dict[str, np.ndarray] = {
        "eval_point": to_u32(eval_point),
        "chips": np.array(sorted(openings)),
        "t_input": to_u32(st.input_buffer),
        "t_output": to_u32(st.output_buffer),
        "t_sponge": to_u32(st.sponge_state),
        "t_in_pos": np.int32(int(st.in_pos)),
        "t_out_pos": np.int32(int(st.out_pos)),
    }
    for name, ev in openings.items():
        data[f"main:{name}"] = to_u32(ev.main)
        if ev.preprocessed is not None:
            data[f"prep:{name}"] = to_u32(ev.preprocessed)
    np.savez(path, **data)


def _load_gkr_cache(
    path: Path,
) -> tuple[Array, dict[str, ChipEvaluation], Transcript]:
    z = np.load(path)
    eval_point = from_u32(z["eval_point"], EF)
    openings = {
        str(name): ChipEvaluation(
            main=from_u32(z[f"main:{name}"], EF),
            preprocessed=(
                from_u32(z[f"prep:{name}"], EF) if f"prep:{name}" in z else None
            ),
        )
        for name in z["chips"]
    }
    base = fresh_transcript()
    state = DuplexState(
        input_buffer=from_u32(z["t_input"], F),
        output_buffer=from_u32(z["t_output"], F),
        sponge_state=from_u32(z["t_sponge"], F),
        in_pos=jnp.int32(int(z["t_in_pos"])),
        out_pos=jnp.int32(int(z["t_out_pos"])),
    )
    return eval_point, openings, DuplexTranscript(base.permutation, base.rate, state)


def _replay_gkr(shard, shard_dir: Path, main_region, prep_region):
    """The full GKR leg of the pipeline, sealed against the dump's post-GKR
    diag before its outputs are trusted as zerocheck inputs."""
    transcript, proof = replay_gkr(
        shard, shard_dir, main_region, prep_region, pow_bits=_GKR_POW_BITS.value
    )
    post = _parse_kv_lines((shard_dir / "gpu_post_gkr_diag.txt").read_text())
    if not check_match(
        "post_gkr_diag (GKR seal)", clone_diag(transcript), int(post["post_gkr_diag"])
    ):
        print("GKR replay diverged from the dump; zerocheck inputs are invalid.")
        sys.exit(1)
    return proof.eval_point, proof.chip_openings, transcript


def _parse_phase3(path: Path) -> dict[str, dict[str, Array]]:
    """``phase3_chip_opened_values_full.txt`` -> {chip: {prep, main}} (EF)."""
    chips: dict[str, dict[str, Array]] = {}
    name = None
    parts: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("chip ") and stripped.endswith(":"):
            if name is not None:
                chips[name] = {k: _parse_ef_list(" ".join(v)) for k, v in parts.items()}
            name = stripped[len("chip ") : -1]
            parts = {"prep": [], "main": []}
        elif stripped.startswith("prep["):
            parts["prep"].append(stripped.split("=", 1)[1])
        elif stripped.startswith("main["):
            parts["main"].append(stripped.split("=", 1)[1])
    if name is not None:
        chips[name] = {k: _parse_ef_list(" ".join(v)) for k, v in parts.items()}
    return chips


def _check_opened_values(zc: ZerocheckProof, main_region, shard_dir: Path) -> bool:
    """Each chip's final per-column openings against the dump, main and prep
    separately (the driver folds ``[main | prep]`` traces; the dump labels
    the two ranges)."""
    ref = _parse_phase3(shard_dir / "phase3_chip_opened_values_full.txt")
    ok = check_match("phase3 chip set", sorted(ref), sorted(main_region.chip_names))
    for i, name in enumerate(main_region.chip_names):
        final = zc.finals[i]
        nc = final.shape[0]
        vals = final[:, 0] if final.shape[1] > 0 else jnp.zeros((nc,), dtype=EF)
        mw = int(main_region.chip_widths[i])
        ok_i = check_match(f"openings:{name} main", vals[:mw], ref[name]["main"])
        ok_i &= check_match(f"openings:{name} prep", vals[mw:], ref[name]["prep"])
        ok &= ok_i
    return ok


def _report_univariate_encoding(zc: ZerocheckProof, shard_dir: Path) -> None:
    """Match the dump's 4-value round lines against candidate encodings of our
    coefficient-form polys. Report-only: the challenge chain already pins the
    round-poly bytes, and the dump's encoding has no in-repo consumer to pin
    it against -- once a candidate matches, promote this to a gating check."""
    lines = (shard_dir / "gpu_univariate.txt").read_text().splitlines()
    by_alpha = {}
    for k, line in enumerate(lines):
        efs = _parse_ef_list(line)
        if efs.shape[0] == 5:
            by_alpha[to_u32(efs[4]).tobytes()] = (k, efs[:4])

    rounds = []  # (round index, dumped 4 values)
    for r in range(1, int(zc.msgs.challenge.shape[0])):
        hit = by_alpha.get(to_u32(zc.msgs.challenge[r]).tobytes())
        if hit is None:
            print(f"univariate: round {r} challenge not found in dump lines")
            return
        rounds.append((r, hit[1]))

    one = jnp.ones((), EF)
    candidates = {
        "coeffs[0:4]": lambda p: p[:4],
        "coeffs[1:5]": lambda p: p[1:],
        "evals@{0,1,2,3}": lambda p: jnp.stack(
            [eval_coeffs(p, t * one) for t in (0, 1, 2, 3)]
        ),
        "evals@{0,1,2,4}": lambda p: jnp.stack(
            [eval_coeffs(p, t * one) for t in (0, 1, 2, 4)]
        ),
    }
    for label, fn in candidates.items():
        if all(
            np.array_equal(to_u32(fn(zc.msgs.round_poly[r])), to_u32(want))
            for r, want in rounds
        ):
            print(f"univariate round-poly encoding matched: {label} (rounds 1..21)")
            return
    print("univariate: no candidate encoding matched (informational)")


def main(argv) -> None:
    del argv
    shard_dir = Path(_SHARD_DIR.value)
    shard = load_fixture_shard(shard_dir)
    main_region, prep_region = shard_regions(shard)

    cache = Path(_GKR_CACHE.value) if _GKR_CACHE.value else None
    if cache is not None and cache.suffix != ".npz":
        # np.savez appends .npz to a bare path; normalize so the exists()
        # probe and the write target stay the same file.
        cache = cache.with_name(cache.name + ".npz")
    if cache is not None and cache.exists():
        print(f"loading GKR outputs from {cache}")
        eval_point, openings, transcript = _load_gkr_cache(cache)
    else:
        eval_point, openings, transcript = _replay_gkr(
            shard, shard_dir, main_region, prep_region
        )
        if cache is not None:
            _save_gkr_cache(cache, eval_point, openings, transcript)
            print(f"saved GKR outputs to {cache}")

    transcript, zc = prove_shard_zerocheck(
        shard.main_trace_data.chips,
        main_region,
        prep_region,
        shard.main_trace_data.public_values,
        eval_point,
        openings,
        transcript,
        max_log_row_count=MAX_LOG_ROW_COUNT,
    )

    state = _parse_kv_lines(
        (shard_dir / "gpu_zerocheck_state.txt").read_text().split("\nchip ")[0]
    )
    lam = _parse_kv_lines((shard_dir / "phase3_lambda.txt").read_text())
    z_row = _parse_ef_list((shard_dir / "gpu_z_row.txt").read_text())

    ok = check_match(
        "batching_challenge",
        zc.batching_challenge,
        _parse_ef_list(state["batching_challenge"])[0],
    )
    ok &= check_match(
        "gkr_opening_batch_challenge",
        zc.gkr_opening_batch_challenge,
        _parse_ef_list(state["gkr_opening_batch_challenge"])[0],
    )
    ok &= check_match("lambda", zc.lambda_, _parse_ef_list(lam["lambda"])[0])
    ok &= check_match(
        "num_rounds", int(zc.msgs.challenge.shape[0]), int(state["num_rounds"])
    )

    # The joint claim seeds round 0's p(0) + p(1) identity, so the first
    # round poly carries it: claimed_sum = c0 + sum(c).
    p0 = zc.msgs.round_poly[0]
    ok &= check_match(
        "claimed_sum", p0[0] + jnp.sum(p0), _parse_ef_list(state["claimed_sum"])[0]
    )
    # z_row is the challenge list reversed (SP1's jagged_point order). Every
    # challenge is a sponge image of all bytes observed before it, so 22
    # matches pin the entire emitted round-poly stream, round 0 included.
    ok &= check_match("zc_sumcheck_point (z_row)", zc.msgs.challenge[::-1], z_row)
    ok &= check_match(
        "final_eval",
        eval_coeffs(zc.msgs.round_poly[-1], zc.msgs.challenge[-1]),
        _parse_ef_list(state["final_eval"])[0],
    )
    ok &= _check_opened_values(zc, main_region, shard_dir)
    _report_univariate_encoding(zc, shard_dir)

    if not ok:
        sys.exit(1)
    print("zerocheck stage byte-match: ALL OK")


if __name__ == "__main__":
    flags.mark_flag_as_required("shard_dir")
    app.run(main)
