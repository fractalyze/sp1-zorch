# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Convert an SP1-reference phase dump into the ``gpu_fibonacci`` byte-match
fixture layout.

The producer is the pinned SP1 fork (``fractalyze/sp1`` @ ``ref``) driven with
``SP1_DUMP_PHASES`` + ``SP1_DUMP_ZC_BUFFERS`` (see ``README.md``); this module is
the deterministic dump -> fixture step. It emits Montgomery-form ``u32`` arrays
that the byte-match tests load via ``jax.lax.bitcast_convert_type``.

Three conversions are load-bearing (each established by byte-match against the
dump's own golden):

1. SP1 serialises field elements **canonically** in JSON / Debug-txt; the device
   ``.bin`` buffers are **Montgomery**. Everything sourced from JSON/txt is
   converted canonical -> Montgomery (via zk_dtypes); the ``.bin`` dense passes
   through unchanged.
2. The sumcheck ``point_and_eval[0]`` (the bound point) is stored in the reverse
   of the round feed order, so the replayed challenges are reversed back.
3. ``dense_eval`` is ``expected_eval`` (D at the folded point), not the summand
   evaluation in ``point_and_eval[1]``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import namedtuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax
from zk_dtypes import koalabear_mont, koalabearx4_mont

# SP1 packs each region to a multiple of 2^LOG_STACKING_HEIGHT (the SMCS stacking
# height); the open folds the same 2^LOG_STACKING_HEIGHT message domain.
LOG_STACKING_HEIGHT = 21


def _to_mont(canon_u32: np.ndarray) -> np.ndarray:
    """Canonical u32 limbs -> Montgomery raw u32 limbs. zk_dtypes owns the field:
    ``convert_element_type`` reads the canonical value into the Montgomery field,
    ``bitcast_convert_type`` reads back its raw limbs (the inverse of
    ``shard_prover.serialize._field_bytes``). Per base-field limb, so EF elements
    pass their ``(N, 4)`` limbs straight through."""
    bf = lax.convert_element_type(jnp.asarray(canon_u32, jnp.uint32), koalabear_mont)
    return np.asarray(lax.bitcast_convert_type(bf, jnp.uint32))


def _ef_json(elems: list[dict]) -> np.ndarray:
    """``[{"value": [u32 x4]}, ...]`` (canonical) -> ``(N, 4)`` Montgomery u32."""
    return _to_mont(np.asarray([e["value"] for e in elems], dtype=np.uint32))


def _ef_txt(path: Path, prefix: str) -> np.ndarray:
    """Parse ``<prefix>...BinomialExtensionField { value: [..] }`` lines (canonical)
    -> ``(N, 4)`` Montgomery u32."""
    rows = [
        [int(x) for x in re.search(r"value:\s*\[([0-9,\s]+)\]", ln).group(1).split(",")]
        for ln in path.read_text().splitlines()
        if ln.startswith(prefix) and "value" in ln
    ]
    return _to_mont(np.asarray(rows, dtype=np.uint32))


def _poly_coeffs(polys: list[dict]) -> np.ndarray:
    """``univariate_polys`` -> ``(rounds, degree+1, 4)`` Montgomery u32."""
    return _to_mont(
        np.asarray(
            [[c["value"] for c in p["coefficients"]] for p in polys], dtype=np.uint32
        )
    )


# --- zerocheck pieces ---------------------------------------------------
#
# The zerocheck fixture is a separate directory (consumed by
# ``sp1_zorch/zerocheck/jagged_byte_match_test.py``) that also holds the shared
# dense the jagged/open stages borrow. Every conversion below is byte-match
# verified against the dump's own zerocheck goldens (``gpu_shard_proof.json``).


def _ef(canon_u32: np.ndarray) -> jnp.ndarray:
    """Canonical u32 limbs -> extension-field array (for in-converter arithmetic
    on the GKR claims / eq adjustment); the (N, 4) limbs collapse to (N,) EF."""
    return lax.bitcast_convert_type(jnp.asarray(_to_mont(canon_u32)), koalabearx4_mont)


def _ef_value_lines(path: Path) -> np.ndarray:
    """Every ``value: [..]`` row of a Debug-printed EF dump -> (N, 4) canonical."""
    return np.asarray(
        [
            [
                int(x)
                for x in re.search(r"value:\s*\[([0-9,\s]+)\]", ln).group(1).split(",")
            ]
            for ln in path.read_text().splitlines()
            if "value" in ln
        ],
        dtype=np.uint32,
    )


def _state_field(path: Path, key: str) -> np.ndarray:
    """``<key>=BinomialExtensionField { value: [..] }`` -> (4,) canonical u32."""
    for ln in path.read_text().splitlines():
        if ln.strip().startswith(key + "="):
            m = re.search(r"value:\s*\[([0-9,\s]+)\]", ln)
            return np.asarray([int(x) for x in m.group(1).split(",")], dtype=np.uint32)
    raise KeyError(f"{key} not in {path.name}")


def _u32(a: jnp.ndarray) -> np.ndarray:
    return np.asarray(lax.bitcast_convert_type(a, jnp.uint32))


TraceLayout = namedtuple("TraceLayout", "names heights widths")


def _trace_layout(dump: Path) -> tuple[TraceLayout, TraceLayout]:
    """``gpu_traces/trace_layout.txt`` -> (main, prep) chip schedules."""
    main = TraceLayout([], [], [])
    prep = TraceLayout([], [], [])
    for ln in (dump / "gpu_traces" / "trace_layout.txt").read_text().splitlines():
        m = re.match(r"(main|prep) (\w+):poly_size=(\d+),num_polys=(\d+)", ln)
        if not m:
            continue
        dst = main if m.group(1) == "main" else prep
        dst.names.append(m.group(2))
        dst.heights.append(int(m.group(3)))
        dst.widths.append(int(m.group(4)))
    return main, prep


def _region_dict(layout: TraceLayout, round_counts) -> dict:
    """SMCS region metadata: per-chip + trailing pad ``row_counts``/
    ``column_counts`` (straight from the structure hash) and the cumulative
    real-chip ``chip_starts``."""
    rc = [r for r, _ in round_counts]
    cc = [c for _, c in round_counts]
    assert rc[:-2] == list(layout.heights), (rc[:-2], layout.heights)
    starts = [0]
    for h, w in zip(layout.heights, layout.widths):
        starts.append(starts[-1] + h * w)
    return {
        "chip_starts": starts,
        "row_counts": rc,
        "column_counts": cc,
        "log_stacking_height": LOG_STACKING_HEIGHT,
        "chip_names": list(layout.names),
    }


def _chip_openings(op: dict) -> jnp.ndarray:
    """One chip's GKR openings as ``[main | prep]`` (main-first, matching the
    claim's beta-weighting)."""

    def storage(block):
        return _ef(
            np.asarray([e["value"] for e in block["evaluations"]["storage"]], np.uint32)
        )

    main_ev = storage(op["main_trace_evaluations"])
    prep = op["preprocessed_trace_evaluations"]
    return jnp.concatenate([main_ev, storage(prep)]) if prep is not None else main_ev


def _chip_claims(chip_openings: dict, main_names, beta_ef) -> np.ndarray:
    """Each chip's GKR opening claim: the beta-power weighting of its
    ``[main | prep]`` column openings (``sp1_zorch.zerocheck.stage``), from
    ``logup_evaluations.chip_openings``. Returns (num_chips, 4) Montgomery u32."""
    evals = [_chip_openings(chip_openings[name]) for name in main_names]
    powers, cur = [], beta_ef  # [beta^1 .. beta^max_width]
    for _ in range(max(e.shape[0] for e in evals)):
        powers.append(cur)
        cur = cur * beta_ef
    powers = jnp.stack(powers)
    return _u32(jnp.stack([jnp.sum(powers[: e.shape[0]] * e) for e in evals]))


def _eq_final(zeta_ef: jnp.ndarray, point_ef: jnp.ndarray) -> jnp.ndarray:
    """eq(zeta, point) = prod_i ((1-z_i)(1-p_i) + z_i p_i) — the per-chip tail of
    the folded openings (the test appends it to every chip's final state)."""
    one = jnp.ones((), zeta_ef.dtype)
    term = (one - zeta_ef) * (one - point_ef) + zeta_ef * point_ef
    acc = one
    for i in range(term.shape[0]):
        acc = acc * term[i]
    return acc


def _chip_finals(dump: Path, main_names, eq_final_u32) -> tuple[np.ndarray, np.ndarray]:
    """``phase3_chip_opened_values_full.txt`` -> padded (num_chips, max_len, 4)
    final states (prep block, main block, then the eq tail) and (num_chips,)
    lengths, matching the driver's ``[prep, main, eq]`` reorder."""
    raw: dict = {}
    cur = None
    for ln in (dump / "phase3_chip_opened_values_full.txt").read_text().splitlines():
        m = re.match(r"chip (\w+):", ln)
        if m:
            cur = m.group(1)
            raw[cur] = {"prep": [], "main": []}
            continue
        v = re.search(r"(prep|main)\[\d+\]=.*value:\s*\[([0-9,\s]+)\]", ln)
        if v:
            raw[cur][v.group(1)].append([int(x) for x in v.group(2).split(",")])
    per_chip, lens = [], []
    for name in main_names:
        rows = raw[name]["prep"] + raw[name]["main"]
        vals = (
            _to_mont(np.asarray(rows, dtype=np.uint32))
            if rows
            else np.zeros((0, 4), np.uint32)
        )
        vals = np.concatenate([vals, eq_final_u32.reshape(1, 4)], axis=0)
        per_chip.append(vals)
        lens.append(vals.shape[0])
    width = max(v.shape[0] for v in per_chip)
    padded = np.zeros((len(per_chip), width, 4), np.uint32)
    for i, v in enumerate(per_chip):
        padded[i, : v.shape[0]] = v
    return padded, np.asarray(lens, dtype=np.int64)


def convert_zerocheck(dump: Path, out: Path) -> None:
    """Emit the zerocheck ``gpu_fibonacci`` fixture (chip schedule, transcript
    challenges, regions + shared dense, and the round-engine goldens) from one
    self-consistent SP1 dump."""
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "outputs").mkdir(parents=True, exist_ok=True)

    main, prep = _trace_layout(dump)
    rounds = json.loads((dump / "gpu_evaluation_proof.json").read_text())
    rounds = rounds["row_counts_and_column_counts"]["rounds"]  # [0]=prep, [1]=main
    shard = json.loads((dump / "gpu_shard_proof.json").read_text())

    # meta: chip order + per-chip real heights.
    (out / "meta.json").write_text(
        json.dumps(
            {
                "chip_names": list(main.names),
                "num_reals": dict(zip(main.names, main.heights)),
            },
            indent=2,
        )
    )

    # Regions + shared dense (already Montgomery in the device dump).
    json.dump(
        _region_dict(main, rounds[1]), (out / "inputs" / "main_region.json").open("w")
    )
    json.dump(
        _region_dict(prep, rounds[0]), (out / "inputs" / "prep_region.json").open("w")
    )
    dense = np.fromfile(dump / "data_dense_full.bin", dtype=np.uint32)
    raw0 = sum(r * c for r, c in rounds[0])
    raw1 = sum(r * c for r, c in rounds[1])
    assert dense.shape[0] == raw0 + raw1, (dense.shape[0], raw0, raw1)
    np.save(out / "inputs" / "prep_dense.npy", dense[:raw0])
    np.save(out / "inputs" / "main_dense.npy", dense[raw0:])

    # Transcript challenges (canonical txt -> Montgomery).
    zeta = _ef_value_lines(dump / "gpu_z_row.txt")  # (22, 4) == GKR eval point
    np.save(out / "inputs" / "zeta.npy", _to_mont(zeta))
    np.save(
        out / "inputs" / "lambda.npy",
        _to_mont(_state_field(dump / "phase3_lambda.txt", "lambda")),
    )
    alpha = _state_field(dump / "gpu_zerocheck_state.txt", "batching_challenge")
    beta = _state_field(dump / "gpu_zerocheck_state.txt", "gkr_opening_batch_challenge")
    np.save(out / "inputs" / "batching_challenge.npy", _to_mont(alpha))
    np.save(out / "inputs" / "gkr_opening_batch_challenge.npy", _to_mont(beta))
    pv = np.fromfile(dump / "gpu_traces" / "public_values.bin", dtype=np.uint32)
    np.save(out / "inputs" / "public_values.npy", pv)  # device .bin is Montgomery

    # chip_claims = beta-power weighting of the GKR openings.
    chip_openings = shard["logup_gkr_proof"]["logup_evaluations"]["chip_openings"]
    np.save(
        out / "inputs" / "chip_claims.npy",
        _chip_claims(chip_openings, main.names, _ef(beta)[()]),
    )

    # Round-engine goldens.
    zc = shard["zerocheck_proof"]
    np.save(out / "outputs" / "round_polys.npy", _poly_coeffs(zc["univariate_polys"]))
    point_values = zc["point_and_eval"][0]["values"]  # the bound sumcheck point
    zc_point = _ef_json(point_values)  # (22, 4)
    np.save(out / "outputs" / "zc_sumcheck_point.npy", zc_point)
    point_ef = _ef(np.asarray([x["value"] for x in point_values], np.uint32))
    eq = _u32(_eq_final(_ef(zeta), point_ef))
    finals, lens = _chip_finals(dump, main.names, eq)
    np.save(out / "outputs" / "chip_final_states.npy", finals)
    np.save(out / "outputs" / "chip_final_lens.npy", lens)


# --- stacked BaseFold open pieces ---------------------------------------
#
# These augment the jagged-eval fixture dir (the open test reads the same
# directory). The open's Fiat-Shamir challenges are replayed from the SP1-fork
# dump files (``basefold_*.txt``, emitted by the fri.rs dump instrumentation);
# the goldens come straight from ``pcs_proof.basefold_proof``.
#
# ``fri_raw_roots`` is deliberately not emitted: it is zorch's *pre-binding*
# Merkle digest, an internal artifact with no SP1 reference (SP1 stores only the
# separator-bound ``fri_commitments``). The refresh drops that lone open_test
# assertion — ``fri_commitments`` is the SP1 byte-match.


def _openings(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """One SMCS batch opening -> (rows ``(Q, width)`` Mont u32, per-level paths
    ``(levels, Q, 8)``)."""
    vd = entry["values"]["dimensions"]  # [Q, width]
    rows = _to_mont(np.asarray(entry["values"]["storage"], dtype=np.uint32)).reshape(
        vd[0], vd[1]
    )
    pd = entry["proof"]["paths"]
    q, levels = int(pd["dimensions"][0]), int(pd["dimensions"][1])
    paths = _to_mont(np.asarray(pd["storage"], dtype=np.uint32))
    return rows, paths.reshape(q, levels, -1).transpose(1, 0, 2)


def convert_open(dump: Path, out: Path) -> None:
    """Augment the jagged fixture dir with the stacked-open pieces: meta basefold
    config, the replayed open challenges, and the ``basefold_proof`` goldens.

    Run after :func:`convert` (it augments that step's ``meta.json`` /
    ``challenges.npz``)."""
    ep = json.loads((dump / "gpu_evaluation_proof.json").read_text())
    bfp = ep["pcs_proof"]["basefold_proof"]
    bevals = ep["pcs_proof"]["batch_evaluations"]["rounds"]

    # log_blowup = bound-tensor height - stacking height (the codeword is the
    # 2^LOG_STACKING_HEIGHT message domain blown up by 2^log_blowup).
    comp0 = bfp["component_polynomials_query_openings_and_proofs"][0]["proof"]
    log_blowup = int(comp0["log_tensor_height"]) - LOG_STACKING_HEIGHT
    query_idx = np.loadtxt(
        dump / "basefold_query_indices.txt", dtype=np.uint32
    ).reshape(-1)

    # meta.json: add the basefold config + per-round stacking height the open
    # test reads (convert() writes only the row/column counts).
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text())
    for rnd in meta["rounds"]:
        rnd["log_stacking_height"] = LOG_STACKING_HEIGHT
    meta["basefold"] = {
        "num_queries": int(query_idx.shape[0]),
        "pow_bits": 0,  # the dump forces FRI pow_bits=0 (deterministic)
        "log_blowup": int(log_blowup),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # challenges.npz: add the replayed open challenges to convert()'s jagged keys.
    ch_path = out / "outputs" / "challenges.npz"
    ch = dict(np.load(ch_path)) if ch_path.exists() else {}
    ch["batch_challenges"] = _to_mont(
        _ef_value_lines(dump / "basefold_batch_challenges.txt")
    )
    ch["fri_betas"] = _to_mont(_ef_value_lines(dump / "basefold_fri_betas.txt"))
    ch["query_indices"] = query_idx
    np.savez(ch_path, **ch)

    # Proof goldens.
    for r, rd in enumerate(bevals):
        mle0 = _ef_json(rd["evaluations"]["storage"])
        np.savez(out / "outputs" / f"batch_evals_r{r}.npz", mle0=mle0)
    np.save(
        out / "outputs" / "fri_commitments.npy",
        _to_mont(np.asarray(bfp["fri_commitments"], np.uint32)),
    )
    np.save(
        out / "outputs" / "final_poly.npy",
        _to_mont(np.asarray(bfp["final_poly"]["value"], np.uint32)),
    )
    np.save(
        out / "outputs" / "pow_witness.npy",
        _to_mont(np.asarray(bfp["pow_witness"], np.uint32)),
    )
    for r, entry in enumerate(bfp["component_polynomials_query_openings_and_proofs"]):
        rows, paths = _openings(entry)
        np.savez(
            out / "outputs" / f"component_openings_r{r}.npz",
            rows=rows,
            **{f"proof_l{lvl}": paths[lvl] for lvl in range(paths.shape[0])},
        )
    for i, entry in enumerate(bfp["query_phase_openings_and_proofs"]):
        rows, paths = _openings(entry)
        np.savez(
            out / "outputs" / f"query_openings_f{i}.npz",
            rows=rows,
            **{f"proof_l{lvl}": paths[lvl] for lvl in range(paths.shape[0])},
        )


def convert(dump: Path, out: Path) -> None:
    """Emit the ``gpu_fibonacci`` fixture (jagged eval pieces + shared dense) from
    one self-consistent SP1 dump directory."""
    proof = json.loads((dump / "gpu_evaluation_proof.json").read_text())
    rounds = proof["row_counts_and_column_counts"]["rounds"]
    row_counts = [[rc for rc, _ in r] for r in rounds]
    column_counts = [[cc for _, cc in r] for r in rounds]

    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "outputs").mkdir(parents=True, exist_ok=True)

    # meta.json — only row/column counts are consumed by the sumcheck byte-match.
    (out / "meta.json").write_text(
        json.dumps(
            {
                "rounds": [
                    {"row_counts": rc, "column_counts": cc}
                    for rc, cc in zip(row_counts, column_counts, strict=True)
                ]
            },
            indent=2,
        )
    )

    # Inputs: z_row, per-round real column claims (drop the trailing pad column
    # per round: claims_r{r} excludes the last ``column_counts[-2:]`` columns).
    np.save(
        out / "inputs" / "z_row.npy",
        _ef_txt(dump / "phase4_sumcheck_claim.txt", "z_row["),
    )
    cols = _ef_txt(dump / "phase4_column_claims.txt", "col[")
    start = 0
    for r, ccs in enumerate(column_counts):
        n_real = sum(ccs) - (ccs[-2] + ccs[-1])
        np.save(out / "inputs" / f"claims_r{r}.npy", cols[start : start + n_real])
        start += sum(ccs)  # skip the round's pad columns too
    assert start == cols.shape[0], (start, cols.shape[0])

    # The bound sumcheck point is stored in the reverse of the round feed order.
    # challenges.npz carries the *feed* order (reversed); the *_point goldens are
    # the round's emitted point (non-reversed, the convention the round produces).
    z_col = _ef_txt(dump / "phase4_z_col.txt", "z_col[")
    inner = proof["jagged_eval_proof"]["partial_sumcheck_proof"]
    outer_point = _ef_json(proof["sumcheck_proof"]["point_and_eval"][0]["values"])
    inner_point = _ef_json(inner["point_and_eval"][0]["values"])
    outer_alphas = outer_point[::-1]
    inner_alphas = inner_point[::-1]
    np.savez(
        out / "outputs" / "challenges.npz",
        z_col=z_col,
        outer_alphas=outer_alphas,
        inner_alphas=inner_alphas,
    )

    # Golden outputs (outer Hadamard sumcheck + inner branching-program sumcheck).
    sp = proof["sumcheck_proof"]
    np.save(out / "outputs" / "outer_sumcheck_claim.npy", _ef_json([sp["claimed_sum"]]))
    np.save(
        out / "outputs" / "outer_sumcheck_polys.npy",
        _poly_coeffs(sp["univariate_polys"]),
    )
    np.save(out / "outputs" / "outer_sumcheck_point.npy", outer_point)
    np.save(out / "outputs" / "dense_eval.npy", _ef_json([proof["expected_eval"]]))
    np.save(out / "outputs" / "inner_claimed_sum.npy", _ef_json([inner["claimed_sum"]]))
    np.save(
        out / "outputs" / "inner_sumcheck_polys.npy",
        _poly_coeffs(inner["univariate_polys"]),
    )
    np.save(out / "outputs" / "inner_point.npy", inner_point)

    # Shared dense D = prep || main (already Montgomery in the device dump). The
    # split point is round 0's raw packed area; the zerocheck fixture consumes
    # these too (the dense is one shard, shared across stages).
    dense = np.fromfile(dump / "data_dense_full.bin", dtype=np.uint32)
    raw0 = sum(rc * cc for rc, cc in rounds[0])
    raw1 = sum(rc * cc for rc, cc in rounds[1])
    assert dense.shape[0] == raw0 + raw1, (dense.shape[0], raw0, raw1)
    np.save(out / "inputs" / "prep_dense.npy", dense[:raw0])
    np.save(out / "inputs" / "main_dense.npy", dense[raw0:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, type=Path, help="SP1 dump directory")
    ap.add_argument("--out", type=Path, help="jagged-eval fixture output directory")
    ap.add_argument(
        "--zerocheck-out",
        type=Path,
        help="zerocheck fixture output directory (chip schedule, regions, shared dense)",
    )
    args = ap.parse_args()
    if not (args.out or args.zerocheck_out):
        ap.error("nothing to do: pass --out and/or --zerocheck-out")
    if args.out:
        convert(args.dump, args.out)  # jagged-eval pieces + shared dense
        convert_open(args.dump, args.out)  # stacked-open pieces (augments the above)
    if args.zerocheck_out:
        convert_zerocheck(args.dump, args.zerocheck_out)


if __name__ == "__main__":
    main()
