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
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax
from zk_dtypes import koalabear_mont


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
    ap.add_argument("--out", required=True, type=Path, help="fixture output directory")
    args = ap.parse_args()
    convert(args.dump, args.out)


if __name__ == "__main__":
    main()
