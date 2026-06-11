//! Phase 3 — fixture emission, ported stage-by-stage from `convert.py`.
//!
//! Each function reproduces one fixture directory's `inputs/`, `outputs/` and
//! `meta.json` from the in-process [`Captured`] data (proof struct + recorder log
//! + dense buffer), matching the committed Montgomery-`u32` bytes exactly. The
//! per-file contract below is the porting spec (see the `convert.py` map + the
//! repo's `tools/fixture-gen/README.md`). Bodies are wired one stage at a time,
//! each gated on a green byte-match (Tasks 4/5/6 on issue #86).
//!
//! Shared conventions to preserve from `convert.py`:
//!  * **Montgomery** `u32` everywhere (device `.bin` buffers pass through; JSON/
//!    txt-sourced field elements were canonical→Mont — here they come straight
//!    from the proof struct, already Montgomery in memory).
//!  * Dense split: `data_dense_full[..raw0]` = prep, `[raw0..]` = main, where
//!    `raw0 = Σ_{round0} row_count·col_count`. Written into BOTH dirs (jagged/open
//!    read prep/main_dense from the zerocheck dir via the shared filegroup).
//!  * Sumcheck point reversal: `*_alphas` (challenges.npz) carry the reversed
//!    feed order; `*_point` goldens keep the round-emitted order.
//!  * `dense_eval` golden = `expected_eval` (not `point_and_eval[1]`).

use std::path::Path;

use sp1_gpu_utils::Ext;

use crate::driver::Captured;
use crate::npy::{ext_mont_limbs, ext_rows_mont, felt_row_mont, write_npy_u32};

type Result = std::io::Result<()>;

/// `sp1_zorch/zerocheck/testdata/gpu_fibonacci/` — from `convert_zerocheck`.
///
/// meta.json: `chip_names`, `num_reals` (name→height).
/// inputs/: `main_region.json`, `prep_region.json` (chip_starts/row_counts/
///   column_counts/log_stacking_height/chip_names); `main_dense.npy`,
///   `prep_dense.npy` (device-Mont passthrough, split at raw0); `zeta.npy`
///   (22,4 — GKR eval point), `lambda.npy` (4), `batching_challenge.npy` (α,4),
///   `gkr_opening_batch_challenge.npy` (β,4), `public_values.npy` (187),
///   `chip_claims.npy` (35,4 — recomputed β-power weighting of each chip's
///   `[main|prep]` GKR openings from `logup_evaluations.chip_openings`).
/// outputs/: `round_polys.npy` (22,5,4 — zerocheck univariate_polys),
///   `zc_sumcheck_point.npy` (22,4 — point_and_eval[0], stored reversed),
///   `chip_final_states.npy` (35,247,4 — `[prep,main,eq]` reordered + zero-pad,
///   eq tail = eval_eq(zeta, point)), `chip_final_lens.npy` (35, int64).
///
/// Provenance: zeta/lambda/α/β from `captured.log` EF samples; round_polys /
/// point from `proof.zerocheck_proof`; chip openings from
/// `proof.logup_gkr_proof.logup_evaluations`; dense + public_values from
/// `captured`.
pub fn zerocheck(c: &Captured, dir: &Path) -> Result {
    let inputs = dir.join("inputs");
    let outputs = dir.join("outputs");
    std::fs::create_dir_all(&inputs)?;
    std::fs::create_dir_all(&outputs)?;

    // inputs/public_values.npy — raw Montgomery passthrough.
    write_npy_u32(
        &inputs.join("public_values.npy"),
        &[c.public_values.len()],
        &felt_row_mont(&c.public_values),
    )?;

    // inputs/{prep,main}_dense.npy — split the packed dense `D` at the round-0
    // boundary. TODO(derive): raw0 = Σ_{round 0} row_count·col_count from
    // `proof.evaluation_proof.row_counts_and_column_counts.rounds[0]`; for
    // gpu_fibonacci the prep block is 2_097_152 elements.
    const RAW0: usize = 2_097_152;
    write_npy_u32(
        &inputs.join("prep_dense.npy"),
        &[RAW0],
        &felt_row_mont(&c.host_dense[..RAW0]),
    )?;
    write_npy_u32(
        &inputs.join("main_dense.npy"),
        &[c.host_dense.len() - RAW0],
        &felt_row_mont(&c.host_dense[RAW0..]),
    )?;

    // outputs/round_polys.npy (rounds, degree+1, 4) — zerocheck univariate polys.
    let zc = &c.proof.zerocheck_proof;
    let rounds = zc.univariate_polys.len();
    let deg_plus_1 = zc
        .univariate_polys
        .first()
        .map_or(0, |p| p.coefficients.len());
    let mut round_polys = Vec::with_capacity(rounds * deg_plus_1 * 4);
    for poly in &zc.univariate_polys {
        for &coeff in &poly.coefficients {
            round_polys.extend_from_slice(&ext_mont_limbs(coeff));
        }
    }
    write_npy_u32(
        &outputs.join("round_polys.npy"),
        &[rounds, deg_plus_1, 4],
        &round_polys,
    )?;

    // outputs/zc_sumcheck_point.npy (rounds, 4) — point_and_eval[0] (stored
    // reversed vs the round feed order — the golden keeps the stored order).
    let point: Vec<Ext> = zc.point_and_eval.0.values().iter().copied().collect();
    write_npy_u32(
        &outputs.join("zc_sumcheck_point.npy"),
        &[point.len(), 4],
        &ext_rows_mont(&point),
    )?;

    // TODO(next session): the challenge inputs (zeta / lambda / batching_challenge /
    // gkr_opening_batch_challenge) via positional reading of `c.log`; chip_claims
    // (β-power weighting of `logup_gkr_proof.logup_evaluations.chip_openings`);
    // chip_final_states/lens ([prep,main,eq] reorder + eq tail); main/prep regions;
    // meta.json (chip_names + num_reals). See the module doc + convert.py.
    Ok(())
}

/// `sp1_zorch/jagged/testdata/gpu_fibonacci/` (jagged-eval pieces) — from `convert`.
///
/// meta.json: `rounds` = [{row_counts, column_counts, log_stacking_height}].
/// inputs/: `z_row.npy` (22,4), `claims_r0.npy` (25,4), `claims_r1.npy`
///   (1699,4 — real columns; pad cols dropped); shared `prep_dense`/`main_dense`.
/// outputs/: `challenges.npz` {z_col(11,4), outer_alphas(23,4)=outer_point[::-1],
///   inner_alphas(48,4)=inner_point[::-1]}; `outer_sumcheck_claim.npy` (1,4),
///   `outer_sumcheck_polys.npy` (23,3,4), `outer_sumcheck_point.npy` (23,4),
///   `dense_eval.npy` (1,4 = expected_eval), `inner_claimed_sum.npy` (1,4),
///   `inner_sumcheck_polys.npy` (48,3,4), `inner_point.npy` (48,4).
///
/// Provenance: from `proof.evaluation_proof` — `sumcheck_proof` (outer),
/// `jagged_eval_proof.partial_sumcheck_proof` (inner), `expected_eval`,
/// `row_counts_and_column_counts`; `z_col` from `captured.log`.
pub fn jagged(_c: &Captured, _dir: &Path) -> Result {
    todo!("Task 5 — port convert; byte-match jagged/testdata/gpu_fibonacci")
}

/// Stacked-open pieces — from `convert_open` (augments the jagged dir).
///
/// meta.json: adds `log_stacking_height` per round + `basefold` {num_queries:52,
///   pow_bits:0, log_blowup:2}.
/// outputs/: augments `challenges.npz` with {batch_challenges(2,4), fri_betas
///   (21,4), query_indices(52,)}; `batch_evals_r{0,1}.npz` (mle0),
///   `fri_commitments.npy` (21,8), `final_poly.npy` (4), `pow_witness.npy` (()),
///   `component_openings_r{0,1}.npz` (rows + proof_l0..l22),
///   `query_openings_f0..f20.npz` (rows + shrinking proof_l*). NOT emitted:
///   `fri_raw_roots` (zorch-internal, no SP1 reference).
///
/// Provenance: the four in-scope basefold challenges from `captured.log`
/// (`batch_challenges`/`fri_betas` = EF samples, `query_indices` = bit_samples,
/// `pow_witness` = grind_witnesses) — these replace the `fri.rs` #29 dumps; the
/// goldens (commitments/openings/final_poly/batch_evals) from
/// `proof.evaluation_proof.pcs_proof` (`StackedBasefoldProof`).
pub fn open(_c: &Captured, _dir: &Path) -> Result {
    todo!("Task 6 — port convert_open; byte-match the open pieces")
}
