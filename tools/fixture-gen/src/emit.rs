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

use slop_algebra::{AbstractExtensionField, AbstractField};
use sp1_gpu_utils::{Ext, Felt};

use crate::driver::Captured;
use crate::npy::{ext_mont_limbs, ext_rows_mont, felt_row_mont, write_npy_i64, write_npy_u32};
use crate::recorder::RecorderLog;

type Result = std::io::Result<()>;

/// SMCS stacking height — every region is packed to a multiple of `2^this`, and
/// the open folds the same message domain (mirrors `convert.py::LOG_STACKING_HEIGHT`).
const LOG_STACKING_HEIGHT: usize = 21;

/// `batching_challenge` (α), `gkr_opening_batch_challenge` (β) and `lambda` are
/// the three Fiat–Shamir challenges sampled — in this order, consecutively — at
/// the zerocheck setup, after the GKR phase. They are not in the proof struct
/// (the verifier re-derives them), so we read them positionally from the
/// recorder's protocol-ordered EF-sample stream. The base index is fixed for the
/// fork pin; the byte-match guards against drift. β/λ are expressed as offsets
/// off α so their "consecutive" relationship can't rot independently.
const ZC_ALPHA_SAMPLE: usize = 501;
const ZC_BETA_SAMPLE: usize = ZC_ALPHA_SAMPLE + 1;
const ZC_LAMBDA_SAMPLE: usize = ZC_ALPHA_SAMPLE + 2;

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

    let proof = &c.proof;

    // The fixture's chip schedule is the chips actually opened this shard — a
    // strict subset of the machine's full chip list, in the canonical (BTreeMap,
    // alphabetical) order the proof commits them. `chip_names` / regions / the
    // per-chip goldens all index by this order.
    let chip_names: Vec<&str> = proof
        .opened_values
        .chips
        .keys()
        .map(String::as_str)
        .collect();

    // Round row/column counts: rounds[0] = preprocessed traces, rounds[1] = main
    // traces; each is the real chips followed by two SMCS padding entries.
    let rounds = &proof.evaluation_proof.row_counts_and_column_counts.rounds;
    let prep_round = &rounds[0];
    let main_round = &rounds[1];

    // meta.json — chip schedule + per-chip real (main) heights.
    let num_reals: Vec<(&str, usize)> = chip_names
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, main_round[i].0))
        .collect();
    std::fs::write(dir.join("meta.json"), meta_json(&chip_names, &num_reals))?;

    // Regions: main covers every active chip; prep only the chips that carry a
    // preprocessed trace (non-empty preprocessed opening).
    let prep_names: Vec<&str> = chip_names
        .iter()
        .copied()
        .filter(|n| !proof.opened_values.chips[*n].preprocessed.local.is_empty())
        .collect();
    std::fs::write(
        inputs.join("main_region.json"),
        region_json(&chip_names, main_round),
    )?;
    std::fs::write(
        inputs.join("prep_region.json"),
        region_json(&prep_names, prep_round),
    )?;

    // inputs/public_values.npy — raw Montgomery passthrough.
    write_npy_u32(
        &inputs.join("public_values.npy"),
        &[c.public_values.len()],
        &felt_row_mont(&c.public_values),
    )?;

    // inputs/{prep,main}_dense.npy — split the packed dense `D` at the round-0
    // boundary raw0 = Σ_{round 0} row_count·col_count.
    let raw0: usize = prep_round.iter().map(|(r, cc)| r * cc).sum();
    let raw1: usize = main_round.iter().map(|(r, cc)| r * cc).sum();
    assert_eq!(
        raw0 + raw1,
        c.host_dense.len(),
        "dense split raw0+raw1 vs |D|"
    );
    write_npy_u32(
        &inputs.join("prep_dense.npy"),
        &[raw0],
        &felt_row_mont(&c.host_dense[..raw0]),
    )?;
    write_npy_u32(
        &inputs.join("main_dense.npy"),
        &[raw1],
        &felt_row_mont(&c.host_dense[raw0..]),
    )?;

    // Transcript challenges. zeta = the GKR eval point (carried in the proof
    // struct); lambda/α/β = the consecutive zerocheck-setup EF samples.
    let le = &proof.logup_gkr_proof.logup_evaluations;
    let zeta: Vec<Ext> = le.point.iter().copied().collect();
    write_npy_u32(
        &inputs.join("zeta.npy"),
        &[zeta.len(), 4],
        &ext_rows_mont(&zeta),
    )?;

    let ef = ef_samples(&c.log);
    assert!(
        ef.len() > ZC_LAMBDA_SAMPLE,
        "recorder captured only {} EF samples (need > {})",
        ef.len(),
        ZC_LAMBDA_SAMPLE
    );
    let alpha = ef[ZC_ALPHA_SAMPLE];
    let beta = ef[ZC_BETA_SAMPLE];
    let lambda = ef[ZC_LAMBDA_SAMPLE];
    write_npy_u32(
        &inputs.join("batching_challenge.npy"),
        &[4],
        &ext_mont_limbs(alpha),
    )?;
    write_npy_u32(
        &inputs.join("gkr_opening_batch_challenge.npy"),
        &[4],
        &ext_mont_limbs(beta),
    )?;
    write_npy_u32(&inputs.join("lambda.npy"), &[4], &ext_mont_limbs(lambda))?;

    // chip_claims — β-power weighting of each chip's [main | prep] GKR openings.
    let claims = chip_claims(c, &chip_names, beta);
    write_npy_u32(
        &inputs.join("chip_claims.npy"),
        &[claims.len(), 4],
        &ext_rows_mont(&claims),
    )?;

    // outputs/round_polys.npy (rounds, degree+1, 4) — zerocheck univariate polys.
    let zc = &proof.zerocheck_proof;
    let n_rounds = zc.univariate_polys.len();
    let deg_plus_1 = zc
        .univariate_polys
        .first()
        .map_or(0, |p| p.coefficients.len());
    let mut round_polys = Vec::with_capacity(n_rounds * deg_plus_1 * 4);
    for poly in &zc.univariate_polys {
        for &coeff in &poly.coefficients {
            round_polys.extend_from_slice(&ext_mont_limbs(coeff));
        }
    }
    write_npy_u32(
        &outputs.join("round_polys.npy"),
        &[n_rounds, deg_plus_1, 4],
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

    // outputs/chip_final_states + lens — each chip's folded openings as
    // [prep, main, eq] (eq = eq(zeta, point), the same tail on every chip),
    // zero-padded to the widest chip.
    let eq = eq_eval(&zeta, &point);
    let (states, lens, width) = chip_finals(c, &chip_names, eq);
    write_npy_u32(
        &outputs.join("chip_final_states.npy"),
        &[chip_names.len(), width, 4],
        &states,
    )?;
    write_npy_i64(
        &outputs.join("chip_final_lens.npy"),
        &[chip_names.len()],
        &lens,
    )?;

    Ok(())
}

/// Reconstruct the protocol-ordered EF challenge stream from the recorder log:
/// each maximal run of base `sample()` draws is chunked into 4-limb EF elements
/// (a trailing remainder of < 4 in a run is discarded, matching the openvm
/// reference's positional walk — a flat "chunk all samples by 4" would mask a
/// misaligned interleave).
fn ef_samples(log: &RecorderLog) -> Vec<Ext> {
    let mut out = Vec::new();
    let (vals, is_sample) = (&log.values, &log.is_sample);
    let n = vals.len();
    let mut i = 0;
    while i < n {
        if !is_sample[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && is_sample[j] {
            j += 1;
        }
        let mut k = i;
        while k + 4 <= j {
            out.push(<Ext as AbstractExtensionField<Felt>>::from_base_slice(
                &vals[k..k + 4],
            ));
            k += 4;
        }
        i = j;
    }
    out
}

/// Each chip's GKR opening claim: the β-power weighting Σ_k β^(k+1)·opening_k of
/// its `[main | prep]` column openings (main first), from
/// `logup_evaluations.chip_openings` (mirrors `sp1_zorch.zerocheck.stage`).
fn chip_claims(c: &Captured, chip_names: &[&str], beta: Ext) -> Vec<Ext> {
    let openings = &c.proof.logup_gkr_proof.logup_evaluations.chip_openings;
    chip_names
        .iter()
        .map(|&n| {
            let ce = &openings[n];
            let mut evals = ce.main_trace_evaluations.to_vec();
            if let Some(prep) = &ce.preprocessed_trace_evaluations {
                evals.extend(prep.to_vec());
            }
            // Σ_k β^(k+1)·opening_k, β as a running power per chip.
            let mut pow = beta;
            evals
                .iter()
                .map(|&x| {
                    let term = pow * x;
                    pow *= beta;
                    term
                })
                .sum()
        })
        .collect()
}

/// eq(zeta, point) = Π_i ((1−z_i)(1−p_i) + z_i·p_i) — the folded-opening tail the
/// reference appends to every chip's final state.
fn eq_eval(zeta: &[Ext], point: &[Ext]) -> Ext {
    let one = Ext::one();
    zeta.iter()
        .zip(point)
        .map(|(&z, &p)| (one - z) * (one - p) + z * p)
        .product()
}

/// Each chip's final state `[prep, main, eq]` (preprocessed openings, then main
/// openings, then the shared eq tail), zero-padded to the widest chip. Returns
/// the flattened `(n_chips, width, 4)` Montgomery limbs, the per-chip lengths,
/// and `width`.
fn chip_finals(c: &Captured, chip_names: &[&str], eq: Ext) -> (Vec<u32>, Vec<i64>, usize) {
    let chips = &c.proof.opened_values.chips;
    let per_chip: Vec<Vec<Ext>> = chip_names
        .iter()
        .map(|&n| {
            let co = &chips[n];
            let mut row = co.preprocessed.local.clone();
            row.extend(co.main.local.iter().copied());
            row.push(eq);
            row
        })
        .collect();
    let width = per_chip.iter().map(Vec::len).max().unwrap_or(0);
    let lens: Vec<i64> = per_chip.iter().map(|r| r.len() as i64).collect();
    // Each row → its Montgomery limbs, zero-padded to `width` (Montgomery zero is
    // the raw 0 limb, so the pad needs no conversion).
    let flat: Vec<u32> = per_chip
        .iter()
        .flat_map(|row| {
            let mut limbs = ext_rows_mont(row);
            limbs.resize(width * 4, 0);
            limbs
        })
        .collect();
    (flat, lens, width)
}

/// `meta.json` matching Python `json.dumps(..., indent=2)` (no trailing newline).
fn meta_json(chip_names: &[&str], num_reals: &[(&str, usize)]) -> String {
    let mut s = String::from("{\n  \"chip_names\": [\n");
    for (i, n) in chip_names.iter().enumerate() {
        let sep = if i + 1 < chip_names.len() { "," } else { "" };
        s.push_str(&format!("    \"{n}\"{sep}\n"));
    }
    s.push_str("  ],\n  \"num_reals\": {\n");
    for (i, (n, h)) in num_reals.iter().enumerate() {
        let sep = if i + 1 < num_reals.len() { "," } else { "" };
        s.push_str(&format!("    \"{n}\": {h}{sep}\n"));
    }
    s.push_str("  }\n}");
    s
}

/// A region dict matching Python `json.dump` (compact, `", "` / `": "`
/// separators, no trailing newline). `names`/`round` are the real chips and the
/// full round (real chips + 2 padding entries); `chip_starts` accumulates over
/// the real chips only.
fn region_json(names: &[&str], round: &[(usize, usize)]) -> String {
    let mut starts = vec![0usize];
    for &(r, cc) in &round[..names.len()] {
        starts.push(starts.last().unwrap() + r * cc);
    }
    let starts = starts
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let rows = round
        .iter()
        .map(|&(r, _)| r.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let cols = round
        .iter()
        .map(|&(_, c)| c.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let names = names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{\"chip_starts\": [{starts}], \"row_counts\": [{rows}], \"column_counts\": [{cols}], \
         \"log_stacking_height\": {LOG_STACKING_HEIGHT}, \"chip_names\": [{names}]}}"
    )
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
