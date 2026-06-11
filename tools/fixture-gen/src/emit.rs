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

use slop_algebra::{AbstractExtensionField, AbstractField, UnivariatePolynomial};
use sp1_gpu_utils::{Ext, Felt};

use crate::driver::Captured;
use crate::npy::{
    ext_mont_limbs, ext_rows_mont, felt_mont_u32, felt_row_mont, write_npy_i64, write_npy_u32,
    write_npz, NpzEntry,
};
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

/// `z_col` (11 EF) — the jagged-eval column challenges, sampled after the
/// zerocheck setup. Like the zerocheck α/β/λ they are not in the proof struct
/// (the verifier re-derives them), so they are read positionally from the
/// recorder's protocol-ordered EF-sample stream. The byte-match guards the
/// offset against protocol drift. (The open's basefold challenges are instead
/// end-anchored — they are the last EF draws — see [`open`].)
const JAGGED_Z_COL_SAMPLE: usize = 526;
const JAGGED_Z_COL_LEN: usize = 11;

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
    let (round_polys, rp_shape) = poly_limbs(&zc.univariate_polys);
    write_npy_u32(&outputs.join("round_polys.npy"), &rp_shape, &round_polys)?;

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
/// meta.json: `rounds` = [{row_counts, column_counts}] (open augments each round
///   with `log_stacking_height` + the `basefold` block — see [`open`]).
/// inputs/: `z_row.npy` (22,4 — the zerocheck bound point the jagged eval opens
///   at), `claims_r0.npy` (25,4), `claims_r1.npy` (1699,4). The shared
///   `prep_dense`/`main_dense` are NOT written here — they live in the zerocheck
///   dir and are borrowed via the Bazel filegroup (one dense per shard).
/// outputs/: `challenges.npz` {z_col(11,4), outer_alphas(23,4)=outer_point[::-1],
///   inner_alphas(48,4)=inner_point[::-1]} (open appends its basefold challenges);
///   `outer_sumcheck_claim.npy` (1,4), `outer_sumcheck_polys.npy` (23,3,4),
///   `outer_sumcheck_point.npy` (23,4), `dense_eval.npy` (1,4 = expected_eval),
///   `inner_claimed_sum.npy` (1,4), `inner_sumcheck_polys.npy` (48,3,4),
///   `inner_point.npy` (48,4).
///
/// Provenance: from `proof.evaluation_proof` — `sumcheck_proof` (outer),
/// `jagged_eval_proof.partial_sumcheck_proof` (inner), `expected_eval`,
/// `row_counts_and_column_counts`. `z_row` is `zerocheck_proof.point_and_eval[0]`
/// and the per-round column claims are the chips' `{preprocessed,main}.local`
/// openings (`opened_values`); `z_col` comes from `captured.log`.
pub fn jagged(c: &Captured, dir: &Path) -> Result {
    let inputs = dir.join("inputs");
    let outputs = dir.join("outputs");
    std::fs::create_dir_all(&inputs)?;
    std::fs::create_dir_all(&outputs)?;

    let ep = &c.proof.evaluation_proof;
    let rounds = &ep.row_counts_and_column_counts.rounds;

    // meta.json — row/column counts only; `open` rewrites it with the per-round
    // stacking height + basefold config.
    std::fs::write(dir.join("meta.json"), jagged_meta_json(rounds, None))?;

    // inputs/z_row.npy — the zerocheck bound point (identical to the zerocheck
    // dir's `zc_sumcheck_point`; the row-variable point the jagged eval opens at).
    let z_row = c.proof.zerocheck_proof.point_and_eval.0.values().as_slice();
    write_npy_u32(
        &inputs.join("z_row.npy"),
        &[z_row.len(), 4],
        &ext_rows_mont(z_row),
    )?;

    // inputs/claims_r{0,1}.npy — the per-round real column claims, i.e. each
    // chip's preprocessed (round 0) / main (round 1) column openings concatenated
    // in chip-schedule (BTreeMap) order. The dense's pad columns carry no opening,
    // so this concatenation already excludes them (no explicit pad drop needed).
    let chips = c.proof.opened_values.chips.values();
    let claims_r0: Vec<Ext> = chips
        .clone()
        .flat_map(|co| co.preprocessed.local.iter().copied())
        .collect();
    let claims_r1: Vec<Ext> = chips.flat_map(|co| co.main.local.iter().copied()).collect();
    write_npy_u32(
        &inputs.join("claims_r0.npy"),
        &[claims_r0.len(), 4],
        &ext_rows_mont(&claims_r0),
    )?;
    write_npy_u32(
        &inputs.join("claims_r1.npy"),
        &[claims_r1.len(), 4],
        &ext_rows_mont(&claims_r1),
    )?;

    // outputs — outer (Hadamard) sumcheck goldens.
    let sp = &ep.sumcheck_proof;
    write_ext_scalar(&outputs.join("outer_sumcheck_claim.npy"), sp.claimed_sum)?;
    let (op_flat, op_shape) = poly_limbs(&sp.univariate_polys);
    write_npy_u32(
        &outputs.join("outer_sumcheck_polys.npy"),
        &op_shape,
        &op_flat,
    )?;
    let outer_point = sp.point_and_eval.0.values().as_slice();
    write_npy_u32(
        &outputs.join("outer_sumcheck_point.npy"),
        &[outer_point.len(), 4],
        &ext_rows_mont(outer_point),
    )?;

    // dense_eval = expected_eval (D at the folded point, not point_and_eval[1]).
    write_ext_scalar(&outputs.join("dense_eval.npy"), ep.expected_eval)?;

    // inner (jagged-eval branching-program) sumcheck goldens.
    let inner = &ep.jagged_eval_proof.partial_sumcheck_proof;
    write_ext_scalar(&outputs.join("inner_claimed_sum.npy"), inner.claimed_sum)?;
    let (ip_flat, ip_shape) = poly_limbs(&inner.univariate_polys);
    write_npy_u32(
        &outputs.join("inner_sumcheck_polys.npy"),
        &ip_shape,
        &ip_flat,
    )?;
    let inner_point = inner.point_and_eval.0.values().as_slice();
    write_npy_u32(
        &outputs.join("inner_point.npy"),
        &[inner_point.len(), 4],
        &ext_rows_mont(inner_point),
    )?;

    // challenges.npz — the jagged challenges; `open` re-emits this file with its
    // basefold challenges appended (mirrors `convert` then `convert_open`).
    let ef = ef_samples(&c.log);
    write_npz(
        &outputs.join("challenges.npz"),
        &jagged_challenge_entries(c, &ef),
    )?;

    Ok(())
}

/// Stacked-open pieces — from `convert_open` (augments the jagged dir).
///
/// meta.json: rewritten to add `log_stacking_height` per round + a `basefold`
///   block {num_queries, pow_bits:0, log_blowup}.
/// outputs/: re-emits `challenges.npz` (the jagged keys plus {batch_challenges
///   (2,4), fri_betas(21,4), query_indices(52,)}); `batch_evals_r{0,1}.npz`
///   (mle0), `fri_commitments.npy` (21,8), `final_poly.npy` (4), `pow_witness.npy`
///   (()), `component_openings_r{0,1}.npz` (rows + proof_l0..l22),
///   `query_openings_f0..f20.npz` (rows + shrinking proof_l*). NOT emitted:
///   `fri_raw_roots` (zorch-internal pre-binding digest, no SP1 reference).
///
/// Provenance: `batch_challenges`/`fri_betas` are the *tail* EF samples of
/// `captured.log` and `query_indices` are its `bit_samples` — these three replace
/// the fork's `fri.rs` #29 dumps. The goldens (`fri_commitments`/`final_poly`/
/// `pow_witness`/`batch_evaluations`/openings) come straight from
/// `proof.evaluation_proof.pcs_proof` (`StackedBasefoldProof`); `pow_witness` is a
/// proof-struct field (the grind witness, 0 at pow_bits=0), not a log entry.
pub fn open(c: &Captured, dir: &Path) -> Result {
    let outputs = dir.join("outputs");
    std::fs::create_dir_all(&outputs)?;

    let ep = &c.proof.evaluation_proof;
    let pcs = &ep.pcs_proof;
    let bfp = &pcs.basefold_proof;
    let rounds = &ep.row_counts_and_column_counts.rounds;

    // meta.json — augment the jagged rounds with the per-round stacking height +
    // the basefold config. `log_blowup` = a component opening's tensor height over
    // the stacking height; `num_queries` = the count of FRI query-index draws.
    let num_queries = c.log.bit_samples.len();
    let log_blowup = bfp.component_polynomials_query_openings_and_proofs[0]
        .proof
        .log_tensor_height
        - LOG_STACKING_HEIGHT;
    std::fs::write(
        dir.join("meta.json"),
        jagged_meta_json(rounds, Some((num_queries, log_blowup))),
    )?;

    // challenges.npz — the jagged challenges plus the replayed basefold open
    // challenges. batch_challenges (one per commitment round) and fri_betas (one
    // per FRI fold) are the *last* EF draws of the protocol, so anchor them to the
    // end of the sample stream rather than an absolute offset; query_indices are
    // the FRI `sample_bits` draws (raw indices, not field elements).
    let ef = ef_samples(&c.log);
    let n_fri = bfp.fri_commitments.len();
    let n_batch = rounds.len();
    let fri_betas = ef_slice(&ef, ef.len() - n_fri, n_fri);
    let batch_challenges = ef_slice(&ef, ef.len() - n_fri - n_batch, n_batch);
    let query_indices: Vec<u32> = c.log.bit_samples.iter().map(|&i| i as u32).collect();
    let mut challenges = jagged_challenge_entries(c, &ef);
    challenges.push(NpzEntry::u32(
        "batch_challenges",
        vec![batch_challenges.len(), 4],
        ext_rows_mont(&batch_challenges),
    ));
    challenges.push(NpzEntry::u32(
        "fri_betas",
        vec![fri_betas.len(), 4],
        ext_rows_mont(&fri_betas),
    ));
    challenges.push(NpzEntry::u32(
        "query_indices",
        vec![query_indices.len()],
        query_indices,
    ));
    write_npz(&outputs.join("challenges.npz"), &challenges)?;

    // batch_evals_r{r}.npz — the stacked-PCS per-round batch evaluations (mle0).
    for (r, mle) in pcs.batch_evaluations.rounds.iter().enumerate() {
        let vals = mle.to_vec();
        write_npz(
            &outputs.join(format!("batch_evals_r{r}.npz")),
            &[NpzEntry::u32(
                "mle0",
                vec![vals.len(), 4],
                ext_rows_mont(&vals),
            )],
        )?;
    }

    // Basefold proof goldens. fri_commitments are 21 Merkle roots (8 Felts each).
    let mut fc = Vec::with_capacity(n_fri * 8);
    for digest in &bfp.fri_commitments {
        for &f in digest.iter() {
            fc.push(felt_mont_u32(f));
        }
    }
    write_npy_u32(&outputs.join("fri_commitments.npy"), &[n_fri, 8], &fc)?;
    write_npy_u32(
        &outputs.join("final_poly.npy"),
        &[4],
        &ext_mont_limbs(bfp.final_poly),
    )?;
    write_npy_u32(
        &outputs.join("pow_witness.npy"),
        &[],
        &[felt_mont_u32(bfp.pow_witness)],
    )?;

    // Component / query-phase Merkle openings: opened rows + per-level path digests.
    for (r, op) in bfp
        .component_polynomials_query_openings_and_proofs
        .iter()
        .enumerate()
    {
        write_npz(
            &outputs.join(format!("component_openings_r{r}.npz")),
            &opening_entries(
                op.values.storage.as_slice(),
                op.values.sizes(),
                op.proof.paths.storage.as_slice(),
                op.proof.paths.sizes(),
            ),
        )?;
    }
    for (i, op) in bfp.query_phase_openings_and_proofs.iter().enumerate() {
        write_npz(
            &outputs.join(format!("query_openings_f{i}.npz")),
            &opening_entries(
                op.values.storage.as_slice(),
                op.values.sizes(),
                op.proof.paths.storage.as_slice(),
                op.proof.paths.sizes(),
            ),
        )?;
    }

    Ok(())
}

/// Flatten sumcheck univariate polynomials to `(rounds, degree+1, 4)` Montgomery
/// limbs (each poly's `coefficients`).
fn poly_limbs(polys: &[UnivariatePolynomial<Ext>]) -> (Vec<u32>, [usize; 3]) {
    let deg_plus_1 = polys.first().map_or(0, |p| p.coefficients.len());
    let mut flat = Vec::with_capacity(polys.len() * deg_plus_1 * 4);
    for p in polys {
        for &coeff in &p.coefficients {
            flat.extend_from_slice(&ext_mont_limbs(coeff));
        }
    }
    (flat, [polys.len(), deg_plus_1, 4])
}

/// Write a single EF element as a `(1, 4)` `.npy` of Montgomery limbs.
fn write_ext_scalar(path: &Path, e: Ext) -> Result {
    write_npy_u32(path, &[1, 4], &ext_mont_limbs(e))
}

/// A contiguous run `[start, start+len)` of the EF-sample stream, as challenges.
fn ef_slice(ef: &[Ext], start: usize, len: usize) -> Vec<Ext> {
    assert!(
        start + len <= ef.len(),
        "ef_samples[{start}..{}] out of range (have {})",
        start + len,
        ef.len()
    );
    ef[start..start + len].to_vec()
}

/// The three jagged-eval challenge arrays for `challenges.npz`: `z_col` (column
/// challenges read positionally from the recorder log) plus the outer/inner
/// sumcheck points *reversed* — the prover feeds the bound point in reverse, so
/// the challenge order is the reverse of the stored `*_point` goldens.
fn jagged_challenge_entries(c: &Captured, ef: &[Ext]) -> Vec<NpzEntry> {
    let z_col = ef_slice(ef, JAGGED_Z_COL_SAMPLE, JAGGED_Z_COL_LEN);
    let ep = &c.proof.evaluation_proof;
    let mut outer_alphas = ep
        .sumcheck_proof
        .point_and_eval
        .0
        .values()
        .as_slice()
        .to_vec();
    outer_alphas.reverse();
    let mut inner_alphas = ep
        .jagged_eval_proof
        .partial_sumcheck_proof
        .point_and_eval
        .0
        .values()
        .as_slice()
        .to_vec();
    inner_alphas.reverse();
    vec![
        NpzEntry::u32("z_col", vec![z_col.len(), 4], ext_rows_mont(&z_col)),
        NpzEntry::u32(
            "outer_alphas",
            vec![outer_alphas.len(), 4],
            ext_rows_mont(&outer_alphas),
        ),
        NpzEntry::u32(
            "inner_alphas",
            vec![inner_alphas.len(), 4],
            ext_rows_mont(&inner_alphas),
        ),
    ]
}

/// One Merkle opening → its `.npz` members: `rows` (opened values `(Q, width)`)
/// and per-level `proof_l{lvl}` path digests `(Q, 8)`. The path tensor is stored
/// `(Q, levels)` of 8-wide digests; we transpose it to per-level `(Q, 8)` arrays
/// (mirrors `convert.py::_openings`).
fn opening_entries(
    values: &[Felt],
    values_sizes: &[usize],
    paths: &[[Felt; 8]],
    paths_sizes: &[usize],
) -> Vec<NpzEntry> {
    let (q, width) = (values_sizes[0], values_sizes[1]);
    let mut entries = vec![NpzEntry::u32("rows", vec![q, width], felt_row_mont(values))];
    let (pq, levels) = (paths_sizes[0], paths_sizes[1]);
    for lvl in 0..levels {
        let mut data = Vec::with_capacity(pq * 8);
        for qi in 0..pq {
            for &f in paths[qi * levels + lvl].iter() {
                data.push(felt_mont_u32(f));
            }
        }
        entries.push(NpzEntry::u32(format!("proof_l{lvl}"), vec![pq, 8], data));
    }
    entries
}

/// `meta.json` for the jagged dir, byte-matching Python `json.dumps(indent=2)`
/// (no trailing newline). `basefold = Some((num_queries, log_blowup))` adds the
/// open augmentation — the per-round `log_stacking_height` and the `basefold`
/// block; `None` writes the bare jagged rounds (`convert`'s pre-`convert_open`
/// form).
fn jagged_meta_json(rounds: &[Vec<(usize, usize)>], basefold: Option<(usize, usize)>) -> String {
    let mut s = String::from("{\n  \"rounds\": [\n");
    for (ri, round) in rounds.iter().enumerate() {
        s.push_str("    {\n");
        push_int_array(&mut s, "row_counts", round.iter().map(|&(r, _)| r));
        s.push_str(",\n");
        push_int_array(&mut s, "column_counts", round.iter().map(|&(_, c)| c));
        if basefold.is_some() {
            s.push_str(&format!(
                ",\n      \"log_stacking_height\": {LOG_STACKING_HEIGHT}\n"
            ));
        } else {
            s.push('\n');
        }
        let rsep = if ri + 1 < rounds.len() { "," } else { "" };
        s.push_str(&format!("    }}{rsep}\n"));
    }
    s.push_str("  ]");
    if let Some((num_queries, log_blowup)) = basefold {
        s.push_str(&format!(
            ",\n  \"basefold\": {{\n    \"num_queries\": {num_queries},\n    \"pow_bits\": 0,\n    \"log_blowup\": {log_blowup}\n  }}\n"
        ));
    } else {
        s.push('\n');
    }
    s.push('}');
    s
}

/// Append `      "<key>": [` + one indented int per line + `      ]` (a 2-space
/// JSON int array at round-field depth, no trailing newline).
fn push_int_array(s: &mut String, key: &str, vals: impl Iterator<Item = usize>) {
    let vals: Vec<usize> = vals.collect();
    s.push_str(&format!("      \"{key}\": [\n"));
    for (i, v) in vals.iter().enumerate() {
        let sep = if i + 1 < vals.len() { "," } else { "" };
        s.push_str(&format!("        {v}{sep}\n"));
    }
    s.push_str("      ]");
}
