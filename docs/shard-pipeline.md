# Shard pipeline: Rounds, stages, and SP1 dump vocabulary

The shard proof is one `zorch.round.Round` chain threading a single duplex
transcript. Every building block follows zorch's Round contract — a prover
round maps `(carry, transcript) -> (carry, transcript, msg)` — and a stage is
just a named sub-chain whose carry is a claim being reduced. Chains nest
(`ProveChain` is itself a `Round`), so the whole pipeline, each stage, and
each per-variable sumcheck step are the same shape at different scales. That
compositionality is the point of building on zorch: stages are assembled from
Rounds, not hand-rolled as monoliths.

## Stages

| Stage (this repo) | Round composition | Claim carried | Module | rsp byte-match |
|---|---|---|---|---|
| Trace commit | SMCS merkle commit over the jagged dense packing (no sumcheck rounds; transcript observes vk, public values, commitment, chip metadata) | — (seeds the transcript) | `sp1_zorch/commit` | `commit:verify_trace_commit` |
| LogUp-GKR | A chain of layer Rounds (output layer → input layer), each layer a chain of per-variable sumcheck rounds | Per-layer running claim, ending in trace-column openings at the final evaluation point | `sp1_zorch/logup_gkr` | `logup_gkr:verify_first_layer`, `logup_gkr:verify_gkr_prove` |
| Zerocheck | One jagged multi-chip sumcheck: 22 homogeneous per-variable rounds over `eq * (constraint RLC + GKR column term)` | In: every chip's constraint zero-sum + its GKR opening claim; out: one claim at the sumcheck point | `sp1_zorch/zerocheck` | `zerocheck:verify_zerocheck` |
| Jagged evaluation | Outer/inner sumcheck rounds + stacked BaseFold opening | In: the zerocheck point claim (row point) + column claims; out: the PCS opening | planned (`zorch/pcs/jagged`) | planned |
| Assembly | Proof layout + bincode serialization + `sp1_verify_shard` FFI | — | planned | planned |

The issue tracker maps stages to fractalyze/sp1-zorch#13's plan (#16–#22).

## SP1 reference dump vocabulary

The byte-match reference is a dump captured from SP1's instrumented prover
(capture recipe: whir-zorch `sp1/testing/testdata/rsp/README.md`). SP1's
instrumentation calls the stages **phases** — its `tracing` span boundaries —
and several dump files carry that prefix. The numbering:

| SP1 term | This repo's stage |
|---|---|
| phase 1 | Trace commit |
| phase 2 | LogUp-GKR |
| phase 3 | Zerocheck |
| phase 4 | Jagged evaluation / PCS opening |

**Convention: "phase N" appears in this repo only when citing SP1 dump
artifacts (file names, capture spans). Our own code, docs, and PRs name
stages by their Round composition as in the table above.**

Per-file map (one rsp shard directory):

| Dump file | Stage | Contents / consumer |
|---|---|---|
| `gpu_traces/*.bin`, `*.meta` | input | Per-chip main traces + dims; `shard_prover.fixture_loader` |
| `gpu_vk.txt`, `gpu_commitment.txt` | Trace commit | vk, main commitment; preamble observes + `verify_trace_commit` |
| `gpu_pre_gkr_diag.txt`, `gpu_post_grind_diag.txt`, `gpu_post_gkr_diag.txt` | LogUp-GKR | Challenger checkpoints (one cloned squeeze each); seal the transcript before/after the stage |
| `gpu_gkr_state.txt` | LogUp-GKR | Grind witness, alpha, beta seeds, output MLEs, z1 |
| `gpu_first_layer.txt` | LogUp-GKR | Input-layer buffer (the one round `gkr_sumcheck_rounds.txt` does not log) |
| `gkr_sumcheck_rounds.txt` | LogUp-GKR | Per-layer lambda + claim, output to input |
| `gpu_individual_column_evals.txt` | LogUp-GKR → Zerocheck | Flat per-column openings at the GKR point (the zerocheck claim inputs) |
| `gpu_zerocheck_state.txt` | Zerocheck | Batching + GKR opening-batch challenges, joint claimed sum, round count, final eval |
| `phase3_lambda.txt` | Zerocheck | Chip-RLC lambda |
| `phase3_chip_opened_values_full.txt` | Zerocheck | Per-chip main/prep opened values at the sumcheck point |
| `gpu_z_row.txt` | Zerocheck | The sumcheck point, reversed (SP1's jagged row point) |
| `gpu_univariate.txt`, `gpu_sumcheck_finalize.txt` | cross-stage | One line/block per per-variable sumcheck round across all stages (round polys + sampled challenge; finalize diagnostics). Neither logs a stage's round 0 |
| `phase4_column_claims.txt`, `phase4_sumcheck_claim.txt`, `phase4_z_col.txt` | Jagged evaluation | Column claims, reduced claim, column point |
| `gpu_evaluation_proof.json` | Jagged evaluation | The serialized evaluation proof (jagged eval + stacked BaseFold PCS) |

Files not listed (ad-hoc `gpu_nrv*_*.bin` buffers) are point-in-time debug
captures with no consumer here.
