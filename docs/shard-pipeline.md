# Shard pipeline: Rounds, stages, and SP1 dump vocabulary

The shard proof is one `zorch.round.Round` chain threading a single duplex
transcript; a **stage** is a named sub-chain whose carry is a claim being
reduced. The Round contract and chain composition are zorch's — see zorch
`round.py` and `docs/sumcheck.md`. This page adds only the SP1 layer: which
Rounds each stage composes, and how SP1's reference-dump vocabulary maps onto
them.

## Stages

| Stage (this repo) | Round composition | Claim carried | Module | rsp byte-match |
|---|---|---|---|---|
| Trace commit | SMCS merkle commit over the jagged dense packing (no sumcheck rounds; transcript observes vk, public values, commitment, chip metadata) | — (seeds the transcript) | `sp1_zorch/commit` | `shard_prover:verify_prove_shard` (`--max_stage=1`) |
| LogUp-GKR | A chain of layer Rounds (output layer → input layer), each layer a chain of per-variable sumcheck rounds | Per-layer running claim, ending in trace-column openings at the final evaluation point | `sp1_zorch/logup_gkr` | `logup_gkr:verify_first_layer`, `logup_gkr:verify_gkr_prove` |
| Zerocheck | One jagged multi-chip sumcheck: 22 homogeneous per-variable rounds over `eq * (constraint RLC + GKR column term)`, then the per-chip opened values absorbed into the transcript | In: every chip's constraint zero-sum + its GKR opening claim; out: one claim at the sumcheck point + the opened values there (the evaluation stage's per-column claims) | `sp1_zorch/zerocheck` | `zerocheck:verify_zerocheck` |

Each stage runnable above gates one stage's math; `shard_prover:verify_prove_shard`
gates the *composition* — it runs the assembled `prove_shard_chain` over a dump
and seals it on the chain's own trace commitment plus the zerocheck point, which
transitively pins the full composed transcript.

Two stages follow zerocheck — jagged evaluation (outer/inner sumcheck +
stacked BaseFold opening, consuming the zerocheck point claim) and proof
assembly/serialization; see fractalyze/sp1-zorch#13 for the plan.

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
| `gpu_traces/*.bin`, `*.meta` | input | Per-chip main traces + dims (`.meta` alone for zero-real chips; `public_values.bin` rides alongside); `shard_prover.fixture_loader` |
| `gpu_vk.txt`, `gpu_commitment.txt` | Trace commit | vk, main commitment; preamble observes the vk, `verify_prove_shard` (`--max_stage=1`) byte-matches the main commitment (the preprocessed commit is setup-bound in the vk, covered transitively by the full-chain open) |
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
