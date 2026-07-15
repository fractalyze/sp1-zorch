# Architecture: the shard ProveChain and SP1 dump vocabulary

`prove_shard_chain` is a `ProveChain` of four **Stages** — trace commit,
LogUp-GKR, zerocheck, jagged evaluation — threading one duplex transcript and a
single **Bridge** (`ShardBridge`). `verify_shard_chain` is the `VerifyChain`
dual: one verifier Stage per prover Stage. This page names how the shard proof
maps onto Stage / Round / Bridge, then maps SP1's reference-dump terms onto the
Stages.

## Stage / Round / Bridge in this repo

- **Stage** — one step of the shard's heterogeneous sequence, a `Round` subclass
  named `*Stage`: `TraceCommitStage`, `LogupGkrStage`, `ShardZerocheckStage`,
  `ShardJaggedEvalStage`, plus the `PreambleStage` the trace-commit Stage and the
  byte-match replay share. Each Stage runs its own inner Rounds.
- **Round** — kept for the genuine inner rounds a Stage scans (per-variable
  sumcheck, GKR layers): `JaggedGkrLayerRound`, `OpenedValuesRound`, … Only the
  stage level was renamed off `Round`.
- **Bridge** — the state a Stage hands the next: `ShardBridge` (prover),
  `ShardVerifierBridge` (verifier). It holds only what a later Stage reads from
  an earlier one — a Stage writes its own fields via `replace` and passes the
  rest through; static config (vk, SMCS, chips) lives on the Stage, not the
  Bridge. It rides the chain's carry slot, so the Stage signatures read
  `bridge: ShardBridge`.

## Stages

| Stage (this repo) | Round composition | Claim carried | Module | rsp byte-match |
|---|---|---|---|---|
| Trace commit (`TraceCommitStage`) | SMCS merkle commit over the jagged dense packing (no sumcheck rounds; transcript observes vk, public values, commitment, chip metadata via `PreambleStage`) | — (seeds the transcript) | `sp1_zorch/commit` | `shard_prover:verify_prove_shard` (`--max_stage=1`) |
| LogUp-GKR (`LogupGkrStage`) | A chain of layer Rounds (output layer → input layer), each layer a chain of per-variable sumcheck rounds | Per-layer running claim, ending in trace-column openings at the final evaluation point | `sp1_zorch/logup_gkr` | `logup_gkr:verify_first_layer`, `logup_gkr:verify_gkr_prove` |
| Zerocheck (`ShardZerocheckStage`) | One jagged multi-chip sumcheck: 22 homogeneous per-variable rounds over `eq * (constraint RLC + GKR column term)`, then the per-chip opened values absorbed into the transcript | In: every chip's constraint zero-sum + its GKR opening claim; out: one claim at the sumcheck point + the opened values there (the evaluation Stage's per-column claims) | `sp1_zorch/zerocheck` | `zerocheck:verify_zerocheck` |
| Jagged evaluation (`ShardJaggedEvalStage`) | Outer/inner sumcheck reducing the committed trace to `D(z_final)`, then the stacked BaseFold open of `D` at that point | In: the zerocheck point + per-column claims off the Bridge; out: the evaluation proof (jagged eval + stacked BaseFold PCS) | `zorch/pcs/jagged` | `shard_prover:verify_prove_shard` |

Each Stage runnable above gates one Stage's math; `shard_prover:verify_prove_shard`
gates the *composition* — it runs the assembled `prove_shard_chain` over a dump
and seals it on the chain's own trace commitment plus the zerocheck point, which
transitively pins the full composed transcript. Proof assembly / serialization
consumes the chain's message list (see `shard_prover/serialize.py`).

## SP1 reference dump vocabulary

The byte-match reference is a dump captured from SP1's instrumented prover
(capture recipe: whir-zorch `sp1/testing/testdata/rsp/README.md`). SP1's
instrumentation calls the Stages **phases** — its `tracing` span boundaries —
and several dump files carry that prefix. The numbering:

| SP1 term | This repo's Stage |
|---|---|
| phase 1 | Trace commit |
| phase 2 | LogUp-GKR |
| phase 3 | Zerocheck |
| phase 4 | Jagged evaluation / PCS opening |

**Convention: "phase N" appears in this repo only when citing SP1 dump
artifacts (file names, capture spans). Our own code, docs, and PRs name the
levels Stage / Round / Bridge as above.**

Per-file map (one rsp shard directory):

| Dump file | Stage | Contents / consumer |
|---|---|---|
| `gpu_traces/*.bin`, `*.meta` | input | Per-chip main traces + dims (`.meta` alone for zero-real chips; `public_values.bin` rides alongside); `shard_prover.fixture_loader` |
| `gpu_vk.txt`, `gpu_commitment.txt` | Trace commit | vk, main commitment; preamble observes the vk, `verify_prove_shard` (`--max_stage=1`) byte-matches the main commitment (the preprocessed commit is setup-bound in the vk, covered transitively by the full-chain open) |
| `gpu_pre_gkr_diag.txt`, `gpu_post_grind_diag.txt`, `gpu_post_gkr_diag.txt` | LogUp-GKR | Challenger checkpoints (one cloned squeeze each); seal the transcript before/after the Stage |
| `gpu_gkr_state.txt` | LogUp-GKR | Grind witness, alpha, beta seeds, output MLEs, z1 |
| `gpu_first_layer.txt` | LogUp-GKR | Input-layer buffer (the one round `gkr_sumcheck_rounds.txt` does not log) |
| `gkr_sumcheck_rounds.txt` | LogUp-GKR | Per-layer lambda + claim, output to input |
| `gpu_individual_column_evals.txt` | LogUp-GKR → Zerocheck | Flat per-column openings at the GKR point (the zerocheck claim inputs) |
| `gpu_zerocheck_state.txt` | Zerocheck | Batching + GKR opening-batch challenges, joint claimed sum, round count, final eval |
| `phase3_lambda.txt` | Zerocheck | Chip-RLC lambda |
| `phase3_chip_opened_values_full.txt` | Zerocheck | Per-chip main/prep opened values at the sumcheck point |
| `gpu_z_row.txt` | Zerocheck | The sumcheck point, reversed (SP1's jagged row point) |
| `gpu_univariate.txt`, `gpu_sumcheck_finalize.txt` | cross-stage | One line/block per per-variable sumcheck round across all Stages (round polys + sampled challenge; finalize diagnostics). Neither logs a Stage's round 0 |
| `phase4_column_claims.txt`, `phase4_sumcheck_claim.txt`, `phase4_z_col.txt` | Jagged evaluation | Column claims, reduced claim, column point |
| `gpu_evaluation_proof.json` | Jagged evaluation | The serialized evaluation proof (jagged eval + stacked BaseFold PCS) |

Files not listed (ad-hoc `gpu_nrv*_*.bin` buffers) are point-in-time debug
captures with no consumer here.
