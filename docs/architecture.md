# Architecture: the shard stage composition and SP1 dump vocabulary

`ShardProver` is a composite **Stage** running four phases — trace commit,
LogUp-GKR, zerocheck, jagged evaluation — over one duplex transcript, reducing
the public shard statement to the trivial claim. `ShardVerifier` is its dual:
one verifier role per prover role. This page names how the shard proof maps
onto Stage / Round, then maps SP1's reference-dump terms onto the phases.

## Stage / Round in this repo

- **Stage** — one claim reduction, a `ProverStage` / `VerifierStage` pair:
  `LogupGkrProver`, `ZerocheckProver`, `JaggedPcsProver` and their duals, all
  composed by `ShardProver` / `ShardVerifier`. Each Stage runs its own inner
  Rounds.
- **Round** — the genuine inner rounds a Stage scans (per-variable sumcheck,
  GKR layers): `JaggedGkrLayerRound`, `OpenedValuesRound`, …
- **Seams** — what crosses between phases is a *claim*, not a shared carry:
  `GkrOutputClaim` then `TraceEvaluationClaim`, each a phase's reduced claim and
  the next phase's source claim, so both roles derive the same thing. Trace
  commit and the preamble absorb are not stages — a committer runs before any
  claim exists, and `absorb_preamble` only touches the transcript, so it is one
  shared function both roles call. Prover-only products (the commit digest
  trees) stay locals in `ShardProver.prove` and reach the opening through its
  witness, never through a claim.

## Stages

| Stage (this repo) | Round composition | Claim carried | Module | rsp byte-match |
|---|---|---|---|---|
| Trace commit (`TraceCommitter`) | SMCS merkle commit over the jagged dense packing (no sumcheck rounds; transcript observes vk, public values, commitment, chip metadata via `absorb_preamble`) | — (seeds the transcript) | `sp1_zorch/commit` | `shard_prover:verify_prove_shard` (`--max_stage=1`) |
| LogUp-GKR (`LogupGkrProver`) | A chain of layer Rounds (output layer → input layer), each layer a chain of per-variable sumcheck rounds | Per-layer running claim, ending in trace-column openings at the final evaluation point | `sp1_zorch/logup_gkr` | `logup_gkr:verify_first_layer`, `logup_gkr:verify_gkr_prove` |
| Zerocheck (`ZerocheckProver`) | One jagged multi-chip sumcheck: 22 homogeneous per-variable rounds over `eq * (constraint RLC + GKR column term)`, then the per-chip opened values absorbed into the transcript | In: every chip's constraint zero-sum + its GKR opening claim; out: one claim at the sumcheck point + the opened values there (the evaluation Stage's per-column claims) | `sp1_zorch/zerocheck` | `zerocheck:verify_zerocheck` |
| Jagged evaluation (`JaggedPcsProver`) | Outer/inner sumcheck reducing the committed trace to `D(z_final)`, then the stacked BaseFold open of `D` at that point | In: the zerocheck point + per-column claims off its source claim; out: the evaluation proof (jagged eval + stacked BaseFold PCS) | `zorch/pcs/jagged` | `shard_prover:verify_prove_shard` |

Each Stage runnable above gates one Stage's math; `shard_prover:verify_prove_shard`
gates the *composition* — it runs the assembled `ShardProver` over a dump and
seals it on the prove's own trace commitment plus the zerocheck point, which
transitively pins the full composed transcript. Proof assembly / serialization
consumes the named sections of `ShardProof` (see `shard_prover/serialize.py`).

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
levels Stage / Round as above.**

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
