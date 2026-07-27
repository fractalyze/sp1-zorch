# Architecture: the shard stage composition and SP1 dump vocabulary

`ShardProver` reduces the public shard statement to the trivial claim over one
duplex transcript, and `ShardVerifier` is its dual — one verifier role per
prover role. This page names the levels the proof is built from, then maps
SP1's reference-dump terms onto them.

## Stage / Round

- **Stage** — one claim reduction, as a `ProverStage` / `VerifierStage` pair.
  There are three, composed by `ShardProver` / `ShardVerifier`; each runs its
  own inner Rounds. The table below is the roster.
- **Round** — the genuine inner rounds a Stage scans (per-variable sumcheck,
  GKR layers): `JaggedGkrLayerRound`, `OpenedValuesRound`, …
- **Seams** — what crosses between Stages is a *claim*, not a shared carry:
  `GkrOutputClaim` then `TraceEvaluationClaim`, each one Stage's reduced claim
  and the next one's source claim, so both roles derive the same thing.
- **Statement vs configuration** — a claim carries what varies per shard, the
  roles carry what does not, and the two trace dimensions fall on either side
  of that line. `ChipMetadata` (which chips, and each one's real row count)
  changes shard to shard, so it is claim data and rides every claim down the
  chain. Column counts are fixed by each chip's AIR — SP1's `chip.width()` —
  so `ChipWidths` is role configuration, as are the security parameters
  (blowup, query count, grind bits). The row counts are held as values and
  `ChipMetadata.preamble_stream` derives the absorb stream from them, so the
  transcript's view and the structural view cannot disagree.

## The PCS brackets the Stages

The trace commit is not a fourth Stage. It is the jagged PCS's own commit half:
`JaggedPcsProver.commit` packs and Merkle-commits the regions, and
`JaggedPcsProver.prove` — the Stage — opens what it committed. Fiat-Shamir
forces the halves apart, because the commitment must bind the transcript
before LogUp-GKR draws a challenge and the open cannot run until zerocheck
produces the point to open at.

`JaggedCommitData` spans that gap: the scheme's own prover data (the digest
trees and the round SMCS commitments), held as a local in `ShardProver.prove`
and handed to the open through its witness. It rides no claim, because a claim
is what both roles derive and the verifier never sees a digest tree.

Between the halves both roles call one shared `bind_commitment`, which absorbs
SP1's preamble stream and names the roots the opening is checked against, so an
ordering edit cannot land in one Fiat-Shamir stream and not the other. The
commit half carries no claim of its own; it is byte-matched by
`shard_prover:verify_prove_shard --max_phase=1` and unit-tested for structure
in `commit:trace_commit_test`.

## Stages

| Stage | Round composition | Claim carried | Module | rsp byte-match |
|---|---|---|---|---|
| LogUp-GKR (`LogupGkrProver`) | A chain of layer Rounds (output layer → input layer), each layer a chain of per-variable sumcheck rounds | Per-layer running claim, ending in trace-column openings at the final evaluation point | `sp1_zorch/logup_gkr` | `logup_gkr:verify_first_layer`, `logup_gkr:verify_gkr_prove` |
| Zerocheck (`ZerocheckProver`) | One jagged multi-chip sumcheck: 22 homogeneous per-variable rounds over `eq * (constraint RLC + GKR column term)`, then the per-chip opened values absorbed into the transcript | In: every chip's constraint zero-sum + its GKR opening claim; out: one claim at the sumcheck point + the opened values there (the evaluation Stage's per-column claims) | `sp1_zorch/zerocheck` | `zerocheck:verify_zerocheck` |
| Jagged opening (`JaggedPcsProver.prove`) | Outer/inner sumcheck reducing the committed trace to `D(z_final)`, then the stacked BaseFold open of `D` at that point | In: the zerocheck point + per-column claims off its source claim; out: the evaluation proof (jagged eval + stacked BaseFold PCS) | `zorch/pcs/jagged` | `shard_prover:verify_prove_shard` |

Each runnable above gates one Stage's math; `shard_prover:verify_prove_shard`
gates the *composition* — it runs the assembled `ShardProver` over a dump and
seals it on the prove's own trace commitment plus the zerocheck point, which
transitively pins the full composed transcript. Proof assembly / serialization
consumes the named sections of `ShardProof` (see `shard_prover/serialize.py`).

## SP1 reference dump vocabulary

The byte-match reference is a dump captured from SP1's instrumented prover
(capture recipe: whir-zorch `sp1/testing/testdata/rsp/README.md`). SP1's
instrumentation calls its `tracing` span boundaries **phases**, and several
dump files carry that prefix. The numbering:

| SP1 phase | This repo |
|---|---|
| phase 1 | the PCS commit half |
| phase 2 | LogUp-GKR Stage |
| phase 3 | Zerocheck Stage |
| phase 4 | Jagged opening Stage |

**Convention: "phase" names an SP1 tracing span and nothing else** — dump file
names, capture spans, and the byte-match harness that times itself against
them (`--max_phase`, the `[phase X]` lines). Our own levels are Stage and
Round. The table above is not a synonym list: phase 1 is a PCS commit, not a
Stage.

Per-file map (one rsp shard directory):

| Dump file | Stage | Contents / consumer |
|---|---|---|
| `gpu_traces/*.bin`, `*.meta` | input | Per-chip main traces + dims (`.meta` alone for zero-real chips; `public_values.bin` rides alongside); `shard_prover.fixture_loader` |
| `gpu_vk.txt`, `gpu_commitment.txt` | Trace commit | vk, main commitment; preamble observes the vk, `verify_prove_shard` (`--max_phase=1`) byte-matches the main commitment (the preprocessed commit is setup-bound in the vk, covered transitively by the full-chain open) |
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
