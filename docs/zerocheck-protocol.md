# Zerocheck protocol map

The one-page map of the zerocheck stage: the claim it proves, the three
batchings, the per-round schedule, and the round-poly point accounting.
Each fact's authoritative statement lives at its definition site — this
page exists because the structure is cross-file (claim assembly, round
engine, coefficient layouts, and verifier dual live in different modules)
and no single docstring can show how they compose. Stage vocabulary and
the SP1 dump mapping are [`shard-pipeline.md`](shard-pipeline.md)'s; the
pinned SP1 reference permalinks live in the module docstrings they size.

| Module | Role |
|---|---|
| [`prover.py`](../sp1_zorch/zerocheck/prover.py) | Stage glue: challenge sampling, zeta, claim derivation, trace slicing, transcript tail |
| [`jagged.py`](../sp1_zorch/zerocheck/jagged.py) | The round engine: `JaggedZerocheckSummand` (protocol content) + `prove_jagged_zerocheck` (SP1-schedule driver) |
| [`coeffs.py`](../sp1_zorch/zerocheck/coeffs.py) | SP1's batching-coefficient layouts and the marked constraint fold |
| [`verifier.py`](../sp1_zorch/zerocheck/verifier.py) | The zorch-native dual: round replay + final oracle check |

## The joint claim

One single-phase sumcheck over the `nrv` row variables proves every chip's
constraint zero-sum **and** its GKR opening claim together:

```
sum_{x in {0,1}^nrv} eq(zeta, x) * sum_c lambda_c * (
    C_{alpha_c}(trace_c(x)) - C_{alpha_c}(0_row) * geq_c(x)
    + sum_j beta**(j+1) * col_{c,j}(x)
)  =  sum_c lambda_c * v_c
```

Per chip the constraint part sums to zero (AIR constraints vanish on real
witness rows — the zerocheck statement) and the column part to `v_c`, the
chip's LogUp-GKR opening claim at `zeta`. The GKR opening claims are folded
INTO the sumcheck claim; there is no separate opening pass. Per-term
rationale: `JaggedZerocheckSummand`'s docstring. The claim seeds `v_c` are
derived once in `prover.gkr_opening_claims` and consumed by both the prover
(sumcheck seed) and the verifier dual (claimed-sum re-derivation).

## Three batchings, three challenges

Sampled inside the stage, after LogUp-GKR, in this order
(`prover.sample_stage_challenges`):

| Challenge | Batches | Coefficient layout | Defined at |
|---|---|---|---|
| `alpha` (wire: `batching_challenge`) | a chip's K constraints | descending powers `[a^(K-1), ..., a, 1]`; empty for a lookup-only chip (K = 0) | `coeffs.rlc_coeffs` |
| `beta` (wire: `gkr_opening_batch_challenge`) | a chip's `[main \| prep]` columns | ascending powers skipping 1: `[b, b^2, ...]` | `coeffs.gkr_powers` |
| `lambda` | across chips | descending powers (the same `rlc_coeffs`) | `JaggedZerocheckSummand.combine_chips` |

Chips are lambda-RLC'd **anew every round, never folded** — a chip index is
a batch axis, not a sumcheck variable — so the reduction is single-phase
and rounds = nrv. There is no interaction/batch coordinate (the contrast
with LogUp-GKR's two-phase row + interaction reduction).

## The per-round schedule

One `lax.scan` body (`prove_jagged_zerocheck.round_step`); the round binds
`zeta[n-1-round]` — SP1's back-to-front variable order (its `jagged_point`
is the challenge list reversed):

1. **Pair split** of each chip's fixed-width buffer — the LSB stride-2
   fold (`zorch.sumcheck.prover.split_pairs`). The buffer width (num_real
   rounded up to a multiple of 4, SP1's round-0 alignment) is schedule,
   not protocol: it keeps every per-chip op shape-invariant so one
   compiled kernel serves all rounds.
2. **Computed round-poly evaluations** at t in {0, 2, 4} per chip
   (`JaggedZerocheckSummand.chip_evals`): the alpha-folded constraint
   evaluation plus the beta-weighted column term, eq-weighted, rows past
   the live bound masked to zero.
3. **Correction + scaling** per t-point (`chip_evals`' `correct`): scale
   by the bound eq factor and subtract the zero-extension leak
   `C_alpha(0_row) * geq` (see "Two paddings" below).
4. **Assembly and RLC**: Gruen coefficient assembly
   (`zorch.sumcheck.gruen.round_coeffs_from_matrix`), then the cross-chip
   lambda-RLC (`combine_chips`) — only the RLC'd poly hits the wire.
   Observe, sample one extension challenge (SP1's `sample_ext_element`
   rule), fold: buffers to `p0 + r*diff`, per-chip claims and eq mass via
   `zorch.sumcheck.gruen.fold_round_scalars`.

## Round-poly point accounting

Degree 4 = max AIR constraint degree 3, times the bound eq factor's
degree 1 (`JaggedZerocheckSummand.DEGREE`; the constraint-degree bound is
the caller's contract). Five coefficients per round are pinned by:

- **computed** — s(0), s(2), s(4): SP1's `sum_as_poly_in_last_variable`
  choice of materialized points (`JaggedZerocheckSummand.extra_ts`);
- **free** — s(1) = claim − s(0) (the claim identity) and an implicit zero
  at the bound eq factor's root b = (1−z)/(1−2z) (the Gruen compression),
  both assembled by `zorch.sumcheck.gruen`.

**Round 0 drops the constraint term at t = 0 only** — that term IS the
statement being proven — while the column term, which does not vanish
there, is kept (`chip_evals`' round-0 mask). Round polys travel in
COEFFICIENT form — SP1's wire encoding, checked by
`zorch.sumcheck.verifier.CoeffsSumcheckRound`; zorch's dense sumcheck
driver sends evaluation-form messages instead.

## Two paddings, one correction

Two different paddings meet in this stage; only one is corrected:

- **Trace-internal padding** (rows below `num_real`): SP1's trace-gen makes
  these rows constraint-satisfying, so they are summed as real rows — no
  correction. `num_real` is the committed height, padding included.
- **Zero-extension** (rows at or above `num_real`): all chips share one
  `{0,1}^nrv` hypercube, so a short chip's tail rows are its trace MLE's
  canonical zero-extension. `C_alpha(0_row)` is generally nonzero and would
  leak into the sum; the summand subtracts `C_alpha(0_row) * geq` with
  `zorch.poly.geq.VirtualGeq`'s closed form, so no `2^nrv` indicator is
  ever materialized. On real rows geq = 0; on zero-extension rows the
  constraint term and the correction cancel exactly.

Every jagged summand carries such a padding correction; LogUp-GKR's dual is
the neutral-fraction virtual mass (`zorch.logup_gkr.jagged_prover`). The
concept lives on the `zorch.sumcheck.gruen.GruenSummand` seam; the math is
each summand's own.

## Versus the LogUp-GKR jagged engine

The generic round machinery is shared by definition, not by analogy — both
engines are `GruenSummand` instances consuming `zorch.sumcheck.gruen`
(free-point assembly, coefficient conversion, post-round fold) and
`zorch.sumcheck.prover` (pair split, zero re-extension). What remains
different is protocol content:

| | Zerocheck | LogUp-GKR jagged |
|---|---|---|
| Phase structure | single-phase, rounds = nrv; chips RLC'd every round | two-phase: row then interaction variables |
| Degree / extra points | 4, extras {2, 4} (SP1's {0, 2, 4}) | 3, extra {1/2} |
| Round-0 rule | constraint term dropped at t = 0, column term kept | — |
| Padding correction | zero-extension leak: `−C_alpha(0_row) * geq` | fold-neutral fraction's virtual mass |
| Claim seeds | GKR openings `v_c`, folded into the sumcheck claim | per-layer running claim |
