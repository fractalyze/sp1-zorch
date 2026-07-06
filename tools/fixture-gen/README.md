# `tools/fixture-gen/` — SP1-reference byte-match fixture generator

Reproducibly (re)generates the `gpu_fibonacci` byte-match fixtures from the
**pinned SP1 reference fork**, so the committed golden has a single authoritative
provenance. A single Rust crate links the reference GPU shard prover, runs one
in-process fibonacci shard prove with a **recording challenger**, and writes the
committed Montgomery-`u32` `.npy`/`.npz` files directly — no Python convert step,
no scattered debug dumps.

(This supersedes the earlier dump-and-Python-convert pipeline of sp1-zorch #76 /
PR #83 and the fork's `fri.rs` Fiat-Shamir dumps of sp1 #29.)

## Pin

SP1 fork `fractalyze/sp1` @ `ref` =
`c8b528bbc3cd4ee15b1e5f7ffa4b4ca697453c55` — the upstream base plus two changes:
`refactor(merkle): generalize Poseidon2SP1Field16Kernels over GC` (a phantom-param
generalization, no behavior change) so a downstream `IopCtx` — the recording
challenger — can instantiate the shard prover without forking prover source; and
`chore(sp1-gpu): retire dead SP1_DUMP_PHASES/SP1_DUMP_ZC_BUFFERS writes` (removes
debug dumps whose only consumer was the deleted `convert.py`; fixtures come from
the recording challenger, not the dumps, so output is unchanged).
Determinism: the basefold prover is built with FRI `pow_bits = 0`, so the prove
(and therefore the fixtures) is byte-identical run to run.

The crate's backend deps are `git`-pinned to `fractalyze/sp1` at the ref above
(see `Cargo.toml`); for local iteration against a sibling fork checkout, swap
them to `path` deps temporarily.

## Recipe (cargo, outside Bazel; needs CUDA GPU + SP1 toolchain)

```sh
cd tools/fixture-gen
export CUDA_HOME=/usr/local/cuda-12.9 CUDA_PATH=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_ARCHS=120                 # sm_120 (RTX 5090); 89 Ada / 90 Hopper
cargo build --release

# Regenerate both fixture dirs in place from one GPU prove:
./target/release/fixture-gen \
  --zerocheck-out ../../sp1_zorch/zerocheck/testdata/gpu_fibonacci \
  --out           ../../sp1_zorch/jagged/testdata/gpu_fibonacci
```

`--out` emits the jagged-eval pieces **and** the stacked-open pieces (the open
augments the jagged dir's `meta.json` / `challenges.npz`); `--zerocheck-out`
emits the zerocheck chip schedule, transcript challenges, regions, and the shared
dense `D`. At least one is required. A clean `git status` after regenerating in
place is the byte-match: same pin → identical bytes.

### Iterating without re-proving

The GPU prove is the slow, shared-resource step. Cache one capture and iterate
the emit stages offline (no GPU):

```sh
./target/release/fixture-gen --dump-cache /tmp/cap.json --out <dir>   # prove once
./target/release/fixture-gen --from-cache /tmp/cap.json --out <dir>   # emit only
```

## Provenance notes (load-bearing)

Field elements are emitted as raw **Montgomery** `u32`: the device `.bin` buffers
(dense `D`, public values) and the proof-struct elements are already Montgomery in
memory, so the on-disk bytes match the committed golden directly (the byte-match
compares Montgomery limbs, no tolerances).

A few conventions the emit code preserves (each verified by byte-match):

1. The sumcheck `point_and_eval[0]` is stored **reversed** vs the round feed
   order — `challenges.npz` `*_alphas` carry the reversed feed order, the `*_point`
   goldens keep the round-emitted (non-reversed) point.
2. `dense_eval` golden = `expected_eval` (D at the folded point), not
   `point_and_eval[1]`.
3. `chip_claims` are not in the proof directly — they are the β-power weighting of
   each chip's `[main | prep]` GKR openings (`logup_evaluations.chip_openings`),
   recomputed exactly as `sp1_zorch/zerocheck/prover.py` does.

The open's Fiat-Shamir challenges (`batch_challenges`, `fri_betas`,
`query_indices`) are recovered from the recording challenger's transcript log
rather than from proof fields; `pow_witness` is a proof-struct field (0 at
`pow_bits = 0`). `fri_raw_roots` is deliberately not emitted: it is zorch's
pre-binding Merkle digest, an internal artifact with no SP1 reference (SP1 stores
only the separator-bound `fri_commitments`).

The `.npz` files are written to byte-match NumPy's `savez` exactly (it opens each
member with `force_zip64=True` — Zip64 local headers, plain central directory);
see `src/npy.rs::write_npz`.
