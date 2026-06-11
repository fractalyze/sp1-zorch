# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""rsp byte-match harness for the layered LogUp-GKR prove -- a runnable.

Rebuilds the shard preamble transcript (vk -> public values -> the dump's
main commitment -> chip metadata), runs ``prove_logup_gkr``, and compares
every transcript-derived value against the reference dump:

- ``gpu_pre_gkr_diag.txt`` / ``gpu_post_grind_diag.txt`` -- challenger
  checkpoints (each diag is one cloned squeeze, non-destructive here since
  the transcript is functional);
- ``gpu_gkr_state.txt`` -- witness, alpha, beta seeds, output MLEs, z1;
- ``gkr_sumcheck_rounds.txt`` -- per-layer lambda + claim, output to input.

The challenge derivation (alpha / seeds / z1) is replayed on transcript
clones purely for diagnosis -- when a layer diverges, the replay says
whether the drift was already in the head challenges or in the layer
stream. It must mirror ``prove_logup_gkr``'s head exactly.

    bazel run //sp1_zorch/logup_gkr:verify_gkr_prove -- \\
        --shard_dir=/path/to/rsp_dump/shardN

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
from absl import app, flags

from sp1_zorch.logup_gkr.head import (
    GrindRound,
    HeadChallengesRound,
    OutputBindRound,
)
from sp1_zorch.logup_gkr.prover import LogupGkrProof, num_beta_values
from sp1_zorch.logup_gkr.public_values import eval_public_values
from sp1_zorch.shard_prover.fixture_loader import (
    _parse_ef_list,
    _parse_kv_lines,
    check_match,
    load_fixture_shard,
)
from sp1_zorch.shard_prover.replay import (
    clone_diag,
    preamble_transcript,
    replay_gkr,
    shard_regions,
)
from zorch.transcript import Transcript

_SHARD_DIR = flags.DEFINE_string(
    "shard_dir", None, "rsp shard dump directory (e.g. .../rsp_dump/shard1)."
)
_GKR_POW_BITS = flags.DEFINE_integer(
    "gkr_pow_bits",
    12,
    "GKR grind bits (SP1 hardcodes GKR_GRINDING_BITS = 12).",
)


def _replay_challenges_to_z1(
    transcript: Transcript,
    proof: LogupGkrProof,
    num_betas: int,
    state: dict[str, str],
) -> bool:
    """Diagnostic replay from the preamble through z1 on a transcript clone.

    Threads the same head Rounds ``prove_logup_gkr`` runs (one schedule
    definition, ``sp1_zorch.logup_gkr.head``); the per-value checks say
    which challenge first diverged when a later anchor mismatches. Runs
    after the prove since the z1 leg absorbs the circuit output."""
    # pow_bits=0: advance the recorded witness's stream without re-judging it.
    _, transcript, _ = GrindRound(proof.witness)(None, transcript)
    ok = check_match(
        "post_grind_diag", clone_diag(transcript), int(state["witness_diag"])
    )
    _, transcript, head = HeadChallengesRound(num_betas)(None, transcript)
    ok &= check_match("alpha", head.alpha, _parse_ef_list(state["alpha"])[0])
    for i in range(head.beta_seeds.shape[0]):
        ok &= check_match(
            f"beta_seed[{i}]",
            head.beta_seeds[i],
            _parse_ef_list(state[f"beta_seed[{i}]"])[0],
        )
    _, transcript, z1 = OutputBindRound(proof.circuit_output)(None, transcript)
    z1_keys = sum(1 for k in state if k.startswith("z1["))
    ok &= check_match("z1 count", z1.shape[0], z1_keys)
    for i in range(min(z1.shape[0], z1_keys)):
        ok &= check_match(f"z1[{i}]", z1[i], _parse_ef_list(state[f"z1[{i}]"])[0])
    return ok


def _check_outputs(proof: LogupGkrProof, state: dict[str, str]) -> bool:
    num = proof.circuit_output.numerator
    den = proof.circuit_output.denominator
    ok = check_match("num_len", num.shape[0], int(state["num_len"]))
    ok &= check_match("den_len", den.shape[0], int(state["den_len"]))
    for label, got in (("output_num", num), ("output_den", den)):
        n_keys = sum(1 for k in state if k.startswith(f"{label}["))
        want = jnp.concatenate(
            [_parse_ef_list(state[f"{label}[{i}]"]) for i in range(n_keys)]
        )
        ok &= check_match(f"{label} ({n_keys} entries)", got[: want.shape[0]], want)
    return ok


def _check_public_values_leg(
    proof: LogupGkrProof,
    shard,
    preamble: Transcript,
    num_betas: int,
) -> bool:
    """The output-layer bus-balance leg on a real shard (acceptance criterion:
    the public-values digest byte-matches SP1).

    Re-derives the head challenges the way ``verify_logup_gkr`` does — the
    prover sampled and discarded the public-values challenge; the verifier
    folds the public-values constraints under it — then checks SP1's two
    output-layer conditions: the constraint accumulator folds to zero and the
    circuit's cumulative sum ``sum(num/den)`` cancels the interaction digest.
    A from-scratch re-prove of an unbalanced witness changes that sum and
    fails the equality; a valid shard balances exactly."""
    _, transcript, _ = GrindRound(proof.witness)(None, preamble)
    _, transcript, head = HeadChallengesRound(num_betas)(None, transcript)
    public_values = shard.main_trace_data.public_values
    accumulator, digest = eval_public_values(
        public_values, head.pv_challenge, head.alpha, head.betas
    )
    out = proof.circuit_output
    output_cumulative_sum = jnp.sum(out.numerator / out.denominator)
    ok = check_match(
        "pv constraint accumulator == 0", accumulator, jnp.zeros((), accumulator.dtype)
    )
    ok &= check_match(
        "bus balance: sum(num/den) == -pv_digest", output_cumulative_sum, -digest
    )
    return ok


def _check_rounds(proof: LogupGkrProof, shard_dir: Path) -> bool:
    blocks = (shard_dir / "gkr_sumcheck_rounds.txt").read_text().split("--- round ---")
    rounds = [_parse_kv_lines(b, skip_unkeyed=True) for b in blocks if b.strip()]
    # The dump logs one block per round EXCEPT the input layer's (its buffer
    # is gpu_first_layer.txt, dumped separately) -- so blocks anchor every
    # round but the last; the trace openings anchor that one.
    ok = check_match(
        "round count (+1 unlogged input round)",
        len(proof.round_proofs),
        len(rounds) + 1,
    )
    for i, (rp, ref) in enumerate(zip(proof.round_proofs, rounds)):
        want_lam = _parse_ef_list(ref["lambda"])[0]
        want_claim = _parse_ef_list(ref["claim"])[0]
        ok_i = bool(jnp.all(rp.lam == want_lam))
        ok_i &= bool(jnp.all(rp.claim == want_claim))
        print(
            f"{'OK ' if ok_i else 'MISMATCH'} round {i} (nrv={ref['nrv']}) lambda/claim"
        )
        if not ok_i:
            print(f"  lam  got:  {rp.lam}\n  lam  want: {want_lam}")
            print(f"  claim got:  {rp.claim}\n  claim want: {want_claim}")
            return False
        ok &= ok_i
    return ok


def main(argv) -> None:
    del argv
    shard_dir = Path(_SHARD_DIR.value)
    shard = load_fixture_shard(shard_dir)
    state = _parse_kv_lines(
        (shard_dir / "gpu_gkr_state.txt").read_text(), skip_unkeyed=True
    )
    diag = _parse_kv_lines((shard_dir / "gpu_pre_gkr_diag.txt").read_text())
    grind = _parse_kv_lines((shard_dir / "gpu_post_grind_diag.txt").read_text())
    state["witness_diag"] = grind["post_grind_diag"]

    main_region, prep_region = shard_regions(shard)

    preamble = preamble_transcript(shard, shard_dir)
    ok = check_match("pre_gkr_diag", clone_diag(preamble), int(diag["pre_gkr_diag"]))

    num_betas = num_beta_values(shard.main_trace_data.chips)
    print(f"num_beta_values={num_betas}")

    transcript, proof = replay_gkr(
        shard, shard_dir, main_region, prep_region, pow_bits=_GKR_POW_BITS.value
    )

    ok &= _check_outputs(proof, state)
    ok &= _check_rounds(proof, shard_dir)
    ok &= _check_public_values_leg(proof, shard, preamble, num_betas)
    # One scalar seals the whole stage: the post-GKR diag samples the
    # challenger after every round poly, opening, and trace eval absorbed,
    # so a match here means the full opening stream byte-matched too.
    post = _parse_kv_lines((shard_dir / "gpu_post_gkr_diag.txt").read_text())
    ok &= check_match(
        "post_gkr_diag", clone_diag(transcript), int(post["post_gkr_diag"])
    )
    if not ok:
        # Localize the drift: per-challenge replay names the first value
        # that diverged. Skipped on success -- round 0's lambda already
        # binds the whole head stream through z1, and the replay re-absorbs
        # the full output MLEs (the stage's costliest observe).
        _replay_challenges_to_z1(preamble, proof, num_betas, state)

    if not ok:
        sys.exit(1)
    print("layered GKR prove byte-match: ALL OK")


if __name__ == "__main__":
    flags.mark_flag_as_required("shard_dir")
    app.run(main)
