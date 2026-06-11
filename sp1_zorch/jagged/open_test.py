# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match the stacked BaseFold open against the SP1 pipeline dump.

Drives ``stacked_basefold_open`` over the vendored ``gpu_fibonacci`` regions
(preprocessed + main, committed separately) with a scripted transcript replaying
the dumped basefold challenges (the RLC weights, the per-round FRI fold betas,
the query positions, and the proof-of-work witness), then byte-matches the
structural proof the open produces: per-round batch evaluations, the FRI
fold-layer roots (SP1 separator-bound), the final poly, the proof-of-work
witness, and the component + query Merkle openings. The duplex encoding that
derives those challenges is the pipeline's concern (verified elsewhere via the
real transcript), so replaying them here isolates the open's commit/fold/open
math — same precedent as ``prover_test`` for the sumcheck half. Mont-u32, no
tolerances.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as BF
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.jagged.open import StackedRound, stacked_basefold_open
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params

_FIXTURE = Path(__file__).parent / "testdata" / "gpu_fibonacci"
# The committed dense lives with the zerocheck fixture (same shard); the open
# reproves D(z_final) over its preprocessed + main regions.
_ZC_INPUTS = (
    Path(__file__).parent.parent / "zerocheck" / "testdata" / "gpu_fibonacci" / "inputs"
)


def _smcs() -> SingleMatrixCommitmentScheme:
    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _from_u32(u32, dtype):
    return jax.lax.bitcast_convert_type(jnp.asarray(u32, dtype=jnp.uint32), dtype)


def _u32(a) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


def _raw_area(round_meta) -> int:
    return sum(
        int(r) * int(c)
        for r, c in zip(
            round_meta["row_counts"], round_meta["column_counts"], strict=True
        )
    )


def _out(name):
    return np.load(_FIXTURE / "outputs" / name)


class _ScriptedTranscript:
    """Replays the dumped basefold challenges so the byte-match exercises the
    open's commit/fold/open math against the reference run's Fiat-Shamir outcomes
    rather than re-deriving them (the duplex encoding is the pipeline's concern).
    Mirrors ``prover_test._ScriptedTranscript`` for the sumcheck half.

    ``samples`` is the flat base-field squeeze stream the open consumes in order
    (each extension challenge is four base squeezes); ``witness`` is the dumped
    proof-of-work witness the grind returns."""

    def __init__(self, samples, witness):
        self._next = list(samples)
        self._witness = witness

    def observe(self, values):
        return self

    def sample(self, n=1):
        return self, jnp.stack([self._next.pop(0) for _ in range(n)])

    def grind(self, pow_bits):
        return self, self._witness


class StackedOpenByteMatchTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        meta = json.loads((_FIXTURE / "meta.json").read_text())
        log_s = int(meta["rounds"][0]["log_stacking_height"])
        cfg = meta["basefold"]
        S = 1 << log_s

        smcs = _smcs()
        code = BitReversedReedSolomon(
            message_len=S, blowup=1 << int(cfg["log_blowup"]), dtype=BF
        )

        prep = _from_u32(
            np.load(_ZC_INPUTS / "prep_dense.npy")[: _raw_area(meta["rounds"][0])], BF
        )
        main = _from_u32(
            np.load(_ZC_INPUTS / "main_dense.npy")[: _raw_area(meta["rounds"][1])], BF
        )

        def build_round(dense):
            k = dense.shape[0] // S
            # Column-major stacking: dense block k is column k. Encode all columns
            # in one batched FFT (the code rides the leading axis), bit-reversed
            # exactly as trace_commit writes the committed codeword.
            mle = dense.reshape(k, S).T  # [S, K]
            codeword = code.encode(dense.reshape(k, S)).T  # [S*blowup, K]
            _root, digest_layers = smcs.commit(codeword)
            return StackedRound(mle=mle, codeword=codeword, digest_layers=digest_layers)

        rounds = [build_round(prep), build_round(main)]

        # Scripted squeeze stream: the RLC weights then each fold round's beta,
        # all as four base squeezes per extension challenge, then one base squeeze
        # per query (the canonical value the open masks to a position).
        ch = _out("challenges.npz")
        ef_stream = _from_u32(
            np.concatenate(
                [ch["batch_challenges"].reshape(-1), ch["fri_betas"].reshape(-1)]
            ),
            BF,
        )
        query_stream = jnp.asarray(ch["query_indices"], dtype=jnp.uint32).astype(BF)
        samples = list(ef_stream) + list(query_stream)
        witness = _from_u32(_out("pow_witness.npy"), BF)

        z_final = _from_u32(_out("outer_sumcheck_point.npy"), EF)
        dense_eval = _from_u32(_out("dense_eval.npy"), EF)

        cls.proof, _ = stacked_basefold_open(
            smcs,
            code,
            rounds,
            z_final,
            dense_eval,
            log_s,
            num_queries=int(cfg["num_queries"]),
            pow_bits=int(cfg["pow_bits"]),
            transcript=_ScriptedTranscript(samples, witness),
        )

    def _assert_match(self, got, exp_u32, name, allow_zero=False):
        exp = np.asarray(exp_u32, dtype=np.uint32).reshape(-1)
        if not allow_zero:
            self.assertGreater(int(exp.sum()), 0, f"degenerate fixture {name}")
        got = _u32(got)
        self.assertEqual(got.shape, exp.shape, f"{name} shape")
        mism = np.nonzero(got != exp)[0]
        self.assertEqual(mism.size, 0, f"{name} diverged at u32 {mism[:8]}")

    def test_batch_evals(self):
        for r in range(2):
            self._assert_match(
                self.proof.batch_evals[r], _out(f"batch_evals_r{r}.npz")["mle0"],
                f"batch_evals_r{r}",
            )

    # No fri_raw_roots check: the raw (pre-binding) fold-layer root is a
    # zorch-internal digest with no SP1 reference — SP1's proof carries only the
    # separator-bound root, byte-matched by test_fri_commitments below.

    def test_fri_commitments(self):
        self._assert_match(self.proof.fri_commitments, _out("fri_commitments.npy"), "fri_commitments")

    def test_univariate_messages(self):
        self._assert_match(
            self.proof.univariate_messages,
            _out("univariate_messages.npy"),
            "univariate_messages",
        )

    def test_final_poly(self):
        self._assert_match(self.proof.final_poly, _out("final_poly.npy"), "final_poly")

    def test_pow_witness(self):
        # pow_bits == 0 on this dev fixture, so the witness is the canonical zero.
        self._assert_match(
            self.proof.pow_witness, _out("pow_witness.npy"), "pow_witness", allow_zero=True
        )

    def test_component_openings(self):
        for r in range(2):
            rows, paths = self.proof.component_openings[r]
            dump = _out(f"component_openings_r{r}.npz")
            self._assert_match(rows, dump["rows"], f"component_openings_r{r}.rows")
            for lvl, path in enumerate(paths):
                self._assert_match(path, dump[f"proof_l{lvl}"], f"component_r{r}.proof_l{lvl}")

    def test_query_openings(self):
        for i, (rows, paths) in enumerate(self.proof.query_openings):
            dump = _out(f"query_openings_f{i}.npz")
            self._assert_match(rows, dump["rows"], f"query_openings_f{i}.rows")
            for lvl, path in enumerate(paths):
                self._assert_match(path, dump[f"proof_l{lvl}"], f"query_f{i}.proof_l{lvl}")


if __name__ == "__main__":
    absltest.main()
