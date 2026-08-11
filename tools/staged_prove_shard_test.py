# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Staged runner cold-path units: stage order + release points on a stub
prover, and the flag-surface parse smoke. The GPU prove + golden byte-match
run through the ``//tools:staged_prove_shard`` runnable itself."""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest, flagsaver

from tools import staged_prove_shard as sps


class _StubProver:
    """Records the stage dispatch order the runner drives."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        outer = self

        class _Opening:
            def commit(self, witness):
                outer._calls.append("commit")
                return "commitment", "commit_data"

            def prove(self, claim, witness, transcript):
                outer._calls.append("opening")
                return SimpleNamespace(
                    transcript=transcript + ["opening"],
                    reduced_claim="trivial",
                    reduction_proof="jagged_proof",
                )

        class _Gkr:
            def prove(self, claim, witness, transcript):
                outer._calls.append("gkr")
                return SimpleNamespace(
                    transcript=transcript + ["gkr"],
                    reduced_claim="gkr_claim",
                    reduction_proof="gkr_proof",
                )

        class _Zerocheck:
            def prove(self, claim, witness, transcript):
                outer._calls.append("zerocheck")
                outer.zerocheck_source_claim = claim
                return SimpleNamespace(
                    transcript=transcript + ["zerocheck"],
                    reduced_claim="evaluation",
                    reduction_proof="zc_proof",
                )

        self.opening = _Opening()
        self.gkr = _Gkr()
        self.zerocheck = _Zerocheck()


def _claim():
    return SimpleNamespace(public_values="pv", chip_metadata="meta")


class ProveStagedTest(absltest.TestCase):
    def _run(self, n: int, checks=()):
        calls: list[str] = []
        prover = _StubProver(calls)
        releases: list[int] = []
        with (
            mock.patch.object(
                sps,
                "bind_commitment",
                lambda transcript, claim, commitment: (transcript + ["bind"], "roots"),
            ),
            mock.patch.object(
                sps, "_release_stage", lambda: releases.append(len(calls))
            ),
            mock.patch.object(sps, "ShardProof", lambda *sections: sections),
        ):
            out = sps._prove_staged(prover, _claim(), "witness", [], n, checks)
        return prover, calls, releases, out

    def test_full_chain_order_and_threading(self) -> None:
        prover, calls, releases, (proof, sections, evaluation, commit_data) = self._run(
            4
        )
        # The composite ShardProver.prove order: commit -> GKR -> zerocheck ->
        # jagged opening.
        self.assertEqual(calls, ["commit", "gkr", "zerocheck", "opening"])
        # One release point between consecutive stages (after 1, 2, 3 calls).
        self.assertEqual(releases, [1, 2, 3])
        self.assertEqual(
            sections, ["commitment", "gkr_proof", "zc_proof", "jagged_proof"]
        )
        self.assertEqual(proof, ("commitment", "gkr_proof", "zc_proof", "jagged_proof"))
        self.assertEqual(evaluation, "evaluation")
        self.assertEqual(commit_data, "commit_data")
        # The zerocheck consumed the GKR's reduced claim as its source claim.
        self.assertEqual(prover.zerocheck_source_claim.gkr, "gkr_claim")

    def test_max_phase_prefix_stops_early_without_proof(self) -> None:
        _, calls, releases, (proof, sections, evaluation, _) = self._run(2)
        self.assertEqual(calls, ["commit", "gkr"])
        self.assertEqual(releases, [1])
        self.assertEqual(sections, ["commitment", "gkr_proof"])
        self.assertIsNone(proof)
        self.assertIsNone(evaluation)

    def test_commit_only(self) -> None:
        _, calls, releases, (proof, sections, _, commit_data) = self._run(1)
        self.assertEqual(calls, ["commit"])
        self.assertEqual(releases, [])
        self.assertEqual(sections, ["commitment"])
        self.assertIsNone(proof)
        self.assertEqual(commit_data, "commit_data")

    def test_phase_check_runs_per_stage_and_fail_fast_exits(self) -> None:
        seen: list[str] = []

        def _ok(section):
            seen.append(section)
            return True

        self._run(2, checks=[_ok, _ok])
        self.assertEqual(seen, ["commitment", "gkr_proof"])
        with self.assertRaises(SystemExit):
            self._run(2, checks=[_ok, lambda section: False])


class FlagParseTest(absltest.TestCase):
    def test_flag_surface_parses(self) -> None:
        with flagsaver.flagsaver():
            sps.flags.FLAGS(
                [
                    "staged_prove_shard",
                    "--shard_dir=/dump/shard1,/dump/shard2",
                    "--group_manifest_json=/dump/group_manifest.json",
                    "--zc_class_json=/dump/zc.json",
                    "--gkr_class_json=/dump/gkr.json",
                    "--runs=3",
                    "--max_phase=2",
                    "--noproof_sha256",
                    "--ffi_verify",
                    "--jaxprof_dir=/tmp/prof",
                ]
            )
            self.assertEqual(sps._SHARD_DIR.value, "/dump/shard1,/dump/shard2")
            self.assertEqual(
                sps._GROUP_MANIFEST_JSON.value, "/dump/group_manifest.json"
            )
            self.assertEqual(sps._ZC_CLASS_JSON.value, "/dump/zc.json")
            self.assertEqual(sps._GKR_CLASS_JSON.value, "/dump/gkr.json")
            self.assertEqual(sps._RUNS.value, 3)
            self.assertEqual(sps._MAX_PHASE.value, 2)
            self.assertFalse(sps._PROOF_SHA256.value)
            self.assertTrue(sps._FFI_VERIFY.value)
            self.assertEqual(sps._JAXPROF_DIR.value, "/tmp/prof")

    def test_defaults(self) -> None:
        with flagsaver.flagsaver():
            sps.flags.FLAGS(["staged_prove_shard", "--shard_dir=/dump/shard1"])
            self.assertEqual(sps._RUNS.value, 1)
            self.assertEqual(sps._MAX_PHASE.value, 4)
            self.assertTrue(sps._PROOF_SHA256.value)
            self.assertFalse(sps._FFI_VERIFY.value)
            self.assertIsNone(sps._GROUP_MANIFEST_JSON.value)


if __name__ == "__main__":
    absltest.main()
