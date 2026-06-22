# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Pure-logic tests for the bench devenv preflight (no jax / GPU / git)."""

from pathlib import Path

from absl.testing import absltest

from sp1_zorch.commit.bench_preflight import (
    SourceState,
    jax_warnings,
    module_pin_commit,
    parse_override_path,
    source_warnings,
)


def _state(branch="main", behind=0, on_main=True, dirty=False):
    return SourceState(
        path=Path("/abs/zorch"),
        branch=branch,
        head="abc1234",
        dirty=dirty,
        behind=behind,
        on_main_lineage=on_main,
    )


class ParseOverridePathTest(absltest.TestCase):
    def test_active_line(self):
        text = "common --override_module=zorch=/home/me/zorch\n"
        self.assertEqual(parse_override_path(text), "/home/me/zorch")

    def test_commented_line_ignored(self):
        text = "# common --override_module=zorch=/home/me/zorch\n"
        self.assertIsNone(parse_override_path(text))

    def test_no_override(self):
        self.assertIsNone(parse_override_path("build --config=cuda\n"))

    def test_last_definition_wins(self):
        text = (
            "common --override_module=zorch=/first\n"
            "build --override_module=zorch=/second\n"
        )
        self.assertEqual(parse_override_path(text), "/second")

    def test_trailing_flags_after_path(self):
        text = "common --override_module=zorch=/p --config=cuda\n"
        self.assertEqual(parse_override_path(text), "/p")

    def test_other_module_not_matched(self):
        text = "common --override_module=jax=/home/me/jax\n"
        self.assertIsNone(parse_override_path(text))


class SourceWarningsTest(absltest.TestCase):
    def test_clean_main_no_warnings(self):
        self.assertEqual(source_warnings(_state()), [])

    def test_off_main_branch_warns(self):
        w = source_warnings(_state(branch="perf/encode-ntt-shard0", on_main=False))
        self.assertLen(w, 1)
        self.assertIn("off origin/main lineage", w[0])

    def test_behind_main_warns(self):
        w = source_warnings(_state(behind=37))
        self.assertLen(w, 1)
        self.assertIn("37 commit(s) behind", w[0])

    def test_off_main_takes_precedence_over_behind(self):
        # An off-main branch reports the branch problem, not a behind-count.
        w = source_warnings(_state(branch="dead", behind=5, on_main=False))
        self.assertLen(w, 1)
        self.assertIn("off origin/main lineage", w[0])

    def test_dirty_adds_warning(self):
        w = source_warnings(_state(dirty=True))
        self.assertLen(w, 1)
        self.assertIn("uncommitted changes", w[0])

    def test_off_main_and_dirty_both_warn(self):
        w = source_warnings(_state(branch="dead", on_main=False, dirty=True))
        self.assertLen(w, 2)


class JaxWarningsTest(absltest.TestCase):
    def test_patched_warns(self):
        self.assertLen(jax_warnings(True), 1)
        self.assertIn("bit_reverse_output", jax_warnings(True)[0])

    def test_stock_no_warning(self):
        self.assertEqual(jax_warnings(False), [])

    def test_unknown_no_warning(self):
        self.assertEqual(jax_warnings(None), [])


class ModulePinCommitTest(absltest.TestCase):
    _MODULE = """
bazel_dep(name = "zorch", version = "0.0.0")
git_override(
    module_name = "zorch",
    remote = "https://github.com/fractalyze/zorch.git",
    commit = "522bcacd8f9d36ad1a36aee5b08fdc08ff364f93",
)
"""

    def test_extracts_zorch_commit(self):
        self.assertEqual(
            module_pin_commit(self._MODULE),
            "522bcacd8f9d36ad1a36aee5b08fdc08ff364f93",
        )

    def test_missing_module_returns_none(self):
        self.assertIsNone(module_pin_commit(self._MODULE, module="nope"))

    def test_no_git_override_returns_none(self):
        self.assertIsNone(module_pin_commit('bazel_dep(name = "zorch")'))


if __name__ == "__main__":
    absltest.main()
