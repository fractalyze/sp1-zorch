# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Recursion warm: spec/builder/CLI units + the cache-key identity contract.

The load-bearing test is ``KeyIdentityTest``: a really-executing run of the
recursion stage chain and a ``warm_worker``-intercepted run of the SAME chain
must request identical persistent-cache keys — the warm adds ZERO entries to
the cache the real run filled. ``frx.jit`` interception happens at decoration
(module import) time, so the two arms cannot share a process: each runs in a
subprocess (the warm arm imports ``warm_worker`` first, exactly as the
production worker does), both over the same spec-built shard and a tiny
machine. The mechanism is machine-value- and device-agnostic — the same code
path a CUDA warm at the production combine/shrink statics runs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import textwrap  # noqa: E402
from pathlib import Path  # noqa: E402

from absl.testing import absltest, parameterized  # noqa: E402

from sp1_zorch.shard_prover import warm_recursion as wr  # noqa: E402
from sp1_zorch.shard_prover import warm_recursion_cache as wrc  # noqa: E402

# A CPU-sized recursion machine for the identity arms: the statics are jit
# inputs like the production 20/21 combine values, so tiny values exercise the
# same keying while keeping both subprocess chains seconds-cheap.
TINY_MACHINE = wr.RecursionMachine(
    log_stacking_height=4,
    log_blowup=1,
    num_queries=2,
    fri_pow_bits=1,
    max_log_row_count=5,
)

# Spec entry for the identity arms: BaseAlu with keygen-taller prep (the
# recursion prep/main height split), Select main-only. Even heights — GKR's
# even/odd row split needs them.
TINY_SPEC = [
    {
        "name": "tiny",
        "stage": "combine",
        "public_values_len": 8,
        "chips": {
            "BaseAlu": {"rows": 4, "cols": 4, "prep_rows": 8, "prep_cols": 3},
            "Select": {"rows": 8, "cols": 4},
        },
    }
]


def tiny_chips() -> dict:
    """rw-shaped tiny chips for the identity arms, keyed like the recursion
    manifest: constraint-less (the stub policy SP1's lookup chips share) but
    carrying one typed send interaction each, so the LogUp-GKR zones stay
    live in the chain."""
    from rw_constraints import Interaction, InteractionInfo, VirtualPairCol

    from sp1_zorch.shard_prover.chip_loader import make_chip_stub

    def chip(name: str, num_cols: int) -> object:
        c = make_chip_stub(name, num_cols)
        c._interaction_info = {
            "t": InteractionInfo(
                fn="t",
                bus="memory_bus",
                kind="send",
                tuple_width=2,
                interaction=Interaction(
                    values=(VirtualPairCol.single_main(1),),
                    multiplicity=VirtualPairCol.single_main(0),
                    kind=3,
                    is_send=True,
                ),
                sp1_index=0,
            )
        }
        return c

    # num_cols counts prep + main, the rw manifest convention the resolution
    # rule matches against: BaseAlu 4 main + 3 prep, Select 4 main.
    return {"base_alu": chip("base_alu", 7), "select": chip("select", 4)}


def _write_spec(entries: list) -> str:
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, dir=os.environ.get("TEST_TMPDIR")
    )
    json.dump(entries, f)
    f.close()
    return f.name


class StageMachinesTest(absltest.TestCase):
    """The machine statics are SP1 constants (components.rs / fri_params.rs);
    a drift computes disjoint cache keys against the production prove."""

    def test_combine_machine_matches_sp1(self) -> None:
        m = wr.COMBINE_MACHINE
        self.assertEqual(
            (20, 2, 124, 16, 21),
            (
                m.log_stacking_height,
                m.log_blowup,
                m.num_queries,
                m.fri_pow_bits,
                m.max_log_row_count,
            ),
        )

    def test_shrink_machine_matches_sp1(self) -> None:
        m = wr.SHRINK_MACHINE
        self.assertEqual(
            (18, 3, 94, 22, 19),
            (
                m.log_stacking_height,
                m.log_blowup,
                m.num_queries,
                m.fri_pow_bits,
                m.max_log_row_count,
            ),
        )

    def test_normalize_runs_on_the_combine_machine(self) -> None:
        self.assertIs(wr.STAGE_MACHINES["normalize"], wr.COMBINE_MACHINE)

    def test_gkr_grind_is_stage_independent(self) -> None:
        self.assertEqual(12, wr.GKR_GRINDING_BITS)


class SpecTest(parameterized.TestCase):
    def test_loads_valid_spec(self) -> None:
        entries = wr.load_spec(_write_spec(TINY_SPEC))
        self.assertLen(entries, 1)
        e = entries[0]
        self.assertEqual("tiny", e.name)
        self.assertIs(e.machine(), wr.COMBINE_MACHINE)
        self.assertEqual(
            wr.ChipShape(rows=4, cols=4, prep_rows=8, prep_cols=3),
            e.chips["BaseAlu"],
        )
        self.assertEqual(wr.ChipShape(rows=8, cols=4), e.chips["Select"])
        self.assertEqual(8, e.public_values_len)

    def test_name_defaults_to_stage(self) -> None:
        spec = [dict(TINY_SPEC[0], stage="shrink")]
        del spec[0]["name"]
        (e,) = wr.load_spec(_write_spec(spec))
        self.assertEqual("shrink", e.name)
        self.assertIs(e.machine(), wr.SHRINK_MACHINE)

    @parameterized.named_parameters(
        ("unknown_stage", {"stage": "wrap"}, "unknown stage"),
        (
            "unknown_chip",
            {"chips": {"KeccakPermute": {"rows": 4, "cols": 4}}},
            "not a canonical recursion chip",
        ),
        (
            "nonpositive_shape",
            {"chips": {"BaseAlu": {"rows": 0, "cols": 4}}},
            "must be positive",
        ),
        ("no_chips", {"chips": {}}, "no chips"),
        ("zero_pv", {"public_values_len": 0}, "public_values_len"),
    )
    def test_rejects_malformed_entry(self, override: dict, msg: str) -> None:
        spec = [dict(TINY_SPEC[0], **override)]
        with self.assertRaisesRegex(ValueError, msg):
            wr.load_spec(_write_spec(spec))

    def test_rejects_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            wr.load_spec(_write_spec(TINY_SPEC + TINY_SPEC))


class BuildShardTest(absltest.TestCase):
    """The builder mirrors the production recursion assembly: sorted SP1-name
    chip order, main at emitted heights / prep at keygen heights, uncommitted
    host-encoded arrays, the fixed-width VK, width-matched chip resolution."""

    def _entry(self) -> wr.StageShapes:
        return wr.load_spec(_write_spec(TINY_SPEC))[0]

    def test_shapes_and_order(self) -> None:
        from zk_dtypes import koalabear_mont as BF

        shard = wr.build_recursion_shard(self._entry(), tiny_chips())
        traces = shard.main_trace_data.traces
        self.assertEqual(("BaseAlu", "Select"), traces.chip_order)
        self.assertEqual((4, 4), traces.per_chip["BaseAlu"].array.shape)
        self.assertEqual(4, traces.per_chip["BaseAlu"].num_real)
        self.assertEqual((8, 4), traces.per_chip["Select"].array.shape)
        self.assertEqual({"BaseAlu"}, set(shard.preprocessed_traces))
        self.assertEqual((8, 3), shard.preprocessed_traces["BaseAlu"].shape)
        self.assertEqual((8,), shard.main_trace_data.public_values.shape)
        self.assertEqual(BF, traces.per_chip["BaseAlu"].array.dtype)
        self.assertEqual(BF, shard.preprocessed_traces["BaseAlu"].dtype)

    def test_arrays_are_uncommitted(self) -> None:
        shard = wr.build_recursion_shard(self._entry(), tiny_chips())
        for arr in (
            shard.main_trace_data.traces.per_chip["BaseAlu"].array,
            shard.preprocessed_traces["BaseAlu"],
            shard.main_trace_data.public_values,
            shard.vk.preprocessed_commit,
        ):
            self.assertFalse(arr.committed)

    def test_vk_has_the_fixed_recursion_widths(self) -> None:
        vk = wr.build_recursion_shard(self._entry(), tiny_chips()).vk
        self.assertEqual((8,), vk.preprocessed_commit.shape)
        self.assertEqual((3,), vk.pc_start.shape)
        self.assertEqual((7,), vk.cum_sum_x.shape)
        self.assertEqual((7,), vk.cum_sum_y.shape)
        self.assertEqual(0, vk.enable_untrusted)

    def test_width_match_attaches_rw_chip_mismatch_stubs(self) -> None:
        chips = tiny_chips()
        shard = wr.build_recursion_shard(self._entry(), chips)
        got = shard.main_trace_data.chips
        # BaseAlu: 4 main + 3 prep == num_cols 7 -> the rw chip itself.
        self.assertIs(chips["base_alu"], got["BaseAlu"])
        # Select: widths match too.
        self.assertIs(chips["select"], got["Select"])
        # A width drift falls back to a constraint-less stub.
        drifted = wr.StageShapes(
            name="d",
            stage="combine",
            chips={
                "BaseAlu": wr.ChipShape(rows=4, cols=5, prep_rows=8, prep_cols=3),
                "Select": wr.ChipShape(rows=8, cols=4),
            },
            public_values_len=8,
        )
        stubbed = wr.build_recursion_shard(drifted, chips).main_trace_data.chips
        self.assertIsNot(chips["base_alu"], stubbed["BaseAlu"])
        self.assertEqual(5, stubbed["BaseAlu"].num_cols)

    def test_requires_some_prep(self) -> None:
        entry = wr.StageShapes(
            name="noprep",
            stage="combine",
            chips={"Select": wr.ChipShape(rows=8, cols=4)},
            public_values_len=8,
        )
        with self.assertRaisesRegex(ValueError, "preprocessed"):
            wr.build_recursion_shard(entry, tiny_chips())

    def test_production_chips_resolve_by_manifest_width(self) -> None:
        # The real recursion/v1 chips attach whenever the spec's main + prep
        # width equals the rw manifest num_cols — the production rule over the
        # production data (base_alu: 11 columns total).
        chips = wr.load_recursion_chips()
        self.assertEqual(set(wr.RECURSION_NAME_MAP.values()), set(chips))
        n = chips["base_alu"].num_cols
        entry = wr.StageShapes(
            name="p",
            stage="combine",
            chips={
                "BaseAlu": wr.ChipShape(rows=4, cols=n - 3, prep_rows=8, prep_cols=3)
            },
            public_values_len=8,
        )
        shard = wr.build_recursion_shard(entry, chips)
        self.assertIs(chips["base_alu"], shard.main_trace_data.chips["BaseAlu"])


class CliTest(absltest.TestCase):
    def test_selects_all_entries_by_default(self) -> None:
        spec = wr.load_spec(_write_spec(TINY_SPEC))
        self.assertEqual(spec, wrc.select_entries(spec, ""))

    def test_selects_named_entries(self) -> None:
        two = [
            TINY_SPEC[0],
            dict(TINY_SPEC[0], name="shrinky", stage="shrink"),
        ]
        spec = wr.load_spec(_write_spec(two))
        self.assertEqual(
            ["shrinky"], [e.name for e in wrc.select_entries(spec, "shrinky")]
        )

    def test_unknown_entry_raises(self) -> None:
        spec = wr.load_spec(_write_spec(TINY_SPEC))
        with self.assertRaisesRegex(ValueError, "unknown spec entries"):
            wrc.select_entries(spec, "typo")

    def test_worker_cmd_targets_the_recursion_worker(self) -> None:
        cmd = wrc.worker_cmd("/tmp/spec.json", "normalize")
        self.assertEqual(
            [
                sys.executable,
                "-m",
                "sp1_zorch.shard_prover.warm_recursion_worker",
                "/tmp/spec.json",
                "normalize",
            ],
            cmd,
        )

    def test_worker_env_sets_both_cache_spellings_and_real_backend(self) -> None:
        env = wrc.worker_env({"JAX_PLATFORMS": "cpu", "KEEP": "1"}, "/c")
        self.assertEqual("/c", env["FRX_COMPILATION_CACHE_DIR"])
        self.assertEqual("/c", env["JAX_COMPILATION_CACHE_DIR"])
        self.assertNotIn("JAX_PLATFORMS", env)
        self.assertEqual("1", env["KEEP"])


# The two identity arms share this driver body; only the interception import
# differs. Values are irrelevant to cache keys, so the zero-filled spec shard
# serves both.
_ARM_SCRIPT = textwrap.dedent(
    """\
    import os, sys

    os.environ["JAX_PLATFORMS"] = "cpu"
    mode, cache, spec_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "warm":
        # Production order: the intercept patches frx.jit BEFORE any chain
        # import binds a decorator.
        from sp1_zorch.shard_prover import warm_worker as ww
    import frx

    frx.config.update("jax_compilation_cache_dir", cache)
    frx.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    frx.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    from sp1_zorch.shard_prover import warm_recursion as wr
    from sp1_zorch.shard_prover.warm_recursion_test import TINY_MACHINE, tiny_chips

    entry = {e.name: e for e in wr.load_spec(spec_path)}["tiny"]
    shard = wr.build_recursion_shard(entry, tiny_chips())
    wr.prove_stage_chain(TINY_MACHINE, shard, gkr_pow_bits=1)
    if mode == "warm":
        n_failed = ww._drain_compiles()
        assert n_failed == 0, f"{n_failed} zone compiles failed"
        assert ww._stats["compiled"] > 0, "warm drove no zones"
        print(f"ZONES={ww._stats['compiled']}", flush=True)
    print("ARM_OK", flush=True)
    """
)


# Entry names the compile-only WRAPPER itself may mint beyond the chain's own
# set: the zeros standing in for zone outputs are eager ``fnp.zeros`` /
# dtype-view glue (``warm_worker._zone_zeros``), each an eager micro-jit. They
# appear here only because the test persists sub-second compiles
# (min_compile_time 0) to observe the tiny chain at all; a production cache
# (default threshold) never stores them, and a prove never asks for them.
_WARM_GLUE_ENTRY_PREFIXES = ("jit_broadcast_in_dim-", "jit_convert_element_type-")


class KeyIdentityTest(absltest.TestCase):
    """THE contract: the warm-driven recursion chain requests exactly the
    persistent-cache keys a really-executing recursion prove computes.

    Same filename = same cache key, and the key folds in the cache dir path
    itself (``xla_gpu_per_fusion_autotune_cache_dir`` rides in the hashed
    compile options), so both arms MUST share one cache dir — which is also
    the production contract: seed and prove mount the same dir.
    """

    def _run_arm(self, mode: str, cache: str, spec_path: str, script: str) -> str:
        env = dict(os.environ)
        # The arm must resolve sp1_zorch AND every dep exactly as this
        # process does (a divergent resolution is itself a key drift), so it
        # inherits this interpreter's sys.path order verbatim.
        root = str(Path(wr.__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join([root] + [p for p in sys.path if p])
        proc = subprocess.run(
            [sys.executable, script, mode, cache, spec_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        self.assertEqual(
            0,
            proc.returncode,
            f"{mode} arm failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        self.assertIn("ARM_OK", proc.stdout)
        return proc.stdout

    def test_warm_adds_zero_entries_to_a_real_proves_cache(self) -> None:
        tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR")))
        cache = tmp / "cache"
        cache.mkdir()
        spec_path = _write_spec(TINY_SPEC)
        script = tmp / "arm.py"
        script.write_text(_ARM_SCRIPT)

        def entries() -> set[str]:
            return {p.name for p in cache.rglob("*") if p.is_file()}

        self._run_arm("real", str(cache), spec_path, str(script))
        real = entries()
        self.assertGreater(len(real), 0, "real chain wrote no cache entries")
        # Guard against a vacuous pass: the real run must have persisted the
        # chain's own zone executables (sp1-zorch zone fns are underscored).
        self.assertTrue(
            any(n.startswith("jit__") for n in real),
            f"no chain-zone entries in the real cache: {sorted(real)[:10]}",
        )

        out = self._run_arm("warm", str(cache), spec_path, str(script))
        drift = {
            n for n in entries() - real if not n.startswith(_WARM_GLUE_ENTRY_PREFIXES)
        }
        self.assertEmpty(
            drift,
            "warm-driven recursion chain computed keys the real prove does "
            f"not: {sorted(drift)}",
        )
        zones = int(out.split("ZONES=")[1].split()[0])
        self.assertGreater(zones, 0)


if __name__ == "__main__":
    absltest.main()
