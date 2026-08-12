# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Warm-worker commitment alignment, on CPU.

The persistent-cache key carries each parameter's committed/uncommitted state
(``allow_spmd_sharding_propagation_to_parameters``: committed arg = sharding
specified = ``False``). In the real prove, commitment originates at eager
``device_put``s (the transcript host-FS round trips) and jax propagates it
zone to zone — a jit's outputs are committed exactly when an arg arrived
committed. These tests pin the compile-only wrapper to that rule: zone zeros
carry the placement the real results would, a warm chain HITS the entries a
really-executing chain wrote, and the deviceless translation keeps the keying
without a target device. The mechanism is device-generic, so CPU exercises
the same code path a CUDA warm runs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import itertools  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402

import frx  # noqa: E402
import frx.numpy as fnp  # noqa: E402
import jax  # noqa: E402
from jax.sharding import SingleDeviceSharding  # noqa: E402

from sp1_zorch.shard_prover import warm_worker as ww  # noqa: E402


def _entries(cache_dir: str) -> int:
    return sum(1 for p in Path(cache_dir).rglob("*") if p.is_file())


def _cpu() -> jax.Device:
    return jax.local_devices(backend="cpu")[0]


# Distinct bodies so each zone keys its own persistent-cache module. The
# ``tag`` constant makes a chain's modules unique to one test: identical
# modules would be served by process-global executable caches across tests
# and never reach the per-test cache dir.
def _make_zones(tag: float) -> tuple[Callable[..., Any], ...]:
    def zone1(a: Any) -> Any:
        return a * tag

    def zone2(prior: Any, host: Any) -> Any:
        return prior + host * tag

    def zone3(a: Any) -> Any:
        return a - tag

    def zone4(uncommitted_prior: Any, committed_prior: Any) -> Any:
        return uncommitted_prior * committed_prior + tag

    return zone1, zone2, zone3, zone4


_zone1, _zone2, _zone3, _zone4 = _make_zones(2.0)

# Per-test zone tags for CacheKeyAlignmentTest; 2.0 is the placement tests'.
_TAGS = itertools.count(3)


class ZoneZerosPlacementTest(absltest.TestCase):
    def test_host_args_yield_uncommitted_zeros(self) -> None:
        out = ww._compile_only_jit(_zone1)(np.ones((3,), np.float32))
        self.assertEqual(ww._drain_compiles(), 0)
        self.assertFalse(out.committed)
        np.testing.assert_array_equal(np.asarray(out), np.zeros((3,), np.float32))

    def test_committed_arg_yields_committed_zeros(self) -> None:
        seed = jax.device_put(fnp.ones((3,), np.float32), _cpu())
        out = ww._compile_only_jit(_zone2)(seed, np.float32(1.0))
        self.assertEqual(ww._drain_compiles(), 0)
        self.assertTrue(out.committed)
        # jax carries commitment through eager glue, zone to zone.
        self.assertTrue((out * 2 + 1).committed)


class CacheKeyAlignmentTest(absltest.TestCase):
    """The warm chain must hit the persistent-cache entries a really-executing
    chain writes — committed and uncommitted lineages both."""

    def setUp(self) -> None:
        super().setUp()
        # The persistent cache keeps an in-memory layer; drop it so each
        # test's fresh cache dir sees real writes, not in-memory hits.
        from frx._src import compilation_cache as _cc

        _cc.reset_cache()
        self._tag = float(next(_TAGS))
        self._cache = tempfile.mkdtemp()
        self._prev = (
            frx.config.jax_compilation_cache_dir,
            frx.config.jax_persistent_cache_min_compile_time_secs,
            frx.config.jax_persistent_cache_min_entry_size_bytes,
        )
        frx.config.update("jax_compilation_cache_dir", self._cache)
        frx.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        frx.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

    def tearDown(self) -> None:
        frx.config.update("jax_compilation_cache_dir", self._prev[0])
        frx.config.update("jax_persistent_cache_min_compile_time_secs", self._prev[1])
        frx.config.update("jax_persistent_cache_min_entry_size_bytes", self._prev[2])
        super().tearDown()

    def _chain(self, jit_fn: Callable[..., Any]) -> tuple[Any, Any, Any]:
        """The four-zone chain under ``jit_fn``: a committed lineage (a
        ``device_put`` seed, the prove's host-FS round trip), an uncommitted
        host lineage, and a zone mixing both. Zones are unique per test
        (``_make_zones``); both drivers of one test share them."""
        z1, z2, z3, z4 = _make_zones(self._tag)
        seed = jax.device_put(fnp.ones((4,), np.float32), _cpu())
        r1 = jit_fn(z1)(seed)
        r2 = jit_fn(z2)(r1, np.float32(3.0))
        u1 = jit_fn(z3)(np.ones((4,), np.float32))
        jit_fn(z4)(u1, r2)
        return r1, r2, u1

    def test_warm_chain_hits_staged_entries(self) -> None:
        r1, r2, u1 = self._chain(ww._real_jit)
        self.assertTrue(r1.committed)
        self.assertTrue(r2.committed)
        self.assertFalse(u1.committed)
        n_staged = _entries(self._cache)
        self.assertGreaterEqual(n_staged, 4)

        w1, w2, wu1 = self._chain(ww._compile_only_jit)
        self.assertEqual(ww._drain_compiles(), 0)
        self.assertEqual(w1.committed, r1.committed)
        self.assertEqual(w2.committed, r2.committed)
        self.assertEqual(wu1.committed, u1.committed)
        self.assertEqual(_entries(self._cache), n_staged)

    def test_committedness_is_a_different_key(self) -> None:
        # Guards the hit test against passing vacuously: were the flag vector
        # ever dropped from the cache key, this starts failing and the
        # commitment alignment machinery can go. Lowers THIS test's own zone2
        # — byte-identical body to the chain's compile — so the only key
        # delta left is the per-parameter commitment flag (chain: committed
        # r1; here: uncommitted zeros).
        self._chain(ww._real_jit)
        n_staged = _entries(self._cache)
        _, z2, _, _ = _make_zones(self._tag)
        lowered = ww._real_jit(z2).lower(fnp.zeros((4,), np.float32), np.float32(3.0))
        lowered.compile()
        self.assertGreater(_entries(self._cache), n_staged)

    def test_deviceless_translation_keys_match_committed_args(self) -> None:
        # Stand in for the compile-only topology device with the CPU device:
        # the translation (committed leaf -> spec sharded on _topo_dev) is
        # target-agnostic, and on CPU the resulting key must equal the
        # committed-concrete-arg key the really-executing chain produced.
        self._chain(ww._real_jit)
        n_staged = _entries(self._cache)
        ww._topo_dev = _cpu()
        try:
            self._chain(ww._compile_only_jit)
            self.assertEqual(ww._drain_compiles(), 0)
        finally:
            ww._topo_dev = None
        self.assertEqual(_entries(self._cache), n_staged)


class ChainRcTest(absltest.TestCase):
    """The worker's exit code must carry the harness verdict: absl's app.run
    always leaves via SystemExit, and the staged sweep exits with a
    failed-shards payload when any shard's chain died."""

    def test_failure_payload_is_nonzero(self) -> None:
        self.assertEqual(ww._chain_rc("failed shards: ['shard3']"), 1)
        self.assertEqual(ww._chain_rc(1), 1)
        self.assertEqual(ww._chain_rc(2), 1)

    def test_clean_exit_is_zero(self) -> None:
        self.assertEqual(ww._chain_rc(0), 0)
        self.assertEqual(ww._chain_rc(None), 0)


class LoweringArgsTest(absltest.TestCase):
    def test_no_translation_without_topology(self) -> None:
        committed = jax.device_put(fnp.ones((2,), np.float32), _cpu())
        args, kwargs = ww._lowering_args((committed,), {"k": committed})
        self.assertIs(args[0], committed)
        self.assertIs(kwargs["k"], committed)

    def test_retargets_committed_leaves_only(self) -> None:
        cpu = _cpu()
        committed = jax.device_put(fnp.ones((2, 2), np.float32), cpu)
        uncommitted = fnp.ones((2,), np.float32)
        ww._topo_dev = cpu
        try:
            args, kwargs = ww._lowering_args(
                (committed, uncommitted, 5), {"k": committed}
            )
        finally:
            ww._topo_dev = None
        self.assertIsInstance(args[0], jax.ShapeDtypeStruct)
        self.assertEqual(args[0].shape, (2, 2))
        self.assertEqual(args[0].sharding, SingleDeviceSharding(cpu))
        self.assertIs(args[1], uncommitted)
        self.assertEqual(args[2], 5)
        self.assertIsInstance(kwargs["k"], jax.ShapeDtypeStruct)


if __name__ == "__main__":
    absltest.main()
