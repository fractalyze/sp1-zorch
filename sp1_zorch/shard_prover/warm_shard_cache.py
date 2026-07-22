# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Analyze a shard dump for optimal grouped compilation, then fill the
persistent compile cache in parallel.

The prove chain's heavy stages compile keyed on ``(chip set, class, static
tuples)`` — NOT the shard's runtime heights (they ride as a traced int32
vector). So every shard of one chip-set class shares one executable, and the
cache can be filled once per *distinct class* rather than once per shard.

Two phases:

  analyze  Scan ``--shard_dir`` (a dump holding ``shard*`` subdirs), derive
           each shard's zerocheck ``TotalCapClass``, LogUp-GKR ``GkrCapClass``,
           and jagged class, then group by chip set. Emits the group manifest
           (``--out_manifest``) and a compile plan: the distinct executables
           and a parallel partition.

  warm     (``--warm``) Compile-only fill: fan out ``warm_worker`` processes
           (one shard each) that drive the real prove chain but lower+compile
           every zone WITHOUT executing a kernel (``warm_worker`` intercepts
           ``frx.jit``), all writing the shared ``--cache_dir``. A real prove
           later hits every entry with zero recompiles. XLA still autotunes
           on-device during compile, so a worker peaks at ~2 GiB (46M area) to
           ~8 GiB (400M) — far below the ~20 GiB execute; concurrency is capped
           by ``--mem_budget_gib``. One shard per PROCESS because the cuda_async
           pool fragments and is never returned within a process (survives
           ``clear_caches``), so a long-lived worker OOMs after a couple of big
           shards.

Grouping policy (memory-aware, matches the single-process prove):
  * Zerocheck area_cap is pinned to the chip-set group MAX only when the
    group's area spread is tight (min/max > ``--group_area_ratio``); a wide
    group would price the small shards' zerocheck buffer at the big shard's
    area. One shared zerocheck compile per tight group (the #284-pole stage).
  * GKR stays each shard's OWN class — a group-max GKR class inflates the
    pyramid ~1.5x and OOMs the big shards on execute.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from absl import app, flags

from sp1_zorch.shard_prover.verify_prove_shard import (
    load_fixture_shard,
    shard_regions,
)
from sp1_zorch.zerocheck.jagged import TotalCapClass
from sp1_zorch.logup_gkr.circuit import GkrCapClass

_DUMP_DIR = flags.DEFINE_string(
    "dump_dir", None, "Dump directory holding shard* subdirs (or a comma list "
    "of shard dirs).", required=True)
_OUT_MANIFEST = flags.DEFINE_string(
    "out_manifest", None, "Write the per-shard group-class manifest here "
    "(the --group_manifest_json verify_prove_shard consumes).")
_GROUP_AREA_RATIO = flags.DEFINE_float(
    "group_area_ratio", 0.4, "Share one zerocheck compile across a chip set "
    "only when its min/max area_cap exceeds this (else each shard keeps its "
    "own area to avoid over-pricing small shards).")
_GKR_CLUSTER_RATIO = flags.DEFINE_float(
    "gkr_cluster_ratio", 1.15, "Cluster shards of one chip set to share a GKR "
    "compile when the cluster's per-chip-max class inflates no member's total "
    "GKR height beyond this. The merged class rides the shard's LogUp-GKR prove "
    "too, so the bound is EXECUTE memory (the pyramid scales with total height) "
    "— keep it tight (~1.1-1.2x) so a cluster still fits the card. 1.0 disables "
    "(each shard its own GKR class).")
_WARM = flags.DEFINE_bool(
    "warm", False, "After analysis, compile-only fill the cache (phase 2): fan "
    "out warm_worker subprocesses that lower+compile every zone WITHOUT "
    "executing (~2 GiB each), all writing --cache_dir.")
_CACHE_DIR = flags.DEFINE_string(
    "cache_dir", None, "Persistent compile cache dir the warm workers fill "
    "(and a real prove later hits). Required with --warm.")
_JOBS = flags.DEFINE_integer(
    "jobs", 8, "Max parallel warm workers. The effective count is capped by "
    "--mem_budget_gib / (biggest shard's compile-only peak) so concurrent "
    "workers fit the card; small-shard dumps use the full --jobs.")
_MEM_BUDGET_GIB = flags.DEFINE_float(
    "mem_budget_gib", 30.0, "Device-memory budget the concurrent workers must "
    "fit (sum of their est peaks).")
_WORKER_MEM_FRACTION = flags.DEFINE_float(
    "worker_mem_fraction", 0.5, "Per-worker cuda_async pool cap "
    "(XLA_PYTHON_CLIENT_MEM_FRACTION): releases autotune scratch between zones. "
    "Must exceed a shard's single-zone autotune need (~13.5 GiB at 400M area, "
    "so 0.5=16 GiB on a 32 GB card); with N concurrent workers keep N*frac<~1.")


def _shard_dirs() -> list[Path]:
    v = _DUMP_DIR.value
    if "," in v:
        return [Path(p) for p in v.split(",")]
    root = Path(v)
    subs = sorted(
        (p for p in root.glob("shard*") if p.is_dir()),
        key=lambda p: int(p.name.replace("shard", "")),
    )
    return subs or [root]


def _shard_class(sd: Path) -> dict:
    """Derive one shard's (chip set, zerocheck, GKR, jagged) class — the compile
    keys, no GPU. Mirrors verify_prove_shard._verify_shard's class block."""
    shard = load_fixture_shard(sd)
    main_region, prep_region = shard_regions(shard)
    main = shard.main_trace_data
    order = list(main.traces.chip_order)
    num_reals = [int(main.traces.per_chip[n].num_real) for n in order]
    prep_w = (
        {n: int(prep_region.chip_widths[k])
         for k, n in enumerate(prep_region.chip_names)}
        if prep_region is not None else {}
    )
    chip_cols = [int(main_region.chip_widths[i]) + prep_w.get(name, 0)
                 for i, name in enumerate(order)]
    zc = TotalCapClass.from_heights(num_reals, chip_cols)
    gkr = GkrCapClass.from_heights([int(h) for h in main_region.chip_heights])
    regions_jc = [r for r in (prep_region, main_region) if r is not None]
    jl = sum(sum(int(c) for c in r.column_counts) for r in regions_jc)
    jks = [int(r.dense.shape[0]) >> int(r.log_stacking_height) for r in regions_jc]
    area = sum(int(r.dense.shape[0]) for r in regions_jc)
    return {
        "order": order,
        "area_cap": int(zc.area_cap),
        "gkr_heights": {n: int(h) for n, h in zip(order, gkr.chip_heights)},
        "jagged": {"L": jl, "n_d": (area - 1).bit_length() + 1, "K": jks,
                   "rlc_bits": max(sum(jks) - 1, 0).bit_length()},
    }


def _analyze(dirs: list[Path]) -> tuple[dict, dict]:
    """Return (per-shard classes, chip-set groups)."""
    classes = {}
    for sd in dirs:
        classes[sd.name] = _shard_class(sd)
        c = classes[sd.name]
        print(f"{sd.name}: chips={len(c['order'])} area_cap={c['area_cap']} "
              f"K={c['jagged']['K']} L={c['jagged']['L']}", flush=True)
    groups = defaultdict(list)
    for name, c in classes.items():
        groups[tuple(c["order"])].append(name)
    return classes, groups


def _cluster_gkr(order: tuple, shards: list, classes: dict) -> list[list[str]]:
    """Bin shards of one chip set into GKR clusters: a cluster's per-chip-max
    (merged) class must inflate no member's total GKR height beyond
    --gkr_cluster_ratio, so the shared class still fits each member's execute
    memory. Greedy, largest-first (a jumbo shard opens its own cluster and only
    admits ones close to it)."""
    ratio = _GKR_CLUSTER_RATIO.value
    tot = {s: sum(classes[s]["gkr_heights"].values()) for s in shards}
    if ratio <= 1.0:
        return [[s] for s in shards]

    def merged_total(members: list) -> int:
        return sum(max(classes[s]["gkr_heights"][n] for s in members)
                   for n in order)

    clusters: list[list[str]] = []
    for s in sorted(shards, key=lambda s: -tot[s]):
        for cl in clusters:
            m = merged_total(cl + [s])
            if all(m <= ratio * tot[x] for x in cl + [s]):
                cl.append(s)
                break
        else:
            clusters.append([s])
    return clusters


def _plan(classes: dict, groups: dict) -> dict:
    """Assign each shard its group + cluster classes; count distinct compiles."""
    ratio = _GROUP_AREA_RATIO.value
    manifest = {}
    plan = []
    for order, shards in groups.items():
        areas = [classes[s]["area_cap"] for s in shards]
        tight = len(shards) > 1 and (min(areas) / max(areas)) > ratio
        area_pin = max(areas) if tight else None
        # GKR: cluster similar shards to share a compile; each cluster's merged
        # per-chip-max class rides every member's prove (fits execute memory).
        gkr_clusters = _cluster_gkr(order, shards, classes)
        for cl in gkr_clusters:
            gmax = {n: max(classes[s]["gkr_heights"][n] for s in cl) for n in order}
            for s in cl:
                manifest.setdefault(s, {})["gkr"] = gmax
        zc_variants = 1 if tight else len({a for a in areas})
        for s in shards:
            manifest.setdefault(s, {})["area_cap"] = (
                area_pin if tight else classes[s]["area_cap"])
        plan.append({
            "chips": len(order), "shards": sorted(shards, key=_snum),
            "tight_zerocheck_group": tight, "area_pin": area_pin,
            "distinct_zerocheck_compiles": zc_variants,
            "distinct_gkr_compiles": len(gkr_clusters),
        })
    return {"manifest": manifest, "plan": plan}


def _snum(s: str) -> int:
    return int(s.replace("shard", ""))


def main(argv):
    del argv
    dirs = _shard_dirs()
    print(f"=== analyzing {len(dirs)} shards ===", flush=True)
    classes, groups = _analyze(dirs)
    out = _plan(classes, groups)
    print(f"\n=== {len(groups)} chip-set groups (compile boundary) ===")
    tot_zc = tot_gkr = 0
    for g in sorted(out["plan"], key=lambda g: -len(g["shards"])):
        tot_zc += g["distinct_zerocheck_compiles"]
        tot_gkr += g["distinct_gkr_compiles"]
        tag = "GROUP" if g["tight_zerocheck_group"] else "own"
        print(f"  {tag:>5} {g['chips']:>2}ch {len(g['shards']):>2}sh "
              f"{[_snum(s) for s in g['shards']]}: "
              f"zc_compiles={g['distinct_zerocheck_compiles']} "
              f"gkr_compiles={g['distinct_gkr_compiles']} area_pin={g['area_pin']}")
    print(f"\ndistinct compiles to fill: {tot_zc} zerocheck + {tot_gkr} GKR "
          f"(+ per-chipset trace/open zones) vs {len(classes)} shards naive")
    manifest_path = _OUT_MANIFEST.value
    if _WARM.value and manifest_path is None:
        # The warm needs the manifest on disk for the workers; default beside
        # the cache so grouped-zerocheck compiles match the real prove.
        manifest_path = str(Path(_CACHE_DIR.value) / "group_manifest.json")
    if manifest_path:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(out["manifest"]))
        print(f"wrote manifest -> {manifest_path}")
    if _WARM.value:
        _warm(dirs, classes, manifest_path)


def _est_peak_gib(area_cap: int) -> float:
    """Conservative compile-only device peak with the pool-release cap (autotune
    ON): the cap keeps a big shard's scratch from accumulating, so the peak
    holds at the single-zone max — ~2 GiB at 46M, ~11.5 GiB at 402M.
    Overestimate a little so the peak-aware scheduler never packs into an OOM;
    two 400M shards (~11.5 GiB each measured) then run concurrently on a 32 GB
    card, small ones pack further."""
    return 3.0 + area_cap / 30e6  # 400M -> ~16.4 GiB (fits its ~13.5 GiB single-zone autotune)


def _warm(dirs: list[Path], classes: dict, manifest_path: str) -> None:
    if not _CACHE_DIR.value:
        raise ValueError("--warm requires --cache_dir")
    cache = _CACHE_DIR.value
    Path(cache).mkdir(parents=True, exist_ok=True)
    # Workers run concurrently and each peaks at its LARGEST shard, so cap the
    # worker count by the memory budget: n * max_peak <= budget. Then LPT-pack
    # (largest first into the least-loaded worker) to balance compile time;
    # same-class compiles dedup in the shared cache.
    # ONE shard per worker PROCESS: the cuda_async pool fragments and is never
    # returned within a process (survives clear_caches), so a long-lived worker
    # OOMs after a couple of big shards. A fresh process per shard resets the
    # pool. Cap concurrency by the memory budget (each peaks at its shard).
    # Biggest first, so a heavy shard grabs the card early and runs ~solo while
    # small ones pack around it. Peak-aware: launch a worker only if the sum of
    # running peaks (+ its own) still fits the budget — so a ~24 GiB 400M shard
    # never coexists with another big one, but several small shards do.
    shards = sorted((str(sd) for sd in dirs),
                    key=lambda s: -classes[Path(s).name]["area_cap"])
    peaks = {s: _est_peak_gib(classes[Path(s).name]["area_cap"]) for s in shards}
    budget = _MEM_BUDGET_GIB.value
    # Cap each worker's cuda_async pool so freed autotune scratch is RELEASED
    # between the ~95 zone compiles instead of accumulating (autotune stays ON,
    # so the warmed executable matches a normal prove and runs fast). Without
    # this a 400M shard's scratch piles up past 32 GiB; with it the peak holds
    # at ~11.5 GiB. cuda_async allocator required.
    env = dict(os.environ,
               FRX_COMPILATION_CACHE_DIR=cache, JAX_COMPILATION_CACHE_DIR=cache,
               XLA_PYTHON_CLIENT_ALLOCATOR="cuda_async",
               XLA_PYTHON_CLIENT_MEM_FRACTION=str(_WORKER_MEM_FRACTION.value))
    # Analysis (this process) runs CPU-only via JAX_PLATFORMS=cpu so it grabs no
    # device memory; workers need the GPU, so drop the override for them.
    env.pop("JAX_PLATFORMS", None)
    print(f"=== warming {len(dirs)} shards, peak-aware pool (<= {budget:.0f} GiB, "
          f"<= {_JOBS.value} procs); est peaks "
          f"{peaks[shards[0]]:.0f}..{peaks[shards[-1]]:.0f} GiB ===", flush=True)
    pending = list(shards)
    running: dict = {}  # Popen -> (shard, peak)
    ok = fail = 0
    while pending or running:
        launched = True
        while launched and pending:
            launched = False
            used = sum(pk for _, pk in running.values())
            s = pending[0]
            # Always allow one worker even if a lone big shard exceeds budget.
            if len(running) < _JOBS.value and (
                    not running or used + peaks[s] <= budget):
                pending.pop(0)
                p = subprocess.Popen(
                    [sys.executable, "-m", "sp1_zorch.shard_prover.warm_worker",
                     s, manifest_path or ""], env=env)
                running[p] = (s, peaks[s])
                launched = True
        for p in list(running):
            if p.poll() is not None:
                s, _ = running.pop(p)
                if p.returncode == 0:
                    ok += 1
                else:
                    fail += 1
                    print(f"  warm worker for {Path(s).name} exited "
                          f"{p.returncode}", flush=True)
        if running:
            time.sleep(2)
    print(f"=== warm done: {ok}/{ok + fail} shards ok; "
          f"cache entries: {sum(1 for _ in Path(cache).rglob('*') if _.is_file())} ===",
          flush=True)


if __name__ == "__main__":
    app.run(main)
