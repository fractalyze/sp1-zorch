# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Analyze a shard dump for optimal grouped compilation, then fill the
persistent compile cache in parallel.

The prove chain's heavy stages compile keyed on ``(chip set, class, static
tuples)`` — NOT the shard's runtime heights (they ride as a traced int32
vector). So every shard of one chip-set class shares one executable, and the
cache can be filled once per *distinct class* rather than once per shard.

Two steps:

  analyze  Scan ``--dump_dir`` (a dump holding ``shard*`` subdirs), derive
           each shard's zerocheck ``TotalCapClass``, LogUp-GKR ``GkrCapClass``,
           and jagged class, then group by chip set. Emits the group manifest
           (``--out_manifest``) and a compile plan: the distinct executables
           and the dispatch order.

  warm     (``--warm``) Compile-only fill: fan out ``warm_worker`` processes
           (one shard each) that drive the real prove chain but lower+compile
           every zone WITHOUT executing a kernel (``warm_worker`` intercepts
           ``frx.jit``), all writing the shared ``--cache_dir``. A real prove
           of a WARMED shard later hits every entry with zero recompiles.
           ``--warm_per_class`` (default) warms only a greedy cover of the
           dump's class-keyed compile keys — one representative shard per
           distinct class — since every other shard of a class would re-trace
           and cache-hit those entries. One zone is per-shard, not
           class-keyed: ``_jagged_pack_jit`` keys on each region's exact
           row-count tuples, so a non-selected shard's first prove pays that
           one compile cold — a cheap concat/pad graph, not a class compile
           (see ``_COVER_KINDS``).
           ``--nowarm_per_class`` warms every shard and fills it too. XLA still
           autotunes on-device during compile, so a worker peaks at ~2 GiB
           (46M area) to ~18 GiB (400M, two compile threads) — below the
           ~29 GiB execute; concurrency is capped by ``--mem_budget_gib``,
           one shard per worker process.

Grouping policy (memory-aware, matches the single-process prove):
  * Zerocheck area_cap is pinned to the chip-set group MAX only when the
    group's area spread is tight (min/max > ``--group_area_ratio``); a wide
    group would price the small shards' zerocheck buffer at the big shard's
    area. One shared zerocheck compile per tight group (the #284-pole stage).
  * GKR: one pinned GkrCapClass per chip-set group (heights = per-chip max,
    slot_cap = group max) — the pyramid keys on slot_cap so the pin does not
    inflate it, and first-layer inflation is transient.

Proving against the cache:
  * The prove must run with the SAME ``XLA_FLAGS`` as the warm — compilation
    flags are part of the persistent-cache key (only dump flags are excluded).
  * One shard per prove PROCESS at big (~400M) areas. A batched
    ``--shard_dir=a,b,...`` prove would amortize beautifully (the second
    shard's trace is ~0.1 s vs minutes — shared executables stay loaded and
    device-resident) but reliably OOMs on the second shard's first ~11 GiB
    alloc: bytes_in_use is small after shard one, yet the cuda_async pool
    stays reserved near the ~29 GiB execute peak, and no
    XLA_PYTHON_CLIENT_MEM_FRACTION release threshold tried (1.0/0.6/0.15)
    releases it. Root cause not yet isolated.
  * Host-RAM budget: one ptxas on a big constraint cone peaks at ~28 GiB RSS
    (secp256k1 cones far worse, fractalyze/xla#312), so
    ``--xla_gpu_force_compilation_parallelism`` multiplies into host OOMs on
    cold fills — leave it unset and let cross-zone (WARM_COMPILE_THREADS) and
    cross-worker concurrency carry the parallelism.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from absl import app, flags

from sp1_zorch.logup_gkr.circuit import GkrCapClass, build_gkr_chips
from sp1_zorch.shard_prover.compile_classes import (
    jagged_class,
    resolve_classes,
    tight_classes,
)
from sp1_zorch.shard_prover.fixture_loader import load_fixture_shard
from sp1_zorch.shard_prover.replay import shard_regions
from sp1_zorch.zerocheck.jagged import TotalCapClass

_DUMP_DIR = flags.DEFINE_string(
    "dump_dir",
    None,
    "Dump directory holding shard* subdirs (or a comma list " "of shard dirs).",
    required=True,
)
_OUT_MANIFEST = flags.DEFINE_string(
    "out_manifest",
    None,
    "Write the per-shard group-class manifest here "
    "(the --group_manifest_json //tools:staged_prove_shard consumes).",
)
_GROUP_AREA_RATIO = flags.DEFINE_float(
    "group_area_ratio",
    0.4,
    "Share one zerocheck compile across a chip set "
    "only when its min/max area_cap exceeds this (else each shard keeps its "
    "own area to avoid over-pricing small shards).",
)
_WARM = flags.DEFINE_bool(
    "warm",
    False,
    "After analysis, compile-only fill the cache (phase 2): fan "
    "out warm_worker subprocesses that lower+compile every zone WITHOUT "
    "executing (~2 GiB each), all writing --cache_dir.",
)
_CACHE_DIR = flags.DEFINE_string(
    "cache_dir",
    None,
    "Persistent compile cache dir the warm workers fill "
    "(and a real prove later hits). Required with --warm.",
)
_JOBS = flags.DEFINE_integer(
    "jobs",
    8,
    "Max parallel warm workers per GPU. The effective count is capped by "
    "--mem_budget_gib / (biggest shard's compile-only peak) so concurrent "
    "workers fit the card; small-shard dumps use the full --jobs.",
)
_MEM_BUDGET_GIB = flags.DEFINE_float(
    "mem_budget_gib",
    30.0,
    "Device-memory budget the concurrent workers must " "fit (sum of their est peaks).",
)
_WORKER_MEM_FRACTION = flags.DEFINE_float(
    "worker_mem_fraction",
    0.5,
    "Per-worker cuda_async pool cap "
    "(XLA_PYTHON_CLIENT_MEM_FRACTION): releases autotune scratch between zones. "
    "Must exceed a shard's single-zone autotune need (~13.5 GiB at 400M area, "
    "so 0.5=16 GiB on a 32 GB card); with N concurrent workers keep N*frac<~1.",
)
_GPUS = flags.DEFINE_string(
    "gpus",
    "",
    "Comma-separated GPU ids to spread the warm across (e.g. "
    "'0,1'). Chip-set groups are dispatched dynamically, costliest first — "
    "whichever GPU drains its queue takes the next group; a whole group stays "
    "on one GPU (concurrent same-class compiles would race the cache). Empty: "
    "one pool on the inherited CUDA_VISIBLE_DEVICES.",
)
_WARM_PER_CLASS = flags.DEFINE_bool(
    "warm_per_class",
    True,
    "Warm one representative shard per distinct compile class instead of "
    "every shard: a class's remaining shards would only re-trace and "
    "cache-hit every class-keyed zone. Their one per-shard zone (the jagged "
    "pack, keyed on exact row counts) stays a cheap cold compile on first "
    "prove. False warms all shards, filling that zone too.",
)
_FRONT_SHARDS = flags.DEFINE_string(
    "front_shards",
    "",
    "Comma-separated shard dir names moved to the head of the dispatch order "
    "(named shards first, in the given order; the rest keep the area-desc "
    "order). On a single GPU the group cost queue is not consulted and "
    "area-desc fires the keccak class (~93 zones, the wall pole) late; "
    "front-loading it overlaps its long compile with everything else.",
)
_PEAK_OVERRIDES_JSON = flags.DEFINE_string(
    "peak_overrides_json",
    "",
    "JSON file mapping shard dir name -> measured device peak GiB, applied "
    "verbatim in place of the area-formula estimate for the listed shards — "
    "the operator owns any safety margin. Pool-peak readings miss "
    "compile/autotune transients (jagged rematerialization floors ~13 GiB "
    "at 400M area), so an override below the real transient can over-pack "
    "and kill worker chains; prefer conservative values.",
)


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
    """Derive one shard's (chip set, zerocheck, GKR, jagged) class — the
    compile keys, no GPU — via the shared ``compile_classes`` math the staged
    prove harness prints as its class census lines."""
    shard = load_fixture_shard(sd)
    main_region, prep_region = shard_regions(shard)
    main = shard.main_trace_data
    order = list(main.traces.chip_order)
    num_reals = [int(main.traces.per_chip[n].num_real) for n in order]
    gkr_chips = build_gkr_chips(main.chips, order)
    zc, gkr, slot_bound = tight_classes(
        main_region, prep_region, order, num_reals, gkr_chips
    )
    return {
        "order": order,
        "area_cap": int(zc.area_cap),
        "gkr_heights": {n: int(h) for n, h in zip(order, gkr.chip_heights)},
        "gkr_slot_bound": int(slot_bound),
        "jagged": jagged_class(main_region, prep_region),
    }


def _analyze(dirs: list[Path]) -> tuple[dict, dict]:
    """Return (per-shard classes, chip-set groups)."""
    names = [sd.name for sd in dirs]
    if len(set(names)) != len(names):
        raise ValueError("--dump_dir entries must have unique shard basenames")
    classes = {}
    for sd in dirs:
        classes[sd.name] = _shard_class(sd)
        c = classes[sd.name]
        print(
            f"{sd.name}: chips={len(c['order'])} area_cap={c['area_cap']} "
            f"K={c['jagged']['K']} L={c['jagged']['L']}",
            flush=True,
        )
    groups = defaultdict(list)
    for name, c in classes.items():
        groups[tuple(c["order"])].append(name)
    return classes, groups


def _plan(classes: dict, groups: dict) -> dict:
    """Assign each shard its group + cluster classes; count distinct compiles."""
    ratio = _GROUP_AREA_RATIO.value
    manifest: dict[str, Any] = {}
    plan = []
    for order, shards in groups.items():
        areas = [classes[s]["area_cap"] for s in shards]
        tight = len(shards) > 1 and (min(areas) / max(areas)) > ratio
        area_pin = max(areas) if tight else None
        # GKR: one class per chip-set group. The pyramid keys on slot_cap (pin
        # the group-max tight bound — heights don't inflate it), and the
        # heights-keyed first-layer/open zones tolerate the per-chip-max pin
        # (their inflation is transient) — so the whole group shares one
        # compile set (GkrCapClass, sp1-zorch#290).
        gmax = {n: max(classes[s]["gkr_heights"][n] for s in shards) for n in order}
        slot_pin = max(classes[s]["gkr_slot_bound"] for s in shards)
        zc_variants = 1 if tight else len({a for a in areas})
        for s in shards:
            manifest.setdefault(s, {})["gkr"] = gmax
            manifest[s]["gkr_slot_cap"] = slot_pin
            manifest[s]["area_cap"] = area_pin if tight else classes[s]["area_cap"]
        plan.append(
            {
                "chips": len(order),
                "shards": sorted(shards, key=_snum),
                "tight_zerocheck_group": tight,
                "area_pin": area_pin,
                "distinct_zerocheck_compiles": zc_variants,
                "distinct_gkr_compiles": 1,
            }
        )
    return {"manifest": manifest, "plan": plan}


def _snum(s: str) -> int:
    return int(s.replace("shard", ""))


# The CLASS-KEYED compile-key kinds one warmed shard fills, mirroring the
# prove chain: zerocheck rounds key on (chip set, TotalCapClass.area_cap); the
# LogUp-GKR zones on (chip set, GkrCapClass heights + slot_cap); the
# trace/open zones on the chip set; the class-keyed jagged zones (eval/open)
# on the derived (L, n_d, K) class — keyed per chip set here, a conservative
# refinement of the cache's own key. The chipset kind alone does not
# distinguish dense region shapes — the jagged kind's (L, n_d, K) is what
# keeps a same-chip-set different-shape rider off a cold commit/open compile,
# so it must not be coarsened.
#
# Deliberately absent: the jagged PACK zone. ``_jagged_pack_jit`` keys on each
# region's exact row-count tuple (``rc_rounds``/``cc_rounds`` — the per-chip
# heights themselves), a per-shard static that no subset short of every shard
# can cover. It sits outside the cover contract: under ``--warm_per_class`` a
# non-selected shard's first prove pays that one compile cold — a cheap
# concat/pad graph, not a class compile.
_COVER_KINDS = ("zerocheck", "gkr", "chipset", "jagged")


def compile_cover_keys(name: str, classes: dict, manifest: dict) -> dict:
    """Shard ``name``'s effective class-keyed compile keys (``_COVER_KINDS``)
    under ``manifest`` — resolved by ``compile_classes.resolve_classes``, the
    SAME field-by-field resolution the staged prove harness applies to its
    ``--group_manifest_json``, so the cover fills exactly the classes a
    manifest-driven prove requests (a partial entry pins only what it names)."""
    c = classes[name]
    order = tuple(c["order"])
    tc, gkr = resolve_classes(
        order,
        TotalCapClass(area_cap=int(c["area_cap"])),
        GkrCapClass(
            tuple(int(c["gkr_heights"][n]) for n in order), int(c["gkr_slot_bound"])
        ),
        manifest_entry=manifest.get(name),
    )
    slot = gkr.slot_cap
    j = c["jagged"]
    return {
        "zerocheck": (order, int(tc.area_cap)),
        "gkr": (order, gkr.chip_heights, None if slot is None else int(slot)),
        "chipset": order,
        "jagged": (order, int(j["L"]), int(j["n_d"]), tuple(int(k) for k in j["K"])),
    }


def select_warm_shards(
    classes: dict, manifest: dict, per_class: bool = True
) -> list[str]:
    """The shard subset ``--warm`` compiles: every shard, or (``per_class``)
    a greedy cover of the dump's class-keyed compile keys — one
    representative per distinct effective zerocheck class (its area-max
    carrier, so a manifest-less lone run of the representative pins the same
    class; the grouping policy itself rides in ``manifest``), extended by a
    carrier for any GKR / chip-set / jagged key the zerocheck picks leave
    uncovered. The per-shard jagged pack zone is outside the cover (see
    ``_COVER_KINDS``)."""
    if not per_class:
        return sorted(classes, key=_snum)
    keys = {n: compile_cover_keys(n, classes, manifest) for n in classes}
    # Area-max first: each class's chosen carrier is its biggest member, and
    # cover-extension picks are deterministic.
    by_area = sorted(classes, key=lambda n: (-classes[n]["area_cap"], _snum(n)))
    selected: list[str] = []
    covered: set[tuple] = set()  # (kind, key) pairs the selection fills
    for kind in _COVER_KINDS:
        for name in by_area:
            if (kind, keys[name][kind]) not in covered:
                selected.append(name)
                covered.update((k, keys[name][k]) for k in _COVER_KINDS)
    check_warm_cover(selected, classes, manifest)
    return sorted(selected, key=_snum)


def check_warm_cover(selected: Sequence[str], classes: dict, manifest: dict) -> None:
    """RAISE unless ``selected`` compiles every class-keyed key
    (``_COVER_KINDS``) an all-shards warm would — the cover contract, checked
    rather than hoped for. The per-shard jagged pack zone is outside the
    contract by construction: its key is invisible to ``classes`` and only an
    all-shards warm fills it."""
    keys = {n: compile_cover_keys(n, classes, manifest) for n in classes}
    covered = {(k, keys[n][k]) for n in selected for k in _COVER_KINDS}
    missing = [
        (kind, n, keys[n][kind])
        for n in sorted(classes, key=_snum)
        for kind in _COVER_KINDS
        if (kind, keys[n][kind]) not in covered
    ]
    if missing:
        raise ValueError(
            "per-class warm selection misses compile keys the all-shards "
            f"warm fills: {missing}"
        )


def _selection_banner(selected: Sequence[str], total: int) -> str:
    return f"warming {len(selected)} of {total} shards (compile-key cover): " + str(
        list(selected)
    )


def main(argv: Sequence[str]) -> None:
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
        print(
            f"  {tag:>5} {g['chips']:>2}ch {len(g['shards']):>2}sh "
            f"{[_snum(s) for s in g['shards']]}: "
            f"zc_compiles={g['distinct_zerocheck_compiles']} "
            f"gkr_compiles={g['distinct_gkr_compiles']} area_pin={g['area_pin']}"
        )
    print(
        f"\ndistinct compiles to fill: {tot_zc} zerocheck + {tot_gkr} GKR "
        f"(+ per-chipset trace/open zones) vs {len(classes)} shards naive"
    )
    manifest_path = _OUT_MANIFEST.value
    if _WARM.value and not _CACHE_DIR.value:
        raise ValueError("--warm requires --cache_dir")
    if _WARM.value and manifest_path is None:
        # The warm needs the manifest on disk for the workers; default beside
        # the cache so grouped-zerocheck compiles match the real prove.
        manifest_path = str(Path(_CACHE_DIR.value) / "group_manifest.json")
    if manifest_path:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(manifest_path).write_text(json.dumps(out["manifest"]))
        print(f"wrote manifest -> {manifest_path}")
    if _WARM.value:
        selected = select_warm_shards(
            classes, out["manifest"], per_class=_WARM_PER_CLASS.value
        )
        if _WARM_PER_CLASS.value:
            print(_selection_banner(selected, len(dirs)), flush=True)
        keep = set(selected)
        dirs = [d for d in dirs if d.name in keep]
        groups = {o: [s for s in ss if s in keep] for o, ss in groups.items()}
        _warm(dirs, classes, groups, manifest_path)


def _est_peak_gib(area_cap: int) -> float:
    """Conservative compile-only device peak with the pool-release cap and the
    default two compile threads (autotune ON): measured 18.3 GiB at 402M,
    2.1 GiB at 46M. Overestimate a little so the peak-aware scheduler never
    packs into an OOM."""
    return 4.0 + area_cap / 28e6  # 400M -> ~18.3 GiB (measured at 402M)


def front_load(shards: Sequence[str], front: Sequence[str]) -> list[str]:
    """Dispatch order with the ``front`` shard dir names moved to the head,
    in the given order; the rest keep their relative order (stable sort)."""
    rank = {n: i for i, n in enumerate(front)}
    return sorted(shards, key=lambda s: rank.get(Path(s).name, len(rank)))


_LAUNCH_HEADROOM_GIB = 1.5


def launch_allowed(free_gib: float | None, peak_gib: float) -> bool:
    """Whether a newcomer may launch next to running workers, judged by the
    card's MEASURED free VRAM. The estimate-sum budget bounds steady usage,
    but cuda_async pools retain freed memory up to each worker's own peak, so
    a newcomer's CUDA context + cuDNN handle initialize against the real free
    memory — launched into a full card, the worker dies at its first compile
    (``RunBackend`` RET_CHECKs ``dnn_support`` when ``cudnnCreate`` cannot
    allocate). ``None`` (unreadable) falls back to estimate-only packing."""
    return free_gib is None or free_gib >= peak_gib + _LAUNCH_HEADROOM_GIB


def _gpu_free_gib(gpu_id: str | None) -> float | None:
    """Free VRAM (GiB) on the target GPU via one nvidia-smi call, or ``None``
    when it can't be read (no id, no tool, or a query error)."""
    dev = gpu_id or (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    if not dev:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={dev}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:  # noqa: BLE001 — any read failure degrades to estimates
        return None


def shard_peaks(
    shards: Sequence[str], classes: dict, overrides_path: str = ""
) -> dict[str, float]:
    """Per-shard device-peak GiB the scheduler packs against: the measured
    override verbatim for shards listed in ``overrides_path``
    (``--peak_overrides_json``), the area-formula estimate otherwise."""
    overrides = json.loads(Path(overrides_path).read_text()) if overrides_path else {}
    return {
        s: (
            float(overrides[Path(s).name])
            if Path(s).name in overrides
            else _est_peak_gib(classes[Path(s).name]["area_cap"])
        )
        for s in shards
    }


def _group_queue(classes: dict, groups: dict) -> list[list[str]]:
    """Chip-set groups as a work queue, costliest first. Cost = the group's
    distinct cold zerocheck compiles (a tight group prices as its one pinned
    compile) + one group GKR at the max area + a per-shard rider term (even a
    full cache-hit shard pays its trace+lower, ~0.3x a 400M cold compile in
    these area units). The queue is dispatched dynamically — whichever GPU
    frees up takes the next group — so real per-card speed differences and
    estimate error self-correct; the cost only sets dispatch ORDER. A group
    never splits across GPUs: concurrent same-class compiles race the cache
    and duplicate the work."""
    rider = 120e6
    costed = []
    for _, shards in groups.items():
        areas = [classes[s]["area_cap"] for s in shards]
        tight = len(shards) > 1 and min(areas) / max(areas) > _GROUP_AREA_RATIO.value
        zc_cost = max(areas) if tight else sum(set(areas))
        costed.append((zc_cost + max(areas) + rider * len(shards), shards))
    ordered = [shards for _, shards in sorted(costed, key=lambda t: -t[0])]
    for cost, shards in sorted(costed, key=lambda t: -t[0]):
        print(
            f"  queue: est {cost / 1e6:.0f}M {[_snum(s) for s in shards]}", flush=True
        )
    return ordered


def _warm(dirs: list[Path], classes: dict, groups: dict, manifest_path: str) -> None:
    cache = _CACHE_DIR.value
    Path(cache).mkdir(parents=True, exist_ok=True)
    # One shard per worker process; per GPU, workers launch peak-aware —
    # only while the sum of running est peaks (+ the candidate's) fits the
    # budget — so a big shard runs ~solo while small ones pack around it.
    # Biggest shard first within a group, so its cold class compiles while
    # the riders queue behind it.
    shards = sorted(
        (str(sd) for sd in dirs), key=lambda s: -classes[Path(s).name]["area_cap"]
    )
    front = [n.strip() for n in _FRONT_SHARDS.value.split(",") if n.strip()]
    if front:
        shards = front_load(shards, front)
    peaks = shard_peaks(shards, classes, _PEAK_OVERRIDES_JSON.value)
    budget = _MEM_BUDGET_GIB.value
    # Cap each worker's cuda_async pool so freed autotune scratch is RELEASED
    # between the ~95 zone compiles instead of accumulating (autotune stays ON,
    # so the warmed executable matches a normal prove and runs fast). Without
    # this a 400M shard's scratch piles up past 32 GiB; with it the peak holds
    # at ~11.5 GiB. cuda_async allocator required.
    env = dict(
        os.environ,
        FRX_COMPILATION_CACHE_DIR=cache,
        JAX_COMPILATION_CACHE_DIR=cache,
        XLA_PYTHON_CLIENT_ALLOCATOR="cuda_async",
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(_WORKER_MEM_FRACTION.value),
    )
    # Analysis (this process) runs CPU-only via JAX_PLATFORMS=cpu so it grabs no
    # device memory; workers need the GPU, so drop the override for them.
    env.pop("JAX_PLATFORMS", None)
    print(
        f"=== warming {len(dirs)} shards, peak-aware pool (<= {budget:.0f} GiB, "
        f"<= {_JOBS.value} procs/GPU); est peaks "
        f"{peaks[shards[0]]:.0f}..{peaks[shards[-1]]:.0f} GiB ===",
        flush=True,
    )
    gpus = [g.strip() for g in _GPUS.value.split(",") if g.strip()] or [None]
    by_name = {Path(s).name: s for s in shards}
    group_q = (
        [[by_name[n] for n in grp] for grp in _group_queue(classes, groups)]
        if gpus != [None]
        else [list(shards)]
    )
    pending: dict = {g: [] for g in gpus}  # per-GPU shard queue
    running: dict = {}  # Popen -> (shard, peak, gpu)
    ok = fail = 0
    while any(pending.values()) or group_q or running:
        for g in gpus:
            queue = pending[g]
            # Steal the next group when this GPU's queue drains — dispatch
            # follows actual completion, not a static estimate.
            if not queue and group_q:
                queue.extend(
                    sorted(
                        group_q.pop(0), key=lambda s: -classes[Path(s).name]["area_cap"]
                    )
                )
            launched = True
            while launched and queue:
                launched = False
                mine = [(s, pk) for s, pk, pg in running.values() if pg == g]
                used = sum(pk for _, pk in mine)
                # First FITTING shard, not the head: a small shard must not
                # wait behind a blocked big head.
                pick = None
                for cand in queue:
                    if not mine or used + peaks[cand] <= budget:
                        pick = cand
                        break
                if pick is None:
                    break
                s = pick
                # Always allow one worker even if a lone big shard exceeds
                # the budget.
                if len(mine) < _JOBS.value and (not mine or used + peaks[s] <= budget):
                    if mine and not launch_allowed(_gpu_free_gib(g), peaks[s]):
                        break  # real free VRAM says no — retry next poll
                    queue.remove(s)
                    wenv = env if g is None else dict(env, CUDA_VISIBLE_DEVICES=g)
                    p = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "sp1_zorch.shard_prover.warm_worker",
                            s,
                            manifest_path or "",
                        ],
                        env=wenv,
                    )
                    running[p] = (s, peaks[s], g)
                    launched = True
        for p in list(running):
            if p.poll() is not None:
                s, _, _ = running.pop(p)
                if p.returncode == 0:
                    ok += 1
                else:
                    fail += 1
                    print(
                        f"  warm worker for {Path(s).name} exited " f"{p.returncode}",
                        flush=True,
                    )
        if running:
            time.sleep(2)
    entries = sum(1 for _ in Path(cache).rglob("*") if _.is_file())
    print(
        f"=== warm done: {ok}/{ok + fail} shards ok; " f"cache entries: {entries} ===",
        flush=True,
    )
    # A warm that filled nothing is a broken prewarm, not a success: callers
    # (donor reseed, bring-up prewarm) treat exit 0 as "cache is served" and
    # then eat the full cold penalty silently (sp1-zorch#341).
    if fail or not ok or entries == 0:
        sys.exit(f"warm FAILED: {fail} worker(s) failed, {entries} cache entries")


if __name__ == "__main__":
    app.run(main)
