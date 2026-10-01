#!/usr/bin/env python3
"""
Generate baselines.json with reference cuts and wall times.

Default reference is official KaHyPar (cut_kKaHyPar_sea20, direct k-way); pass
--tool hmetis to anchor to hMETIS 2.0pre1 instead. Every reference result is
re-scored with our own checker (utils/hgr.py) so the baseline and the candidates
are measured by exactly the same definition of cut and balance. A run whose
partition violates our capacity is rejected rather than silently used as a
(too easy) baseline - with KaHyPar this never happens (it enforces the balance
constraint as hard, like we do); with hMETIS on large-net instances it can.

Protocol per (instance, k): one run per seed in SEEDS, cut = median of the
re-scored cuts, time = median wall seconds. That mirrors how candidates are
scored (median over the same seeds), so baseline and candidate see the same
amount of luck.

    python3 make_baselines.py [--tool kahypar] [--out baselines.json]
"""

import argparse
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import statistics
import tempfile
import time

BASE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from utils.hgr import evaluate_partition  # noqa: E402

HMETIS = os.environ.get("HMETIS_BIN", "hmetis2.0pre1")            # hMETIS 2.0pre1 binary
KAHYPAR_BIN = os.environ.get("KAHYPAR_BIN", "KaHyPar")           # KaHyPar application binary
KAHYPAR_PRESET = os.environ.get("KAHYPAR_PRESET", "cut_kKaHyPar_sea20.ini")  # KaHyPar preset

DATASET_DIRS = [
    os.path.join(os.environ.get("HGP_DATASET_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset")), "ISPD98"),
    os.path.join(os.environ.get("HGP_DATASET_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset")), "Titan23"),
]

# Must match evaluator.py's TRAIN / KS / SEEDS / EPS: these anchors define both
# the cut reference and the per-cell time budgets of the evolution loop.
TRAIN = ["ibm01", "ibm05", "ibm09", "ibm13", "ibm17",
         "neuron", "cholesky_mc", "minres", "openCV", "gsm_switch"]
KS = [2, 3, 4]
SEEDS = [1, 2, 3]
EPS = 0.02


def hgr_path(inst):
    for d in DATASET_DIRS:
        p = os.path.join(d, f"{inst}.hgr")
        if os.path.exists(p):
            return p
    sys.exit(f"{inst}.hgr not found in {DATASET_DIRS}")


def run_hmetis(hgr, k, eps, nruns, timeout=1800):
    """One hMETIS invocation (nruns internal restarts). Returns (partition, seconds)."""
    with tempfile.TemporaryDirectory() as d:
        local = os.path.join(d, os.path.basename(hgr))
        shutil.copy2(hgr, local)
        cmd = [
            HMETIS,
            "-ptype=kway",     # kway's ufactor is "largest part / average - 1", matching our eps
            "-otype=cut",
            f"-ufactor={eps * 100:g}",
            f"-nruns={nruns}",
            local,
            str(k),
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"hmetis failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
        out = f"{local}.part.{k}"
        if not os.path.exists(out):
            raise RuntimeError(f"hmetis produced no {out}\n{proc.stdout[-2000:]}")
        with open(out) as f:
            return [int(x) for x in f.read().split()], elapsed


def run_kahypar(hgr, k, eps, seed, timeout=3600):
    """One KaHyPar run (direct k-way, sea20 preset). Returns (partition, seconds).

    Runs in a temp dir because -w true writes the partition file next to the
    input. The cut is never taken from KaHyPar's own report.
    """
    with tempfile.TemporaryDirectory() as d:
        local = os.path.join(d, os.path.basename(hgr))
        shutil.copy2(hgr, local)
        cmd = [KAHYPAR_BIN, "-h", local, "-k", str(k), "-e", str(eps),
               "-o", "cut", "-m", "direct", "-p", KAHYPAR_PRESET,
               f"--seed={seed}", "-w", "true"]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=d)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"KaHyPar failed ({proc.returncode}): {proc.stderr[-1500:]}")
        parts = glob.glob(os.path.join(d, "*.part*")) + glob.glob(os.path.join(d, "*.KaHyPar"))
        if not parts:
            raise RuntimeError(f"KaHyPar wrote no partition file. stdout: {proc.stdout[-800:]}")
        with open(parts[0]) as f:
            return [int(x) for x in f.read().split()], elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["kahypar", "hmetis"], default="kahypar")
    ap.add_argument("--nruns", type=int, default=10,
                    help="hMETIS internal restarts per run (hmetis mode only)")
    ap.add_argument("--instances", nargs="*", default=TRAIN,
                    help="instances to anchor; defaults to the training set")
    ap.add_argument("--out", default=str(BASE_DIR / "baselines.json"))
    args = ap.parse_args()

    binary = KAHYPAR_BIN if args.tool == "kahypar" else HMETIS
    if not os.path.exists(binary):
        sys.exit(f"{args.tool} binary not found at {binary}")

    baselines = {}
    print(f"{'key':20}{'cut':>8}{'time(s)':>9}  per-seed (cut,s)")
    for inst in args.instances:
        hgr = hgr_path(inst)
        for k in KS:
            key = f"{inst}_k{k}"
            cuts, times = [], []
            for seed in SEEDS:
                if args.tool == "kahypar":
                    part, secs = run_kahypar(hgr, k, EPS, seed)
                else:
                    part, secs = run_hmetis(hgr, k, EPS, args.nruns)
                # Re-score with OUR definition, not the tool's report.
                r = evaluate_partition(hgr, part, k, EPS)
                if not r["feasible"]:
                    sys.exit(f"{args.tool} baseline for {key} seed {seed} violates our "
                             f"capacity ({r['reason']}); it cannot anchor the budget.")
                cuts.append(r["cut"])
                times.append(secs)
            cut = int(statistics.median(cuts))
            wall = statistics.median(times)
            print(f"{key:20}{cut:>8}{wall:>9.2f}  "
                  + " ".join(f"({c},{t:.1f})" for c, t in zip(cuts, times)), flush=True)
            baselines[key] = {"cut": cut, "time": round(wall, 3)}

    meta = {
        "_source": ("KaHyPar official build, preset cut_kKaHyPar_sea20, direct k-way"
                    if args.tool == "kahypar" else
                    f"hMETIS 2.0pre1 -ptype=kway -otype=cut -ufactor={EPS * 100:g} "
                    f"-nruns={args.nruns}"),
        "_eps": EPS,
        "_train": args.instances,
        "_ks": KS,
        "_seeds": SEEDS,
        "_note": ("cuts recomputed by utils/hgr.evaluate_partition, not taken from the "
                  "tool's output; cut and time are medians over the seeds and anchor "
                  "the evaluator's per-cell quality reference and time budget"),
    }
    with open(args.out, "w") as f:
        json.dump({**meta, **baselines}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
