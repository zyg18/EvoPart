#!/usr/bin/env python3
"""
Full benchmark of an evolved program on ISPD98 + Titan23.

360 runs: 40 instances x k in {2,3,4} x seeds 1-3, each re-scored by
utils/hgr.py. Appends to the CSV after every run so progress survives a kill.
Runs WORKERS single-threaded processes at a time (default 4: low enough that
memory-bandwidth contention on this 64-core box is negligible, so the wall
times stay comparable to the standalone hMETIS / KaHyPar measurements).

    python3 bench_best.py build_best/our_hparter best ours_full_benchmark.csv
"""

import concurrent.futures
import csv
import os
import pathlib
import subprocess
import sys
import threading
import time

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE))
from utils.hgr import evaluate_partition, read_partition, read_hgr  # noqa: E402

EXE, SYSTEM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
WORKERS = int(os.environ.get("BENCH_WORKERS", "4"))
_ROOT = os.environ.get("HGP_DATASET_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset"))
DIRS = [os.path.join(_ROOT, "ISPD98"), os.path.join(_ROOT, "Titan23")]
KS = [2, 3, 4]
SEEDS = [1, 2, 3]
EPS = 0.02
RUN_LIMIT = 3600
TMP = pathlib.Path(os.environ.get("BENCH_TMP", f"/tmp/bench_{SYSTEM}"))
TMP.mkdir(parents=True, exist_ok=True)

out_path = BASE / OUT
done = set()
if out_path.exists():
    with open(out_path) as f:
        for r in csv.DictReader(f):
            done.add((r["instance"], int(r["k"]), int(r["run"])))
    print(f"resuming: {len(done)} runs already recorded", flush=True)
else:
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(["instance", "k", "system", "run", "cut", "seconds"])

jobs = []
for d in DIRS:
    for hgr_name in sorted(os.listdir(d)):
        if not hgr_name.endswith(".hgr"):
            continue
        inst = hgr_name[:-4]
        for k in KS:
            for seed in SEEDS:
                if (inst, k, seed) not in done:
                    jobs.append((inst, os.path.join(d, hgr_name), k, seed))

lock = threading.Lock()


def run(job):
    inst, hgr, k, seed = job
    n = read_hgr(hgr)[0]
    part_file = TMP / f"{inst}_{k}_{seed}.parts"
    t0 = time.time()
    cut = ""
    try:
        p = subprocess.run(
            [EXE, hgr, f"--k={k}", f"--eps={EPS}", f"--seed={seed}", f"--out={part_file}"],
            capture_output=True, text=True, timeout=RUN_LIMIT)
        secs = time.time() - t0
        if p.returncode == 0 and part_file.exists():
            r = evaluate_partition(hgr, read_partition(part_file, n), k, EPS)
            if r["feasible"]:
                cut = r["cut"]
            else:
                print(f"  INFEASIBLE {inst} k{k} s{seed}: {r['reason']}", flush=True)
        else:
            print(f"  FAIL {inst} k{k} s{seed}: exit {p.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        secs = float(RUN_LIMIT)
        print(f"  TIMEOUT {inst} k{k} s{seed}", flush=True)
    finally:
        if part_file.exists():
            part_file.unlink()
    with lock:
        with open(out_path, "a", newline="") as f:
            csv.writer(f).writerow([inst, k, SYSTEM, seed, cut, round(secs, 2)])
        print(f"{inst} k{k} s{seed}: cut={cut} t={secs:.1f}s", flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
    list(pool.map(run, jobs))
print("ALL DONE", flush=True)
