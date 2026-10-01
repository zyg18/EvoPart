"""
Evaluator for the hypergraph-partitioning example.

Contract with OpenEvolve: `evaluate(program_path)` receives the path of a text
file holding the packed evolvable sources. We unpack it next to the fixed files,
compile, run, and score.

Scoring is deliberately defensive:
  * the C++ program's own RESULT line is ignored for scoring - every candidate is
    re-scored from the partition vector it wrote, using utils/hgr.py;
  * a cell (instance, k) scores 0 as soon as ANY of its seeds fails: infeasible,
    malformed, crashed, or killed at the hard limit;
  * every failure path returns combined_score 0.0 instead of raising, because a
    raised exception costs `evaluator.max_retries` re-runs of the same broken code.

quality(instance, k) = baseline_cut / median cut over the seeds    (unbounded above)
cut_quality          = geometric mean of quality over all (instance, k)
time_term            = clamp(0.02 * log2(time_vs_kahypar), -0.14, +0.14)
combined_score       = cut_quality - length_penalty - time_term

where time_vs_kahypar is the geometric mean over cells of (our median wall time /
KaHyPar wall time). Cut and time are therefore separate: the geometric mean is a
pure cut comparison against official KaHyPar (cut_kKaHyPar_sea20, direct k-way),
and runtime enters once, additively, at a fixed exchange rate - each doubling of
wall time relative to KaHyPar costs 0.02, each halving earns 0.02, saturating at
32x either way. combined_score = 1.0 means "KaHyPar's cut at KaHyPar's speed".

The hard limit (BETA x KaHyPar time per cell, floored/capped) is a cliff, not a
gradient: a run past it is killed and its cell scores 0, which zeroes the
geometric mean. It sits at 64x so that a slow but working program (10-30x
KaHyPar) still has at least 2x headroom under measurement noise.

Single-threading is enforced, not requested. Every run's CPU time (rusage of that
child and its reaped descendants, obtained per child via os.wait4) is compared
with its wall time; a serial process cannot exceed cpu = wall, real parallelism
lands at 5-16x, and a violating run zeroes the whole evaluation with an explicit
reason. Because runs of one program execute concurrently (see RUN_WORKERS), the
per-child rusage from wait4 is essential: RUSAGE_CHILDREN aggregates every
reaped child of this process and would misattribute CPU time across runs.

Besides combined_score, three behavioural metrics are attached for MAP-Elites
and duplicate detection (see config.yaml's feature_dimensions):
  * log_speed   - clamp(log2(time_vs_kahypar), -2, 5); the grid's speed axis.
  * k_profile   - q3 / sqrt(q2 * q4) over per-k cut-quality geomeans; >1 means
    relatively better at k=3 than at the power-of-two ks. The grid's skill axis.
  * behavior_fp - hash of the full-grid cut vector, folded into a float. Two
    programs with identical cuts everywhere are behaviourally the same
    algorithm; start_evolution.py uses this to keep one per island.

Cascade stages share one compilation and one set of runs per program: a small
in-process cache keyed by the packed source's sha1 keeps the last few build
directories and per-(instance, k, seed) results, so stage 3 re-uses everything
stage 1 and 2 already measured instead of recompiling and re-running it.
"""

import atexit
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

BASE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from openevolve.evaluation_result import EvaluationResult  # noqa: E402
from utils.code_to_query import PARSE_ERROR_HINT, unpack  # noqa: E402
from utils.hgr import evaluate_partition, geometric_mean, read_partition  # noqa: E402

# The training set spans both suites, so instances are looked up across both
# dataset directories (first hit wins; names are unique across the suites).
_DATASET_ROOT = os.environ.get("HGP_DATASET_DIR", str(BASE_DIR.parents[1] / "dataset"))
DATASET_DIRS = [os.path.join(_DATASET_ROOT, "ISPD98"), os.path.join(_DATASET_ROOT, "Titan23")]
PRISTINE = BASE_DIR / "initial_program"
BASELINES = json.loads((BASE_DIR / "baselines.json").read_text())

# Mixed-suite training set: five ISPD98 instances (13k-185k vertices) plus five
# Titan23 instances spanning 92k-493k vertices and both structural families
# (cholesky_mc / minres: linear-algebra datapaths; neuron / openCV: stream
# processing; gsm_switch: >60-pin nets carry over half the pins). A grid drawn
# from one suite, or from one or two Titan23 designs, lets the program
# specialise to it and lose on the rest. The other 13 ISPD98 and 17 Titan23
# instances stay unseen by the evolution loop.
TRAIN = ["ibm01", "ibm05", "ibm09", "ibm13", "ibm17",
         "neuron", "cholesky_mc", "minres", "openCV", "gsm_switch"]
KS = [2, 3, 4]
SEEDS = [1, 2, 3]
EPS = 0.02

# Stage-2 gate: a mixed grid, 1 seed per cell - all three k on the smallest
# instance plus one mid-size and one large ISPD cell at the harder ks. Cheap
# (the Titan cells stay stage-3 only) but not blind to k=3/4.
STAGE2_CELLS = [("ibm01", 2), ("ibm01", 3), ("ibm01", 4), ("ibm09", 3), ("ibm13", 4)]

COMPILE_TIMEOUT = 180          # seconds

# Hard limit, as a multiple of the KaHyPar wall time for the same (instance, k).
# A cliff for runaway programs, not a penalty: the graded pressure on runtime is
# the additive time term below. 64x leaves >2x headroom against contention
# noise for a program running 10-30x slower than KaHyPar. Floor: a cell where
# KaHyPar is very fast must not get a cap that measurement noise alone would
# trip. Ceiling: KaHyPar takes 76-685s on neuron/openCV/gsm_switch; 64x is
# hours. Under the evaluator's own concurrency (RUN_WORKERS runs, times
# parallel_evaluations programs) the big instances slow down 25-35% from
# memory contention, so the ceiling keeps ~2x over a ~1000s standalone run.
BETA = 64.0
MIN_HARD = 60.0                # seconds
MAX_HARD = 2400.0              # seconds

# Time term: TIME_TERM_PER_DOUBLING per doubling of the grid-wide wall-time
# ratio to KaHyPar, clamped to +/-TIME_TERM_MAX (i.e. saturating at 2^7 = 128x
# either way; the clamp only matters at the extremes). Negative when faster than
# KaHyPar, so being faster is rewarded, not just not-penalised. 0.02 per doubling
# means a halving of runtime is worth the same as a 2% cut improvement.
TIME_TERM_PER_DOUBLING = 0.02
TIME_TERM_MAX = 0.14            # saturates at 2^7 = 128x either way

# Single-thread enforcement (see module docstring). Judge: per-run CPU time vs
# wall time. Thresholds sit in the dead zone between "serial" (cpu <= wall) and
# "parallel" (cpu >= 3x wall); the absolute term covers timer granularity on
# sub-second runs. Measured on ibm01 k2: serial excess -0.000s, a 20-thread
# program +12.7s.
PARALLEL_RATIO = 1.25          # cpu > ratio * wall ...
PARALLEL_SLACK = 0.05          # ... and cpu - wall > this many seconds

# How many runs of ONE program execute concurrently. Each run is single-threaded
# (enforced), so this is the core budget per evaluation; with
# evaluator.parallel_evaluations concurrent evaluations the machine sees
# RUN_WORKERS x parallel_evaluations busy cores. Overridable per run via env.
RUN_WORKERS = int(os.environ.get("HGP_RUN_WORKERS", "12"))
RUN_POLL_S = 0.02              # wait4 polling interval; bounds the timing granularity

# How much of the program's stdout is echoed back through artifacts.
STDOUT_TAIL_BYTES = 2000
STDOUT_KEEP_BYTES = 65536      # read back at most this much of a run's stdout

# Code-size pressure. combined_score loses LENGTH_PENALTY_PER_CHAR per packed
# character (capped). Sized to break exact ties toward the shorter program and
# to make padding cost something, while staying below a real quality difference:
# one cut unit on a small instance moves combined_score by ~3e-4, the same as
# ~30 KB of source at this rate. Cell replacement in MAP-Elites is strictly
# ">", so without this a dead-code-removal child can never displace its parent.
LENGTH_PENALTY_PER_CHAR = 1e-8
LENGTH_PENALTY_MAX = 0.003

# Bounds for the feature axes, so one outlier cannot squeeze the MAP-Elites bins
# for everyone (binning is min-max over observed values). log_speed spans
# 4x faster than KaHyPar to 32x slower.
K_PROFILE_MIN, K_PROFILE_MAX = 0.5, 2.0
LOG_SPEED_MIN, LOG_SPEED_MAX = -2.0, 5.0

# Append-only record of every evaluation, for watching a run in progress.
# One JSON object per line; see show_log.py. Each record is written with a single
# write() call so concurrent workers (parallel_evaluations > 1) cannot interleave.
EVAL_LOG = pathlib.Path(os.environ.get("HGP_EVAL_LOG", BASE_DIR / "eval_log.jsonl"))

# --- cross-stage reuse cache -------------------------------------------------
# Keyed by sha1 of the packed source. Each entry keeps the compiled workdir,
# the build artifacts, and every (instance, k, seed) result measured so far, so
# the cascade compiles once and never repeats a run. Per worker process (the
# cascade for one program runs entirely inside one worker). Bounded, LRU.
_CACHE = collections.OrderedDict()   # sha -> {"workdir", "build_art", "runs"}
_CACHE_MAX = 2


def _cache_evict():
    while len(_CACHE) > _CACHE_MAX:
        _, old = _CACHE.popitem(last=False)
        shutil.rmtree(old["workdir"], ignore_errors=True)


@atexit.register
def _cache_clear():
    for entry in _CACHE.values():
        shutil.rmtree(entry["workdir"], ignore_errors=True)
    _CACHE.clear()


def _hgr_path(inst):
    for d in DATASET_DIRS:
        p = os.path.join(d, f"{inst}.hgr")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{inst}.hgr not found in {DATASET_DIRS}")


def _hard_limit(inst, k):
    """Kill line in seconds for one (instance, k) cell."""
    t = BASELINES[f"{inst}_k{k}"]["time"]
    return min(max(BETA * t, MIN_HARD), MAX_HARD)


def _zero(reason, **artifacts):
    """Uniform failure result: score 0, with enough context for the next prompt."""
    return EvaluationResult(
        metrics={"combined_score": 0.0, "compile_ok": 0.0,
                 "time_vs_kahypar": 0.0,
                 # Neutral feature coordinates so a failed program still maps
                 # into the MAP-Elites grid (worst speed bin, neutral profile).
                 "log_speed": LOG_SPEED_MAX, "k_profile": 1.0, "behavior_fp": 0.0},
        artifacts={"failure": reason, **artifacts},
    )


def _build(program_path, workdir):
    """Unpack + compile. Returns (ok, artifacts)."""
    text = pathlib.Path(program_path).read_text()
    try:
        missing = unpack(text, workdir, PRISTINE)
    except Exception:
        return False, {"unpack_error": traceback.format_exc()[-4000:], "hint": PARSE_ERROR_HINT}

    art = {}
    if missing:
        # Not fatal - those files fell back to the pristine version - but the model
        # needs to know its edit did not land.
        art["unparsed_files"] = ", ".join(missing)
        art["hint"] = PARSE_ERROR_HINT

    try:
        proc = subprocess.run(
            ["make"], cwd=workdir, capture_output=True, text=True, timeout=COMPILE_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        art["compile_stderr"] = f"compilation exceeded {COMPILE_TIMEOUT}s"
        return False, art

    if proc.returncode != 0:
        art["compile_stderr"] = proc.stderr[-8000:]
        return False, art

    return True, art


def _read_tail(path, limit):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _spawn_and_reap(cmd, cwd, stdout_path, stderr_path, hard_limit):
    """
    Run one child under a wall-clock limit and return
    (exit code or None if killed, wall seconds, cpu seconds).

    The child is reaped with os.wait4 so the rusage is that of THIS child (plus
    any descendants it reaped itself), which stays correct when several runs
    execute concurrently in this process. RUSAGE_CHILDREN would not.
    """
    with open(stdout_path, "wb") as fo, open(stderr_path, "wb") as fe:
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=fo, stderr=fe)
        deadline = t0 + hard_limit
        killed = False
        while True:
            pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid != 0:
                break
            if time.time() >= deadline:
                killed = True
                proc.kill()
                pid, status, ru = os.wait4(proc.pid, 0)
                break
            time.sleep(RUN_POLL_S)
        elapsed = time.time() - t0
    # Tell Popen the child is gone, so its finaliser does not waitpid() again.
    proc.returncode = os.waitstatus_to_exitcode(status)
    cpu = ru.ru_utime + ru.ru_stime
    return (None if killed else proc.returncode), elapsed, cpu


def _run_one(exe, workdir, inst, k, seed, hard_limit):
    """
    One partitioning run, scored independently of what the program reports.
    Returns (cut or None, wall seconds, cpu seconds, note, stdout tail).

    stdout is returned so the evaluator can feed a tail of it back through the
    artifacts channel: whatever the program prints is shown to the LLM that
    edits it next, which lets the program keep notes for its own successors
    (phase timings, decisions taken, ...). Scoring never reads it.
    """
    hgr = _hgr_path(inst)
    tag = f"{inst}_k{k}_s{seed}"
    out = os.path.join(workdir, f"{tag}.parts")
    so, se = os.path.join(workdir, f"{tag}.stdout"), os.path.join(workdir, f"{tag}.stderr")

    rc, elapsed, cpu = _spawn_and_reap(
        [exe, hgr, f"--k={k}", f"--eps={EPS}", f"--seed={seed}", f"--out={out}"],
        workdir, so, se, hard_limit)
    stdout = _read_tail(so, STDOUT_KEEP_BYTES)

    # Parallelism is judged first: a program that is both parallel and slow
    # must be told about the parallelism, which is the rule it broke.
    if cpu > PARALLEL_RATIO * elapsed and cpu - elapsed > PARALLEL_SLACK:
        return None, elapsed, cpu, (
            f"PARALLEL EXECUTION DETECTED: {cpu:.1f}s CPU in {elapsed:.1f}s wall "
            f"(~{cpu / max(elapsed, 1e-6):.0f} cores). The program must be "
            f"single-threaded; remove all threads/processes/OpenMP."), stdout

    if rc is None:
        return None, float(hard_limit), cpu, f"killed at hard limit {hard_limit:.1f}s", stdout
    if rc != 0:
        err = _read_tail(se, 4000).strip()[-300:]
        return None, elapsed, cpu, f"exit {rc}: {err}", stdout
    if not os.path.exists(out):
        return None, elapsed, cpu, "no partition file written", stdout

    try:
        n_parts = read_partition(out, _num_vertices(hgr))
    except Exception as e:
        return None, elapsed, cpu, f"bad partition file: {e}", stdout

    r = evaluate_partition(hgr, n_parts, k, EPS)
    if not r["feasible"]:
        return None, elapsed, cpu, f"infeasible: {r['reason']}", stdout
    return r["cut"], elapsed, cpu, "", stdout


def _num_vertices(hgr):
    from utils.hgr import read_hgr

    return read_hgr(hgr)[0]


def _run_pending(exe, workdir, runs, cells):
    """
    Execute every (instance, k, seed) of `cells` not already in `runs`,
    RUN_WORKERS at a time, longest expected cells first (their KaHyPar time is
    the best available proxy) so the pool's makespan is the slowest cell rather
    than the slowest cell plus whatever was queued behind it.
    """
    pending = [(inst, k, seed) for inst, k, seeds in cells for seed in seeds
               if (inst, k, seed) not in runs and f"{inst}_k{k}" in BASELINES]
    if not pending:
        return
    pending.sort(key=lambda rk: -BASELINES[f"{rk[0]}_k{rk[1]}"]["time"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, RUN_WORKERS)) as pool:
        futs = {pool.submit(_run_one, exe, workdir, inst, k, seed, _hard_limit(inst, k)): (inst, k, seed)
                for inst, k, seed in pending}
        for fut in concurrent.futures.as_completed(futs):
            runs[futs[fut]] = fut.result()


def _score(entry, cells, art):
    """Run the grid (reusing cached runs) and fold it into cut_quality + extras."""
    workdir, runs = entry["workdir"], entry["runs"]
    exe = os.path.join(workdir, "our_hparter")
    if not os.path.exists(exe):
        return None, {}

    _run_pending(exe, workdir, runs, cells)

    qualities, slowdowns, detail = [], [], {}
    last_stdout = None  # (label, text) of the most recent run that produced output
    for inst, k, seeds in cells:
        key = f"{inst}_k{k}"
        base_entry = BASELINES.get(key)
        if base_entry is None:
            continue
        base, ref_time = base_entry["cut"], base_entry["time"]
        hard = _hard_limit(inst, k)

        cuts, walls, cpus, notes = [], [], [], []
        for seed in seeds:
            cut, elapsed, cpu, note, out_text = runs[(inst, k, seed)]
            if note.startswith("PARALLEL EXECUTION DETECTED"):
                # One violation voids the whole evaluation - no partial credit
                # for parallel speedup.
                art["parallelism"] = f"{key} seed {seed}: {note}"
                return None, {}
            if out_text.strip():
                last_stdout = (f"{key} seed {seed}", out_text)
            if cut is None:
                notes.append(f"seed{seed}: {note}")
            else:
                cuts.append(cut)
                walls.append(elapsed)
                cpus.append(cpu)

        # Any failed seed zeroes the cell (and with it the geometric mean): the
        # kill line is a deliberate cliff, and a program that finishes 2 of 3
        # seeds is not "a bit slower", it is unreliable at this size.
        if notes:
            qualities.append(0.0)
            detail[key] = {"failed": notes, "hard_limit_s": round(hard, 1),
                           "kahypar_time_s": ref_time}
            if cuts:
                detail[key]["cuts_of_finished_seeds"] = cuts
            continue

        med_cut = statistics.median(cuts)
        med_time = statistics.median(walls)
        quality = base / max(med_cut, 1)
        slowdown = med_time / max(ref_time, 1e-6)

        qualities.append(quality)
        slowdowns.append(slowdown)
        detail[key] = {
            "cut_median": med_cut, "cuts": cuts, "baseline_cut": base,
            "quality": round(quality, 4),
            "time_median_s": round(med_time, 2),
            "cpu_median_s": round(statistics.median(cpus), 2),
            "kahypar_time_s": ref_time,
            "time_vs_kahypar": round(slowdown, 2),
            "hard_limit_s": round(hard, 1),
        }

    art["per_instance"] = json.dumps(detail, indent=2)

    # Per-k digest, so "k=3 is the weak axis" is legible at a glance instead of
    # having to be inferred from 15 JSON entries. Same numbers, just grouped:
    # one line for cut quality, one for wall time.
    by_k_time, by_k_quality = {}, {}
    for key, v in detail.items():
        if "quality" not in v:
            continue
        k = int(key.rsplit("_k", 1)[1])
        by_k_quality.setdefault(k, []).append(v["quality"])
        by_k_time.setdefault(k, []).append(v["time_vs_kahypar"])
    if len(by_k_quality) > 1:
        # No editorializing beyond the grouping (e.g. no "weakest cell" callout):
        # naming the same target every round would herd all lineages onto it.
        art["per_k_summary"] = (
            "cut quality vs KaHyPar  " + " | ".join(
                f"k={k}: {geometric_mean(v):.4f} ({len(v)} cells)"
                for k, v in sorted(by_k_quality.items()))
            + "\nwall time vs KaHyPar    " + " | ".join(
                f"k={k}: {geometric_mean(v):.2f}x"
                for k, v in sorted(by_k_time.items()))
        )

    # The program's own stdout, echoed back to whoever edits it next. One run's
    # worth (the last of the grid - the largest instance at the highest k), tail
    # only, so a chatty program cannot flood the prompt.
    if last_stdout is not None:
        label, text = last_stdout
        art["stdout_tail"] = f"[from run {label}]\n" + text[-STDOUT_TAIL_BYTES:]

    # k_profile: cut skill at k=3 relative to the geometric mean of the
    # power-of-two ks. Quality-only (no time), so the two feature axes stay
    # orthogonal. Neutral 1.0 when any k is missing or entirely failed.
    q = {k: geometric_mean(v) for k, v in by_k_quality.items() if v}
    if all(kk in q and q[kk] > 0 for kk in (2, 3, 4)):
        k_profile = q[3] / math.sqrt(q[2] * q[4])
        k_profile = max(K_PROFILE_MIN, min(K_PROFILE_MAX, k_profile))
    else:
        k_profile = 1.0

    # Behaviour fingerprint over the exact cuts produced (per cell, per seed).
    # Folded to 52 bits so it survives the float-typed metrics dict exactly.
    fp_src = json.dumps({key: detail[key].get("cuts") for key in sorted(detail)},
                        sort_keys=True)
    behavior_fp = float(int(hashlib.sha1(fp_src.encode()).hexdigest()[:13], 16))

    extra = {
        # How many times slower than KaHyPar, geometric mean over the cells that
        # finished. 1.0 means matching KaHyPar wall time.
        "time_vs_kahypar": geometric_mean(slowdowns) if slowdowns else 0.0,
        "k_profile": k_profile,
        "behavior_fp": behavior_fp,
    }
    return geometric_mean(qualities), extra


def _log(stage, program_path, combined, extra, art, failure=None):
    """Append one line describing this evaluation. Never let logging break scoring."""
    try:
        text = pathlib.Path(program_path).read_text()
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,
            # The evaluator is not told the program id, so identify the candidate
            # by a hash of its source; equal hash means byte-identical program.
            "prog": hashlib.sha1(text.encode()).hexdigest()[:10],
            "combined_score": None if combined is None else round(combined, 4),
            **{k: round(v, 4) for k, v in (extra or {}).items()},
        }
        if failure:
            rec["failure"] = failure
        if "per_instance" in art:
            cells = json.loads(art["per_instance"])
            rec["cells"] = {
                key: {
                    "cut": v.get("cut_median"),
                    "cuts": v.get("cuts"),
                    "t": v.get("time_median_s"),
                    "cpu": v.get("cpu_median_s"),
                    "base": v.get("baseline_cut"),
                    "q": v.get("quality"),
                    "x": v.get("time_vs_kahypar"),
                    **({"failed": v["failed"]} if "failed" in v else {}),
                }
                for key, v in cells.items()
            }
        with open(EVAL_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _evaluate(program_path, stage, cells, full_grid=False):
    try:
        text = pathlib.Path(program_path).read_text()
        sha = hashlib.sha1(text.encode()).hexdigest()

        entry = _CACHE.get(sha)
        if entry is not None and not os.path.exists(os.path.join(entry["workdir"], "our_hparter")):
            entry = None    # workdir vanished (tmp cleanup); rebuild
        if entry is None:
            workdir = tempfile.mkdtemp(prefix="hgp_")
            ok, build_art = _build(program_path, workdir)
            if not ok:
                shutil.rmtree(workdir, ignore_errors=True)
                _log(stage, program_path, 0.0, {}, build_art, failure="compile")
                return _zero("compilation failed", **build_art)
            entry = {"workdir": workdir, "build_art": build_art, "runs": {}}
            _CACHE[sha] = entry
            _cache_evict()
        else:
            _CACHE.move_to_end(sha)

        # Build artifacts (e.g. an unparsed-files warning) must reach the final
        # stage's result too, not just the stage that happened to compile.
        art = dict(entry["build_art"])

        combined, extra = _score(entry, cells, art)
        if combined is None:
            if "parallelism" in art:
                _log(stage, program_path, 0.0, {}, art, failure="parallel")
                return _zero("parallel execution detected - the program must be "
                             "single-threaded", **art)
            _log(stage, program_path, 0.0, {}, art, failure="no binary")
            return _zero("binary missing after successful make", **art)

        packed_chars = len(text)
        penalty = min(LENGTH_PENALTY_MAX, LENGTH_PENALTY_PER_CHAR * packed_chars)

        # Time term on the grid-wide wall-time ratio. Skipped when no cell
        # finished (tv == 0): log2 would explode, and the score is 0 anyway.
        tv = extra.get("time_vs_kahypar", 0.0)
        time_term = 0.0
        log_speed = LOG_SPEED_MAX
        if tv > 0:
            time_term = max(-TIME_TERM_MAX,
                            min(TIME_TERM_MAX, TIME_TERM_PER_DOUBLING * math.log2(tv)))
            log_speed = max(LOG_SPEED_MIN, min(LOG_SPEED_MAX, math.log2(tv)))

        # The fingerprint is only comparable across programs measured on the
        # same cells; restrict it to full-grid results (0 disables dedup).
        if not full_grid or combined <= 0:
            extra["behavior_fp"] = 0.0

        final = max(0.0, combined - penalty - time_term) if combined > 0 else 0.0
        art["code_size"] = (
            f"{packed_chars} chars packed; length penalty {penalty:.4f} subtracted "
            f"from combined_score ({LENGTH_PENALTY_PER_CHAR:g}/char, cap {LENGTH_PENALTY_MAX})"
        )
        art["speed"] = (
            f"wall time {tv:.2f}x KaHyPar (geomean over cells) -> time term {time_term:+.4f} "
            f"subtracted from combined_score ({TIME_TERM_PER_DOUBLING:g} per doubling, "
            f"clamped to +/-{TIME_TERM_MAX}; negative means a bonus for being faster than "
            f"KaHyPar). cut_quality {combined:.4f} - length {penalty:.4f} - time "
            f"{time_term:+.4f} = combined_score {final:.4f}"
        )
        extra = {**extra, "length_penalty": penalty, "time_term": time_term,
                 "log_speed": log_speed}

        _log(stage, program_path, final, extra, art)
        return EvaluationResult(
            metrics={
                "combined_score": float(final),
                "compile_ok": 1.0,
                # Pure cut quality (geomean of baseline_cut / median cut), before
                # the length and time terms.
                "cut_quality": float(combined),
                **{k: float(v) for k, v in extra.items()},
            },
            artifacts=art,
        )
    except Exception:
        return _zero("evaluator exception", traceback=traceback.format_exc()[-4000:])


# --- OpenEvolve entry points -------------------------------------------------
# Cascade stages widen the grid. Each stage returns a combined_score on the same
# scale (ratio to the KaHyPar baseline), so a candidate stopped early is still
# comparable to one that ran the full grid. Compilation and every individual
# run are shared across stages through _CACHE, so widening only pays for the
# cells not yet measured.

def evaluate_stage1(program_path):
    """Cheapest gate: does it compile and produce one valid partition?"""
    return _evaluate(program_path, "stage1", [("ibm01", 2, [1])])


def evaluate_stage2(program_path):
    """Mixed small grid, 1 seed per cell: all ks on ibm01 plus two harder cells."""
    return _evaluate(program_path, "stage2", [(i, k, [1]) for i, k in STAGE2_CELLS])


def evaluate_stage3(program_path):
    """Full grid: 10 instances x k in {2,3,4} x 3 seeds = 90 runs."""
    return _evaluate(program_path, "stage3",
                     [(i, k, list(SEEDS)) for i in TRAIN for k in KS],
                     full_grid=True)


def evaluate(program_path):
    """Used when cascade_evaluation is off."""
    return evaluate_stage3(program_path)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if not target:
        # Score the pristine initial program.
        from utils.code_to_query import pack

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(pack(PRISTINE))
            target = f.name

    stage = sys.argv[2] if len(sys.argv) > 2 else "3"
    fn = {"1": evaluate_stage1, "2": evaluate_stage2, "3": evaluate_stage3}[stage]
    res = fn(target)
    print(json.dumps(res.metrics, indent=2))
    for key, val in res.artifacts.items():
        print(f"\n--- {key} ---\n{val}")
