#!/usr/bin/env python3
"""
Read eval_log.jsonl, which the evaluator appends to on every evaluation.

    python3 show_log.py                # one line per evaluation, newest last
    python3 show_log.py --full         # 30-cell breakdown for the best evaluation
    python3 show_log.py --full -n 3    # ... for each of the last 3 full-grid runs
    python3 show_log.py --prog a1b2c3  # ... for one specific candidate
    python3 show_log.py --follow       # tail the log as the run progresses

Safe to run while evolution is in flight: the file is append-only.
"""

import argparse
import json
import pathlib
import time

BASE_DIR = pathlib.Path(__file__).parent
DEFAULT_LOG = BASE_DIR / "eval_log.jsonl"

CELL_ORDER = [f"{i}_k{k}"
              for i in ("ibm01", "ibm05", "ibm09", "ibm13", "ibm17",
                        "neuron", "cholesky_mc", "minres", "openCV", "gsm_switch")
              for k in (2, 3, 4)]


def load(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a partially flushed last line while following
    return out


def summary(records):
    print(f"{'#':>4} {'time':19} {'prog':10} {'stage':8} {'score':>7} {'vs KaHyPar':>10} "
          f"{'timeterm':>8}  note")
    best = -1.0
    for i, r in enumerate(records):
        score = r.get("combined_score")
        mark = ""
        if score is not None and r.get("stage") == "stage3" and score > best:
            best, mark = score, "  <-- best full grid"
        print(f"{i:>4} {r.get('ts',''):19} {r.get('prog',''):10} {r.get('stage',''):8} "
              f"{'-' if score is None else f'{score:7.4f}'} "
              f"{r.get('time_vs_kahypar','-'):>10} {r.get('time_term','-'):>8}"
              f"  {r.get('failure','')}{mark}")


def detail(r):
    print(f"\n=== {r.get('ts')}  prog {r.get('prog')}  stage {r.get('stage')}  "
          f"combined_score {r.get('combined_score')} ===")
    cells = r.get("cells")
    if not cells:
        print(f"  no cells ({r.get('failure', 'unknown failure')})")
        return
    print(f"  {'cell':11}{'cut':>8}{'KaHyPar':>8}{'ratio':>8}{'time(s)':>9}"
          f"{'cpu(s)':>9}{'xKaHyPar':>9}   cuts per seed")
    for key in [k for k in CELL_ORDER if k in cells] + \
               [k for k in cells if k not in CELL_ORDER]:
        v = cells[key]
        cut, base = v.get("cut"), v.get("base")
        ratio = f"{cut / base:.2f}x" if cut and base else "-"
        print(f"  {key:11}{str(cut):>8}{str(base):>8}{ratio:>8}{str(v.get('t')):>9}"
              f"{str(v.get('cpu')):>9}{str(v.get('x')):>9}   {v.get('cuts')}"
              f"{'  FAILED: ' + str(v['failed']) if 'failed' in v else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG)
    ap.add_argument("--full", action="store_true", help="show the per-cell breakdown")
    ap.add_argument("-n", type=int, default=1, help="how many records to detail")
    ap.add_argument("--prog", help="detail a specific candidate by its prog hash prefix")
    ap.add_argument("--follow", action="store_true", help="keep printing new records")
    args = ap.parse_args()

    records = load(args.log)
    if not records:
        print(f"no records in {args.log} yet")
        return

    if args.prog:
        for r in [x for x in records if x.get("prog", "").startswith(args.prog)]:
            detail(r)
        return

    if args.full:
        full = [r for r in records if r.get("cells")]
        # Best first, so `--full` alone shows the strongest candidate so far.
        full.sort(key=lambda r: r.get("combined_score") or -1, reverse=True)
        for r in full[: args.n]:
            detail(r)
        return

    summary(records)

    if args.follow:
        seen = len(records)
        try:
            while True:
                time.sleep(5)
                records = load(args.log)
                for r in records[seen:]:
                    score = r.get("combined_score")
                    print(f"{seen:>4} {r.get('ts',''):19} {r.get('prog',''):10} "
                          f"{r.get('stage',''):8} "
                          f"{'-' if score is None else f'{score:7.4f}'}"
                          f"  {r.get('failure','')}")
                    seen += 1
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
