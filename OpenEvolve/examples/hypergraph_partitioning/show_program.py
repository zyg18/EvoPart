#!/usr/bin/env python3
"""
Read the program records OpenEvolve writes into a checkpoint.

Each programs/<uuid>.json holds the full candidate: packed source, metrics,
artifacts, its changes description, and the prompt/response that produced it.
Raw JSON is unreadable at 40 KB a file, so this prints the parts separately.

    python3 show_program.py                    # list every program in the newest checkpoint
    python3 show_program.py best               # the run's best program
    python3 show_program.py <id>               # one program, by id (prefix of its uuid)
    python3 show_program.py <id> --desc        # what the model says it changed
    python3 show_program.py <id> --code        # the packed partition.cpp
    python3 show_program.py <id> --file partition.cpp
    python3 show_program.py <id> --response    # the raw LLM reply (the diffs it emitted)
    python3 show_program.py <id> --prompt      # what it was asked
    python3 show_program.py <id> --artifacts   # per-cell cut/time from its evaluation
    python3 show_program.py --tree             # parent -> child lineage with scores
"""

import argparse
import json
import pathlib
import re
import sys

BASE_DIR = pathlib.Path(__file__).parent
OUT = BASE_DIR / "openevolve_output"


def newest_checkpoint(root=None):
    root = root or (OUT / "checkpoints")
    best = None
    for p in root.glob("checkpoint_*"):
        m = re.fullmatch(r"checkpoint_(\d+)", p.name)
        if p.is_dir() and m:
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, p)
    return best[1] if best else None


def load_all(ckpt):
    progs = {}
    for f in sorted((ckpt / "programs").glob("*.json")):
        d = json.loads(f.read_text())
        progs[d["id"]] = d
    return progs


def resolve(progs, key):
    if key == "best":
        info = json.loads((OUT / "best" / "best_program_info.json").read_text())
        key = info["id"]
    hits = [p for pid, p in progs.items() if pid.startswith(key)]
    if not hits:
        sys.exit(f"no program id starting with {key!r}; try `show_program.py` to list them")
    if len(hits) > 1:
        sys.exit(f"{key!r} is ambiguous: {[p['id'][:8] for p in hits]}")
    return hits[0]


def brief(p):
    m = p.get("metrics", {})
    return (f"{p['id'][:8]}  gen {p.get('generation', '?'):>2}  "
            f"score {m.get('combined_score', float('nan')):.4f}  "
            f"vs KaHyPar {m.get('time_vs_kahypar', float('nan')):.2f}x  "
            f"parent {(p.get('parent_id') or '-')[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program", nargs="?", help="id prefix, or 'best'")
    ap.add_argument("--checkpoint", type=pathlib.Path, default=None)
    ap.add_argument("--desc", action="store_true", help="changes description")
    ap.add_argument("--code", action="store_true", help="full packed source")
    ap.add_argument("--file", help="one file out of the packed source")
    ap.add_argument("--response", action="store_true", help="raw LLM reply")
    ap.add_argument("--prompt", action="store_true", help="prompt it was given")
    ap.add_argument("--artifacts", action="store_true", help="evaluation artifacts")
    ap.add_argument("--tree", action="store_true", help="lineage")
    args = ap.parse_args()

    ckpt = args.checkpoint or newest_checkpoint()
    if ckpt is None:
        sys.exit("no checkpoint found under openevolve_output/checkpoints")
    progs = load_all(ckpt)
    print(f"# checkpoint {ckpt.name}, {len(progs)} programs\n", file=sys.stderr)

    if args.tree:
        children = {}
        for p in progs.values():
            children.setdefault(p.get("parent_id"), []).append(p)
        roots = [p for p in progs.values() if p.get("parent_id") not in progs]

        def walk(p, depth):
            print("  " * depth + brief(p))
            for c in sorted(children.get(p["id"], []),
                            key=lambda x: x.get("generation", 0)):
                walk(c, depth + 1)

        for r in sorted(roots, key=lambda x: x.get("generation", 0)):
            walk(r, 0)
        return

    if not args.program:
        for p in sorted(progs.values(),
                        key=lambda x: x.get("metrics", {}).get("combined_score", 0),
                        reverse=True):
            print(brief(p))
        return

    p = resolve(progs, args.program)

    if args.desc:
        print(p.get("changes_description", "(none)"))
    elif args.code:
        print(p["code"])
    elif args.file:
        m = re.search(r"^\*\s*" + re.escape(args.file) + r"\s*\*\s*:\s*\n@@@\s*\n(.*?)\n?@@@",
                      p["code"], re.MULTILINE | re.DOTALL)
        if not m:
            names = re.findall(r"^\*\s*([^*\n]+?)\s*\*\s*:", p["code"], re.MULTILINE)
            sys.exit(f"{args.file!r} not in this program. Available: {names}")
        print(m.group(1))
    elif args.response:
        pr = p.get("prompts", {}).get("diff_user", {})
        for r in pr.get("responses", []) or ["(no response stored)"]:
            print(r)
    elif args.prompt:
        pr = p.get("prompts", {}).get("diff_user", {})
        print("========== SYSTEM ==========")
        print(pr.get("system", "(none)"))
        print("\n========== USER ==========")
        print(pr.get("user", "(none)"))
    elif args.artifacts:
        raw = p.get("artifacts_json")
        if not raw:
            print("(no artifacts)")
        else:
            art = json.loads(raw)
            for k, v in art.items():
                print(f"--- {k} ---\n{v}\n")
    else:
        print(brief(p))
        print(f"\nmetrics: {json.dumps(p.get('metrics', {}), indent=2)}")
        print(f"code: {len(p.get('code', ''))} chars   "
              f"changes_description: {len(p.get('changes_description', ''))} chars")
        print("\nuse --desc / --code / --file X / --response / --prompt / --artifacts")


if __name__ == "__main__":
    main()
