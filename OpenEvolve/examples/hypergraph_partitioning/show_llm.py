#!/usr/bin/env python3
"""
Explain what OpenEvolve did with each captured LLM response.

`llm_calls/` holds every exchange (see utils/llm_capture.py). This replays the
framework's own parsing on each one - the same extract_diffs and
split_diffs_by_target it uses in process_parallel - so a discarded iteration
can be attributed to a specific cause instead of a log line.

    python3 show_llm.py                  # verdict per call, oldest first
    python3 show_llm.py 3                # detail call #3: every SEARCH block
    python3 show_llm.py 3 --response     # the raw reply
    python3 show_llm.py 3 --prompt       # what it was asked
"""

import argparse
import json
import pathlib
import re
import sys

BASE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent.parent))

from openevolve.config import Config  # noqa: E402
from openevolve.utils.code_utils import (  # noqa: E402
    apply_diff_blocks,
    extract_diffs,
    split_diffs_by_target,
)

DIFF_PATTERN = Config().diff_pattern

# The prompt embeds both diff targets in fenced blocks; recovering them lets us
# reproduce the routing decision exactly. Neither the C++ nor the description
# contains a fence, so the first closing ``` is the right one.
CODE_RE = re.compile(r"^# Current Program\n```[^\n]*\n(.*?)\n```", re.S | re.M)
DESC_RE = re.compile(r"^# Changes Description\n.*?\n```[^\n]*\n(.*?)\n```", re.S | re.M)


def load(log_dir):
    return [
        (f, json.loads(f.read_text()))
        for f in sorted(log_dir.glob("*.json"))
    ]


def analyse(rec):
    """Reproduce process_parallel's verdict for one response."""
    out = {"marker_count": 0, "blocks": 0, "code": 0, "desc": 0, "unmatched": 0,
           "desc_applied": 0, "verdict": "", "detail": None}

    if "response" not in rec:
        out["verdict"] = f"LLM call failed: {rec.get('error', '?')}"
        return out

    resp = rec["response"]
    out["marker_count"] = resp.count("<<<<<<< SEARCH")
    blocks = extract_diffs(resp, DIFF_PATTERN)
    out["blocks"] = len(blocks)
    if not blocks:
        out["verdict"] = "DISCARDED: no valid diffs found"
        return out

    code_text = CODE_RE.search(rec["user"])
    desc_text = DESC_RE.search(rec["user"])
    if not code_text or not desc_text:
        out["verdict"] = "cannot recover prompt targets (prompt layout changed?)"
        return out
    code_text, desc_text = code_text.group(1), desc_text.group(1)

    try:
        code_blocks, desc_blocks, unmatched = split_diffs_by_target(
            blocks, code_text=code_text, changes_description_text=desc_text
        )
    except ValueError as exc:
        out["verdict"] = f"DISCARDED: {exc}"
        return out

    out.update(code=len(code_blocks), desc=len(desc_blocks), unmatched=len(unmatched))
    new_desc, applied = apply_diff_blocks(desc_text, desc_blocks)
    out["desc_applied"] = applied
    out["detail"] = (code_blocks, desc_blocks, unmatched, desc_text, new_desc)

    if applied == 0:
        out["verdict"] = ("DISCARDED: no SEARCH block matched the description"
                          if not desc_blocks else "DISCARDED: description diff did not apply")
    elif not new_desc.strip():
        out["verdict"] = "DISCARDED: description became empty"
    elif new_desc.strip() == desc_text.strip():
        out["verdict"] = "DISCARDED: description unchanged after applying"
    elif not code_blocks:
        out["verdict"] = "ACCEPTED but code untouched (description-only edit)"
    else:
        out["verdict"] = "ACCEPTED"
    return out


def preview(search, n=2):
    lines = [ln for ln in search.split("\n") if ln.strip()][:n]
    return " / ".join(ln.strip()[:70] for ln in lines) or "(blank)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?", type=int, help="detail one call")
    ap.add_argument("--dir", type=pathlib.Path, default=BASE_DIR / "llm_calls")
    ap.add_argument("--response", action="store_true")
    ap.add_argument("--prompt", action="store_true")
    args = ap.parse_args()

    calls = load(args.dir)
    if not calls:
        sys.exit(f"no captured calls in {args.dir}")

    if args.index is None:
        print(f"{'#':>3} {'ts':19} {'model':18} {'resp':>7} {'blk':>4} {'code':>5} "
              f"{'desc':>5} {'unm':>4}  verdict")
        for i, (_, rec) in enumerate(calls):
            a = analyse(rec)
            print(f"{i:>3} {rec['ts']:19} {rec['model']:18} "
                  f"{len(rec.get('response', '')):>7} {a['blocks']:>4} {a['code']:>5} "
                  f"{a['desc']:>5} {a['unmatched']:>4}  {a['verdict']}")
        return

    _, rec = calls[args.index]
    if args.response:
        print(rec.get("response", rec.get("error", "(nothing)")))
        return
    if args.prompt:
        print("========== SYSTEM ==========")
        print(rec["system"])
        print("\n========== USER ==========")
        print(rec["user"])
        return

    a = analyse(rec)
    print(f"call {args.index}  {rec['ts']}  {rec['model']}")
    print(f"response {len(rec.get('response', ''))} chars, "
          f"{a['marker_count']} '<<<<<<< SEARCH' markers, "
          f"{a['blocks']} parsed blocks")
    print(f"verdict: {a['verdict']}\n")
    if not a["detail"]:
        return
    code_blocks, desc_blocks, unmatched, desc_text, new_desc = a["detail"]
    for label, blocks in (("-> code", code_blocks), ("-> description", desc_blocks),
                          ("-> MATCHED NOTHING", unmatched)):
        for s, r in blocks:
            print(f"{label:20} SEARCH {preview(s)}")
            print(f"{'':20} REPLACE {preview(r)}")
    print(f"\ndescription: {len(desc_text)} chars -> {len(new_desc)} chars, "
          f"{a['desc_applied']} block(s) applied")


if __name__ == "__main__":
    main()
