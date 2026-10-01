#!/usr/bin/env python3
"""
Launch evolution for the hypergraph partitioner.

The CLI entry point (openevolve-run.py) takes a single initial-program file, so
this script packs the evolvable C++ sources into one .txt first and then drives
the Python API, the same shape as examples/tsp_tour_minimization.

    python3 start_evolution.py                     # fresh run, or resume latest
    python3 start_evolution.py --iterations 50
    python3 start_evolution.py --no-resume         # ignore existing checkpoints
"""

import argparse
import asyncio
import pathlib
import re
import sys

BASE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from openevolve import OpenEvolve  # noqa: E402
from openevolve.config import Config  # noqa: E402
from utils.code_to_query import pack  # noqa: E402
from utils.llm_capture import install as install_llm_capture  # noqa: E402


def install_behavior_dedup():
    """
    Wrap ProgramDatabase.add with a per-island behavioural duplicate filter.

    The evaluator attaches behavior_fp: a hash of the full-grid cut vector,
    folded into a float. Two programs with the same fingerprint produced
    identical cuts on every (instance, k, seed) - behaviourally the same
    algorithm, whatever the source diff looks like. Keeping both in one island
    wastes a grid cell and stacks the inspiration lists with copies.

    Policy: a duplicate is dropped unless it beats the program it
    duplicates - a strictly higher combined_score (same cuts, so this means
    faster or shorter) or, at equal score, strictly shorter code. A winning
    duplicate is added normally and the program it duplicates is evicted from
    the island: removed from the island set and from any MAP-Elites cell it
    owns, then handed to the framework's _remove_program_if_orphaned, exactly
    what database.add does to a program displaced from its cell. The evicted
    copy has identical cuts and is slower, so nothing the search could use is
    lost. (Admitting only strictly-shorter clones would instead discard the
    same-cuts-but-faster candidates that outscore the library best.)

    The rejection mirrors the framework's own _is_novel path in database.add:
    the program is recorded in self.programs but joins no island, and add()
    returns its id as usual. Scope is per island, so migration (which re-adds
    a copy with target_island set) still works - the migrant is only refused
    where its behaviour already lives.

    A wrapper rather than a framework patch, same pattern as llm_capture.
    """
    import logging

    from openevolve.database import ProgramDatabase

    orig_add = ProgramDatabase.add
    logger = logging.getLogger("behavior_dedup")

    def add(self, program, iteration=None, target_island=None):
        # Framework-generated failure results (stage timeout, retries exhausted)
        # carry metrics like {"error": 0.0} without our feature dimensions, and
        # _calculate_feature_coords raises on a missing dimension - which would
        # silently discard the failed program's record (and its error feedback
        # for the next prompt). Backfill neutral coordinates instead.
        if program.metrics is not None:
            program.metrics.setdefault("log_speed", 5.0)   # evaluator.LOG_SPEED_MAX
            program.metrics.setdefault("k_profile", 1.0)

        try:
            fp = float((program.metrics or {}).get("behavior_fp", 0.0) or 0.0)
        except (TypeError, ValueError):
            fp = 0.0
        if fp > 0.0:
            # Resolve the destination island exactly like database.add does.
            island_idx = target_island
            if island_idx is None and program.parent_id:
                parent = self.programs.get(program.parent_id)
                if parent is not None and "island" in parent.metadata:
                    island_idx = parent.metadata["island"]
            if island_idx is None:
                island_idx = self.current_island
            island_idx = island_idx % len(self.islands)

            def score(p):
                try:
                    return float((p.metrics or {}).get("combined_score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    return 0.0

            evict = []
            for pid in list(self.islands[island_idx]):
                other = self.programs.get(pid)
                if other is None or other.id == program.id:
                    continue
                try:
                    other_fp = float((other.metrics or {}).get("behavior_fp", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if other_fp != fp:
                    continue
                s_new, s_old = score(program), score(other)
                wins = s_new > s_old or (s_new == s_old and len(program.code) < len(other.code))
                if not wins:
                    logger.info(
                        "behavior-dedup: %s duplicates %s in island %d (fp=%.0f, "
                        "score %.4f <= %.4f), not added",
                        program.id, other.id, island_idx, fp, s_new, s_old,
                    )
                    self.programs[program.id] = program   # mirror the _is_novel path
                    return program.id
                evict.append(other)

            # The newcomer beats every same-fingerprint resident: evict them the
            # way database.add evicts a program displaced from its cell.
            for other in evict:
                logger.info(
                    "behavior-dedup: %s (score %.4f) supersedes duplicate %s (score %.4f) "
                    "in island %d (fp=%.0f), evicting the old copy",
                    program.id, score(program), other.id, score(other), island_idx, fp,
                )
                fmap = self.island_feature_maps[island_idx]
                for key in [k for k, v in fmap.items() if v == other.id]:
                    del fmap[key]
                self.islands[island_idx].discard(other.id)
                self.archive.discard(other.id)
                if other.id != self.best_program_id:
                    self._remove_program_if_orphaned(other.id)
        return orig_add(self, program, iteration=iteration, target_island=target_island)

    ProgramDatabase.add = add


def latest_checkpoint(checkpoints_dir):
    """Newest checkpoint_<n> directory, or None."""
    if not checkpoints_dir.exists():
        return None
    best = None
    for path in checkpoints_dir.glob("checkpoint_*"):
        m = re.fullmatch(r"checkpoint_(\d+)", path.name)
        if path.is_dir() and m:
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, path)
    return str(best[1]) if best else None


async def run(evolve, checkpoint):
    best = await evolve.run(checkpoint_path=checkpoint)
    if best is None:
        print("evolution produced no best program")
        return
    print("\nBest program metrics:")
    for name, value in best.metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, (int, float)) else f"  {name}: {value}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=pathlib.Path, default=BASE_DIR / "initial_program",
                    help="directory holding the C++ project")
    ap.add_argument("--output", type=pathlib.Path, default=BASE_DIR / "openevolve_output")
    ap.add_argument("--config", type=pathlib.Path, default=BASE_DIR / "config.yaml")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override max_iterations from the config")
    ap.add_argument("--no-resume", action="store_true", help="start fresh, ignoring checkpoints")
    ap.add_argument("--llm-calls", type=pathlib.Path, default=BASE_DIR / "llm_calls",
                    help="where to record LLM exchanges; --llm-calls '' disables")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Discarded iterations leave no program record, so capture the raw exchanges
    # first; show_llm.py replays the framework's parsing over them.
    if str(args.llm_calls):
        install_llm_capture(args.llm_calls)
        print(f"recording LLM calls to {args.llm_calls}")

    install_behavior_dedup()
    print("behavior-fingerprint dedup installed (one identical-cut program per island)")

    # Pack the evolvable sources into the single file OpenEvolve expects. The
    # .txt suffix also becomes config.file_suffix, so temporary programs written
    # during evaluation get the right extension.
    initial = args.output / "initial_program.txt"
    initial.write_text(pack(args.source))
    print(f"packed {args.source} -> {initial} ({len(initial.read_text())} chars)")

    config = Config.from_yaml(args.config)
    if args.iterations is not None:
        config.max_iterations = args.iterations

    evolve = OpenEvolve(
        initial_program_path=str(initial),
        evaluation_file=str(BASE_DIR / "evaluator.py"),
        config=config,
        output_dir=str(args.output),
    )

    checkpoint = None if args.no_resume else latest_checkpoint(args.output / "checkpoints")
    print(f"checkpoint: {checkpoint or 'none (fresh run)'}")
    print(f"iterations: {config.max_iterations}")

    asyncio.run(run(evolve, checkpoint))


if __name__ == "__main__":
    main()
