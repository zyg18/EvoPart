# Evolving a hypergraph partitioner from scratch

This example evolves a complete multilevel hypergraph partitioner with OpenEvolve, starting from a
stub that only assigns vertex i to part i mod k. Everything the LLM edits is one C++ file,
`initial_program/partition.cpp`; the only fixed file is the makefile.

## Layout

* `initial_program/partition.cpp` - the starting program (parser, round-robin assignment, output).
* `initial_program/makefile` - fixed build (`g++ -std=gnu++17 -O3 -march=native`, target `our_hparter`).
* `evaluator.py` - builds a candidate and runs it on the training grid (10 instances x k in {2,3,4}
  x 3 seeds); recomputes cut and balance from the written partition files; computes the score
  (cut quality - time term - code size), the behavioral MAP-Elites descriptors and the cut-vector
  fingerprint; three cascade tiers with cached runs; returns per-cell measurements and the program's
  stdout as artifacts for the next prompt.
* `baselines.json` - KaHyPar reference cut and wall time per training cell (medians of 3 seeds,
  rescored by `utils/hgr.py`); `make_baselines.py` regenerates it.
* `config.yaml` - OpenEvolve configuration: the system prompt, LLM ensemble (Claude Opus 5 / Fable 5.1
  through the `claude` CLI), islands and MAP-Elites dimensions (`log_speed` x `k_profile`), cascade
  thresholds, and the changes-description mechanism.
* `start_evolution.py` - packs the evolvable file, installs the per-island fingerprint deduplication
  and the LLM-call recorder, and runs OpenEvolve (resumes from the latest checkpoint).
* `utils/` - `code_to_query.py` (pack/unpack of the evolvable file), `hgr.py` (cut and balance),
  `llm_capture.py` (claude CLI backend without tools, records every exchange).
* `run_batches.sh` - runs 200 iterations in batches with rests between them.
* `bench_best.py`, `patch_xlsx.py`, `make_baselines.py` - full-benchmark tools.
* `show_program.py`, `show_log.py`, `show_llm.py` - inspection helpers for a finished run.

## Setup

1. Install OpenEvolve from the repository root: `pip install -e .`
2. The `claude` CLI must be installed and logged in (the run uses `claude -p` with tools disabled).
3. Datasets: put the ISPD98 (`ibm01.hgr` ... `ibm18.hgr`) and Titan23 hypergraphs in hMETIS `.hgr`
   format under `<repo>/dataset/ISPD98/` and `<repo>/dataset/Titan23/`, or point `HGP_DATASET_DIR`
   at a directory with those two subdirectories. The ISPD98 benchmark suite is available from
   the ISPD98 circuit benchmark distribution and Titan23 from the Titan benchmark release; the
   converted `.hgr` files used in the paper are provided on request.
4. Baselines: `baselines.json` is included. To regenerate it, point `HMETIS_BIN`, `KAHYPAR_BIN` and
   `KAHYPAR_PRESET` at your binaries and run `make_baselines.py`.

## Run

    python3 start_evolution.py --iterations 200          # or ./run_batches.sh

Outputs go to `openevolve_output/` (checkpoints, best program) and `llm_calls/` (every prompt and
reply); `eval_log.jsonl` records every evaluation.

## Benchmark a program

    python3 -c "import evaluator, pathlib; evaluator.unpack(pathlib.Path('openevolve_output/best/best_program.txt').read_text(), 'build_best', evaluator.PRISTINE)"
    make -C build_best
    python3 bench_best.py build_best/our_hparter best ours_full_benchmark.csv

`bench_best.py` runs all 40 instances at k in {2,3,4} with 3 seeds and rescored partitions.
