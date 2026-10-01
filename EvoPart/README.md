# EvoPart

A single-file, single-threaded multilevel hypergraph partitioner (1,186 lines of C++ excluding comments and blank lines), evolved from a round-robin stub by an
LLM coding agent (see `../OpenEvolve` for the harness). Objective: minimise the number of cut nets
subject to the balance constraint |V_p| <= floor((1 + eps) * ceil(n / k)) for every part p.

## Build

    make            # g++ -std=gnu++17 -O3 -march=native, produces ./our_hparter

## Run

    ./our_hparter <input.hgr> --k=K --eps=EPS --seed=S --out=<partition file>

* Input: hMETIS `.hgr` format, unweighted (first non-comment line `<num_hyperedges> <num_vertices>`,
  then one line per hyperedge with 1-indexed vertex ids; lines starting with `%` are comments).
* Output: one line per vertex, in vertex order, holding its part id in [0, k).
* Diagnostics (per-phase time breakdown, restarts, V-cycles, flow steps, pair sweeps, a histogram of
  cut nets by size) are printed to stdout; the last line is `RESULT cut=<cut> feasible=<0|1>`.

## Results

Geometric means of per-instance ratios over ISPD98 (18 instances) and Titan23 (22 instances),
k in {2, 3, 4}, 3 seeds, all systems single-threaded, KaHyPar (cut_kKaHyPar_sea20) = 1.00:

| suite   | avg cut | wall time |
|---------|---------|-----------|
| ISPD98  | 0.992   | 1.96      |
| Titan23 | 0.908   | 0.30      |
| all     | 0.945   | 0.70      |

The program was evolved on ten training instances (ibm01, ibm05, ibm09, ibm13, ibm17, neuron,
cholesky_mc, minres, openCV, gsm_switch) at k in {2, 3, 4}; the other 30 instances and k = 5..8 were
never seen during evolution.
