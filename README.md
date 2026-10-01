# EvoPart: an evolved open hypergraph partitioner

**Project page: [`index.html`](index.html)** - a self-contained page with the method, the evolution run, the full
results and the ablations. Download it and open it in a browser, or enable GitHub Pages (Settings > Pages >
branch `main`, folder `/`) to have it served directly.

* `index.html` - the project page.
* `EvoPart/` - the evolved partitioner: one C++ file (1,186 lines excluding comments and blank lines), single-threaded, with a makefile.
* `OpenEvolve/` - the OpenEvolve framework (https://github.com/algorithmicsuperintelligence/openevolve)
  together with the example that produced EvoPart, `OpenEvolve/examples/hypergraph_partitioning/`. The ISPD98 and Titan23
  hypergraphs are not included; see `OpenEvolve/examples/hypergraph_partitioning/README.md`.

See the README in each directory.
