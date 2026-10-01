# EvoPart: an evolved open hypergraph partitioner

**Project page: https://anonymous.4open.science/r/EvoPart-4F6C/index.html** - the page works with JavaScript
on or off. With JavaScript off you see the static version; for the interactive charts, tables and animation,
turn on JavaScript (on Anonymous GitHub, click **JS on** at the top right of the file view) or download
`index.html` and open it in a browser.

* `EvoPart/` - the evolved partitioner: one C++ file (1,186 lines excluding comments and blank lines), single-threaded, with a makefile.
* `OpenEvolve/` - the OpenEvolve framework (https://github.com/algorithmicsuperintelligence/openevolve)
  together with the example that produced EvoPart, `OpenEvolve/examples/hypergraph_partitioning/`. The ISPD98 and Titan23
  hypergraphs are not included; see `OpenEvolve/examples/hypergraph_partitioning/README.md`.

See the README in each directory.
