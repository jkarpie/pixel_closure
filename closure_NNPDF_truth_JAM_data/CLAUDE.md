# Plots are the deliverable, not a garnish

**If an agent is running anything in this package, it should produce figures, not just
tables.** The owner reads these results visually; a number that only ever appeared in a
transcript has not been communicated.

Plot anything a human is meant to judge -- coverage, pulls, a parameter scan, a
before/after comparison -- and hand over the file.

Reuse what is here rather than rebuilding figure plumbing:

* `run_closure.py::_prepare_matplotlib` -- points `MPLCONFIGDIR` at a writable dir
* `run_closure.py::save_figure_both` -- writes PNG **and** PDF from one call
* `run_closure.py::hybrid_xscale` -- log-`x` below 0.1, linear above, joined C1
* `run_closure.py::plot_reproduction` / `plot_comparison`, `plot_ratios.py`,
  `plot_datasets.py` -- existing per-case, cross-Q and dataset figures

Worked example of why: the `t3` "systematic over-estimate" survived several rounds of
tables. What settled it was one figure -- per-field offset against how much that offset
moves when the noise is redrawn -- which separates a noise draw from a machinery bias at a
glance and is nearly unreadable as a table. A second figure (the per-case reproduction
grid) then showed the *mechanism* for the biases that are real: `t8` sitting flat at its
constant prior amplitude for all `x < 1e-2` while the truth falls away, and the valence
fields dipping negative at large `x` to buy back an integral pinned by `cons_norm_*`.
Neither is visible in any summary statistic the suite stores.
