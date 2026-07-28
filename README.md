# minoodle

Probabilistic sampling of sequences from a metagenome assembly graph, as an alternative to
rules-based consensus assembly.

`minoodle` reads as a diminutive of "noodle" — that's a coincidence of naming, not a comment
on the method.

## What it does

Given a metaSPAdes assembly graph (GFA) and the paired-end reads that built it, minoodle runs
sequential Monte Carlo to draw a **weighted** multiset of sequences (paths through the graph)
from a defined target distribution, rather than collapsing the graph to a single rules-based
consensus per locus. Output includes per-sequence importance weights, `log Ẑ`, and per-branch
marginal probabilities.

The target distribution combines a path prior with a pseudo-likelihood built from read
congruence (pair-HMM error model), paired-end fragment congruence, and local coverage
consistency. It is a **pseudo-posterior**, not a true generative posterior over the read set —
reads come from the whole community, not one path — and the sampler is exact with respect to
that defined target, not to some unstated "true" posterior.

## Status

Early stage. See [docs/minoodle-implementation-plan.md](docs/minoodle-implementation-plan.md)
for the full statistical formulation, architecture, milestones, and open decisions. Development
target for v0 is single-sample paired-end short reads, 16S-derived synthetic shotgun data at
k=21.

## License

TBD.
