"""Term C — coverage (§2.7), the first of M4's likelihood terms.

    python -m minoodle.model.coverage ablate ~/Documents/minoodle_run/L1

§2.7 asks for a latent abundance `λ` in particle state, per-unitig depth scored under a
predictive distribution, and a coverage-change hazard so that a genuine change (a repeat, a
chimeric junction) is possible but penalised. That is a Gamma-Poisson: `λ ~ Gamma(a₀, b₀)`
with Poisson counts is exactly the NegBin §2.7 names, so the dispersion `φ` *is* the Gamma
shape and there is no second parameter to carry. One `(a, b)` posterior per side, updated by
conjugacy — bounded state, so the §2.4 contract holds trivially.

**Units: k-mer counts with a k-mer-span exposure, not per-base depth** (M4 finding 1, closing
M3 finding 2). The observation for a unitig is `y = round(cov_kmer · span)` against exposure
`m = span = L-k+1`, which is what metaSPAdes actually counted. Two reasons:

- The `cov_base = cov_kmer · L/(L-k+1)` conversion §5 M3 specifies is *wrong*, and the
  correction is a constant, not a length-dependent one: measured on L0, span-weighted k-mer
  coverage over the unitigs longer than 1 kb is 25.73, and base depth × `(R-k+1)/R` is
  25.73 as well (`R` = read length). A read of length `R` carries `R-k+1` k-mers over `R`
  bases; the unitig's own length never enters. `graph.py` now applies that constant, but the
  term does not need it at all — k-mer counts are the observed quantity.
- Exposure-weighting is what makes short unitigs behave. L0's median unitig is 32 bp against
  k = 22, i.e. a span of 11: as a *rate* it is wildly overdispersed, as a *count with exposure
  11* it simply carries little information, which is the truth.

A repeat traversed twice is scored twice. §2.7's v0 is deliberately a local self-consistency
term and not a global deconvolution — the latter couples all paths and is not incrementally
decomposable (that is v1, §7). Do not "fix" this here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minoodle.exact import PriorParams, set_side
from minoodle.graph import UnitigGraph, code, decode
from minoodle.index import KmerIndex, read_fasta, read_fastq, reference_walk, seed_weights
from minoodle.interfaces import Edge, IncrementalLikelihood, OrientedNode, PathGraph, Seed, Side

# (a, b) of the Gamma rate posterior, one per side.
Post = tuple[float, float]
State = tuple[Post, Post]


@dataclass(frozen=True, slots=True)
class CoverageParams:
    """`shape` is §2.7's `φ`; `hazard` is its coverage-change hazard.

    `mean` defaults to the graph's own span-weighted k-mer coverage (`CoverageTerm` estimates
    it) — an empirical-Bayes prior mean, which is legitimate because the prior over paths is
    what has to stay data-independent (§2.3, D18), not the likelihood's hyperparameters.
    """

    shape: float = 2.0
    hazard: float = 0.05  # 0.01 is too aggressive on a real graph — M4 item 1 finding 4
    mean: float | None = None


DEFAULT_COVERAGE = CoverageParams()


def _log_pred(y: float, m: float, a: float, b: float) -> float:
    """NegBin predictive for a count `y` at exposure `m` under `λ ~ Gamma(a, b)` (rate `b`)."""
    return (
        math.lgamma(a + y)
        - math.lgamma(a)
        - math.lgamma(y + 1.0)
        + a * math.log(b / (b + m))
        + y * math.log(m / (b + m))
    )


class CoverageTerm(IncrementalLikelihood):
    """§2.7 as a per-side Gamma-Poisson with a change hazard."""

    def __init__(self, graph: PathGraph, params: CoverageParams = DEFAULT_COVERAGE):
        self.graph = graph
        self.params = params
        # Orientation-independent, so one entry per unitig keyed off `nodes()`. Cheap enough to
        # do eagerly (L0: 156 unitigs) and it keeps `extend` free of graph lookups.
        self.obs: dict[int, tuple[float, float]] = {}
        for n in graph.nodes():
            m = float(len(graph.unitig_seq(n)) - graph.k + 1)
            self.obs[n.unitig] = (round(graph.unitig_kmer_cov(n) * m), m)
        if params.mean is not None:
            mean = params.mean
        else:
            y = sum(v[0] for v in self.obs.values())
            m = sum(v[1] for v in self.obs.values())
            mean = y / m
        self.a0 = params.shape
        self.b0 = params.shape / mean

    # --- scoring ---------------------------------------------------------------------

    def _score(self, n: OrientedNode, post: Post) -> tuple[Post, float]:
        """Score one unitig's count and return the updated posterior.

        The hazard is a two-component mixture — carry on with the running posterior, or restart
        from the prior — collapsed back to a single `(a, b)` by responsibility.

        **The increment is a log-odds against the prior, not a raw likelihood** (M4 finding 2).
        §2.5 requires this of term A for a reason that applies verbatim to term C: a raw
        `log p(y)` is a large negative number for every unitig, so it is a flat toll on
        *extending* against a STOP that costs nothing, and the term stops competing on branch
        choice and starts competing with the length prior. Measured on L1 before the fix: the
        posterior collapsed onto single-unitig paths and the ablation had no branch to score at
        all. Dividing by the same count's probability under the prior alone leaves exactly the
        Bayes factor for "this unitig continues the abundance I have been tracking", which is
        zero when uninformative and signed the right way otherwise.

        ponytail: the collapse is Bayesian online change-point detection truncated to a
        run-length posterior of two atoms. The full run-length posterior is the upgrade path if
        the ablation says the truncation is losing branches; it is not free (state grows with
        path length, so §2.4 would need a horizon).
        """
        y, m = self.obs[n.unitig]
        a, b = post
        h = self.params.hazard
        null = _log_pred(y, m, self.a0, self.b0)
        lp_keep = _log_pred(y, m, a, b)
        if h <= 0.0:
            return (a + y, b + m), lp_keep - null
        total = float(np.logaddexp(math.log1p(-h) + lp_keep, math.log(h) + null))
        r = math.exp(math.log(h) + null - total)
        return (
            (1.0 - r) * (a + y) + r * (self.a0 + y),
            (1.0 - r) * (b + m) + r * (self.b0 + m),
        ), total - null

    # --- IncrementalLikelihood -------------------------------------------------------

    def init(self, seed: Seed) -> tuple[State, float]:
        # The seed unitig is charged here, once, and seeds *both* frontiers' posteriors
        # (§2.6's score-once rule; M1 finding 2 is why this method has a return channel).
        post, incr = self._score(seed.node, (self.a0, self.b0))
        return (post, post), incr

    def extend(self, st: State, e: Edge, side: Side) -> tuple[State, float]:
        post, incr = self._score(e, st[side])
        return set_side(st, side, post), incr

    def stop_logp(self, st: State, side: Side) -> float:
        del st, side  # coverage says nothing about termination (§2.8)
        return 0.0


# --- ablation (§5 M4 item 1) ---------------------------------------------------------


class NoLikelihood(IncrementalLikelihood):
    """The prior-only arm of any ablation. Every increment is zero."""

    def init(self, seed: Seed) -> tuple[None, float]:
        del seed
        return None, 0.0

    def extend(self, st: None, e: Edge, side: Side) -> tuple[None, float]:
        del st, e, side
        return None, 0.0

    def stop_logp(self, st: None, side: Side) -> float:
        del st, side
        return 0.0


def truth_edges(graph: PathGraph, walk: list[OrientedNode]) -> set[tuple[int, int]]:
    """The oriented edges a reference walk uses, in `branch_marginals`' key space.

    Consecutive pairs that are not graph edges are dropped: `reference_walk` skips k-mers it
    cannot place uniquely, so a repeat leaves a gap and the pair spanning it is not an
    adjacency. Both members of each bidirected twin are included, because the left frontier
    walks the flipped orientation and `branch_marginals` keys it as walked.
    """
    out: set[tuple[int, int]] = set()
    for a, b in itertools.pairwise(walk):
        if code(b) in graph.out_edges(a):
            out.add((code(a), code(b)))
            out.add((code(b.flipped()), code(a.flipped())))
    return out


def true_edge_mass(
    marginals: dict[tuple[int, int], float], truth: set[tuple[int, int]], graph: PathGraph
) -> tuple[float, float]:
    """`(mass on this genome's edges, mass on every edge leaving one of its branch points)`.

    Two restrictions, both load-bearing:

    - **Branching nodes only.** A non-branching node carries no information about the term —
      the path had nowhere else to go — so counting it would dilute the metric with whatever
      the prior did anyway.
    - **Branch points on this genome's walk only.** Otherwise the denominator is the whole
      graph and the metric measures which organism the sampler *visited*, not whether it took
      the right turn once there. On L1 that distinction is everything: the 10:1 minor organism
      holds a small share of the graph's branch points, so a whole-graph denominator reports
      its abundance instead of its accuracy.
    """
    sources = {a for a, _ in truth}
    good = total = 0.0
    for (a, b), w in marginals.items():
        if a not in sources or graph.out_edges(decode(a)).size < 2:
            continue
        total += w
        if (a, b) in truth:
            good += w
    return good, total


def ablate(
    run_dir: Path,
    particles: int,
    islands: int,
    rho: float,
    seed: int,
    hazards: list[float],
    max_bases: int | None,
) -> list[str]:
    """§5 M4's gate for item 1: does the term on its own move mass onto the ground truth?

    Ground truth on a real graph is each reference genome's *walk through the unitig graph*,
    read off the existing k-mer index — no new alignment machinery (`index.reference_walk`).
    Reported per genome, so on L1 the 10:1 minor organism is visible separately; that is the
    whole point of the rung.
    """
    from minoodle import sampler  # local: nothing in `sampler` imports `model`, keep it so

    manifest = json.loads((run_dir / "manifest.json").read_text())
    graph = UnitigGraph.from_gfa(run_dir / "asm" / "assembly_graph_with_scaffolds.gfa")
    index = KmerIndex(graph)

    reads = [r for p in sorted(run_dir.glob("sim_reads_R*.fastq.gz")) for r in read_fastq(p)]
    proposal = sampler.weighted_seed_proposal(graph, seed_weights(index, reads), eps=0.01)
    # `rho` sets the *prior* expected total length to `2/rho` — the fixtures' 0.04 would sample
    # 50-base paths off a 100 kb genome. `max_bases` is a budget the prior cannot supply:
    # per M4 finding 3 the coverage term's increment is unbounded above, so on a self-consistent
    # stretch it out-argues any geometric stop and the walk does not terminate. Both arms run
    # under the same budget, so the comparison is still like-for-like.
    cfg = sampler.SMCConfig(max(1, particles // islands), islands, max_bases, seed=seed)

    arms: dict[str, IncrementalLikelihood] = {"prior only": NoLikelihood()}
    for h in hazards:
        arms[f"hazard {h:g}"] = CoverageTerm(graph, CoverageParams(hazard=h))
    marginals: dict[str, dict[tuple[int, int], float]] = {}
    for label, lik in arms.items():
        res = sampler.run(graph, lik, PriorParams(rho), cfg, proposal=proposal)
        post = res.posterior()
        marginals[label] = sampler.branch_marginals(post)
        # Mean unitigs per path is the diagnostic that matters: an unnormalised term shows up
        # as a collapse to single-unitig paths long before it shows up in the branch metric.
        nodes = np.mean([1 + len(left) + len(right) for _, left, right in post])
        print(
            f"{label:>12}  log Z {res.log_Z:+.3f}  island spread {res.log_Z_spread:.3f}  "
            f"{len(post)} states  {nodes:.2f} unitigs/path"
        )

    failures: list[str] = []
    for ref in manifest["references"]:
        path = Path(ref["path"])
        walk = reference_walk(index, next(read_fasta(path)))
        truth = truth_edges(graph, walk)
        frac = {}
        for label in arms:
            good, total = true_edge_mass(marginals[label], truth, graph)
            frac[label] = good / total if total else float("nan")
        # The denominator is reported alongside: an arm that never walks this organism has no
        # accuracy to report, and that is a different result from getting its branches wrong.
        print(
            f"{path.stem:>28}  {len(walk)} nodes, {len(truth) // 2} edges  "
            + "  ".join(
                f"{k} {v:.4f} (mass {true_edge_mass(marginals[k], truth, graph)[1]:.3f})"
                for k, v in frac.items()
            )
        )
        # `nan` propagates as a loss here, deliberately: an arm that never walked this organism
        # has not raised anything about it, whatever it did elsewhere (finding 5).
        best = max((k for k in arms if k != "prior only"), key=lambda k: frac[k])
        if not frac[best] > frac["prior only"]:
            failures.append(
                f"{path.stem}: coverage did not raise true-edge mass at any hazard "
                f"({frac['prior only']:.4f} -> best {frac[best]:.4f})"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minoodle.model.coverage", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ablate", help="§5 M4 item 1's ablation on a generated dataset")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--particles", type=int, default=4000)
    p.add_argument("--islands", type=int, default=4)
    p.add_argument("--rho", type=float, default=1e-3, help="2/rho is the expected total length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--hazard",
        type=float,
        nargs="+",
        default=[0.05],
        help="one arm per value; on a real graph this is the load-bearing knob, not a nuisance",
    )
    p.add_argument("--max-bases", type=int, default=40_000, help="0 for unbounded (M4 finding 3)")

    args = parser.parse_args(argv)
    failures = ablate(
        args.run_dir.expanduser(),
        args.particles,
        args.islands,
        args.rho,
        args.seed,
        args.hazard,
        args.max_bases or None,
    )
    for f in failures:
        print(f, file=sys.stderr)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
