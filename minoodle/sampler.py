"""SMC over paths (§3, §5 M2).

The fully-adapted step of §3.2, systematic resampling, ancestry tracking and the island model,
written as a direct loop rather than on top of Chopin's `particles` (§4) — the state space is a
discrete bidirected graph and the resampling routine worth borrowing is eight lines.

    python -m minoodle.sampler validate fixtures/manifest.json --particles 100000

Four things carry the whole correctness argument, and all four are weight bookkeeping:

1. **STOP is one of the fully-adapted alternatives, per side.** §3.2's pseudocode enumerates
   out-edges only; a frontier may stop anywhere, so leaving STOP out of the `logsumexp` targets
   a different measure and `log Ẑ` silently disagrees with `log Z`. A particle is absorbed only
   when *both* sides have stopped.
2. **The prior is M1's normalised per-base geometric**, reused verbatim from `exact` rather
   than re-derived here — including `next_side`, so the alternation cannot drift. Two
   implementations of the same formula is two chances to get the length accounting wrong.
3. **Truncation matches the enumerator.** `enumerate_paths` drops an extension that would push
   the sequence past `max_bases` but still scores STOP with `dead_end` meaning *graph*
   out-degree zero, not "nothing legal left". The lost mass (2e-3 on `repeat_twice`, whose
   budget is deliberately tight enough to make that mass visible) is part of the target the
   fixtures encode, so the sampler has to lose exactly the same mass.
4. **The seed is a proposal, not the prior (D18).** Seeds are drawn from `q_seed` and the
   initial log-weight carries `log p_seed - log q_seed`. The target does not depend on
   `q_seed`, so swapping the proposal must not move the posterior — that invariance is the
   test that catches a mis-normalised or partial-support `q`. Under `uniform_seed_proposal`
   (`q = p`) prior-only `log Ẑ` is 0 exactly; under any other proposal it is 0 only in
   expectation.

Per §4.2 the engine takes an injected `uniforms(k) -> ndarray` source and consumes a
predictable number of draws per step (`N` for the choices, one more if resampling triggers), so
a recorded stream replays byte-identically in the Rust port.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from minoodle import diagnostics as diag
from minoodle.exact import (
    DEFAULT_PRIOR,
    TOY_GRAPHS,
    GCBias,
    PriorParams,
    SeedProposal,
    StatePath,
    ToyGraph,
    code,
    decode,
    enumerate_paths,
    next_side,
    seed_logp,
    set_side,
    survive_logp,
    terminal_logp,
    uniform_seed_proposal,
    weighted_seed_proposal,
)
from minoodle.interfaces import (
    IncrementalLikelihood,
    OrientedNode,
    PathGraph,
    Seed,
    Side,
)

Uniforms = Callable[[int], np.ndarray]


@dataclass(frozen=True, slots=True)
class SMCConfig:
    """`n_particles` is *per island*; `n_islands` runs are independent (§3.3 item 4)."""

    n_particles: int = 1000
    n_islands: int = 4
    max_bases: int = 250
    ess_frac: float = 0.5
    seed: int = 0


DEFAULT_CONFIG = SMCConfig()


def _logsumexp(x: list[float]) -> float:
    m = max(x)
    if m == -math.inf:  # every alternative impossible; `m - m` would be a nan
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in x))


def _pick(gammas: list[float], total: float, u: float) -> int:
    acc = 0.0
    for i, g in enumerate(gammas):
        acc += math.exp(g - total)
        if u < acc:
            return i
    return len(gammas) - 1


@dataclass
class IslandResult:
    """One island's particles plus the ancestry needed to reconstruct and audit them."""

    log_Z: float
    log_w: np.ndarray  # incremental weights since the last resampling
    nodes: np.ndarray  # (T, N) node codes, -1 where the step was a STOP or the particle was
    sides: np.ndarray  # (T, N) which frontier that step extended
    parents: list[np.ndarray]
    seeds: list[Seed]  # per surviving particle; rides along through resampling

    def paths(self) -> list[StatePath]:
        lin = diag.lineage(self.parents)
        nodes = np.take_along_axis(self.nodes, lin, axis=1)
        sides = np.take_along_axis(self.sides, lin, axis=1)
        out: list[StatePath] = []
        for i in range(self.log_w.size):
            walks: tuple[list, list] = ([], [])
            for c, s in zip(nodes[:, i], sides[:, i], strict=True):
                if c >= 0:
                    walks[int(s)].append(decode(int(c)))
            out.append((self.seeds[i], tuple(walks[Side.LEFT]), tuple(walks[Side.RIGHT])))
        return out

    def posterior(self) -> dict[StatePath, float]:
        """Deduplicated with *summed* normalised weights (§3.4)."""
        w = diag.normalise(self.log_w)
        out: dict[StatePath, float] = {}
        for p, wi in zip(self.paths(), w, strict=True):
            out[p] = out.get(p, 0.0) + float(wi)
        return out

    def per_locus_ess(self) -> np.ndarray:
        return diag.per_locus_ess(diag.lineage(self.parents), self.log_w)


@dataclass
class SMCResult:
    islands: list[IslandResult]

    @property
    def log_Z(self) -> float:
        z = [i.log_Z for i in self.islands]
        return _logsumexp(z) - math.log(len(z))

    @property
    def log_Z_spread(self) -> float:
        """Between-island standard deviation — the convergence diagnostic, in place of a
        convergence assertion (§3.4)."""
        return float(np.std([i.log_Z for i in self.islands], ddof=1)) if len(self.islands) > 1 else 0.0

    def posterior(self) -> dict[StatePath, float]:
        """Islands pooled in proportion to their own `Ẑ`, which is what makes the pooled
        estimate the ratio estimator rather than an average of ratios."""
        z = np.array([i.log_Z for i in self.islands])
        share = np.exp(z - z.max())
        share /= share.sum()
        out: dict[StatePath, float] = {}
        for s, island in zip(share, self.islands, strict=True):
            for p, w in island.posterior().items():
                out[p] = out.get(p, 0.0) + float(s) * w
        return out

    def min_per_locus_ess(self) -> float:
        return min(float(i.per_locus_ess().min()) for i in self.islands)


def run_island(
    graph: PathGraph,
    likelihood: IncrementalLikelihood,
    params: PriorParams,
    cfg: SMCConfig,
    uniforms: Uniforms,
    proposal: SeedProposal,
) -> IslandResult:
    n = cfg.n_particles
    lp_seed = seed_logp(graph)
    log_q = proposal.log_q

    # Seeds come from `q_seed`, one uniform each, and the weight carries `log p - log q`
    # (D18). Not fully adapted over seeds the way the old start step was over start nodes:
    # that is O(#seeds) per island, which a real graph makes impossible.
    cdf = np.cumsum(proposal.q)
    idx0 = np.minimum(np.searchsorted(cdf, uniforms(n), side="right"), len(cdf) - 1)

    node: list[tuple[OrientedNode, OrientedNode]] = []
    state: list[Any] = []
    pending = np.zeros((n, 2), dtype=np.int64)
    log_w = np.empty(n)
    seeds: list[Seed] = []
    total = np.zeros(n, dtype=np.int64)
    for i, j in enumerate(idx0):
        seed = proposal.seeds[j]
        seed_len = len(graph.unitig_seq(seed.node))
        if seed_len > cfg.max_bases:
            # ponytail: `enumerate_paths` drops these seeds outright, so a proposal that can
            # reach one targets a different measure. Toy graphs never do. M4 revisits when
            # real unitigs meet a real budget — probably by making `max_bases` unbounded.
            raise ValueError(f"seed unitig {seed_len} bp exceeds max_bases {cfg.max_bases}")
        st, incr = likelihood.init(seed)
        seeds.append(seed)
        state.append(st)
        node.append((seed.node.flipped(), seed.node))
        pending[i] = (seed.offset, seed_len - graph.k - seed.offset)
        total[i] = seed_len
        log_w[i] = lp_seed - log_q[j] + incr

    stopped = np.zeros((n, 2), dtype=bool)
    nodes_hist: list[np.ndarray] = []
    sides_hist: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    log_Z = 0.0
    t = 0

    while True:
        u = uniforms(n)
        row_node = np.full(n, -1, dtype=np.int64)
        row_side = np.zeros(n, dtype=np.int8)
        for i in np.flatnonzero(~stopped.all(axis=1)):
            side = next_side(t, bool(stopped[i, 0]), bool(stopped[i, 1]))
            assert side is not None
            out = graph.out_edges(node[i][side])
            # STOP is inside the `logsumexp`, and a dead end leaves it there alone with its
            # forced weight of 0 — not a branch around the alternative set (§3.2).
            gammas = [
                terminal_logp(int(pending[i, side]), out.size == 0, params)
                + likelihood.stop_logp(state[i], side)
            ]
            opts: list[tuple[OrientedNode, Any, int] | None] = [None]
            if out.size:
                base = survive_logp(int(pending[i, side]), params) - math.log(out.size)
                for c in out:
                    nxt = decode(int(c))
                    nb = graph.new_bases(nxt, first=False)
                    if total[i] + nb > cfg.max_bases:
                        continue  # dropped, exactly as `enumerate_paths` drops it
                    st2, incr = likelihood.extend(state[i], nxt, side)
                    gammas.append(base + incr)
                    opts.append((nxt, st2, nb))
            step_total = _logsumexp(gammas)
            if step_total == -math.inf:
                # No bases left for this side to stop in *and* every extension over budget.
                # `enumerate_paths` drops the subtree at that point, so the particle has to
                # die with zero weight rather than pick an impossible alternative.
                log_w[i] = -math.inf
                stopped[i] = True
                continue
            log_w[i] += step_total
            choice = opts[_pick(gammas, step_total, float(u[i]))]
            if choice is None:
                stopped[i, side] = True
            else:
                nxt, state[i], nb = choice
                node[i] = set_side(node[i], side, nxt)
                pending[i, side] = nb
                total[i] += nb
                row_node[i], row_side[i] = code(nxt), int(side)

        nodes_hist.append(row_node)
        sides_hist.append(row_side)
        if diag.ess(log_w) < cfg.ess_frac * n:
            log_Z += _logsumexp(list(log_w)) - math.log(n)
            idx = diag.systematic_resample(log_w, float(uniforms(1)[0]))
            node = [node[j] for j in idx]
            state = [state[j] for j in idx]
            seeds = [seeds[j] for j in idx]
            pending, total, stopped = pending[idx], total[idx], stopped[idx]
            log_w = np.zeros(n)
            parents.append(idx)
        else:
            parents.append(np.arange(n))

        if stopped.all():
            break
        t += 1

    log_Z += _logsumexp(list(log_w)) - math.log(n)
    return IslandResult(
        log_Z, log_w, np.array(nodes_hist), np.array(sides_hist), parents, seeds
    )


def run(
    graph: PathGraph,
    likelihood: IncrementalLikelihood,
    params: PriorParams = DEFAULT_PRIOR,
    cfg: SMCConfig = DEFAULT_CONFIG,
    uniforms: Uniforms | None = None,
    proposal: SeedProposal | None = None,
) -> SMCResult:
    """Islands share one uniform stream but never resample across each other (§3.3 item 4).

    `proposal` defaults to uniform seeding, i.e. `q_seed = p_seed` — the configuration the
    exact `log Ẑ` tests are stated under.
    """
    if uniforms is None:
        rng = np.random.default_rng(cfg.seed)
        uniforms = rng.random
    if proposal is None:
        proposal = uniform_seed_proposal(graph)
    return SMCResult([run_island(graph, likelihood, params, cfg, uniforms, proposal)
                      for _ in range(cfg.n_islands)])


def branch_marginals(dist: dict[StatePath, float]) -> dict[tuple[int, int], float]:
    """`p(edge used)` summed over states (§3.4) — for downstream statistics, more useful than
    the sequences. Takes any `{state: probability}` mapping, so the exact enumeration and the
    sampler are compared through the same function.

    Left-side edges are keyed in the flipped orientation they were actually walked in; the
    two sides are therefore never conflated, and an edge and its twin stay distinguishable.
    """
    out: dict[tuple[int, int], float] = {}
    for (seed, left, right), p in dist.items():
        for walk in ((seed.node.flipped(),) + left, (seed.node,) + right):
            for a, b in itertools.pairwise(walk):
                key = (code(a), code(b))
                out[key] = out.get(key, 0.0) + p
    return out


# --- validation gate (§5 M2, §5 M3.5) ------------------------------------------------------


def _skew(graph: PathGraph, seed: int) -> np.ndarray:
    """A lopsided seed weighting, standing in for coverage-weighted anchors on a toy graph.

    A decade of spread plus a tenth of the seeds at exactly zero — those are the ones only the
    `eps` mixture keeps on support, and they are the case that biases the estimator rather than
    merely inflating its variance. Harsher than this is not a better test, it is just a noisier
    one: the measured margin against a *broken* `p/q` (importance correction dropped) is TV
    0.29 vs 0.03, and adding spread degrades the correct run faster than the broken one.
    """
    rng = np.random.default_rng(seed + 7)
    n = len(graph.seeds())
    w = 10.0 ** rng.uniform(0.0, 1.0, size=n)
    w[rng.random(n) < 0.1] = 0.0
    return w


def sbc_ranks(
    graph: ToyGraph,
    likelihood: IncrementalLikelihood,
    params: PriorParams,
    cfg: SMCConfig,
    stat: dict[StatePath, float],
    exact_pi: dict[StatePath, float],
    reps: int,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rank of an exact draw from π inside the sampler's own draws, over `reps` replicates.

    This is the degenerate form of SBC. Proper simulation-based calibration needs a generative
    likelihood to draw data from; `GCBias` is a bare potential, so until M4 the available test
    is the equivalent statement with the data integrated out: if the sampler's law equals π,
    then an exact draw is exchangeable with `n_draws` sampler draws and its rank is uniform.
    """
    keys = list(exact_pi)
    pi = np.array([exact_pi[k] for k in keys])
    ranks = np.empty(reps, dtype=np.int64)
    for r in range(reps):
        res = run(graph, likelihood, params, cfg, rng.random)
        post = res.posterior()
        paths = list(post)
        w = np.array([post[p] for p in paths])
        draws = rng.choice(len(paths), size=n_draws, p=w / w.sum())
        values = np.array([stat[paths[d]] for d in draws])
        truth = stat[keys[rng.choice(len(keys), p=pi / pi.sum())]]
        ranks[r] = diag.rank(values, rng.random(n_draws), truth, float(rng.random()))
    return ranks


def validate(manifest_path: Path, particles: int, islands: int, seed: int) -> list[str]:
    """Re-run the M2 gate against the committed exact posteriors. Returns failures."""
    manifest = json.loads(manifest_path.read_text())
    by_name = {e["name"]: e for e in manifest["graphs"]}
    failures: list[str] = []

    for factory in TOY_GRAPHS:
        g = factory()
        e = by_name[g.name]
        params = PriorParams(e["rho"])
        lik = GCBias(g, e["beta"])
        exact = enumerate_paths(g, lik, params, e["max_bases"])
        target = exact.posterior()

        cfg = SMCConfig(max(1, particles // islands), islands, e["max_bases"], seed=seed)
        res = run(g, lik, params, cfg)
        got = res.posterior()

        # The gate is TV against what an *exact iid sampler of the same ESS* would incur, not
        # the plan's flat 0.01: on `repeat_twice` (242 atoms) a perfect sampler averages TV
        # 0.0104 at N = 1e5, so the flat threshold is unreachable there by construction. See
        # the M2 notes in CLAUDE.md.
        tv = diag.tv_distance(got, target)
        ess = sum(diag.ess(i.log_w) for i in res.islands)
        ref_mean, ref_hi = diag.multinomial_tv_reference(exact.pi, max(1, int(ess)))
        dz = res.log_Z - e["log_Z"]
        err = max(res.log_Z_spread / math.sqrt(islands), 1e-6)
        bm, bm_exact = branch_marginals(got), branch_marginals(target)
        bm_err = max(
            (abs(v - bm_exact.get(k, 0.0)) for k, v in bm.items()), default=0.0
        )
        print(
            f"{g.name:>15}  TV {tv:.5f} (iid@ESS {ref_mean:.5f}, p99 {ref_hi:.5f})  "
            f"log Z {res.log_Z:+.6f} vs {e['log_Z']:+.6f} (d {dz:+.2e}, island se {err:.2e})  "
            f"branch max err {bm_err:.5f}  ESS {ess:.0f}  "
            f"min per-locus ESS {res.min_per_locus_ess():.0f}"
        )
        if tv > 1.25 * ref_hi:
            failures.append(f"{g.name}: TV {tv:.5f} > 1.25x iid p99 {ref_hi:.5f}")
        # 5x, not 2x: the island spread has 3 degrees of freedom and understates itself often.
        # A real bookkeeping bug moves `log Ẑ` by O(1), not by a few sigma.
        if abs(dz) > 5 * err:
            failures.append(f"{g.name}: log Z off by {dz:+.2e}, island se {err:.2e}")

    rng = np.random.default_rng(seed + 1)
    g = TOY_GRAPHS[1]()
    e = by_name[g.name]
    params = PriorParams(e["rho"])
    lik = GCBias(g, e["beta"])
    exact = enumerate_paths(g, lik, params, e["max_bases"])

    # Proposal invariance (§5 M3.5 item 5): the target does not depend on `q_seed`, so a
    # deliberately skewed proposal must reproduce the same posterior. This is what catches a
    # mis-normalised or partial-support `q`; neither inherited gate would.
    cfg = SMCConfig(max(1, particles // islands), islands, e["max_bases"], seed=seed)
    skewed = weighted_seed_proposal(g, _skew(g, seed), eps=0.05)
    inv = run(g, lik, params, cfg, proposal=skewed)
    tv_inv = diag.tv_distance(inv.posterior(), exact.posterior())
    ess_inv = sum(diag.ess(i.log_w) for i in inv.islands)
    _, ref_inv = diag.multinomial_tv_reference(exact.pi, max(1, int(ess_inv)))
    dz_inv = inv.log_Z - e["log_Z"]
    err_inv = max(inv.log_Z_spread / math.sqrt(islands), 1e-6)
    print(
        f"{'q_seed skewed':>15}  TV {tv_inv:.5f} (iid p99 {ref_inv:.5f})  "
        f"log Z {inv.log_Z:+.6f} (d {dz_inv:+.2e}, island se {err_inv:.2e})  ESS {ess_inv:.0f}"
    )
    # 3x, not the M2 gate's 1.25x: a deliberately bad proposal is *meant* to be less efficient
    # than iid, and the ESS this band is built from is measured after resampling, so it
    # overstates the surviving seed diversity. Calibrated against the failure it exists to
    # catch — dropping the `log p - log q` correction gives TV 0.29 where the correct run
    # gives 0.03, so anywhere in 1.3x-9x separates them.
    if tv_inv > 3 * ref_inv:
        failures.append(f"proposal invariance: TV {tv_inv:.5f} > 3x iid p99 {ref_inv:.5f}")
    if abs(dz_inv) > 5 * err_inv:
        failures.append(f"proposal invariance: log Z off by {dz_inv:+.2e}")

    stat = dict(zip(exact.paths, exact.log_pi_unnorm, strict=True))
    n_draws = 63
    ranks = sbc_ranks(
        g, lik, params, SMCConfig(512, 1, e["max_bases"]), stat, exact.posterior(),
        300, n_draws, rng,
    )
    p = diag.ks_uniform(ranks, n_draws)
    print(f"{'sbc(' + g.name + ')':>15}  KS p = {p:.3f} over {ranks.size} replicates")
    if p < 0.01:
        failures.append(f"sbc: KS p = {p:.4f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minoodle.sampler", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate", help="run the M2 gate against the exact fixtures")
    p.add_argument("manifest", type=Path)
    p.add_argument("--particles", type=int, default=100_000, help="total, split across islands")
    p.add_argument("--islands", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    failures = validate(args.manifest.expanduser(), args.particles, args.islands, args.seed)
    for f in failures:
        print(f, file=sys.stderr)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
