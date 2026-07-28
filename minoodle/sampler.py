"""SMC over paths (§3, §5 M2).

The fully-adapted step of §3.2, systematic resampling, ancestry tracking and the island model,
written as a direct loop rather than on top of Chopin's `particles` (§4) — the state space is a
discrete bidirected graph and the resampling routine worth borrowing is eight lines.

    python -m minoodle.sampler validate fixtures/manifest.json --particles 100000

Three things carry the whole correctness argument, and all three are weight bookkeeping:

1. **STOP is one of the fully-adapted alternatives.** §3.2's pseudocode enumerates out-edges
   only; a path may end at any node, and `exact.enumerate_paths` emits a record at every node
   visited, so leaving STOP out of the `logsumexp` targets a different measure and `log Ẑ`
   silently disagrees with `log Z`.
2. **The prior is M1's normalised per-base geometric**, reused verbatim from `exact` rather
   than re-derived here. Two implementations of the same formula is two chances to get the
   length accounting wrong.
3. **Truncation matches the enumerator.** `enumerate_paths` drops an extension that would push
   the sequence past `max_bases` but still scores STOP with `dead_end` meaning *graph*
   out-degree zero, not "nothing legal left". The lost mass (6e-7 on `repeat_twice`) is part of
   the target the fixtures encode, so the sampler has to lose exactly the same mass.

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
    ToyGraph,
    code,
    decode,
    enumerate_paths,
    start_logp,
    survive_logp,
    terminal_logp,
)
from minoodle.interfaces import IncrementalLikelihood, OrientedNode

Path_ = tuple[OrientedNode, ...]
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
    nodes: np.ndarray  # (T, N) node codes, recorded before each step's resampling
    parents: list[np.ndarray]
    path_len: np.ndarray

    def paths(self) -> list[Path_]:
        lin = diag.lineage(self.parents)
        taken = np.take_along_axis(self.nodes, lin, axis=1)
        return [
            tuple(decode(int(c)) for c in taken[: self.path_len[i], i])
            for i in range(self.log_w.size)
        ]

    def posterior(self) -> dict[Path_, float]:
        """Deduplicated with *summed* normalised weights (§3.4)."""
        w = diag.normalise(self.log_w)
        out: dict[Path_, float] = {}
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

    def posterior(self) -> dict[Path_, float]:
        """Islands pooled in proportion to their own `Ẑ`, which is what makes the pooled
        estimate the ratio estimator rather than an average of ratios."""
        z = np.array([i.log_Z for i in self.islands])
        share = np.exp(z - z.max())
        share /= share.sum()
        out: dict[Path_, float] = {}
        for s, island in zip(share, self.islands, strict=True):
            for p, w in island.posterior().items():
                out[p] = out.get(p, 0.0) + float(s) * w
        return out

    def min_per_locus_ess(self) -> float:
        return min(float(i.per_locus_ess().min()) for i in self.islands)


def run_island(
    graph: ToyGraph,
    likelihood: IncrementalLikelihood,
    params: PriorParams,
    cfg: SMCConfig,
    uniforms: Uniforms,
) -> IslandResult:
    n = cfg.n_particles

    # Fully-adapted start step: the start node is drawn from its exact conditional and the
    # weight picks up the whole normalising constant, including the mass of any start unitig
    # already longer than `max_bases` (dropped, as `enumerate_paths` drops it).
    # ponytail: every start's likelihood state is materialised once per island. Fine at 2n
    # nodes; if M3's graph makes that heavy, make it lazy per sampled node.
    opts0: list[tuple[OrientedNode, Any, int]] = []
    gam0: list[float] = []
    for s in graph.nodes():
        nb = graph.new_bases(s, first=True)
        if nb > cfg.max_bases:
            continue
        st, incr = likelihood.init(s)
        opts0.append((s, st, nb))
        gam0.append(start_logp(graph, s, params) + incr)
    total0 = _logsumexp(gam0)

    u = uniforms(n)
    chosen = [opts0[_pick(gam0, total0, float(ui))] for ui in u]
    node = [c[0] for c in chosen]
    state = [c[1] for c in chosen]
    pending = np.array([c[2] for c in chosen], dtype=np.int64)
    total = pending.copy()
    log_w = np.full(n, total0)
    alive = np.ones(n, dtype=bool)
    path_len = np.ones(n, dtype=np.int64)

    nodes_hist: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    log_Z = 0.0
    first_step = True

    while True:
        if not first_step:
            u = uniforms(n)
            for i in np.flatnonzero(alive):
                nd = node[i]
                out = graph.out_edges(nd)
                gammas = [terminal_logp(int(pending[i]), out.size == 0, params)
                          + likelihood.stop_logp(state[i])]
                opts: list[tuple[OrientedNode, Any, int] | None] = [None]
                if out.size:
                    base = survive_logp(int(pending[i]), params) - math.log(out.size)
                    for c in out:
                        nxt = decode(int(c))
                        nb = graph.new_bases(nxt, first=False)
                        if total[i] + nb > cfg.max_bases:
                            continue
                        st2, incr = likelihood.extend(state[i], nxt)
                        gammas.append(base + incr)
                        opts.append((nxt, st2, nb))
                step_total = _logsumexp(gammas)
                log_w[i] += step_total
                choice = opts[_pick(gammas, step_total, float(u[i]))]
                if choice is None:
                    alive[i] = False
                else:
                    node[i], state[i], nb = choice
                    pending[i] = nb
                    total[i] += nb
                    path_len[i] += 1
        first_step = False

        nodes_hist.append(np.array([code(nd) for nd in node], dtype=np.int64))
        if diag.ess(log_w) < cfg.ess_frac * n:
            log_Z += _logsumexp(list(log_w)) - math.log(n)
            idx = diag.systematic_resample(log_w, float(uniforms(1)[0]))
            node = [node[j] for j in idx]
            state = [state[j] for j in idx]
            pending, total, alive, path_len = (
                pending[idx], total[idx], alive[idx], path_len[idx]
            )
            log_w = np.zeros(n)
            parents.append(idx)
        else:
            parents.append(np.arange(n))

        if not alive.any():
            break

    log_Z += _logsumexp(list(log_w)) - math.log(n)
    return IslandResult(log_Z, log_w, np.array(nodes_hist), parents, path_len)


def run(
    graph: ToyGraph,
    likelihood: IncrementalLikelihood,
    params: PriorParams = DEFAULT_PRIOR,
    cfg: SMCConfig = DEFAULT_CONFIG,
    uniforms: Uniforms | None = None,
) -> SMCResult:
    """Islands share one uniform stream but never resample across each other (§3.3 item 4)."""
    if uniforms is None:
        rng = np.random.default_rng(cfg.seed)
        uniforms = rng.random
    return SMCResult([run_island(graph, likelihood, params, cfg, uniforms)
                      for _ in range(cfg.n_islands)])


def branch_marginals(dist: dict[Path_, float]) -> dict[tuple[int, int], float]:
    """`p(edge used)` summed over paths (§3.4) — for downstream statistics, more useful than
    the sequences. Takes any `{path: probability}` mapping, so the exact enumeration and the
    sampler are compared through the same function."""
    out: dict[tuple[int, int], float] = {}
    for path, p in dist.items():
        for a, b in itertools.pairwise(path):
            key = (code(a), code(b))
            out[key] = out.get(key, 0.0) + p
    return out


# --- validation gate (§5 M2) ---------------------------------------------------------------


def sbc_ranks(
    graph: ToyGraph,
    likelihood: IncrementalLikelihood,
    params: PriorParams,
    cfg: SMCConfig,
    stat: dict[Path_, float],
    exact_pi: dict[Path_, float],
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
        params = PriorParams(e["rho"], e["uniform_start"])
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
    params = PriorParams(e["rho"], e["uniform_start"])
    lik = GCBias(g, e["beta"])
    exact = enumerate_paths(g, lik, params, e["max_bases"])
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
