"""Exact enumerator and toy graphs (§5 M1).

Brute-force ground truth for M2: enumerate every path up to a length bound in a tiny graph,
score it under the §2.3 prior and an `IncrementalLikelihood`, and normalise. Without this
there is no way to tell "the sampler works" from "the sampler produces plausible sequences"
(§6.5), so it comes before the sampler.

    python -m minoodle.exact write --out fixtures
    python -m minoodle.exact verify fixtures/manifest.json

Toy graphs live here rather than in `graph.py` because they are fixtures: the metaSPAdes GFA
loader is a different thing and has its own module. `revcomp`/`code`/`decode` live there —
the CSR arrays are what `code` was always for.

**Prior normalisation.** §2.3 is written as `p_seed · Π_t [(1-ρ)^{len_t} · p_edge] · ρ` per
side, which does not sum to 1 over paths — the trailing `ρ` stops at a unitig boundary having
already survived that unitig's bases. Implemented here is the per-base geometric that §2.3's own
prose specifies: a side ends when the stop lands *within* the last unitig's new bases, so its
terminal factor is `1 - (1-ρ)^{len_T}` (and 1 at a dead end, where STOP is forced, §2.8). That
version sums to exactly 1 over `(seed, path)` on an acyclic graph, which is the test that
catches length-accounting bugs.

**Two-sided (rev 10, M3.5).** A state is a `Seed` plus a left and a right walk. The right walk
runs forward out of `seed.node`, the left forward out of `seed.node.flipped()`, so both are
ordinary walks and neither `graph` nor the prior needs a mirrored code path. Each side carries
its own geometric and its own STOP; total length is a sum of two geometrics, mean `2/ρ`, which
is why `ρ` doubled to 0.04 when this landed. The seed's own unitig is scored once, in
`likelihood.init` (§2.6's score-once rule).

**The prior is now RC-symmetric**, which the one-sided version was not: `Seed(n, o)` and
`Seed(n.flipped(), L-k-o)` carry equal `p_seed` and generate mirror-image path sets. That
removes M1's finding 3.

**Seeding is a proposal, not the prior (D18).** `p_seed` is uniform over oriented k-mer
positions; `SeedProposal` below is `q`, and the sampler carries `log p_seed - log q_seed` in the
initial weight. The enumerator never sees `q` — the target does not depend on it, which is
exactly what the proposal-invariance test asserts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from minoodle.graph import code, decode, revcomp
from minoodle.interfaces import (
    Edge,
    IncrementalLikelihood,
    OrientedNode,
    PathGraph,
    Seed,
    Side,
)

FIXTURE_SCHEMA_VERSION = 2

# A fully absorbed state: the seed, the left walk and the right walk (§2.1).
StatePath = tuple[Seed, tuple[OrientedNode, ...], tuple[OrientedNode, ...]]


class ToyGraph(PathGraph):
    """A hand-built bidirected unitig graph held in memory (§5 M1: ≤ 20 nodes).

    Each forward edge `u→v` installs its twin `v̄→ū` on construction, so orientation handling
    — the classic silent-bug source (§5 M3) — cannot be got wrong in a fixture.
    """

    def __init__(
        self,
        name: str,
        k: int,
        seqs: list[bytes],
        edges: list[tuple[int, int]],
        depths: list[float],
    ):
        self.name = name
        self.k = k
        self._seqs = list(seqs)
        self._depths = [float(d) for d in depths]
        adj: dict[OrientedNode, list[OrientedNode]] = {n: [] for n in self.nodes()}
        for u, v in edges:
            if self._seqs[u][-(k - 1) :] != self._seqs[v][: k - 1]:
                raise ValueError(f"edge {u}->{v} violates the k-1 overlap")
            adj[OrientedNode(u, True)].append(OrientedNode(v, True))
            adj[OrientedNode(v, False)].append(OrientedNode(u, False))
        self._out = {n: np.array([code(m) for m in ms], dtype=np.int64) for n, ms in adj.items()}

    def nodes(self) -> list[OrientedNode]:
        return [OrientedNode(u, f) for u in range(len(self._seqs)) for f in (True, False)]

    def out_edges(self, n: OrientedNode) -> np.ndarray:
        return self._out[n]

    def unitig_seq(self, n: OrientedNode) -> bytes:
        seq = self._seqs[n.unitig]
        return seq if n.forward else revcomp(seq)

    def unitig_depth(self, n: OrientedNode) -> np.ndarray:
        return np.full(len(self._seqs[n.unitig]), self._depths[n.unitig])


def _joints(n_unitigs: int, edges: list[tuple[int, int]]) -> list[int]:
    """Union-find over unitig ports; returns 2·n root ids (in-port, out-port per unitig).

    Every successor of `u` must begin with the same k-1 overlap that `u` ends with, and every
    predecessor of `v` must end with `v`'s, so the ports form equivalence classes.
    """
    parent = list(range(2 * n_unitigs))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ru, rv = find(2 * u), find(2 * v + 1)  # out-port of u, in-port of v
        parent[ru] = rv
    return [find(p) for p in range(2 * n_unitigs)]


def build(
    name: str,
    k: int,
    edges: list[tuple[int, int]],
    body_lens: list[int],
    depths: list[float] | None = None,
    seed: int = 0,
) -> ToyGraph:
    """Build a toy graph whose sequences satisfy every edge's k-1 overlap by construction."""
    rng = np.random.default_rng(seed)
    ports = _joints(len(body_lens), edges)

    def draw(n: int) -> bytes:
        return rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), size=n).tobytes()

    joint = {r: draw(k - 1) for r in sorted(set(ports))}
    seqs = [
        joint[ports[2 * u + 1]] + draw(body) + joint[ports[2 * u]]
        for u, body in enumerate(body_lens)
    ]
    return ToyGraph(name, k, seqs, edges, depths or [10.0] * len(body_lens))


def chain() -> ToyGraph:
    return build("chain", 5, [(0, 1), (1, 2)], [6, 4, 8], seed=1)


def bubble() -> ToyGraph:
    """Source, two arms of different length and depth, sink."""
    return build(
        "bubble",
        5,
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        [6, 4, 10, 6],
        depths=[10.0, 8.0, 2.0, 10.0],
        seed=2,
    )


def nested_bubbles() -> ToyGraph:
    """Outer bubble whose left arm (1) contains a second bubble (3 vs 4)."""
    return build(
        "nested_bubbles",
        5,
        [(0, 1), (0, 2), (1, 3), (1, 4), (3, 5), (4, 5), (5, 6), (2, 6)],
        [6, 4, 12, 4, 6, 4, 8],
        seed=3,
    )


def repeat_twice() -> ToyGraph:
    """Unitig 1 is a repeat: the cycle 1→2→1 lets a path traverse it twice (or more)."""
    return build("repeat_twice", 5, [(0, 1), (1, 2), (2, 1), (1, 3)], [6, 4, 6, 8], seed=4)


TOY_GRAPHS = (chain, bubble, nested_bubbles, repeat_twice)


@dataclass(frozen=True, slots=True)
class PriorParams:
    """§2.3. `rho` is the per-base stop probability, applied *per side*.

    0.04 rather than the one-sided 0.02: total length is a sum of two geometrics with mean
    `2/rho`, so this is the value that leaves expected total length where it was.
    """

    rho: float = 0.04


DEFAULT_PRIOR = PriorParams()


def seed_logp(graph: PathGraph) -> float:
    """`p_seed`, uniform over oriented k-mer positions (§2.3, D18).

    Not coverage-weighted — coverage lives in `q_seed`, or the ablations stop meaning anything
    and the prior stops being RC-symmetric.
    """
    return -math.log(len(graph.seeds()))


def next_side(t: int, stopped_left: bool, stopped_right: bool) -> Side | None:
    """Which frontier step `t` extends; `None` once both have stopped (§2.1).

    Deterministic alternation — right on even `t`, left on odd, skipping a stopped side. It is
    never allowed to depend on the data (§6 item 11): "extend whichever side looks more
    promising" makes the target history-dependent, the same defect as a taboo list. The
    enumerator and the sampler both call *this* function so they cannot drift apart.
    """
    if stopped_left and stopped_right:
        return None
    want = Side.RIGHT if t % 2 == 0 else Side.LEFT
    stopped = (stopped_left, stopped_right)
    return want if not stopped[want] else want.other()


@dataclass(frozen=True)
class SeedProposal:
    """`q_seed` — where the particle budget is spent, *not* part of the target (D18).

    Normalised, not merely proportional: self-normalised weights make the posterior invariant
    to a constant factor in `q`, but `log Ẑ` is not, and `log Ẑ` is what §3.1 justifies the
    whole SMC choice with. Full support is enforced by construction rather than discovered as a
    zero divisor mid-run: a graph k-mer with no read on it still has `p_seed > 0`, and `q = 0`
    against it biases the estimator rather than merely inflating its variance.
    """

    seeds: tuple[Seed, ...]
    q: np.ndarray

    def __post_init__(self) -> None:
        if self.q.shape != (len(self.seeds),):
            raise ValueError("q must be one probability per seed")
        if not (self.q > 0).all():
            raise ValueError("q_seed has a zero where p_seed is positive (D18)")
        if abs(float(self.q.sum()) - 1.0) > 1e-9:
            raise ValueError("q_seed is not normalised (D18)")

    @property
    def log_q(self) -> np.ndarray:
        return np.log(self.q)


def uniform_seed_proposal(graph: PathGraph) -> SeedProposal:
    """`q_seed = p_seed`. The configuration under which prior-only `log Ẑ` is exactly 0."""
    seeds = tuple(graph.seeds())
    return SeedProposal(seeds, np.full(len(seeds), 1.0 / len(seeds)))


def weighted_seed_proposal(
    graph: PathGraph, weights: np.ndarray, eps: float = 0.01
) -> SeedProposal:
    """`q = (1-eps)·weights + eps·uniform` over `graph.seeds()`, normalised.

    `weights` is any non-negative mass per seed — anchored read placements on real data
    (`index.seed_weights`), anything at all in a test. `eps` is what buys full support.

    `eps` also **caps the importance weight at `1/eps`**: a seed with zero anchored reads gets
    `q = eps/n` against `p = 1/n`. That is the number to tune it by. Too small and one lucky
    particle on an unread k-mer dominates `Ẑ`; too large and the budget goes back to being
    spread uniformly, which is the thing the proposal exists to avoid.
    """
    if not 0.0 < eps <= 1.0:
        raise ValueError("eps must be in (0, 1] — eps = 0 is the partial-support bug (D18)")
    seeds = tuple(graph.seeds())
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(seeds),) or (w < 0).any():
        raise ValueError("weights must be one non-negative number per seed")
    if w.sum() <= 0:
        raise ValueError("weights are all zero")
    return SeedProposal(seeds, (1.0 - eps) * (w / w.sum()) + eps / len(seeds))


def survive_logp(pending_bases: int, params: PriorParams) -> float:
    """Survive a unitig's new bases without stopping. The step's other half is `-log outdeg`:
    `p_edge` is uniform on purpose, since branch information belongs in `L`, not the prior
    (§2.3), or the ablations mean nothing.
    """
    return pending_bases * math.log1p(-params.rho)


def terminal_logp(pending_bases: int, dead_end: bool, params: PriorParams) -> float:
    """This side's stop lands within its last unitig's new bases — or is forced at a dead end
    (§2.8). `pending_bases == 0` (a seed flush with a unitig end) correctly gives `-inf` unless
    the side is a dead end: there are no bases for the stop to land in, so the side must extend.
    """
    if dead_end:
        return 0.0
    if pending_bases == 0:
        return -math.inf
    return math.log1p(-((1.0 - params.rho) ** pending_bases))


def _gc(seq: bytes) -> int:
    return seq.count(b"G") + seq.count(b"C")


class GCBias(IncrementalLikelihood):
    """Fixture-only likelihood: `β × (G+C count of the new bases)`.

    Not a model — the real terms arrive one at a time at M4 (§5). This exists so the
    enumerator is exercised with a non-constant `L`, and because GC count is invariant under
    reverse complement it makes the §5 M3 RC-symmetry test bite.
    """

    def __init__(self, graph: PathGraph, beta: float = 0.05):
        self.graph = graph
        self.beta = beta

    def init(self, seed: Seed) -> tuple[None, float]:
        # The seed unitig's bases, scored once for both frontiers (§2.6's score-once rule).
        return None, self.beta * _gc(self.graph.unitig_seq(seed.node))

    def extend(self, st: Any, e: Edge, side: Side) -> tuple[None, float]:
        del st, side  # GC is RC-invariant, so the two frontiers score identically
        return None, self.beta * _gc(self.graph.unitig_seq(e)[self.graph.k - 1 :])

    def stop_logp(self, st: Any, side: Side) -> float:
        del st, side
        return 0.0


def mirror(graph: PathGraph, state: StatePath) -> StatePath:
    """The reverse complement of a two-sided state (§2.1).

    Just `(seed.flipped(), right, left)` — the walks themselves are untouched. That falls out of
    representing the left frontier as a forward walk from `seed.node.flipped()`: the mirrored
    state's left walk starts at `seed.node.flipped().flipped()`, which is where the original's
    right walk started. Getting a mirror this cheap is the payoff for that representation, and
    `test_prior_is_reverse_complement_symmetric` is what it buys.
    """
    seed, left, right = state
    return seed.flipped(len(graph.unitig_seq(seed.node)), graph.k), right, left


@dataclass
class Enumeration:
    """Every `(seed, left, right)` state up to the length bound, with its unnormalised log π."""

    graph_name: str
    paths: list[StatePath]
    log_prior: np.ndarray
    log_lik: np.ndarray
    truncated: int  # extensions dropped at the length bound

    @property
    def log_pi_unnorm(self) -> np.ndarray:
        return self.log_prior + self.log_lik

    @property
    def log_Z(self) -> float:
        return float(_logsumexp(self.log_pi_unnorm))

    @property
    def pi(self) -> np.ndarray:
        return np.exp(self.log_pi_unnorm - self.log_Z)

    def posterior(self) -> dict[StatePath, float]:
        """`{state: π(x)}`, the shape the sampler's output is compared against (§5 M2)."""
        return dict(zip(self.paths, (float(p) for p in self.pi), strict=True))

    @property
    def prior_mass(self) -> float:
        """Σ p(x) over `(seed, path)`. Exactly 1 on an acyclic graph — the M1/M3.5 gate."""
        return float(np.exp(self.log_prior).sum())


def _logsumexp(x: np.ndarray) -> float:
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))


def set_side[T](pair: tuple[T, T], side: Side, value: T) -> tuple[T, T]:
    """Per-side state is a 2-tuple indexed by `Side`; this is the immutable setter."""
    return (value, pair[1]) if side is Side.LEFT else (pair[0], value)


def enumerate_paths(
    graph: PathGraph,
    likelihood: IncrementalLikelihood,
    params: PriorParams = DEFAULT_PRIOR,
    max_bases: int = 250,
    max_paths: int = 100_000,
) -> Enumeration:
    """Depth-first enumeration of every `(seed, left, right)` state within the length bound.

    A record is emitted only when *both* sides have stopped: a state is absorbed, and a state
    with one side still open is an interior node of the recursion, not a path. Under the
    one-sided formulation these coincided, which is why the old code emitted at every visit.

    The alternatives at each step are that side's out-edges plus that side's STOP, exactly as
    the sampler's `logsumexp` sees them (§3.2) — including the two rules that historically
    diverged: a dead end has `[STOP]` alone (forced, weight 0), and truncation drops the
    over-long extension while still scoring STOP by *graph* out-degree. The budget is over
    total length, so the sides reach it asymmetrically.
    """
    name = getattr(graph, "name", "graph")
    paths: list[StatePath] = []
    log_prior: list[float] = []
    log_lik: list[float] = []
    truncated = 0
    lp_seed = seed_logp(graph)

    def walk(
        seed: Seed,
        left: tuple[OrientedNode, ...],
        right: tuple[OrientedNode, ...],
        st: Any,
        lp: float,
        ll: float,
        node: tuple[OrientedNode, OrientedNode],
        pending: tuple[int, int],
        stopped: tuple[bool, bool],
        total: int,
        t: int,
    ) -> None:
        nonlocal truncated
        side = next_side(t, *stopped)
        if side is None:
            paths.append((seed, left, right))
            log_prior.append(lp)
            log_lik.append(ll)
            if len(paths) > max_paths:
                raise RuntimeError(f"{name}: more than {max_paths} states at {max_bases} bases")
            return

        out = graph.out_edges(node[side])
        stop_lp = terminal_logp(pending[side], out.size == 0, params)
        if stop_lp > -math.inf:  # -inf only when the side has no bases left to stop in
            walk(
                seed, left, right, st,
                lp + stop_lp, ll + likelihood.stop_logp(st, side),
                node, pending, set_side(stopped, side, True), total, t + 1,
            )
        if out.size == 0:
            return

        branch_lp = lp + survive_logp(pending[side], params) - math.log(out.size)
        for c in out:
            nxt = decode(int(c))
            new = graph.new_bases(nxt, first=False)
            if total + new > max_bases:
                truncated += 1
                continue
            st2, incr = likelihood.extend(st, nxt, side)
            walk(
                seed,
                left + (nxt,) if side is Side.LEFT else left,
                right if side is Side.LEFT else right + (nxt,),
                st2, branch_lp, ll + incr,
                set_side(node, side, nxt), set_side(pending, side, new), stopped,
                total + new, t + 1,
            )

    for seed in graph.seeds():
        seed_len = len(graph.unitig_seq(seed.node))
        if seed_len > max_bases:
            truncated += 1
            continue
        st, incr = likelihood.init(seed)
        walk(
            seed, (), (), st, lp_seed, incr,
            (seed.node.flipped(), seed.node),
            (seed.offset, seed_len - graph.k - seed.offset),
            (False, False), seed_len, 0,
        )

    return Enumeration(
        graph_name=name,
        paths=paths,
        log_prior=np.array(log_prior),
        log_lik=np.array(log_lik),
        truncated=truncated,
    )


# --- fixtures (§4.2) ------------------------------------------------------------------------


def _arrays(e: Enumeration) -> dict[str, np.ndarray]:
    """Flat, variable-length-free encoding: node codes concatenated plus per-walk lengths."""
    return {
        "seed_node": np.array([code(s.node) for s, _, _ in e.paths], dtype=np.int64),
        "seed_offset": np.array([s.offset for s, _, _ in e.paths], dtype=np.int64),
        "left_flat": np.array([code(n) for _, lt, _ in e.paths for n in lt], dtype=np.int64),
        "left_len": np.array([len(lt) for _, lt, _ in e.paths], dtype=np.int64),
        "right_flat": np.array([code(n) for _, _, rt in e.paths for n in rt], dtype=np.int64),
        "right_len": np.array([len(rt) for _, _, rt in e.paths], dtype=np.int64),
        "log_prior": e.log_prior,
        "log_lik": e.log_lik,
        "log_pi": e.log_pi_unnorm - e.log_Z,
    }


def _digest(arrays: dict[str, np.ndarray]) -> str:
    """sha256 of array *content*, not of the .npz.

    Same lesson as the M0 gzip finding: numpy's zip container embeds a timestamp, so hashing
    the file would make byte-identical fixtures look non-reproducible.
    """
    h = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key])
        h.update(f"{key}|{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


DEFAULT_MAX_BASES = 250

# Per-graph length budget. Two-sided states multiply by the seed count (~40-80 on these
# graphs), so the cyclic one needs a shorter budget to stay enumerable — shrink the problem,
# never the gate (§5 M3.5 item 2).
MAX_BASES = {"repeat_twice": 120}


def build_fixtures(
    params: PriorParams = DEFAULT_PRIOR, beta: float = 0.05
) -> list[tuple[ToyGraph, Enumeration]]:
    out = []
    for factory in TOY_GRAPHS:
        g = factory()
        budget = MAX_BASES.get(g.name, DEFAULT_MAX_BASES)
        out.append((g, enumerate_paths(g, GCBias(g, beta), params, budget)))
    return out


def write_fixtures(out_dir: Path, **kwargs: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = kwargs.get("params", DEFAULT_PRIOR)
    entries = []
    for g, e in build_fixtures(**kwargs):
        arrays = _arrays(e)
        np.savez(out_dir / f"{g.name}.npz", **arrays)
        entries.append(
            {
                "name": g.name,
                "file": f"{g.name}.npz",
                "k": g.k,
                "n_unitigs": len(g._seqs),
                "rho": params.rho,
                "beta": kwargs.get("beta", 0.05),
                "max_bases": MAX_BASES.get(g.name, DEFAULT_MAX_BASES),
                "n_seeds": len(g.seeds()),
                "n_paths": len(e.paths),
                "truncated": e.truncated,
                "log_Z": e.log_Z,
                "prior_mass": e.prior_mass,
                "arrays_sha256": _digest(arrays),
            }
        )
    manifest = out_dir / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": FIXTURE_SCHEMA_VERSION, "graphs": entries}, indent=2) + "\n"
    )
    return manifest


def verify_fixtures(manifest_path: Path) -> list[str]:
    """Re-hash stored arrays and re-run the enumeration. Returns problems (empty == good)."""
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != FIXTURE_SCHEMA_VERSION:
        return [f"schema version {manifest['schema_version']} != {FIXTURE_SCHEMA_VERSION}"]
    problems: list[str] = []
    by_name = {entry["name"]: entry for entry in manifest["graphs"]}
    for factory in TOY_GRAPHS:
        g = factory()
        entry = by_name.get(g.name)
        if entry is None:
            problems.append(f"missing from manifest: {g.name}")
            continue
        stored = dict(np.load(manifest_path.parent / entry["file"]))
        if _digest(stored) != entry["arrays_sha256"]:
            problems.append(f"sha256 mismatch: {entry['file']}")
        e = enumerate_paths(
            g, GCBias(g, entry["beta"]), PriorParams(entry["rho"]), entry["max_bases"]
        )
        fresh = _arrays(e)
        for key, arr in fresh.items():
            if arr.shape != stored[key].shape:
                problems.append(f"{g.name}/{key}: shape {stored[key].shape} -> {arr.shape}")
            elif not np.allclose(arr, stored[key], rtol=0, atol=1e-9):
                problems.append(f"{g.name}/{key}: values differ by more than 1e-9")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minoodle.exact", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_write = sub.add_parser("write", help="enumerate the toy graphs and write fixtures")
    p_write.add_argument("--out", type=Path, default=Path("fixtures"))

    p_verify = sub.add_parser("verify", help="re-enumerate and compare against stored fixtures")
    p_verify.add_argument("manifest", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "write":
        print(write_fixtures(args.out.expanduser()))
        return 0

    problems = verify_fixtures(args.manifest.expanduser())
    for p in problems:
        print(p, file=sys.stderr)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
