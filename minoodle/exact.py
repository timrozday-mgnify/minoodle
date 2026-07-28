"""Exact enumerator and toy graphs (§5 M1).

Brute-force ground truth for M2: enumerate every path up to a length bound in a tiny graph,
score it under the §2.3 prior and an `IncrementalLikelihood`, and normalise. Without this
there is no way to tell "the sampler works" from "the sampler produces plausible sequences"
(§6.5), so it comes before the sampler.

    python -m minoodle.exact write --out fixtures
    python -m minoodle.exact verify fixtures/manifest.json

Toy graphs live here rather than in a `graph.py` because they are fixtures: M3's metaSPAdes
GFA loader is a different thing and gets its own module.

**Prior normalisation.** §2.3 is written as `p_start · Π_t [(1-ρ)^{len_t} · p_edge] · ρ`, which
does not sum to 1 over paths — the trailing `ρ` stops at a unitig boundary having already
survived that unitig's bases. Implemented here is the per-base geometric that §2.3's own prose
specifies: a path ends when the stop lands *within* the last unitig's new bases, so the terminal
factor is `1 - (1-ρ)^{len_T}` (and 1 at a dead end, where STOP is forced, §2.8). That version
sums to exactly 1 on an acyclic graph, which is the test that catches length-accounting bugs.

**The prior is directional, the likelihood is not.** A path and its reverse complement are
different states with different priors (different start node, different out-degrees, different
terminal unitig). Only the likelihood is required to be RC-symmetric (§5 M3).
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

from minoodle.interfaces import Edge, IncrementalLikelihood, OrientedNode, PathGraph

FIXTURE_SCHEMA_VERSION = 1

_COMPLEMENT = bytes.maketrans(b"ACGT", b"TGCA")


def revcomp(seq: bytes) -> bytes:
    return seq.translate(_COMPLEMENT)[::-1]


def code(n: OrientedNode) -> int:
    """Pack an oriented node into an int, as the M3 CSR arrays will."""
    return 2 * n.unitig + int(n.forward)


def decode(c: int) -> OrientedNode:
    return OrientedNode(c >> 1, bool(c & 1))


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

    def new_bases(self, n: OrientedNode, first: bool) -> int:
        """Bases `n` contributes to the path sequence: all of them if it starts the path."""
        return len(self._seqs[n.unitig]) - (0 if first else self.k - 1)

    def path_seq(self, path: tuple[OrientedNode, ...]) -> bytes:
        out = self.unitig_seq(path[0])
        for n in path[1:]:
            out += self.unitig_seq(n)[self.k - 1 :]
        return out


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
    """§2.3. `rho` is the per-base stop probability."""

    rho: float = 0.02
    uniform_start: bool = False


DEFAULT_PRIOR = PriorParams()


def start_logp(graph: ToyGraph, n: OrientedNode, params: PriorParams) -> float:
    """`p_start ∝ unitig length × mean coverage`, uniform under a flag (§2.3)."""
    nodes = graph.nodes()
    if params.uniform_start:
        return -math.log(len(nodes))
    mass = [len(graph._seqs[m.unitig]) * graph._depths[m.unitig] for m in nodes]
    return math.log(len(graph._seqs[n.unitig]) * graph._depths[n.unitig]) - math.log(sum(mass))


def survive_logp(pending_bases: int, params: PriorParams) -> float:
    """Survive a unitig's new bases without stopping. The step's other half is `-log outdeg`:
    `p_edge` is uniform on purpose, since branch information belongs in `L`, not the prior
    (§2.3), or the ablations mean nothing.
    """
    return pending_bases * math.log1p(-params.rho)


def terminal_logp(pending_bases: int, dead_end: bool, params: PriorParams) -> float:
    """Stop lands within the last unitig's new bases — or is forced at a dead end (§2.8)."""
    if dead_end:
        return 0.0
    return math.log1p(-((1.0 - params.rho) ** pending_bases))


def _gc(seq: bytes) -> int:
    return seq.count(b"G") + seq.count(b"C")


class GCBias(IncrementalLikelihood):
    """Fixture-only likelihood: `β × (G+C count of the new bases)`.

    Not a model — the real terms arrive one at a time at M4 (§5). This exists so the
    enumerator is exercised with a non-constant `L`, and because GC count is invariant under
    reverse complement it makes the §5 M3 RC-symmetry test bite.
    """

    def __init__(self, graph: ToyGraph, beta: float = 0.05):
        self.graph = graph
        self.beta = beta

    def init(self, start: OrientedNode) -> tuple[None, float]:
        return None, self.beta * _gc(self.graph.unitig_seq(start))

    def extend(self, st: Any, e: Edge) -> tuple[None, float]:
        del st
        return None, self.beta * _gc(self.graph.unitig_seq(e)[self.graph.k - 1 :])

    def stop_logp(self, st: Any) -> float:
        del st
        return 0.0


@dataclass
class Enumeration:
    """Every path up to the length bound, with its exact unnormalised log π."""

    graph_name: str
    paths: list[tuple[OrientedNode, ...]]
    log_prior: np.ndarray
    log_lik: np.ndarray
    truncated: int  # paths dropped at the length bound (nonzero only on cyclic graphs)

    @property
    def log_pi_unnorm(self) -> np.ndarray:
        return self.log_prior + self.log_lik

    @property
    def log_Z(self) -> float:
        return float(_logsumexp(self.log_pi_unnorm))

    @property
    def pi(self) -> np.ndarray:
        return np.exp(self.log_pi_unnorm - self.log_Z)

    def posterior(self) -> dict[tuple[OrientedNode, ...], float]:
        """`{path: π(x)}`, the shape the sampler's output is compared against (§5 M2)."""
        return dict(zip(self.paths, (float(p) for p in self.pi), strict=True))

    @property
    def prior_mass(self) -> float:
        """Σ p(x) over enumerated paths. Exactly 1 on an acyclic graph."""
        return float(np.exp(self.log_prior).sum())


def _logsumexp(x: np.ndarray) -> float:
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))


def enumerate_paths(
    graph: ToyGraph,
    likelihood: IncrementalLikelihood,
    params: PriorParams = DEFAULT_PRIOR,
    max_bases: int = 250,
    max_paths: int = 100_000,
) -> Enumeration:
    """Depth-first enumeration of all paths whose sequence is at most `max_bases` long.

    Every prefix is itself a path (the walk can stop anywhere), so a record is emitted at each
    node visited, not only at leaves.
    """
    paths: list[tuple[OrientedNode, ...]] = []
    log_prior: list[float] = []
    log_lik: list[float] = []
    truncated = 0

    def walk(
        path: tuple[OrientedNode, ...],
        st: Any,
        lp: float,
        ll: float,
        pending: int,
        total: int,
    ) -> None:
        nonlocal truncated
        out = graph.out_edges(path[-1])
        paths.append(path)
        log_prior.append(lp + terminal_logp(pending, len(out) == 0, params))
        log_lik.append(ll + likelihood.stop_logp(st))
        if len(paths) > max_paths:
            raise RuntimeError(f"{graph.name}: more than {max_paths} paths at {max_bases} bases")
        if len(out) == 0:
            return
        branch_lp = lp + survive_logp(pending, params) - math.log(len(out))
        for c in out:
            nxt = decode(int(c))
            new = graph.new_bases(nxt, first=False)
            if total + new > max_bases:
                truncated += 1
                continue
            st2, incr = likelihood.extend(st, nxt)
            walk(path + (nxt,), st2, branch_lp, ll + incr, new, total + new)

    for start in graph.nodes():
        first = graph.new_bases(start, first=True)
        if first > max_bases:
            truncated += 1
            continue
        st, incr = likelihood.init(start)
        walk((start,), st, start_logp(graph, start, params), incr, first, first)

    return Enumeration(
        graph_name=graph.name,
        paths=paths,
        log_prior=np.array(log_prior),
        log_lik=np.array(log_lik),
        truncated=truncated,
    )


# --- fixtures (§4.2) ------------------------------------------------------------------------


def _arrays(e: Enumeration) -> dict[str, np.ndarray]:
    """Flat, variable-length-free encoding: node codes concatenated plus per-path lengths."""
    return {
        "path_flat": np.array([code(n) for p in e.paths for n in p], dtype=np.int64),
        "path_len": np.array([len(p) for p in e.paths], dtype=np.int64),
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


def build_fixtures(
    params: PriorParams = DEFAULT_PRIOR, beta: float = 0.05, max_bases: int = 250
) -> list[tuple[ToyGraph, Enumeration]]:
    out = []
    for factory in TOY_GRAPHS:
        g = factory()
        out.append((g, enumerate_paths(g, GCBias(g, beta), params, max_bases)))
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
                "uniform_start": params.uniform_start,
                "beta": kwargs.get("beta", 0.05),
                "max_bases": kwargs.get("max_bases", 250),
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
            g,
            GCBias(g, entry["beta"]),
            PriorParams(entry["rho"], entry["uniform_start"]),
            entry["max_bases"],
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
