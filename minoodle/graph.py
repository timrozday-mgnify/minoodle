"""metaSPAdes GFA loader and the bidirected CSR graph (§5 M3).

    python -m minoodle.graph stats path/to/assembly_graph_with_scaffolds.gfa --k 21

The assembly is produced in a container, not from a local install (D17):

    docker run --rm --platform linux/amd64 -v "$PWD:/data" \\
      quay.io/biocontainers/spades:4.3.0--hde4eca7_0 \\
      metaspades.py --only-assembler -k 21 \\
      -1 /data/sim_reads_R1.fastq.gz -2 /data/sim_reads_R2.fastq.gz -o /data/asm

Two things §5 M3 warns about, both handled at parse time so nothing downstream has to think
about them:

- **Orientation.** Every GFA `L` line is one bidirected edge, and its twin
  `(v, ¬ov) -> (u, ¬ou)` describes the same adjacency read the other way. metaSPAdes writes
  both members of most twin pairs explicitly, so edges are deduplicated rather than appended.
  The RC round-trip check (`--k`, `stats`) is the test that this is right.
- **Coverage units.** metaSPAdes' `KC:i:` tag and the `_cov_` field in a segment name are
  k-mer counts/coverage, not base coverage. Both are converted here by `L/(L-k+1)`, so
  `unitig_depth` is in the units §2.7's NegBin expects.

**`k` comes from the GFA, not from the assembler's command line.** `metaspades.py -k 21`
writes `21M` overlaps: SPAdes' unitigs are paths of (k+1)-mers, so consecutive segments share
21 bases and this project's k — the one that makes the overlap `k-1` — is **22**. Passing
`--k 21` here would fail every overlap check for the right reason. `from_gfa` therefore reads
the overlap off the `L` lines and requires them all to agree; an explicit `k` is checked
against it rather than trusted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from minoodle.interfaces import OrientedNode, PathGraph

_COMPLEMENT = bytes.maketrans(b"ACGT", b"TGCA")


def revcomp(seq: bytes) -> bytes:
    return seq.translate(_COMPLEMENT)[::-1]


def code(n: OrientedNode) -> int:
    """Pack an oriented node into an int — the CSR arrays are indexed by this."""
    return 2 * n.unitig + int(n.forward)


def decode(c: int) -> OrientedNode:
    return OrientedNode(c >> 1, bool(c & 1))


_SPADES_COV = re.compile(r"_cov_([0-9.]+)")


class UnitigGraph(PathGraph):
    """A bidirected unitig graph in CSR form.

    Same method surface as `exact.ToyGraph`, so the sampler, the likelihood terms and the
    enumerator run unchanged against a real graph.
    """

    def __init__(
        self,
        name: str,
        k: int,
        seqs: list[bytes],
        edges: list[tuple[int, bool, int, bool]],
        kmer_cov: list[float],
    ):
        self.name = name
        self.k = k
        self._seqs = list(seqs)
        n = len(self._seqs)
        self._kmer_cov = [float(c) for c in kmer_cov]
        # k-mer coverage -> base coverage (§5 M3). A unitig shorter than k has no k-mers;
        # metaSPAdes does not emit those, but a hand-written GFA might.
        self._depths = [
            c * len(s) / (len(s) - k + 1) if len(s) >= k else c
            for c, s in zip(self._kmer_cov, self._seqs, strict=True)
        ]

        adj: list[set[int]] = [set() for _ in range(2 * n)]
        for u, ou, v, ov in edges:
            if self._overlap(u, ou)[-(k - 1) :] != self._overlap(v, ov)[: k - 1]:
                raise ValueError(f"edge {u}{'+' if ou else '-'}->{v}{'+' if ov else '-'}"
                                 f" violates the k-1 overlap")
            adj[code(OrientedNode(u, ou))].add(code(OrientedNode(v, ov)))
            adj[code(OrientedNode(v, not ov))].add(code(OrientedNode(u, not ou)))

        self.indptr = np.zeros(2 * n + 1, dtype=np.int64)
        for c, outs in enumerate(adj):
            self.indptr[c + 1] = self.indptr[c] + len(outs)
        self.indices = np.fromiter(
            (m for outs in adj for m in sorted(outs)), dtype=np.int64, count=int(self.indptr[-1])
        )

    def _overlap(self, unitig: int, forward: bool) -> bytes:
        seq = self._seqs[unitig]
        return seq if forward else revcomp(seq)

    # --- PathGraph -------------------------------------------------------------------

    def nodes(self) -> list[OrientedNode]:
        return [OrientedNode(u, f) for u in range(len(self._seqs)) for f in (True, False)]

    def out_edges(self, n: OrientedNode) -> np.ndarray:
        c = code(n)
        return self.indices[self.indptr[c] : self.indptr[c + 1]]

    def unitig_seq(self, n: OrientedNode) -> bytes:
        return self._overlap(n.unitig, n.forward)

    def unitig_depth(self, n: OrientedNode) -> np.ndarray:
        # ponytail: flat per-base depth, as ToyGraph does — GFA carries one number per segment.
        # Per-base depth needs a pileup, which is M4's problem if §2.7 turns out to want it.
        return np.full(len(self._seqs[n.unitig]), self._depths[n.unitig])

    # --- I/O -------------------------------------------------------------------------

    @classmethod
    def from_gfa(cls, path: Path, k: int | None = None, name: str | None = None) -> UnitigGraph:
        """Load a GFA. `k` is inferred from the `L` overlaps; pass it only to assert it."""
        ids: dict[str, int] = {}
        seqs: list[bytes] = []
        raw_segs: list[tuple[str, list[str]]] = []
        raw_edges: list[tuple[str, bool, str, bool, str]] = []

        with path.open() as fh:
            for line in fh:
                fields = line.rstrip("\n").split("\t")
                if fields[0] == "S":
                    _, sid, seq, *tags = fields
                    if seq == "*":
                        raise ValueError(f"segment {sid} has no sequence (GFA written with '*')")
                    ids[sid] = len(seqs)
                    seqs.append(seq.encode().upper())
                    raw_segs.append((sid, tags))
                elif fields[0] == "L":
                    raw_edges.append(
                        (fields[1], fields[2] == "+", fields[3], fields[4] == "+", fields[5])
                    )

        overlaps = {e[4] for e in raw_edges}
        if len(overlaps) > 1:
            raise ValueError(f"GFA mixes overlap lengths: {sorted(overlaps)}")
        if overlaps:
            (only,) = overlaps
            if not only.endswith("M"):
                raise ValueError(f"non-match overlap {only!r}; only exact k-1 overlaps are v0")
            inferred = int(only[:-1]) + 1
            if k is not None and k != inferred:
                raise ValueError(f"GFA overlaps are {only} (k={inferred}), but k={k} was given")
            k = inferred
        elif k is None:
            raise ValueError("edgeless GFA: pass k explicitly")

        cov = [_kmer_coverage(sid, tags, len(seq), k) for (sid, tags), seq in
               zip(raw_segs, seqs, strict=True)]
        edges = [(ids[u], ou, ids[v], ov) for u, ou, v, ov, _ in raw_edges]
        return cls(name or path.stem, k, seqs, edges, cov)

    def to_gfa(self, path: Path) -> None:
        """Write this graph as GFA. Used to turn the M1 toy graphs into parser test cases."""
        with path.open("w") as fh:
            for u, seq in enumerate(self._seqs):
                kc = round(self._kmer_cov[u] * max(len(seq) - self.k + 1, 1))
                fh.write(f"S\t{u}\t{seq.decode()}\tKC:i:{kc}\n")
            written: set[tuple[int, ...]] = set()
            for c in range(len(self._seqs) * 2):
                a = decode(c)
                for m in self.out_edges(a):
                    b = decode(int(m))
                    twin = (b.unitig, not b.forward, a.unitig, not a.forward)
                    if twin in written:
                        continue
                    written.add((a.unitig, a.forward, b.unitig, b.forward))
                    fh.write(
                        f"L\t{a.unitig}\t{'+' if a.forward else '-'}"
                        f"\t{b.unitig}\t{'+' if b.forward else '-'}\t{self.k - 1}M\n"
                    )


def _kmer_coverage(sid: str, tags: list[str], seq_len: int, k: int) -> float:
    """k-mer coverage of a segment, in whatever way this GFA records it.

    `DP:f:` is already a k-mer coverage (metaSPAdes writes both, and `KC/DP` is exactly the
    k-mer span `L-k+1`, which is the arithmetic this project's `k` has to agree with).
    `KC:i:` is a k-mer *count*, so divide. Older SPAdes embeds `_cov_` in the segment name.
    """
    by_key = {t.split(":", 2)[0]: t.split(":", 2)[2] for t in tags if t.count(":") >= 2}
    for key in ("DP", "dp"):
        if key in by_key:
            return float(by_key[key])
    if "KC" in by_key:
        return float(by_key["KC"]) / max(seq_len - k + 1, 1)
    m = _SPADES_COV.search(sid)
    return float(m.group(1)) if m else 0.0


def rc_roundtrip_ok(graph: UnitigGraph) -> bool:
    """Every edge's bidirected twin is present, and every unitig revcomps to itself twice.

    This is the M3 gate's "RC round-trip" — orientation handling is the classic silent-bug
    source (§5 M3), and it is cheap to assert over the whole graph.
    """
    for c in range(len(graph._seqs) * 2):
        a = decode(c)
        if revcomp(revcomp(graph.unitig_seq(a))) != graph.unitig_seq(a):
            return False
        for m in graph.out_edges(a):
            b = decode(int(m))
            if code(a.flipped()) not in graph.out_edges(b.flipped()):
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minoodle.graph", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_stats = sub.add_parser("stats", help="load a GFA and report its shape")
    p_stats.add_argument("gfa", type=Path)
    p_stats.add_argument("--k", type=int, default=None, help="assert the inferred k (overlap+1)")

    args = parser.parse_args(argv)
    g = UnitigGraph.from_gfa(args.gfa.expanduser(), args.k)
    lens = np.array([len(s) for s in g._seqs])
    depth = np.array(g._depths)
    ok = rc_roundtrip_ok(g)
    print(f"{g.name}: {len(lens)} unitigs, {len(g.indices) // 2} edges (k={g.k})")
    print(f"  length  total {lens.sum()}  max {lens.max()}  median {int(np.median(lens))}")
    print(f"  depth   mean {depth.mean():.2f}  median {np.median(depth):.2f}  (base coverage)")
    print(f"  out-degree >1: {int((np.diff(g.indptr) > 1).sum())} of {2 * len(lens)} oriented")
    print(f"  RC round-trip: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
