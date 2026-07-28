"""k-mer index, read anchors and fragment-length estimation (§5 M3).

    python -m minoodle.index check asm/assembly_graph_with_scaffolds.gfa R1.fq.gz R2.fq.gz

This is the M3 gate: anchor recall ≥ 99% on error-free reads.

**What an anchor is.** Every k-mer of a read votes for a `(oriented node, diagonal)` pair,
where the diagonal is the read's start offset in that node's sequence. The winning diagonal is
the anchor; §2.5's banded forward pass at M4 bands around it. §2.5 also wants the *score* of
the best competing placement cached at index time — that score needs the pair-HMM, which is
M4, so what is cached here is the competing placement's **identity and vote count**. M4 fills
in the score; nothing else about the anchor table changes.

**Fragment length is the outer distance** (§2.6), 5′ of mate 1 to 5′ of mate 2, measured here
from pairs whose mates both place uniquely on one unitig in FR orientation. §2.6 requires this
be measured rather than assumed: the D10 values (400 ± 100) are the prior, not the answer.
"""

from __future__ import annotations

import argparse
import gzip
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from minoodle.graph import UnitigGraph, code, decode, revcomp
from minoodle.interfaces import OrientedNode

# ponytail: a k-mer is a dict key of raw bytes, and a repeated k-mer keeps at most this many
# placements. Both are the "dict -> minimal perfect hash" step the plan defers past M3; the
# ceiling is memory on a 1e6-unitig graph, and M6 is where it gets paid.
MAX_OCCURRENCES = 32


@dataclass(frozen=True, slots=True)
class Anchor:
    """A read's best placement, plus the runner-up §2.5 will score against at M4."""

    node: int  # oriented-node code (graph.code)
    diagonal: int  # read's start offset in that node's sequence, may be negative at an end
    votes: int
    rival_node: int | None
    rival_votes: int

    @property
    def unique(self) -> bool:
        return self.votes > self.rival_votes


class KmerIndex:
    """canonical k-mer -> [(oriented node code, offset)] (§4 `index.py`)."""

    def __init__(self, graph: UnitigGraph):
        self.graph = graph
        self.k = graph.k
        self.table: dict[bytes, list[tuple[int, int]]] = {}
        self.repeated = 0
        for u, seq in enumerate(graph._seqs):
            fwd = code(OrientedNode(u, True))
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i : i + self.k]
                rc = revcomp(kmer)
                if kmer <= rc:
                    key, placement = kmer, (fwd, i)
                else:
                    # the same physical k-mer, read off the reverse-complement node
                    key = rc
                    placement = (code(OrientedNode(u, False)), len(seq) - self.k - i)
                bucket = self.table.setdefault(key, [])
                if len(bucket) < MAX_OCCURRENCES:
                    bucket.append(placement)
                else:
                    self.repeated += 1

    def anchor(self, read: bytes) -> Anchor | None:
        """Vote every k-mer of `read` onto `(oriented node, diagonal)`; return the winner."""
        votes: Counter[tuple[int, int]] = Counter()
        for j in range(len(read) - self.k + 1):
            kmer = read[j : j + self.k]
            rc = revcomp(kmer)
            forward = kmer <= rc
            for node, offset in self.table.get(kmer if forward else rc, ()):
                if not forward:
                    # flip to the node the read actually lies on, in that node's coordinates
                    n = decode(node)
                    node = code(n.flipped())
                    offset = len(self.graph._seqs[n.unitig]) - self.k - offset
                votes[(node, offset - j)] += 1
        if not votes:
            return None
        ranked = votes.most_common(2)
        (node, diagonal), n_votes = ranked[0]
        rival = ranked[1] if len(ranked) > 1 else None
        return Anchor(node, diagonal, n_votes, rival[0][0] if rival else None,
                      rival[1] if rival else 0)


def seed_weights(index: KmerIndex, reads: Iterable[bytes]) -> np.ndarray:
    """Anchored k-mer placements per seed, in `graph.seeds()` order — the `q_seed` weights (D18).

    Every k-mer of an anchored read votes for the seed position it lands on. The total is
    exactly the normaliser §2.3 asks for, and `weighted_seed_proposal` divides by it; nothing
    here needs to be a probability.

    **Both orientations are credited.** A read anchors on one strand, but `Seed(n, o)` and its
    flip are two distinct states of an RC-symmetric target. Weighting only the strand the read
    happened to come off would leave half the state space on the `eps` floor for no reason.
    """
    graph = index.graph
    starts: dict[OrientedNode, int] = {}
    total = 0
    for n in graph.nodes():
        starts[n] = total
        total += len(graph.unitig_seq(n)) - index.k + 1
    w = np.zeros(total)

    for read in reads:
        a = index.anchor(read)
        if a is None:
            continue
        node = decode(a.node)
        span = len(graph.unitig_seq(node)) - index.k
        flip = node.flipped()
        for j in range(len(read) - index.k + 1):
            o = a.diagonal + j
            if 0 <= o <= span:  # reads may overhang a unitig end
                w[starts[node] + o] += 1.0
                w[starts[flip] + span - o] += 1.0
    return w


def read_fastq(path: Path) -> Iterator[bytes]:
    """Yield sequences from a (optionally gzipped) FASTQ."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                yield line.strip().upper()


def fragment_length(index: KmerIndex, a: Anchor, b: Anchor, len_a: int, len_b: int) -> int | None:
    """Outer distance (§2.6) for an FR pair on one unitig, or None if the pair can't say.

    Mate 1 forward on node `n`, mate 2 on `n`'s twin, is the only configuration measured: it
    is unambiguous, and the estimate only needs to be unbiased, not exhaustive.
    """
    if not (a.unique and b.unique):
        return None
    na, nb = decode(a.node), decode(b.node)
    if na.unitig != nb.unitig or na.forward == nb.forward:
        return None
    length = len(index.graph._seqs[na.unitig])

    def span(n: OrientedNode, diagonal: int, read_len: int) -> tuple[int, int]:
        """The read's interval in the unitig's forward-strand coordinates."""
        if n.forward:
            return diagonal, diagonal + read_len
        return length - diagonal - read_len, length - diagonal

    (s1, e1), (s2, e2) = span(na, a.diagonal, len_a), span(nb, b.diagonal, len_b)
    frag = max(e1, e2) - min(s1, s2)  # outer distance, orientation-agnostic
    return frag if frag > 0 else None


@dataclass
class CheckResult:
    reads: int
    anchored: int
    unique: int
    fragments: list[int]

    @property
    def recall(self) -> float:
        return self.anchored / self.reads if self.reads else 0.0

    @property
    def unique_frac(self) -> float:
        return self.unique / self.reads if self.reads else 0.0


def check(index: KmerIndex, r1: Path, r2: Path | None, limit: int | None = None) -> CheckResult:
    res = CheckResult(0, 0, 0, [])
    mates = zip(read_fastq(r1), read_fastq(r2)) if r2 else ((s, None) for s in read_fastq(r1))
    for i, (s1, s2) in enumerate(mates):
        if limit is not None and i >= limit:
            break
        anchors = []
        for s in (s1, s2):
            if s is None:
                continue
            res.reads += 1
            a = index.anchor(s)
            anchors.append(a)
            if a is not None:
                res.anchored += 1
                res.unique += int(a.unique)
        if len(anchors) == 2 and all(a is not None for a in anchors):
            f = fragment_length(index, anchors[0], anchors[1], len(s1), len(s2))  # type: ignore[arg-type]
            if f is not None:
                res.fragments.append(f)
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minoodle.index", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_check = sub.add_parser("check", help="the M3 gate: anchor recall on error-free reads")
    p_check.add_argument("gfa", type=Path)
    p_check.add_argument("r1", type=Path)
    p_check.add_argument("r2", type=Path, nargs="?")
    p_check.add_argument("--k", type=int, default=None, help="assert the inferred k (overlap+1)")
    p_check.add_argument("--limit", type=int, default=None, help="stop after N pairs")

    args = parser.parse_args(argv)
    graph = UnitigGraph.from_gfa(args.gfa.expanduser(), args.k)
    index = KmerIndex(graph)
    res = check(index, args.r1.expanduser(), args.r2.expanduser() if args.r2 else None, args.limit)

    print(f"k={index.k}  {len(index.table)} k-mers indexed"
          f"  ({index.repeated} placements dropped at the {MAX_OCCURRENCES}-occurrence cap)")
    print(f"  reads {res.reads}  anchored {res.anchored} ({res.recall:.4%})"
          f"  uniquely {res.unique_frac:.4%}")
    if res.fragments:
        print(f"  fragment length (outer, §2.6) from {len(res.fragments)} FR pairs:"
              f" mean {statistics.fmean(res.fragments):.1f}"
              f"  var {statistics.variance(res.fragments):.0f}")
    ok = res.recall >= 0.99
    print(f"  M3 gate (recall >= 99%): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
