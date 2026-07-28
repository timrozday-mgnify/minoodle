"""GFA round-trip, orientation and coverage units (§5 M3).

CI has no assembler, so the parser is tested against the M1 toy graphs written out as GFA.
The real L0 graph is exercised by the `minoodle.index check` gate, locally.
"""

from __future__ import annotations

import pytest

from minoodle.exact import GCBias, bubble, chain, mirror, nested_bubbles, repeat_twice
from minoodle.graph import UnitigGraph, code, decode, rc_roundtrip_ok, revcomp
from minoodle.interfaces import OrientedNode, Seed, Side

TOYS = [chain(), bubble(), nested_bubbles(), repeat_twice()]


def as_unitig_graph(toy) -> UnitigGraph:
    """The toy graph, re-expressed in the CSR class. Only forward-forward edges are passed:
    `UnitigGraph` installs each twin itself, exactly as it does for a GFA `L` line."""
    edges = [
        (a.unitig, a.forward, decode(int(m)).unitig, decode(int(m)).forward)
        for a in toy.nodes()
        for m in toy.out_edges(a)
        if a.forward
    ]
    return UnitigGraph(toy.name, toy.k, toy._seqs, edges, toy._depths)


@pytest.mark.parametrize("toy", TOYS, ids=lambda g: g.name)
def test_gfa_roundtrip(tmp_path, toy):
    g = as_unitig_graph(toy)
    path = tmp_path / f"{toy.name}.gfa"
    g.to_gfa(path)
    back = UnitigGraph.from_gfa(path)

    assert back.k == g.k
    assert back._seqs == g._seqs
    for n in g.nodes():
        assert set(back.out_edges(n).tolist()) == set(g.out_edges(n).tolist())
        assert back.unitig_seq(n) == g.unitig_seq(n)
        assert back.unitig_depth(n) == pytest.approx(g.unitig_depth(n))


@pytest.mark.parametrize("toy", TOYS, ids=lambda g: g.name)
def test_matches_toygraph(toy):
    """The CSR class must be a drop-in for `ToyGraph` — the sampler is written against it."""
    g = as_unitig_graph(toy)
    for n in toy.nodes():
        assert set(g.out_edges(n).tolist()) == set(toy.out_edges(n).tolist())
        assert g.unitig_seq(n) == toy.unitig_seq(n)
        assert g.new_bases(n, first=False) == toy.new_bases(n, first=False)


@pytest.mark.parametrize("toy", TOYS, ids=lambda g: g.name)
def test_rc_roundtrip(toy):
    assert rc_roundtrip_ok(as_unitig_graph(toy))


@pytest.mark.parametrize("toy", TOYS, ids=lambda g: g.name)
def test_likelihood_is_rc_symmetric(toy):
    """Two-sided, a state and its `mirror` score equally — see `test_exact.py` for the prior.

    Here it is just the likelihood, walked by hand through the ABC, so the term itself is
    checked independently of the enumerator that also walks it.
    """
    g = as_unitig_graph(toy)
    term = GCBias(g, beta=0.7)

    def log_l(seed: Seed, left, right) -> float:
        st, total = term.init(seed)
        for side, walk in ((Side.LEFT, left), (Side.RIGHT, right)):
            for n in walk:
                st, incr = term.extend(st, n, side)
                total += incr
        return total + term.stop_logp(st, Side.LEFT) + term.stop_logp(st, Side.RIGHT)

    for a in g.nodes():
        seed = Seed(a, 0)
        for m in g.out_edges(a):
            state = (seed, (), (decode(int(m)),))
            assert log_l(*state) == pytest.approx(log_l(*mirror(g, state)), abs=1e-12)


def test_kmer_coverage_converted_to_base_coverage(tmp_path):
    """`KC:i:` is a k-mer count; base coverage is `cov_kmer · R/(R-k+1)` (M4 finding 1).

    The conversion is a constant in the read length, *not* §5 M3's `L/(L-k+1)`: the unitig's
    own length never enters, and here it would have been 2.5x out.
    """
    seq = "ACGT" * 8  # 32 bp, k = 22 -> 11 k-mers
    gfa = tmp_path / "one.gfa"
    gfa.write_text(f"S\t1\t{seq}\tKC:i:220\nS\t2\t{seq}\tDP:f:20\nL\t1\t+\t2\t+\t21M\n")
    with pytest.raises(ValueError):
        UnitigGraph.from_gfa(gfa)  # the two segments don't overlap; the check must fire

    gfa.write_text(f"S\t1\t{seq}\tKC:i:220\n")
    g = UnitigGraph.from_gfa(gfa, k=22)
    assert g._kmer_cov[0] == pytest.approx(20.0)  # 220 counts / 11 k-mers
    assert g.unitig_kmer_cov(OrientedNode(0, True)) == pytest.approx(20.0)
    assert g.unitig_depth(OrientedNode(0, True))[0] == pytest.approx(20.0 * 150 / 129)
    short = UnitigGraph.from_gfa(gfa, k=22, read_len=100)
    assert short.unitig_depth(OrientedNode(0, True))[0] == pytest.approx(20.0 * 100 / 79)


def test_k_is_inferred_and_asserted(tmp_path):
    toy = chain()
    gfa = tmp_path / "chain.gfa"
    as_unitig_graph(toy).to_gfa(gfa)
    assert UnitigGraph.from_gfa(gfa).k == toy.k
    with pytest.raises(ValueError, match="but k="):
        UnitigGraph.from_gfa(gfa, k=toy.k + 1)


def test_overlap_violation_rejected(tmp_path):
    gfa = tmp_path / "bad.gfa"
    gfa.write_text("S\t1\tAAAAAA\nS\t2\tCCCCCC\nL\t1\t+\t2\t+\t5M\n")
    with pytest.raises(ValueError, match="overlap"):
        UnitigGraph.from_gfa(gfa)


def test_missing_sequence_rejected(tmp_path):
    gfa = tmp_path / "star.gfa"
    gfa.write_text("S\t1\t*\tLN:i:6\n")
    with pytest.raises(ValueError, match="no sequence"):
        UnitigGraph.from_gfa(gfa, k=3)


def test_code_decode_roundtrip():
    for u in range(5):
        for f in (True, False):
            n = OrientedNode(u, f)
            assert decode(code(n)) == n
    assert revcomp(revcomp(b"ACGTTGCA")) == b"ACGTTGCA"
