"""k-mer index, anchoring and fragment estimation (§5 M3).

Reads are sliced out of a toy graph's own path sequence, so ground truth is exact and CI needs
no assembler. The real gate — recall on the L0 metaSPAdes graph — is
`python -m minoodle.index check` and is run locally.
"""

from __future__ import annotations

import gzip

import pytest

from minoodle.exact import bubble, chain, repeat_twice
from minoodle.graph import code, decode, revcomp
from minoodle.index import KmerIndex, check, fragment_length
from minoodle.interfaces import OrientedNode
from tests.test_graph import as_unitig_graph


@pytest.mark.parametrize("toy", [chain(), bubble(), repeat_twice()], ids=lambda g: g.name)
def test_every_unitig_anchors_to_itself(toy):
    g = as_unitig_graph(toy)
    idx = KmerIndex(g)
    for n in g.nodes():
        seq = g.unitig_seq(n)
        a = idx.anchor(seq)
        assert a is not None
        # the whole unitig, read off in `n`'s orientation, anchors at diagonal 0 on `n`
        assert (a.node, a.diagonal, a.votes) == (code(n), 0, len(seq) - g.k + 1)


def test_anchor_finds_offset_and_strand():
    g = as_unitig_graph(chain())
    idx = KmerIndex(g)
    n = OrientedNode(2, True)
    seq = g.unitig_seq(n)
    read = seq[3:]
    a = idx.anchor(read)
    assert a is not None and decode(a.node) == n and a.diagonal == 3

    rc = idx.anchor(revcomp(read))
    assert rc is not None and decode(rc.node) == n.flipped()
    assert rc.diagonal == len(seq) - len(read) - 3  # == 0, the RC read starts at the RC end


def test_recall_is_total_on_error_free_reads(tmp_path):
    """The M3 gate's shape, in miniature: reads sliced from a real path must all anchor."""
    toy = repeat_twice()
    g = as_unitig_graph(toy)
    path = (OrientedNode(0, True), OrientedNode(1, True), OrientedNode(2, True),
            OrientedNode(1, True), OrientedNode(3, True))
    seq = g.path_seq(path)
    read_len = 12
    reads = [seq[i : i + read_len] for i in range(len(seq) - read_len + 1)]

    fq = tmp_path / "reads.fastq.gz"
    with gzip.open(fq, "wt") as fh:
        for i, r in enumerate(reads):
            fh.write(f"@r{i}\n{r.decode()}\n+\n{'I' * len(r)}\n")

    res = check(KmerIndex(g), fq, None)
    assert res.reads == len(reads)
    assert res.recall == 1.0


def test_unknown_sequence_does_not_anchor():
    g = as_unitig_graph(chain())
    assert KmerIndex(g).anchor(b"N" * 30) is None


def test_fragment_length_is_the_outer_distance():
    """§2.6: 5′ of mate 1 to 5′ of mate 2, so a planted fragment comes back exactly."""
    g = as_unitig_graph(chain())
    idx = KmerIndex(g)
    n = OrientedNode(2, True)
    seq = g.unitig_seq(n)
    read_len, frag = 6, 12
    start = 1
    m1 = seq[start : start + read_len]
    m2 = revcomp(seq[start + frag - read_len : start + frag])

    a, b = idx.anchor(m1), idx.anchor(m2)
    assert a is not None and b is not None
    assert fragment_length(idx, a, b, len(m1), len(m2)) == frag
    # symmetric in the mates
    assert fragment_length(idx, b, a, len(m2), len(m1)) == frag


def test_same_strand_pair_is_not_measured():
    g = as_unitig_graph(chain())
    idx = KmerIndex(g)
    seq = g.unitig_seq(OrientedNode(2, True))
    a, b = idx.anchor(seq[0:6]), idx.anchor(seq[6:12])
    assert a is not None and b is not None
    assert fragment_length(idx, a, b, 6, 6) is None
