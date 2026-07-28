import math
from pathlib import Path

import numpy as np
import pytest

from minoodle.exact import (
    TOY_GRAPHS,
    GCBias,
    PriorParams,
    bubble,
    build,
    chain,
    enumerate_paths,
    nested_bubbles,
    repeat_twice,
    revcomp,
    start_logp,
    verify_fixtures,
    write_fixtures,
)
from minoodle.interfaces import CompositeLikelihood, OrientedNode

ACYCLIC = (chain, bubble, nested_bubbles)
PARAMS = PriorParams(rho=0.02)
REPO_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _prior_only(graph, params=PARAMS, **kw):
    return enumerate_paths(graph, CompositeLikelihood([]), params, **kw)


@pytest.mark.parametrize("factory", ACYCLIC)
def test_prior_is_a_probability_measure(factory):
    """Σ p(x) == 1 over all paths of an acyclic graph.

    The single test that catches nearly every length-accounting or dead-end bug: if new bases
    are double-counted, if the terminal factor is wrong, or if forced STOP at a dead end is
    missed, this drifts off 1.
    """
    e = _prior_only(factory(), max_bases=10_000)
    assert e.truncated == 0
    assert e.prior_mass == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("factory", ACYCLIC)
def test_prior_mass_is_unaffected_by_the_likelihood(factory):
    g = factory()
    e = enumerate_paths(g, GCBias(g), PARAMS, max_bases=10_000)
    assert e.prior_mass == pytest.approx(1.0, abs=1e-12)
    assert e.pi.sum() == pytest.approx(1.0)


def test_cyclic_graph_mass_approaches_one_as_the_bound_grows():
    g = repeat_twice()
    masses = [_prior_only(g, max_bases=b).prior_mass for b in (40, 120, 400)]
    assert all(m < 1.0 for m in masses)
    assert masses == sorted(masses)
    assert masses[-1] > 0.99


def test_repeat_is_actually_visited_twice():
    e = _prior_only(repeat_twice(), max_bases=200)
    repeat = OrientedNode(1, True)
    assert any(p.count(repeat) >= 2 for p in e.paths)


@pytest.mark.parametrize("factory", TOY_GRAPHS)
def test_likelihood_is_reverse_complement_symmetric(factory):
    """§5 M3's property test, cheap to get now.

    Only the likelihood is RC-symmetric: the prior is directional by construction (different
    start node, out-degrees and terminal unitig), which is a property of the generative process
    and not a bug.
    """
    g = factory()
    e = enumerate_paths(g, GCBias(g), PARAMS, max_bases=120)
    by_path = {p: ll for p, ll in zip(e.paths, e.log_lik, strict=True)}
    checked = 0
    for path, ll in by_path.items():
        rc = tuple(n.flipped() for n in reversed(path))
        assert g.path_seq(rc) == revcomp(g.path_seq(path))
        if rc in by_path:
            assert by_path[rc] == pytest.approx(ll, abs=1e-12)
            checked += 1
    assert checked > 0


def test_two_node_chain_matches_the_closed_form():
    """Independent hand calculation on A→B, so the enumerator is not its own oracle."""
    g = build("pair", 5, [(0, 1)], [6, 4], seed=9)
    params = PriorParams(rho=0.05, uniform_start=True)
    e = _prior_only(g, params, max_bases=10_000)
    p = dict(zip(e.paths, np.exp(e.log_prior), strict=True))

    a, b = OrientedNode(0, True), OrientedNode(1, True)
    q = 1 - params.rho
    start = 1 / 4  # 2 unitigs x 2 orientations, uniform
    len_a = len(g.unitig_seq(a))

    # A alone: stop lands inside A's 10 bases. A then B: survive A, single out-edge, and B is
    # a dead end so it terminates with probability 1.
    assert p[(a,)] == pytest.approx(start * (1 - q**len_a))
    assert p[(a, b)] == pytest.approx(start * q**len_a)
    assert p[(b,)] == pytest.approx(start)  # B forward is a dead end


def test_unbalanced_bubble_prefers_the_short_arm_under_the_prior():
    """Both arms are reachable; with no likelihood the prior only sees length, per §2.3 —
    branch information is deliberately absent from p_edge."""
    g = bubble()
    e = _prior_only(g, max_bases=10_000)
    p = dict(zip(e.paths, np.exp(e.log_prior), strict=True))
    src, short_arm, long_arm, sink = (OrientedNode(i, True) for i in range(4))
    assert len(g.unitig_seq(short_arm)) < len(g.unitig_seq(long_arm))
    assert p[(src, short_arm, sink)] > p[(src, long_arm, sink)]


def test_start_distribution_sums_to_one_and_follows_length_times_depth():
    g = bubble()
    assert sum(math.exp(start_logp(g, n, PARAMS)) for n in g.nodes()) == pytest.approx(1.0)
    heavy, light = OrientedNode(0, True), OrientedNode(2, True)  # depth 10 vs 2
    assert start_logp(g, heavy, PARAMS) > start_logp(g, light, PARAMS)
    # Orientation carries no start mass of its own.
    assert start_logp(g, heavy, PARAMS) == start_logp(g, heavy.flipped(), PARAMS)


def test_edge_overlap_is_validated():
    g = chain()
    bad = list(g._seqs)
    bad[1] = b"TTTT" + bad[1][4:]
    with pytest.raises(ValueError, match="overlap"):
        type(g)("bad", g.k, bad, [(0, 1), (1, 2)], [1.0] * 3)


def test_fixtures_round_trip(tmp_path):
    """Regenerating from scratch must reproduce the committed fixtures to 1e-9 (§4.2)."""
    assert verify_fixtures(REPO_FIXTURES / "manifest.json") == []
    fresh = write_fixtures(tmp_path)
    assert verify_fixtures(fresh) == []


def test_fixture_tampering_is_detected(tmp_path):
    manifest = write_fixtures(tmp_path)
    npz = dict(np.load(tmp_path / "chain.npz"))
    npz["log_pi"] = npz["log_pi"] + 1e-6
    np.savez(tmp_path / "chain.npz", **npz)
    problems = verify_fixtures(manifest)
    assert any("chain" in p for p in problems)
