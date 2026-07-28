"""Exact enumerator, two-sided prior and seed proposal (§5 M1, redone at M3.5)."""

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
    mirror,
    nested_bubbles,
    next_side,
    repeat_twice,
    seed_logp,
    uniform_seed_proposal,
    verify_fixtures,
    weighted_seed_proposal,
    write_fixtures,
)
from minoodle.graph import revcomp, seed_path_seq
from minoodle.interfaces import CompositeLikelihood, OrientedNode, Seed, Side

ACYCLIC = (chain, bubble, nested_bubbles)
PARAMS = PriorParams(rho=0.04)
REPO_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _prior_only(graph, params=PARAMS, **kw):
    return enumerate_paths(graph, CompositeLikelihood([]), params, **kw)


def _by_state(e):
    return dict(zip(e.paths, np.exp(e.log_prior), strict=True))


@pytest.mark.parametrize("factory", ACYCLIC)
def test_prior_is_a_probability_measure(factory):
    """Σ p(x) == 1 over `(seed, path)` on an acyclic graph — the M1/M3.5 gate.

    The single test that catches nearly every length-accounting or dead-end bug: if new bases
    are double-counted, if either side's terminal factor is wrong, if forced STOP at a dead end
    is missed, or if the seed's own unitig is split wrongly between the two frontiers, this
    drifts off 1.
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


def test_each_seed_carries_equal_and_uniform_mass():
    """`p_seed` is uniform over oriented k-mer positions (D18), and each seed's own path set
    normalises to that share — the per-seed version of the gate above."""
    g = chain()
    e = _prior_only(g, max_bases=10_000)
    per_seed: dict[Seed, float] = {}
    for (seed, _, _), p in _by_state(e).items():
        per_seed[seed] = per_seed.get(seed, 0.0) + p
    assert set(per_seed) == set(g.seeds())
    assert seed_logp(g) == pytest.approx(-np.log(len(g.seeds())))
    for p in per_seed.values():
        assert p == pytest.approx(1.0 / len(g.seeds()), abs=1e-12)


def test_cyclic_graph_mass_approaches_one_as_the_bound_grows():
    g = repeat_twice()
    masses = [_prior_only(g, max_bases=b).prior_mass for b in (40, 120, 400)]
    assert all(m < 1.0 for m in masses)
    assert masses == sorted(masses)
    assert masses[-1] > 0.99


def test_repeat_is_actually_visited_twice():
    e = _prior_only(repeat_twice(), max_bases=200)
    repeat = OrientedNode(1, True)
    assert any((left + right).count(repeat) >= 2 for _, left, right in e.paths)


def test_both_one_sided_shapes_occur():
    """A state with one frontier extended and the other stopped at the seed.

    Two-sided paths add a degenerate regime the one-sided formulation had no analogue for
    (§5 M3.5's "watch for"), and it is exactly the sort of path that a toy graph might never
    produce by accident. It does — so assert it, rather than hoping.
    """
    states = _prior_only(nested_bubbles(), max_bases=200).paths
    assert any(left and not right for _, left, right in states)
    assert any(right and not left for _, left, right in states)
    assert any(left and right for _, left, right in states)


def test_next_side_alternates_and_skips_a_stopped_side():
    assert next_side(0, False, False) is Side.RIGHT
    assert next_side(1, False, False) is Side.LEFT
    assert next_side(2, False, False) is Side.RIGHT
    # a stopped side forfeits its turn, it does not pause the alternation
    assert next_side(0, False, True) is Side.LEFT
    assert next_side(1, True, False) is Side.RIGHT
    assert next_side(3, True, False) is Side.RIGHT
    assert next_side(0, True, True) is None
    assert next_side(7, True, True) is None


@pytest.mark.parametrize("factory", TOY_GRAPHS)
def test_prior_and_likelihood_are_reverse_complement_symmetric(factory):
    """Rev 10's headline property: `mirror` is a measure-preserving bijection of the state set.

    The one-sided prior was directional by construction and only the likelihood had to be
    RC-symmetric (M1 finding 3). Two-sided, both are — and this is the test that catches an
    asymmetric alternation rule, since a rule that favoured one frontier would show up as
    unequal mass between a state and its mirror.
    """
    g = factory()
    e = enumerate_paths(g, GCBias(g), PARAMS, max_bases=120)
    scored = {p: (lp, ll) for p, lp, ll in zip(e.paths, e.log_prior, e.log_lik, strict=True)}
    for state, (lp, ll) in scored.items():
        rc = mirror(g, state)
        assert seed_path_seq(g, *rc) == revcomp(seed_path_seq(g, *state))
        assert rc in scored, f"{state} has no mirror"
        assert scored[rc][0] == pytest.approx(lp, abs=1e-12)
        assert scored[rc][1] == pytest.approx(ll, abs=1e-12)


def test_two_node_chain_matches_the_closed_form():
    """Independent hand calculation on A→B, so the enumerator is not its own oracle."""
    g = build("pair", 5, [(0, 1)], [6, 4], seed=9)
    params = PriorParams(rho=0.05)
    p = _by_state(_prior_only(g, params, max_bases=10_000))

    a, b = OrientedNode(0, True), OrientedNode(1, True)
    q = 1 - params.rho
    p_seed = 1 / len(g.seeds())  # 2 unitigs x 2 orientations x their k-mer positions
    span = len(g.unitig_seq(a)) - g.k  # last legal offset in A

    for offset in (0, 3, span):
        seed = Seed(a, offset)
        right_bases = span - offset  # A's bases to the right of the seed k-mer
        # A's twin is a dead end, so the left frontier stops with probability 1. The right
        # frontier either stops inside A's remaining bases or takes A's single out-edge to B,
        # which is itself a dead end and so terminates with probability 1.
        assert p.get((seed, (), ()), 0.0) == pytest.approx(p_seed * (1 - q**right_bases))
        assert p[(seed, (), (b,))] == pytest.approx(p_seed * q**right_bases)

    # offset == span leaves the right frontier no bases to stop in, so it *must* extend
    assert (Seed(a, span), (), ()) not in p
    assert p[(Seed(a, span), (), (b,))] == pytest.approx(p_seed)


def test_unbalanced_bubble_prefers_the_short_arm_under_the_prior():
    """Both arms are reachable; with no likelihood the prior only sees length, per §2.3 —
    branch information is deliberately absent from p_edge."""
    g = bubble()
    p = _by_state(_prior_only(g, max_bases=10_000))
    src, short_arm, long_arm, sink = (OrientedNode(i, True) for i in range(4))
    seed = Seed(src, 0)
    assert len(g.unitig_seq(short_arm)) < len(g.unitig_seq(long_arm))
    assert p[(seed, (), (short_arm, sink))] > p[(seed, (), (long_arm, sink))]


def test_seed_proposal_is_normalised_and_has_full_support():
    g = bubble()
    n = len(g.seeds())
    uniform = uniform_seed_proposal(g)
    assert uniform.q.sum() == pytest.approx(1.0)
    assert np.allclose(uniform.log_q, seed_logp(g))

    w = np.zeros(n)
    w[0] = 1.0  # everything on one seed: the partial-support case eps exists for
    skewed = weighted_seed_proposal(g, w, eps=0.1)
    assert skewed.q.sum() == pytest.approx(1.0)
    assert (skewed.q > 0).all()
    # eps caps the importance weight at 1/eps, which is the number to tune it by
    assert (1 / n) / skewed.q.min() == pytest.approx(1 / 0.1)


def test_seed_proposal_rejects_zero_eps_and_bad_weights():
    g = chain()
    n = len(g.seeds())
    with pytest.raises(ValueError, match="eps"):
        weighted_seed_proposal(g, np.ones(n), eps=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        weighted_seed_proposal(g, -np.ones(n))
    with pytest.raises(ValueError, match="all zero"):
        weighted_seed_proposal(g, np.zeros(n))


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
