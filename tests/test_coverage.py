"""§2.7's coverage term (§5 M4 item 1)."""

import itertools
import math

import numpy as np
import pytest

from minoodle import diagnostics as diag
from minoodle.exact import (
    DEFAULT_PRIOR,
    PriorParams,
    bubble,
    build,
    enumerate_paths,
    mirror,
)
from minoodle.interfaces import CompositeLikelihood, OrientedNode, Seed, Side
from minoodle.model.coverage import (
    CoverageParams,
    CoverageTerm,
    NoLikelihood,
    _log_pred,
    true_edge_mass,
    truth_edges,
)
from minoodle.sampler import SMCConfig, branch_marginals, run

PARAMS = PriorParams(0.08)
MAX_BASES = 60


def _term(graph, **kw):
    return CoverageTerm(graph, CoverageParams(**kw))


def test_predictive_normalises():
    """A NegBin over counts, so it sums to 1 — the cheapest guard against a dropped term."""
    for a, b, m in ((2.0, 0.1, 11.0), (0.5, 1.0, 3.0), (30.0, 1.2, 25.0)):
        total = sum(math.exp(_log_pred(y, m, a, b)) for y in range(20_000))
        assert total == pytest.approx(1.0, abs=1e-6)


def test_predictive_prefers_the_matching_rate():
    """20x under exposure 10 should score best when the posterior mean is 20."""
    scores = {
        mean: _log_pred(200.0, 10.0, 4.0, 4.0 / mean) for mean in (5.0, 20.0, 80.0)
    }
    assert max(scores, key=scores.get) == 20.0


def test_increments_sum_to_the_walk_score():
    """`init` + `extend` is the whole path's score: the §2.4 decomposition, spelled out."""
    g = bubble()
    term = _term(g)
    seed = Seed(OrientedNode(0, True), 0)
    st, total = term.init(seed)
    for side, node in ((Side.RIGHT, OrientedNode(1, True)), (Side.RIGHT, OrientedNode(3, True))):
        st, incr = term.extend(st, node, side)
        total += incr
    # Same three unitigs scored by hand against a posterior updated in the same order.
    post = (term.a0, term.b0)
    by_hand = 0.0
    for node in (OrientedNode(0, True), OrientedNode(1, True), OrientedNode(3, True)):
        post, incr = term._score(node, post)
        by_hand += incr
    assert total == pytest.approx(by_hand, abs=1e-12)


def test_seed_unitig_is_scored_once():
    """§2.6's score-once rule: `init` charges the seed, and neither frontier recharges it."""
    g = bubble()
    term = _term(g)
    st, incr = term.init(Seed(OrientedNode(0, True), 0))
    assert incr == pytest.approx(term._score(OrientedNode(0, True), (term.a0, term.b0))[1])
    # Both sides start from the same posterior — the seed's, not the prior's.
    assert st[Side.LEFT] == st[Side.RIGHT] != (term.a0, term.b0)


def test_stop_is_silent():
    """Coverage may not influence termination (§2.8) — that is the prior's job alone."""
    g = bubble()
    term = _term(g)
    st, _ = term.init(Seed(OrientedNode(0, True), 0))
    assert term.stop_logp(st, Side.LEFT) == 0.0
    assert term.stop_logp(st, Side.RIGHT) == 0.0


def test_score_is_reverse_complement_symmetric():
    """k-mer coverage is orientation-free, so the §5 M3.5 mirror leaves the score alone."""
    g = bubble()
    lik = _term(g)
    e = enumerate_paths(g, lik, PARAMS, MAX_BASES)
    by_state = dict(zip(e.paths, e.log_lik, strict=True))
    for state, ll in by_state.items():
        assert by_state[mirror(g, state)] == pytest.approx(ll, abs=1e-12)


# `code(n) = 2*unitig + forward`, so unitig 0 forward is 1, unitig 1 forward is 3, and so on.
DEEP_ARM = (1, 3)  # 0+ -> 1+, depth 8
SHALLOW_ARM = (1, 5)  # 0+ -> 2+, depth 2


def test_unbalanced_bubble_prefers_the_deep_arm():
    """§5 M4 item 1's stated ablation. `bubble()`'s arms are unitig 1 at depth 8 and unitig 2
    at depth 2, off a source at depth 10 — the deep arm is the one consistent with the source,
    and the term has to move mass onto it."""
    g = bubble()
    prior = branch_marginals(enumerate_paths(g, NoLikelihood(), PARAMS, MAX_BASES).posterior())
    cov = branch_marginals(enumerate_paths(g, _term(g), PARAMS, MAX_BASES).posterior())

    def ratio(bm):
        return bm[DEEP_ARM] / bm[SHALLOW_ARM]

    assert ratio(prior) == pytest.approx(1.0, abs=0.02)  # the prior is uniform over branches
    assert ratio(cov) > 3.0


def test_flat_coverage_leaves_the_branch_alone():
    """The complement of the ablation: two arms of the same depth *and* the same length are
    indistinguishable, so the term must not touch the branch at all. Guards against a
    length- or exposure-dependent bias masquerading as coverage signal.
    """
    g = build("flat", 5, [(0, 1), (0, 2), (1, 3), (2, 3)], [6, 4, 4, 6],
              depths=[10.0] * 4, seed=2)
    bm = branch_marginals(enumerate_paths(g, _term(g), PARAMS, MAX_BASES).posterior())
    assert bm[DEEP_ARM] / bm[SHALLOW_ARM] == pytest.approx(1.0, abs=1e-9)


def test_equal_depth_arms_of_different_length_barely_differ():
    """Unequal exposures do tilt the branch — a longer unitig agreeing with the running
    abundance is more evidence than a short one — but it is a rounding error next to real
    coverage signal (1.2x here against 169x in `test_unbalanced_bubble_prefers_the_deep_arm`).
    """
    g = build("uneven", 5, [(0, 1), (0, 2), (1, 3), (2, 3)], [6, 4, 10, 6],
              depths=[10.0] * 4, seed=2)
    bm = branch_marginals(enumerate_paths(g, _term(g), PARAMS, MAX_BASES).posterior())
    assert 1.0 < bm[DEEP_ARM] / bm[SHALLOW_ARM] < 1.5


def test_hazard_softens_the_preference():
    """The coverage-change hazard exists so a genuine change is possible but penalised (§2.7):
    a bigger hazard must move the deep-arm preference *towards* indifference, never past it."""
    g = bubble()
    ratios = []
    for h in (0.0, 0.05, 0.5):
        bm = branch_marginals(
            enumerate_paths(g, _term(g, hazard=h), PARAMS, MAX_BASES).posterior()
        )
        ratios.append(bm[DEEP_ARM] / bm[SHALLOW_ARM])
    assert ratios[0] > ratios[1] > ratios[2] > 1.0


def test_sampler_matches_the_enumerator_with_the_term_on():
    """The sharpest check available: the incremental decomposition the sampler consumes has to
    reproduce the enumerator's target exactly (M2's standing rule for every new term)."""
    g = bubble()
    lik = CompositeLikelihood([CoverageTerm(g)])
    exact = enumerate_paths(g, lik, PARAMS, MAX_BASES)
    r = run(g, lik, PARAMS, SMCConfig(8000, 2, MAX_BASES, seed=3))
    ess = sum(diag.ess(i.log_w) for i in r.islands)
    _, hi = diag.multinomial_tv_reference(exact.pi, max(1, int(ess)))
    assert diag.tv_distance(r.posterior(), exact.posterior()) < 2.0 * hi
    assert r.log_Z == pytest.approx(exact.log_Z, abs=0.05)


def test_no_likelihood_is_the_prior():
    """The ablation's control arm must be exactly prior-only: `log Ẑ == 0` (D18, uniform q)."""
    g = bubble()
    r = run(g, NoLikelihood(), DEFAULT_PRIOR, SMCConfig(500, 2, 120, seed=1))
    assert r.log_Z == pytest.approx(0.0, abs=1e-12)


def test_truth_edges_drops_non_adjacent_pairs():
    """`reference_walk` gaps at repeats, and a pair spanning a gap is not an edge."""
    g = bubble()
    a, b, c = OrientedNode(0, True), OrientedNode(1, True), OrientedNode(3, True)
    assert truth_edges(g, [a, b]) == {DEEP_ARM, (2, 0)}  # the edge and its bidirected twin
    assert truth_edges(g, [a, c]) == set()  # 0 -> 3 is not an edge; the walk skipped an arm


def test_true_edge_mass_ignores_unbranched_nodes():
    g = bubble()
    marg = {DEEP_ARM: 0.7, SHALLOW_ARM: 0.3, (3, 7): 1.0}  # the last is out of unbranched 1+
    good, total = true_edge_mass(marg, {DEEP_ARM}, g)
    assert (good, total) == (0.7, 1.0)


def test_repeat_traversal_is_self_confirming():
    """M4 item 1 finding 7, pinned: re-traversing a unitig scores *better* each time, because
    the running posterior has already absorbed that unitig's own count.

    This is the term paying for loops rather than preventing them, and it is the root cause of
    the L1 runaway. The assertion is deliberately of the broken behaviour — when finding 7's
    multiplicity model lands, this test must be rewritten to assert the *opposite* (a second
    traversal of a 1x unitig is penalised), and its failure is the signal that it worked.
    """
    g = build("rep", 5, [(0, 1), (1, 2), (2, 1), (1, 3)], [6, 4, 6, 8],
              depths=[10.0] * 4, seed=4)
    term = _term(g)
    st, _ = term.init(Seed(OrientedNode(0, True), 0))
    incs = []
    for _ in range(5):
        st, incr = term.extend(st, OrientedNode(1, True), Side.RIGHT)
        incs.append(incr)
    assert all(b > a for a, b in itertools.pairwise(incs))
    assert all(i > 0 for i in incs)


def test_state_stays_bounded():
    """§2.4: `extend` may not accumulate — the state is two `(a, b)` pairs, whatever the walk."""
    g = bubble()
    term = _term(g)
    st, _ = term.init(Seed(OrientedNode(0, True), 0))
    for _ in range(50):
        st, _ = term.extend(st, OrientedNode(1, True), Side.RIGHT)
    assert np.shape(st) == (2, 2)
