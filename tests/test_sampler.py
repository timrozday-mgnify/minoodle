import json
import math
from pathlib import Path

import numpy as np
import pytest

from minoodle import diagnostics as diag
from minoodle.exact import (
    DEFAULT_PRIOR,
    TOY_GRAPHS,
    GCBias,
    PriorParams,
    bubble,
    build,
    chain,
    enumerate_paths,
    nested_bubbles,
)
from minoodle.interfaces import CompositeLikelihood, OrientedNode
from minoodle.sampler import SMCConfig, branch_marginals, run, run_island, sbc_ranks

ACYCLIC = (chain, bubble, nested_bubbles)
REPO_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MANIFEST = json.loads((REPO_FIXTURES / "manifest.json").read_text())
ENTRY = {e["name"]: e for e in MANIFEST["graphs"]}


def _fixture_run(factory, n=2000, islands=2, seed=0):
    g = factory()
    e = ENTRY[g.name]
    params = PriorParams(e["rho"], e["uniform_start"])
    lik = GCBias(g, e["beta"])
    cfg = SMCConfig(n, islands, e["max_bases"], seed=seed)
    return g, e, run(g, lik, params, cfg), enumerate_paths(g, lik, params, e["max_bases"])


@pytest.mark.parametrize("factory", ACYCLIC)
def test_prior_only_log_z_is_exactly_zero(factory):
    """The sampler's counterpart of M1's `Σ p(x) == 1`, and the sharpest test in the file.

    With no likelihood every fully-adapted step's `logsumexp` is the total continuation mass of
    the prior, which is 1 by construction — so `log Ẑ` is 0 with no Monte Carlo error at all.
    Drop STOP from the alternative set, mis-handle a dead end, or double-count new bases and
    this moves immediately.
    """
    r = run(factory(), CompositeLikelihood([]), DEFAULT_PRIOR, SMCConfig(200, 2, 250, seed=1))
    assert r.log_Z == pytest.approx(0.0, abs=1e-9)


def test_two_node_chain_posterior_matches_the_closed_form():
    """Independent hand calculation on A→B, so the sampler is not checked only against the
    enumerator it is supposed to be validating."""
    g = build("pair", 5, [(0, 1)], [6, 4], seed=9)
    params = PriorParams(rho=0.05, uniform_start=True)
    r = run(g, CompositeLikelihood([]), params, SMCConfig(40_000, 1, 10_000, seed=5))
    post = r.posterior()

    a, b = OrientedNode(0, True), OrientedNode(1, True)
    q = 1 - params.rho
    start = 1 / 4  # 2 unitigs x 2 orientations, uniform
    len_a = len(g.unitig_seq(a))

    assert post[(a,)] == pytest.approx(start * (1 - q**len_a), abs=0.005)
    assert post[(a, b)] == pytest.approx(start * q**len_a, abs=0.005)
    assert post[(b,)] == pytest.approx(start, abs=0.005)  # B forward is a dead end


@pytest.mark.parametrize("factory", TOY_GRAPHS)
def test_matches_the_exact_fixture(factory):
    """TV and `log Ẑ` against `fixtures/`, at a particle count that runs in a second.

    Deliberately loose: the M2 gate proper is `python -m minoodle.sampler validate` at N = 1e5.
    This catches bugs, not the last digit.
    """
    _, e, r, exact = _fixture_run(factory, n=2000, islands=2)
    assert diag.tv_distance(r.posterior(), exact.posterior()) < 0.12
    assert r.log_Z == pytest.approx(e["log_Z"], abs=0.05)


@pytest.mark.parametrize("factory", TOY_GRAPHS)
def test_branch_marginals_agree_with_the_enumeration(factory):
    _, _, r, exact = _fixture_run(factory, n=4000, islands=1)
    got, want = branch_marginals(r.posterior()), branch_marginals(exact.posterior())
    assert set(got) <= set(want)
    for key, w in want.items():
        assert got.get(key, 0.0) == pytest.approx(w, abs=0.06)


def test_systematic_resample_is_stratified_and_deterministic():
    """The defining property: each index appears `floor` or `ceil` of `N·w_i` times, from one
    uniform. Multinomial resampling would not satisfy this — which is why §3.3 forbids it."""
    w = np.array([0.5, 0.25, 0.25] * 4) / 4
    for u in (0.0, 0.3, 0.5, 0.999):
        idx = diag.systematic_resample(np.log(w), u)
        counts = np.bincount(idx, minlength=w.size)
        assert np.all(np.abs(counts - idx.size * w) <= 1)
        assert idx.size == w.size


def test_a_degenerate_weight_vector_collapses_to_one_index():
    log_w = np.array([-1000.0, 0.0, -1000.0])
    assert set(diag.systematic_resample(log_w, 0.4).tolist()) == {1}


def test_recorded_uniforms_replay_identically():
    """§4.2's port contract: the engine consumes an injected stream, so the same stream must
    give byte-identical trajectories, ancestry and weights (that is how Rust is checked at M7).
    """
    g = bubble()
    lik = GCBias(g, 0.05)
    cfg = SMCConfig(300, 1, 250, ess_frac=1.0)  # forces the resampling draw into the stream
    stream = np.random.default_rng(11).random(200_000)

    def replay():
        pos = 0

        def draw(k: int) -> np.ndarray:
            nonlocal pos
            assert pos + k <= stream.size, "recorded stream exhausted"
            pos += k
            return stream[pos - k : pos]

        return draw, lambda: pos

    d1, n1 = replay()
    d2, n2 = replay()
    a = run_island(g, lik, DEFAULT_PRIOR, cfg, d1)
    b = run_island(g, lik, DEFAULT_PRIOR, cfg, d2)
    # The contract is not just "reproducible" but "predictable position": N draws per step plus
    # one per resampling, everything from the injected source and nothing from a private RNG.
    steps = a.nodes.shape[0]
    resamples = sum(1 for p in a.parents if not np.array_equal(p, np.arange(p.size)))
    assert resamples > 0
    assert n1() == n2()
    assert steps * cfg.n_particles < n1() <= steps * (cfg.n_particles + 1)
    assert a.log_Z == b.log_Z
    assert np.array_equal(a.log_w, b.log_w)
    assert np.array_equal(a.nodes, b.nodes)
    assert a.paths() == b.paths()


def test_reconstructed_paths_are_walks_the_graph_allows():
    g = nested_bubbles()
    r = run(g, GCBias(g, 0.05), DEFAULT_PRIOR, SMCConfig(500, 1, 250, seed=7))
    exact = set(enumerate_paths(g, GCBias(g, 0.05), DEFAULT_PRIOR, 250).paths)
    for path in r.islands[0].paths():
        assert path in exact  # ancestry reconstruction, not just the weights


def test_the_toy_graphs_never_trigger_adaptive_resampling():
    """Recorded because it is surprising and because it decides how the rest is tested.

    The fully-adapted step is effective enough here that ESS never falls below N/2 on any toy
    graph — so every default run leaves the resampling and ancestry-permutation code paths
    untouched. They are exercised below with `ess_frac=1.0` instead of waiting for a real graph.
    """
    for factory in TOY_GRAPHS:
        g = factory()
        r = run(g, GCBias(g, 0.05), DEFAULT_PRIOR, SMCConfig(2000, 1, 250, seed=0))
        parents = r.islands[0].parents
        assert all(np.array_equal(p, np.arange(p.size)) for p in parents)
        assert diag.ess(r.islands[0].log_w) > 0.8 * 2000


@pytest.mark.parametrize("factory", TOY_GRAPHS)
def test_resampling_every_step_is_still_unbiased(factory):
    """`ess_frac=1.0` forces resampling at every step, so `log Ẑ` accumulation across
    resamplings, the state permutation and the ancestry bookkeeping all have to be right."""
    g = factory()
    e = ENTRY[g.name]
    params, lik = PriorParams(e["rho"], e["uniform_start"]), GCBias(g, e["beta"])
    cfg = SMCConfig(4000, 2, e["max_bases"], ess_frac=1.0, seed=6)
    r = run(g, lik, params, cfg)
    assert any(not np.array_equal(p, np.arange(p.size)) for p in r.islands[0].parents)
    assert r.log_Z == pytest.approx(e["log_Z"], abs=0.08)
    exact = enumerate_paths(g, lik, params, e["max_bases"])
    assert diag.tv_distance(r.posterior(), exact.posterior()) < 0.2


def test_heavy_truncation_matches_the_enumerator():
    """The sampler must lose exactly the mass `enumerate_paths` loses at `max_bases`, including
    scoring STOP as a non-dead-end when the only thing blocking extension is the bound."""
    g = TOY_GRAPHS[3]()  # repeat_twice, the only cyclic graph
    lik = GCBias(g, 0.05)
    for bound in (30, 45, 70):
        exact = enumerate_paths(g, lik, DEFAULT_PRIOR, bound)
        assert exact.truncated > 0
        r = run(g, lik, DEFAULT_PRIOR, SMCConfig(20_000, 1, bound, seed=8))
        assert r.log_Z == pytest.approx(exact.log_Z, abs=0.01)


def test_per_locus_ess_sees_coalescence():
    """Early-locus ESS must be reported and must be able to fall (§3.3 item 2)."""
    g = nested_bubbles()
    r = run(g, GCBias(g, 0.05), DEFAULT_PRIOR, SMCConfig(400, 1, 250, ess_frac=1.0, seed=2))
    trace = r.islands[0].per_locus_ess()
    assert trace.size == r.islands[0].nodes.shape[0]
    assert np.all(trace <= 400 + 1e-9)
    # Later loci partition the particles more finely than earlier ones, so ESS is non-decreasing
    # along the path; the gap between the ends is exactly the coalescence §3.3 warns about.
    assert np.all(np.diff(trace) >= -1e-6)
    assert trace[0] < trace[-1]

    parents = [np.arange(5), np.zeros(5, dtype=np.int64)]
    assert diag.per_locus_ess(diag.lineage(parents), np.zeros(5))[0] == pytest.approx(1.0)


def test_islands_are_independent_but_agree():
    g = bubble()
    r = run(g, GCBias(g, 0.05), DEFAULT_PRIOR, SMCConfig(1000, 4, 250, seed=4))
    zs = [i.log_Z for i in r.islands]
    assert len(set(zs)) == 4  # no cross-island resampling, no shared state
    assert r.log_Z_spread < 0.05
    assert r.log_Z == pytest.approx(ENTRY["bubble"]["log_Z"], abs=0.05)


def test_calibration_ranks_are_uniform():
    """The M2 gate's SBC line, in the only form available before a generative likelihood.

    `GCBias` is a bare potential — there is no data to draw, so proper simulation-based
    calibration waits for M4. What is testable now is the statement SBC reduces to with the
    data integrated out: an exact draw from π is exchangeable with the sampler's draws, so its
    rank is uniform. A biased sampler pushes the ranks to one end.
    """
    g = bubble()
    lik = GCBias(g, 0.05)
    exact = enumerate_paths(g, lik, DEFAULT_PRIOR, 250)
    stat = dict(zip(exact.paths, exact.log_pi_unnorm, strict=True))
    ranks = sbc_ranks(
        g, lik, DEFAULT_PRIOR, SMCConfig(256, 1, 250), stat, exact.posterior(),
        reps=120, n_draws=31, rng=np.random.default_rng(3),
    )
    assert diag.ks_uniform(ranks, 31) > 0.01


def test_ks_uniform_rejects_a_skewed_rank_histogram():
    rng = np.random.default_rng(0)
    assert diag.ks_uniform(rng.integers(0, 32, size=200), 31) > 0.05
    assert diag.ks_uniform(rng.integers(0, 20, size=200), 31) < 0.01


def test_multinomial_reference_scales_with_n():
    pi = np.full(200, 1 / 200)
    small, _ = diag.multinomial_tv_reference(pi, 1_000)
    big, _ = diag.multinomial_tv_reference(pi, 100_000)
    assert small / big == pytest.approx(math.sqrt(100), rel=0.2)
