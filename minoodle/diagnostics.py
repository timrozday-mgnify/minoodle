"""ESS, ancestry and calibration (§3.3, §3.4, §5 M2).

Separate from `sampler.py` because these are the numbers that decide whether a run's claims
are worth anything, and they need to be testable without running a sampler at all.

The one to keep an eye on is `per_locus_ess`: resampling causes path coalescence, so ESS at
the *current* step can look healthy while every surviving particle shares one ancestor from
100 steps back. That is the failure mode that destroys long-range phasing (§3.3), and it is
invisible unless the ancestry is tracked and reported.
"""

from __future__ import annotations

import math

import numpy as np


def normalise(log_w: np.ndarray) -> np.ndarray:
    """Self-normalised importance weights. Consistent but biased, `O(1/N)` (§3.4)."""
    w = np.exp(np.asarray(log_w, dtype=np.float64) - np.max(log_w))
    return w / w.sum()


def ess(log_w: np.ndarray) -> float:
    """Kish effective sample size, `1 / Σ w²` on normalised weights."""
    w = normalise(log_w)
    return float(1.0 / np.square(w).sum())


def systematic_resample(log_w: np.ndarray, u: float) -> np.ndarray:
    """Systematic resampling from a single uniform (§3.3: systematic or stratified, never
    multinomial — multinomial adds variance for nothing).

    One uniform in, `N` indices out: the whole point of the injected-RNG contract (§4.2) is
    that the stream position is predictable, and this consumes exactly one draw.
    """
    w = normalise(log_w)
    n = w.size
    positions = (np.arange(n) + u) / n
    return np.searchsorted(np.cumsum(w), positions).clip(max=n - 1)


def lineage(parents: list[np.ndarray]) -> np.ndarray:
    """`(T, N)` — `out[t, i]` is the step-`t` index of final particle `i`'s ancestor.

    `parents[t][i]` maps a post-resampling index at step `t` to its pre-resampling index, so
    walking backwards composes the permutations.
    """
    n = parents[0].size
    out = np.empty((len(parents), n), dtype=np.int64)
    a = np.arange(n)
    for t in range(len(parents) - 1, -1, -1):
        a = parents[t][a]
        out[t] = a
    return out


def per_locus_ess(lin: np.ndarray, log_w: np.ndarray) -> np.ndarray:
    """ESS of the final weights grouped by distinct ancestor at each locus (§3.3 item 2).

    A value of 1 at early loci means every final particle descends from one starting particle:
    whatever the run says about long-range structure is one sample, not `N`.
    """
    w = normalise(log_w)
    out = np.empty(lin.shape[0])
    for t in range(lin.shape[0]):
        s = np.bincount(lin[t], weights=w)
        out[t] = 1.0 / np.square(s).sum()
    return out


def tv_distance(p: dict, q: dict) -> float:
    """Total variation over the union of two supports. Atoms missing from one side count."""
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in p.keys() | q.keys())


def multinomial_tv_reference(
    pi: np.ndarray, n: int, reps: int = 64, rng: np.random.Generator | None = None
) -> tuple[float, float]:
    """Mean and 99th-percentile TV that an *exact* iid sampler of size `n` would incur.

    The absolute `TV < 0.01` gate cannot on its own separate bias from Monte Carlo noise —
    on `repeat_twice` (242 atoms) the noise floor is already a sizeable fraction of it. Quoting
    the SMC's TV next to this band is what makes a marginal number readable.
    """
    rng = rng or np.random.default_rng(0)
    counts = rng.multinomial(n, pi, size=reps) / n
    tvs = 0.5 * np.abs(counts - pi).sum(axis=1)
    return float(tvs.mean()), float(np.quantile(tvs, 0.99))


def rank(values: np.ndarray, jitter: np.ndarray, x: float, x_jitter: float) -> int:
    """Randomised rank of `x` among `values`, ties broken by the supplied uniforms.

    Every statistic on a 20-path toy graph is heavily tied, and plain `#{v < x}` on tied values
    is not uniform under the null. Ranking the pairs `(value, jitter)` lexicographically with
    iid uniforms restores exact uniformity on `{0, …, len(values)}`.
    """
    return int(((values < x) | ((values == x) & (jitter < x_jitter))).sum())


def ks_uniform(ranks: np.ndarray, n_draws: int) -> float:
    """p-value of the KS test against `Uniform{0, …, n_draws}` (asymptotic Kolmogorov series)."""
    r = np.sort(np.asarray(ranks, dtype=np.float64))
    m = r.size
    u = (r + 0.5) / (n_draws + 1)
    d = max(float(np.max(np.arange(1, m + 1) / m - u)), float(np.max(u - np.arange(m) / m)))
    lam = (math.sqrt(m) + 0.12 + 0.11 / math.sqrt(m)) * d
    return min(1.0, 2.0 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 64)))
