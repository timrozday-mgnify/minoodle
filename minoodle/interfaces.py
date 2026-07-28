"""Abstract interfaces from §4.1 of the implementation plan.

These mirror the eventual Rust traits (§4.2) — the Rust port at M7 is written by reading
this file, so keep the method signatures honest.

The load-bearing one is `IncrementalLikelihood`: it is the mechanism that enforces the
bounded-window constraint of §2.4. A term that cannot be expressed through it does not go
in v0 (§6.1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class OrientedNode:
    """A unitig visited in a given orientation (§2.1).

    `forward=False` means the reverse complement of the stored unitig sequence. Orientation
    handling is the most common source of silent bugs in bidirected graphs (§5 M3), hence a
    distinct type rather than a signed integer.
    """

    unitig: int
    forward: bool

    def flipped(self) -> OrientedNode:
        return OrientedNode(self.unitig, not self.forward)


# ponytail: an edge in a bidirected unitig graph is fully determined by its target oriented
# node, so it is an alias, not a class. M3 widens it if edges acquire their own payload.
Edge = OrientedNode


class Side(IntEnum):
    """Which frontier of a two-sided path a step extends (§2.1).

    The left frontier is an ordinary *forward* walk from the seed node flipped, so `out_edges`
    and `new_bases` need no per-side variants and the RC symmetry of §2.1 holds by construction
    rather than by case analysis. What the side is needed for is the likelihood: each term keeps
    its own bounded window per frontier (§2.4), and STOP is per side (§3.2).
    """

    LEFT = 0
    RIGHT = 1

    def other(self) -> Side:
        return Side.RIGHT if self is Side.LEFT else Side.LEFT


@dataclass(frozen=True, slots=True)
class Seed:
    """A k-mer position: an oriented unitig plus the k-mer's base offset within it (§2.1).

    `offset` is in `node`'s own orientation, so `Seed(n, o)` and `Seed(n.flipped(), L-k-o)` are
    the same physical k-mer read off the two strands — two distinct states carrying equal
    `p_seed` mass, which is what makes the prior RC-symmetric.
    """

    node: OrientedNode
    offset: int

    def flipped(self, unitig_len: int, k: int) -> Seed:
        return Seed(self.node.flipped(), unitig_len - k - self.offset)


class TokenSpace(ABC):
    """Base-space k-mers and minimizer tuples both implement this (§4.1, M8)."""

    @abstractmethod
    def tokenize(self, seq: bytes) -> list[tuple[int, int]]:
        """Return `(token, offset)` pairs for `seq`."""

    @property
    @abstractmethod
    def k(self) -> int: ...


class PathGraph(ABC):
    """The assembly graph as the sampler sees it (§4.1).

    `k` is the overlap convention: consecutive unitigs on a path share `k-1` bases. Note that
    metaSPAdes `-k 21` writes `21M` overlaps, so its graphs are `k = 22` here (§5 M3).
    """

    k: int

    @abstractmethod
    def nodes(self) -> list[OrientedNode]: ...

    @abstractmethod
    def out_edges(self, n: OrientedNode) -> np.ndarray: ...

    @abstractmethod
    def unitig_seq(self, n: OrientedNode) -> bytes: ...

    @abstractmethod
    def unitig_depth(self, n: OrientedNode) -> np.ndarray:
        """Per-base depth. Note §5 M3: metaSPAdes `cov`/`KC:i` are k-mer based and must be
        converted by `L/(L-k+1)` before use, or the §2.7 NegBin mean is systematically wrong.
        """

    def seeds(self) -> list[Seed]:
        """Every oriented k-mer position (§2.3's `p_seed` support).

        Unitigs overlap by `k-1 < k`, so no k-mer position is shared between two unitigs and
        the support is exactly `Σ_u 2·(L_u - k + 1)` positions, each equally likely.
        """
        return [
            Seed(n, o)
            for n in self.nodes()
            for o in range(len(self.unitig_seq(n)) - self.k + 1)
        ]

    def new_bases(self, n: OrientedNode, first: bool) -> int:
        """Bases `n` contributes to the path sequence: all of them if it starts the path."""
        return len(self.unitig_seq(n)) - (0 if first else self.k - 1)

    def path_seq(self, path: tuple[OrientedNode, ...]) -> bytes:
        out = self.unitig_seq(path[0])
        for n in path[1:]:
            out += self.unitig_seq(n)[self.k - 1 :]
        return out


class IncrementalLikelihood(ABC):
    """The bounded-window contract from §2.4. If a term can't fit this, it's not in v0.

    Implementations carry whatever state they need in an opaque `State` object, but `extend`
    must be computable in bounded time from the last `W = max_fragment + read_len` bases —
    it may not consult the whole path. `stop_logp` is likewise a function of bounded recent
    state (§2.8): it is *not* a likelihood-drop threshold, and must not depend on whether the
    data "look finished".
    """

    @abstractmethod
    def init(self, seed: Seed) -> tuple[Any, float]:
        """Return `(state, log increment)` for the bare seed — its unitig and nothing else.

        Amended at M1 from the §4.1 sketch's `-> State`: the seed unitig's own bases score like
        any other bases, and with no return channel for them the first node was silently free.
        The SMC engine needs the same number as a particle's initial log-weight. Amended again
        at M3.5 from `start: OrientedNode`: this increment is where §2.6's score-once rule pays
        out, since for the first few steps both frontiers' windows cover the seed unitig.
        """

    @abstractmethod
    def extend(self, st: Any, e: Edge, side: Side) -> tuple[Any, float]:
        """Return `(new_state, log increment)` for extending `side`'s frontier along `e`."""

    @abstractmethod
    def stop_logp(self, st: Any, side: Side) -> float:
        """Score `side`'s STOP. One of that side's fully-adapted alternatives (§3.2), not a
        separate case, and not a likelihood-drop threshold (§2.8)."""


class CompositeLikelihood(IncrementalLikelihood):
    """Sum of terms (§4.1). Ablations are free: drop a term from the list."""

    def __init__(self, terms: list[IncrementalLikelihood]):
        self.terms = list(terms)

    def init(self, seed: Seed) -> tuple[tuple[Any, ...], float]:
        states: list[Any] = []
        total = 0.0
        for term in self.terms:
            sub, incr = term.init(seed)
            states.append(sub)
            total += incr
        return tuple(states), total

    def extend(self, st: tuple[Any, ...], e: Edge, side: Side) -> tuple[tuple[Any, ...], float]:
        states: list[Any] = []
        total = 0.0
        for term, sub in zip(self.terms, st, strict=True):
            new_sub, incr = term.extend(sub, e, side)
            states.append(new_sub)
            total += incr
        return tuple(states), total

    def stop_logp(self, st: tuple[Any, ...], side: Side) -> float:
        return sum(t.stop_logp(sub, side) for t, sub in zip(self.terms, st, strict=True))
