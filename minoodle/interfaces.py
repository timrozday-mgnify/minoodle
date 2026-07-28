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


class TokenSpace(ABC):
    """Base-space k-mers and minimizer tuples both implement this (§4.1, M8)."""

    @abstractmethod
    def tokenize(self, seq: bytes) -> list[tuple[int, int]]:
        """Return `(token, offset)` pairs for `seq`."""

    @property
    @abstractmethod
    def k(self) -> int: ...


class PathGraph(ABC):
    """The assembly graph as the sampler sees it (§4.1)."""

    @abstractmethod
    def out_edges(self, n: OrientedNode) -> np.ndarray: ...

    @abstractmethod
    def unitig_seq(self, n: OrientedNode) -> bytes: ...

    @abstractmethod
    def unitig_depth(self, n: OrientedNode) -> np.ndarray:
        """Per-base depth. Note §5 M3: metaSPAdes `cov`/`KC:i` are k-mer based and must be
        converted by `L/(L-k+1)` before use, or the §2.7 NegBin mean is systematically wrong.
        """


class IncrementalLikelihood(ABC):
    """The bounded-window contract from §2.4. If a term can't fit this, it's not in v0.

    Implementations carry whatever state they need in an opaque `State` object, but `extend`
    must be computable in bounded time from the last `W = max_fragment + read_len` bases —
    it may not consult the whole path. `stop_logp` is likewise a function of bounded recent
    state (§2.8): it is *not* a likelihood-drop threshold, and must not depend on whether the
    data "look finished".
    """

    @abstractmethod
    def init(self, start: OrientedNode) -> Any: ...

    @abstractmethod
    def extend(self, st: Any, e: Edge) -> tuple[Any, float]:
        """Return `(new_state, log increment)` for extending along `e`."""

    @abstractmethod
    def stop_logp(self, st: Any) -> float: ...


class CompositeLikelihood(IncrementalLikelihood):
    """Sum of terms (§4.1). Ablations are free: drop a term from the list."""

    def __init__(self, terms: list[IncrementalLikelihood]):
        self.terms = list(terms)

    def init(self, start: OrientedNode) -> tuple[Any, ...]:
        return tuple(t.init(start) for t in self.terms)

    def extend(self, st: tuple[Any, ...], e: Edge) -> tuple[tuple[Any, ...], float]:
        states: list[Any] = []
        total = 0.0
        for term, sub in zip(self.terms, st, strict=True):
            new_sub, incr = term.extend(sub, e)
            states.append(new_sub)
            total += incr
        return tuple(states), total

    def stop_logp(self, st: tuple[Any, ...]) -> float:
        return sum(t.stop_logp(sub) for t, sub in zip(self.terms, st, strict=True))
