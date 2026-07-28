from minoodle.interfaces import (
    CompositeLikelihood,
    IncrementalLikelihood,
    OrientedNode,
    Seed,
    Side,
)


class Counter(IncrementalLikelihood):
    """Fake term: state is a step count, increment is a fixed constant."""

    def __init__(self, incr: float):
        self.incr = incr

    def init(self, seed):
        return 0, self.incr

    def extend(self, st, e, side):
        return st + 1, self.incr

    def stop_logp(self, st, side):
        return -st * self.incr


def test_composite_sums_increments_and_threads_states():
    comp = CompositeLikelihood([Counter(1.0), Counter(0.25)])
    st, incr = comp.init(Seed(OrientedNode(0, True), 0))
    assert st == (0, 0)
    assert incr == 1.25

    st, incr = comp.extend(st, OrientedNode(1, True), Side.RIGHT)
    assert incr == 1.25
    st, incr = comp.extend(st, OrientedNode(2, False), Side.LEFT)
    assert incr == 1.25
    assert st == (2, 2)

    assert comp.stop_logp(st, Side.LEFT) == -2 * 1.0 + -2 * 0.25


def test_composite_of_no_terms_is_zero():
    comp = CompositeLikelihood([])
    st, incr = comp.init(Seed(OrientedNode(0, True), 0))
    assert incr == 0.0
    st, incr = comp.extend(st, OrientedNode(1, True), Side.RIGHT)
    assert incr == 0.0
    assert comp.stop_logp(st, Side.LEFT) == 0.0


def test_oriented_node_flip_round_trips():
    n = OrientedNode(7, True)
    assert n.flipped().flipped() == n
    assert n.flipped() != n
