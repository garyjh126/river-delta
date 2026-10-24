"""The C++ engine (engine/cpp) against the Python engine, which is its answer key."""

import itertools
import random

import numpy as np
import pytest

from engine.cards import value
from engine.cfr import SPOTS, Node, Solver
from engine.native import load
from engine.native import solver as cpp_solver

GAMES = ["mini", "c1", "mini_nl", "c1_nl"]


def py_tree(solver: Solver) -> dict:
    def child(c):
        return ("node", c.seq) if isinstance(c, Node) else ("end", c.folder, tuple(c.contrib))

    return {n.seq: (n.player, tuple(n.contrib), n.actions, [child(c) for c in n.children]) for n in solver.nodes}


def cpp_tree(solver) -> dict:
    nodes = solver.nodes

    def child(c):
        if c.node >= 0:
            return "node", nodes[c.node].seq
        return "end", None if c.folder < 0 else c.folder, tuple(c.contrib)

    return {n.seq: (n.player, tuple(n.contrib), n.actions, [child(c) for c in n.children]) for n in nodes}


@pytest.mark.parametrize("game", GAMES)
def test_same_hands_and_tree(game):
    py, cpp = Solver(SPOTS[game]), cpp_solver(SPOTS[game])
    assert cpp.hands == py.hands
    assert cpp.nodes[0].seq == ""
    assert len(cpp.nodes) == len(py.nodes)
    assert cpp_tree(cpp) == py_tree(py)


@pytest.mark.parametrize("game", GAMES)
def test_same_matrices(game):
    py, cpp = Solver(SPOTS[game]), cpp_solver(SPOTS[game])
    assert np.array_equal(cpp.compatible, py.compatible)
    assert np.array_equal(cpp.showdown, py.showdown)
    assert cpp.deal_prob == py.deal_prob


def test_hand_values_order_like_python():
    """Every five-card hand from C1's 24-card deck, plus a fixed sample from the full deck (for aces
    and the wheel), ranks in the same order under both evaluators."""
    small = [r + s for r in "234567" for s in "cdhs"]
    full = [r + s for r in "23456789TJQKA" for s in "shdc"]
    rng = random.Random(0)
    hands = [list(h) for h in itertools.combinations(small, 5)] + [rng.sample(full, 5) for _ in range(50_000)]
    cpp = load()
    ours = [cpp.hand_value(h) for h in hands]
    theirs = [value(h) for h in hands]
    def dense(vs):  # each value's place among the distinct values
        place = {v: k for k, v in enumerate(sorted(set(vs)))}
        return [place[v] for v in vs]

    assert dense(ours) == dense(theirs)
    assert len(set(theirs)) > 5_000  # many distinct values, so the order is a real check


def in_order(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """m @ v with each row's products added strictly left to right, as the C++ engine adds them."""
    return np.cumsum(m * v, axis=1)[:, -1]


class InOrder(Solver):
    """The Python engine with its matrix products added left to right, instead of in whatever order
    the BLAS library chooses. With this, the two engines agree bit for bit. Without it they can't:
    Mini no-limit is sensitive to the order, and with discounted CFR the two orders' average
    strategies drift 6e-3 apart by 5,000 iterations."""

    def _terminal(self, node, player, opp_reach):
        if node.folder is None:
            return node.contrib[0] * in_order(self.showdown, opp_reach)
        sign = -1 if node.folder == player else 1
        return sign * node.contrib[node.folder] * in_order(self.compatible, opp_reach)


@pytest.mark.parametrize("method", Solver.METHODS)
@pytest.mark.parametrize("game", GAMES)
def test_in_order_is_the_same_engine(game, method):
    """Adding up in order changes only rounding: early on, before any drift, the strategies agree
    (within 1e-14 after 10 iterations; C1 no-limit's discounted CFR drifts to 2.5e-12 by 20)."""
    plain, ordered = Solver(SPOTS[game], method), InOrder(SPOTS[game], method)
    plain.iterate(10)
    ordered.iterate(10)
    assert max(np.abs(plain.average(n) - ordered.average(n)).max() for n in plain.nodes) < 1e-12


# Every iteration at first, then spaced out, up to each game's own iteration count.
CHECKPOINTS = (*range(1, 11), 30, 100, 300, 1000, 3000, 5000)


@pytest.mark.parametrize("method", Solver.METHODS)
@pytest.mark.parametrize("game", GAMES)
def test_same_iterates(game, method):
    """Same algorithm, same iterates, bit for bit."""
    py, cpp = InOrder(SPOTS[game], method), cpp_solver(SPOTS[game], method)
    for checkpoint in (c for c in CHECKPOINTS if c <= SPOTS[game].iterations):
        py.iterate(checkpoint - py.iterations)
        cpp.iterate(checkpoint - cpp.iterations)
        assert cpp.iterations == checkpoint
        for i, node in enumerate(cpp.nodes):
            assert np.array_equal(cpp.average(i), py.average(py.by_seq[node.seq])), (checkpoint, node.seq)
