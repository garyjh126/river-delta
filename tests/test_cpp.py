"""The C++ engine (engine/cpp) against the Python engine, which is its answer key."""

import dataclasses
import itertools
import random

import numpy as np
import pytest

from engine.cards import value
from engine.cfr import SPOTS, Node, Solver, deal_hands
from engine.native import load
from engine.native import solver as cpp_solver



def with_random_ranges(spot, seed: int = 1):
    """`spot` with a random weight on most hands for each player, the rest left out."""
    rng = random.Random(seed)
    hands = deal_hands(spot)
    ranges = tuple(", ".join(f"{h}:{rng.random():.3f}" for h in hands if rng.random() < 0.8) for _ in range(2))
    return dataclasses.replace(spot, ranges=ranges)


# The full-deck river only to 30 iterations: the Python engine takes 0.1 s for each.
CASES = {**SPOTS, "c1_river_ranges": with_random_ranges(SPOTS["c1_river"]),
         "river": dataclasses.replace(SPOTS["river"], iterations=30)}
GAMES = ["mini", "c1", "mini_nl", "c1_nl", "c1_river", "c1_river_ranges", "river"]


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
    py, cpp = Solver(CASES[game]), cpp_solver(CASES[game])
    assert cpp.hands == py.hands
    assert cpp.nodes[0].seq == ""
    assert len(cpp.nodes) == len(py.nodes)
    assert cpp_tree(cpp) == py_tree(py)


@pytest.mark.parametrize("game", GAMES)
def test_same_matrices(game):
    py, cpp = Solver(CASES[game]), cpp_solver(CASES[game])
    assert np.array_equal(cpp.compatible, py.compatible)
    assert np.array_equal(cpp.showdown, py.showdown)
    assert np.array_equal(cpp.ranges, py.ranges)
    assert cpp.deal_prob == pytest.approx(py.deal_prob, rel=1e-14)  # numpy adds up the deals in another order


def test_hand_values_order_like_python():
    """Every five-card hand from C1's 24-card deck, plus fixed samples of five and seven cards from the
    full deck (for aces, the wheel and the best five of seven), ranks in the same order under both
    evaluators."""
    small = [r + s for r in "234567" for s in "cdhs"]
    full = [r + s for r in "23456789TJQKA" for s in "shdc"]
    rng = random.Random(0)
    hands = [list(h) for h in itertools.combinations(small, 5)] + [rng.sample(full, 5) for _ in range(50_000)]
    hands += [rng.sample(full, 7) for _ in range(10_000)]
    cpp = load()
    ours = [cpp.hand_value(h) for h in hands]
    theirs = [value(h) for h in hands]
    def dense(vs):  # each value's place among the distinct values
        place = {v: k for k, v in enumerate(sorted(set(vs)))}
        return [place[v] for v in vs]

    assert dense(ours) == dense(theirs)
    assert len(set(theirs)) > 5_000  # many distinct values, so the order is a real check


# Every iteration at first, then spaced out, up to each game's own iteration count.
CHECKPOINTS = (*range(1, 11), 30, 100, 300, 1000, 3000, 5000)


@pytest.mark.parametrize("method", Solver.METHODS)
@pytest.mark.parametrize("game", GAMES)
def test_same_iterates(game, method):
    """Same algorithm, same iterates: average strategies bit for bit, because both engines add up
    every sum in the same order. Values, best responses and exploitability within 1e-12 chips per
    hand (at most 9e-15 seen): numpy adds up the final sum over hands in a different order."""
    py, cpp = Solver(CASES[game], method), cpp_solver(CASES[game], method)
    for checkpoint in (c for c in CHECKPOINTS if c <= CASES[game].iterations):
        py.iterate(checkpoint - py.iterations)
        cpp.iterate(checkpoint - cpp.iterations)
        assert cpp.iterations == checkpoint
        for i, node in enumerate(cpp.nodes):
            assert np.array_equal(cpp.average(i), py.average(py.by_seq[node.seq])), (checkpoint, node.seq)
        for player in (0, 1):
            assert cpp.value(player) == pytest.approx(py.value(player), abs=1e-12)
            assert cpp.best_response(player) == pytest.approx(py.best_response(player), abs=1e-12)
        assert cpp.exploitability() == pytest.approx(py.exploitability(), abs=1e-12)


@pytest.mark.parametrize("method", Solver.METHODS)
@pytest.mark.parametrize("game", GAMES)
def test_trace_is_the_same_run(game, method):
    """A traced run (the replay page's) ends where an untraced one does, and the regrets it records
    last at each node are the Python engine's, bit for bit."""
    traced, plain, py = cpp_solver(CASES[game], method), cpp_solver(CASES[game], method), Solver(CASES[game], method)
    steps = traced.trace(10)
    plain.iterate(10)
    py.iterate(10)
    for i in range(len(traced.nodes)):
        assert np.array_equal(traced.average(i), plain.average(i))
    last = {s.node: s.regret for s in steps if s.kind.name == "update" and s.regret}
    assert len(last) == len(traced.nodes)
    for i, regret in last.items():
        node = traced.nodes[i]
        assert np.array_equal(np.reshape(regret, (len(node.actions), -1)), py.regret[node.seq]), node.seq
