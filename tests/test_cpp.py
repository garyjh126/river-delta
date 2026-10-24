"""The C++ engine (engine/cpp) against the Python engine, which is its answer key."""

import pytest

from engine.cfr import SPOTS, Node, Solver
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
