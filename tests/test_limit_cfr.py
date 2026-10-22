"""The vector CFR+ engine against OpenSpiel, an independent implementation of the same games."""

import re

import numpy as np
import pyspiel
import pytest

from engine.limit_cfr import SPOTS, Solver

ACTION_ID = {"fold": 0, "check": 1, "call": 1, "bet": 2, "raise": 2}


def info(key: str) -> tuple[int, str, str]:
    """(player, private hand, betting sequence) from an OpenSpiel universal_poker infostate string."""
    player = int(re.search(r"\[Player: (\d)\]", key).group(1))
    return player, re.search(r"\[Private: (\w+)\]", key).group(1), re.search(r"\[Sequences: (\w*)\]", key).group(1)


def as_openspiel(solver: Solver, table: dict) -> dict:
    """Our average strategy keyed like OpenSpiel's policy table `table`."""
    out = {}
    for key, entries in table.items():
        player, hand, seq = info(key)
        node = solver.by_seq[seq]
        assert node.player == player
        avg = solver.average(node)
        probs = {ACTION_ID[a]: float(avg[k, solver.index[hand]]) for k, a in enumerate(node.actions)}
        assert set(probs) == {a for a, _ in entries}, key
        out[key] = [(a, probs[a]) for a, _ in entries]
    return out


@pytest.mark.parametrize("game", ["mini", "c1"])
def test_same_infostates_as_openspiel(game):
    solver = Solver(SPOTS[game])
    table = pyspiel.CFRPlusSolver(pyspiel.load_game(SPOTS[game].openspiel)).tabular_average_policy().policy_table()
    assert len(table) == len(solver.nodes) * len(solver.hands)  # Mini 6 x 10 = 60, C1 6 x 210 = 1,260
    as_openspiel(solver, table)  # every OpenSpiel infostate maps to one of ours, with the same actions


def test_showdowns_match_openspiel():
    """Who wins each Mini deal at showdown after check-check, against OpenSpiel's hand evaluator."""
    solver = Solver(SPOTS["mini"])
    game = pyspiel.load_game(SPOTS["mini"].openspiel)
    checked = 0

    def walk(state):
        nonlocal checked
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a))
            return
        hands = [solver.index[info(state.information_state_string(p))[1]] for p in (0, 1)]
        for _ in range(2):
            state = state.child(1)  # check, check
        assert state.returns()[0] == solver.showdown[hands[0], hands[1]]  # each player has 1 in the pot
        checked += 1

    walk(game.new_initial_state())
    assert checked == 5 * 4 * 3 * 2  # every ordered deal of four cards from five


@pytest.mark.parametrize("game, iterations", [("mini", 100), ("c1", 3)])
def test_iterates_match_openspiel_cfr_plus(game, iterations):
    """Same algorithm, same iterates: our average strategy equals OpenSpiel's CFR+ after as many iterations."""
    spiel = pyspiel.CFRPlusSolver(pyspiel.load_game(SPOTS[game].openspiel))
    for _ in range(iterations):
        spiel.evaluate_and_update_policy()
    theirs = spiel.tabular_average_policy().policy_table()
    solver = Solver(SPOTS[game])
    solver.iterate(iterations)
    ours = as_openspiel(solver, theirs)
    gap = max(abs(p - q) for key in theirs for (_, p), (_, q) in zip(ours[key], theirs[key]))
    assert gap < 1e-9


@pytest.mark.parametrize("game, iterations", [("mini", 1000), ("c1", 100)])
def test_exploitability_matches_openspiel(game, iterations):
    """Our best response and OpenSpiel's give the same exploitability for our strategy."""
    pgame = pyspiel.load_game(SPOTS[game].openspiel)
    solver = Solver(SPOTS[game])
    solver.iterate(iterations)
    template = pyspiel.CFRPlusSolver(pgame).tabular_average_policy().policy_table()
    theirs = pyspiel.exploitability(pgame, pyspiel.TabularPolicy(as_openspiel(solver, template)))
    assert solver.exploitability() == pytest.approx(theirs, abs=1e-9)
    assert solver.exploitability() < 0.01


def test_mini_converges_to_its_value():
    """Mini's equilibrium value is 0 for both seats (OpenSpiel: about -5e-7 after 5,000 CFR+ iterations)."""
    solver = Solver(SPOTS["mini"])
    solver.iterate(5000)
    assert solver.exploitability() < 1e-4
    assert abs(solver.value(0)) < 1e-4
    assert solver.value(0) == pytest.approx(-solver.value(1))


def test_strategies_are_distributions():
    solver = Solver(SPOTS["c1"])
    solver.iterate(20)
    for node in solver.nodes:
        for sigma in (solver.current(node), solver.average(node)):
            assert np.all(sigma >= 0)
            assert np.allclose(sigma.sum(axis=0), 1)
