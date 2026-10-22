"""The vector CFR engine against OpenSpiel, an independent implementation of the same games."""

import re

import numpy as np
import pyspiel
import pytest
from open_spiel.python.algorithms import discounted_cfr

from engine.cfr import SPOTS, Solver

ACTION_ID = {"fold": 0, "check": 1, "call": 1, "bet": 2, "raise": 2, "allin": 3}


def info(key: str) -> tuple[int, str, str]:
    """(player, private hand, betting sequence) from an OpenSpiel universal_poker infostate string."""
    player = int(re.search(r"\[Player: (\d)\]", key).group(1))
    return player, re.search(r"\[Private: (\w+)\]", key).group(1), re.search(r"\[Sequences: (\w*)\]", key).group(1)


def as_openspiel(solver: Solver, keys: dict) -> dict:
    """Our average strategy keyed like OpenSpiel's infostates: {key: [(action id, probability)]}.
    `keys` maps each OpenSpiel infostate to its legal action ids."""
    out = {}
    for key, legal in keys.items():
        player, hand, seq = info(key)
        node = solver.by_seq[seq]
        assert node.player == player
        avg = solver.average(node)
        probs = {ACTION_ID[a]: float(avg[k, solver.index[hand]]) for k, a in enumerate(node.actions)}
        assert set(probs) == set(legal), key
        out[key] = [(a, probs[a]) for a in legal]
    return out


def cfr_plus_table(game: str, iterations: int = 0) -> dict:
    spiel = pyspiel.CFRPlusSolver(pyspiel.load_game(SPOTS[game].openspiel))
    for _ in range(iterations):
        spiel.evaluate_and_update_policy()
    return spiel.tabular_average_policy().policy_table()


GAMES = ["mini", "c1", "mini_nl", "c1_nl"]


@pytest.mark.parametrize("game", GAMES)
def test_same_infostates_as_openspiel(game):
    solver = Solver(SPOTS[game])
    table = cfr_plus_table(game)
    assert len(table) == len(solver.nodes) * len(solver.hands)  # e.g. C1 no-limit: 12 x 210 = 2,520
    as_openspiel(solver, {k: [a for a, _ in v] for k, v in table.items()})  # same actions everywhere


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


@pytest.mark.parametrize("game, iterations", [("mini", 100), ("c1", 3), ("mini_nl", 100), ("c1_nl", 2)])
def test_cfr_plus_iterates_match_openspiel(game, iterations):
    """Same algorithm, same iterates: our average strategy equals OpenSpiel's CFR+ after as many iterations."""
    theirs = cfr_plus_table(game, iterations)
    solver = Solver(SPOTS[game])
    solver.iterate(iterations)
    ours = as_openspiel(solver, {k: [a for a, _ in v] for k, v in theirs.items()})
    gap = max(abs(p - q) for key in theirs for (_, p), (_, q) in zip(ours[key], theirs[key]))
    assert gap < 1e-9


@pytest.mark.parametrize("game", ["mini", "mini_nl"])
def test_dcfr_iterates_match_openspiel(game):
    """Our discounted CFR equals OpenSpiel's DCFRSolver (alpha 3/2, beta 0, gamma 2), iteration for iteration."""
    spiel = discounted_cfr.DCFRSolver(pyspiel.load_game(SPOTS[game].openspiel))
    solver = Solver(SPOTS[game], method="dcfr")
    for _ in range(50):
        spiel.evaluate_and_update_policy()
    solver.iterate(50)
    theirs = spiel.average_policy()
    by_infostate = {info(k)[1:]: k for k in theirs.state_lookup}
    gap = 0.0
    for node in solver.nodes:
        avg = solver.average(node)
        for j, hand in enumerate(solver.hands):
            row = theirs.policy_for_key(by_infostate[hand, node.seq])
            gap = max(gap, max(abs(avg[k, j] - row[ACTION_ID[a]]) for k, a in enumerate(node.actions)))
    assert gap < 1e-9


@pytest.mark.parametrize("game, iterations", [("mini", 1000), ("c1", 100), ("mini_nl", 1000), ("c1_nl", 50)])
def test_exploitability_matches_openspiel(game, iterations):
    """Our best response and OpenSpiel's give the same exploitability for our strategy."""
    pgame = pyspiel.load_game(SPOTS[game].openspiel)
    solver = Solver(SPOTS[game])
    solver.iterate(iterations)
    keys = {k: [a for a, _ in v] for k, v in cfr_plus_table(game).items()}
    theirs = pyspiel.exploitability(pgame, pyspiel.TabularPolicy(as_openspiel(solver, keys)))
    assert solver.exploitability() == pytest.approx(theirs, abs=1e-9)
    assert solver.exploitability() < 0.05


def test_mini_converges_to_its_value():
    """Mini's equilibrium value is 0 for both seats (OpenSpiel: about -5e-7 after 5,000 CFR+ iterations)."""
    solver = Solver(SPOTS["mini"])
    solver.iterate(5000)
    assert solver.exploitability() < 1e-4
    assert abs(solver.value(0)) < 1e-4
    assert solver.value(0) == pytest.approx(-solver.value(1))


@pytest.mark.parametrize("game", ["c1", "c1_nl"])
def test_dcfr_converges(game):
    solver = Solver(SPOTS[game], method="dcfr")
    solver.iterate(300)
    assert solver.exploitability() < 1e-3


@pytest.mark.parametrize("method", Solver.METHODS)
def test_strategies_are_distributions(method):
    solver = Solver(SPOTS["c1_nl"], method=method)
    solver.iterate(20)
    for node in solver.nodes:
        for sigma in (solver.current(node), solver.average(node)):
            assert np.all(sigma >= 0)
            assert np.allclose(sigma.sum(axis=0), 1)


def test_export_is_a_complete_tree():
    solver = Solver(SPOTS["c1_nl"])
    solver.iterate(10)
    tree = solver.export()
    assert set(tree["nodes"]) == {node.seq or "root" for node in solver.nodes}
    for node in tree["nodes"].values():
        assert len(node["probs"]) == len(node["actions"])
        for a in node["actions"]:
            assert (a["next"] in tree["nodes"]) if a["next"] else a["end"]
        assert np.allclose(np.sum(node["probs"], axis=0), 1, atol=5e-4)  # rounded to 4 decimals
