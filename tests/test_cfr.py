"""The vector CFR engine against OpenSpiel, an independent implementation of the same games."""

import dataclasses
import itertools
import random
import re

import numpy as np
import pyspiel
import pytest
from open_spiel.python.algorithms import discounted_cfr

from engine.cards import RANKS
from engine.cfr import SPOTS, Solver, Spot, Terminal

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


GAMES = ["mini", "c1", "mini_nl", "c1_nl", "c1_river"]


@pytest.mark.parametrize("game", GAMES)
def test_same_infostates_as_openspiel(game):
    solver = Solver(SPOTS[game])
    table = cfr_plus_table(game)
    assert len(table) == len(solver.nodes) * len(solver.hands)  # e.g. C1 no-limit: 12 x 210 = 2,520
    as_openspiel(solver, {k: [a for a, _ in v] for k, v in table.items()})  # same actions everywhere


def check_showdown(solver: Solver, state) -> None:
    """At a dealt state: check, check, and OpenSpiel's winner is ours."""
    hands = [solver.index[info(state.information_state_string(p))[1]] for p in (0, 1)]
    for _ in range(2):
        state = state.child(1)  # check, check
    assert state.returns()[0] == solver.showdown[hands[0], hands[1]]  # each player has 1 in the pot


@pytest.mark.parametrize("game, deals", [("mini", 5 * 4 * 3 * 2), ("c1_river", 19 * 18 * 17 * 16)])
def test_showdowns_match_openspiel(game, deals):
    """Who wins each deal at showdown after check-check, against OpenSpiel's hand evaluator: every
    ordered deal of four cards from those left (on the river, the best five of seven)."""
    solver = Solver(SPOTS[game])
    checked = 0

    def walk(state):
        nonlocal checked
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a))
            return
        check_showdown(solver, state)
        checked += 1

    walk(pyspiel.load_game(SPOTS[game].openspiel).new_initial_state())
    assert checked == deals


def test_full_deck_showdowns_match_openspiel():
    """The same on a full-deck river, for a sample of 3,000 deals."""
    spot = Spot(ranks=RANKS, suits="cdhs", board=("Ks", "8d", "4c", "2h", "7s"), iterations=0, stack=20)
    solver = Solver(spot)
    game = pyspiel.load_game(spot.openspiel)
    rng = random.Random(0)
    for _ in range(3000):
        state = game.new_initial_state()
        while state.is_chance_node():
            state = state.child(rng.choice(state.legal_actions()))
        check_showdown(solver, state)


class MatrixSolver(Solver):
    """The engine with the matrix products at its terminal nodes, as it had before the O(n) sums."""

    def _terminal(self, node, player, opp_reach):
        if node.folder is None:
            return node.contrib[0] * (self.showdown @ opp_reach)
        sign = -1 if node.folder == player else 1
        return sign * node.contrib[node.folder] * (self.compatible @ opp_reach)


# Mini's regrets tie exactly and CFR+ floors them at 0, so the way 1e-16 of rounding goes picks a
# strategy. The engine's O(n) sums at terminal nodes round differently from OpenSpiel's sums over
# deals, and on Mini its iterates part from OpenSpiel's within a few iterations; the matrix
# products round as OpenSpiel's sums do. So Mini's iterates are checked with them, and every other
# game's with the engine itself.
def iterating(game: str):
    return MatrixSolver if game == "mini" else Solver


@pytest.mark.parametrize("game, iterations", [("mini", 100), ("c1", 3), ("mini_nl", 100), ("c1_nl", 2), ("c1_river", 3)])
def test_cfr_plus_iterates_match_openspiel(game, iterations):
    """Same algorithm, same iterates: our average strategy equals OpenSpiel's CFR+ after as many iterations."""
    theirs = cfr_plus_table(game, iterations)
    solver = iterating(game)(SPOTS[game])
    solver.iterate(iterations)
    ours = as_openspiel(solver, {k: [a for a, _ in v] for k, v in theirs.items()})
    gap = max(abs(p - q) for key in theirs for (_, p), (_, q) in zip(ours[key], theirs[key]))
    assert gap < 1e-9


@pytest.mark.parametrize("game", ["mini", "mini_nl"])
def test_dcfr_iterates_match_openspiel(game):
    """Our discounted CFR equals OpenSpiel's DCFRSolver (alpha 3/2, beta 0, gamma 2), iteration for iteration."""
    spiel = discounted_cfr.DCFRSolver(pyspiel.load_game(SPOTS[game].openspiel))
    solver = iterating(game)(SPOTS[game], method="dcfr")
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


@pytest.mark.parametrize("game, iterations", [("mini", 1000), ("c1", 100), ("mini_nl", 1000), ("c1_nl", 50),
                                             ("c1_river", 100)])
def test_exploitability_matches_openspiel(game, iterations):
    """Our best response and OpenSpiel's give the same exploitability for our strategy."""
    pgame = pyspiel.load_game(SPOTS[game].openspiel)
    solver = Solver(SPOTS[game])
    solver.iterate(iterations)
    keys = {k: [a for a, _ in v] for k, v in cfr_plus_table(game).items()}
    theirs = pyspiel.exploitability(pgame, pyspiel.TabularPolicy(as_openspiel(solver, keys)))
    assert solver.exploitability() == pytest.approx(theirs, abs=1e-9)
    assert solver.exploitability() < 0.05


def openspiel_values(game, policy: dict) -> dict:
    """Walk every history of an OpenSpiel game, both players following `policy`. Per infostate:
    the chance-and-opponent reach summed over its histories, and per action that reach times the
    acting player's expected return after the action (OpenSpiel's returns, blinds included)."""
    table = {}

    def walk(state, reach):  # reach: chance, player 0, player 1
        if state.is_terminal():
            return np.array(state.returns())
        if state.is_chance_node():
            return sum(p * walk(state.child(a), (reach[0] * p, reach[1], reach[2])) for a, p in state.chance_outcomes())
        player, key = state.current_player(), state.information_state_string()
        others = reach[0] * reach[2 - player]
        entry = table.setdefault(key, [0.0, {}])
        entry[0] += others
        value = 0
        for a, p in policy[key]:
            after = list(reach)
            after[1 + player] *= p
            child = walk(state.child(a), after)
            entry[1][a] = entry[1].get(a, 0) + others * child[player]
            value = value + p * child
        return value

    walk(game.new_initial_state(), (1.0, 1.0, 1.0))
    return table


@pytest.mark.parametrize("game", ["mini", "mini_nl"])
def test_evs_match_openspiel(game):
    """Each hand's EV for each action, and the opponent's reach behind it, from a walk over every
    deal and betting line with OpenSpiel's rules and returns. Blinds are dead money in our EV."""
    spot = SPOTS[game]
    solver = Solver(spot)
    solver.iterate(200)
    keys = {k: [a for a, _ in v] for k, v in cfr_plus_table(game).items()}
    table = openspiel_values(pyspiel.load_game(spot.openspiel), as_openspiel(solver, keys))
    evs = solver.evs()
    defined = undefined = 0
    for key, (weight, totals) in table.items():
        _, hand, seq = info(key)
        node, i = solver.by_seq[seq], solver.index[hand]
        action_ev, opp_reach = evs[seq]
        assert opp_reach[i] == pytest.approx(len(solver.hands) * weight, abs=1e-12)  # each hand is dealt 1/n of the time
        for k, a in enumerate(node.actions):
            if weight > 0:
                assert action_ev[k, i] == pytest.approx(totals[ACTION_ID[a]] / weight + spot.blind, abs=1e-9)
                defined += 1
            else:
                assert np.isnan(action_ev[k, i])
                undefined += 1
    assert defined > 0
    assert defined + undefined == sum(len(n.actions) for n in solver.nodes) * len(solver.hands)


def test_evs_add_up():
    """Player 1's EVs at the root average to its value plus the blind it has in already; folding
    without having bet is worth 0; the opponent reaches the root with every hand."""
    solver = Solver(SPOTS["c1_nl"], method="dcfr")
    solver.iterate(100)
    evs = solver.evs()
    action_ev, opp_reach = evs[""]
    hand_ev = (solver.average(solver.root) * action_ev).sum(axis=0)
    assert hand_ev.mean() == pytest.approx(solver.value(0) + solver.spot.blind, abs=1e-12)
    assert np.all(opp_reach == 1)
    folds = 0
    for node in solver.nodes:
        if "fold" in node.actions and node.contrib[node.player] == solver.spot.blind:
            fold_ev = evs[node.seq][0][node.actions.index("fold")]
            assert np.all((fold_ev == 0) | np.isnan(fold_ev))
            folds += 1
    assert folds > 0


def walk_deals(solver: Solver) -> tuple[float, dict]:
    """The vector engine's sums done the slow way: every deal one at a time, both players following
    the average strategy. Returns player 0's expected chips, each deal weighted by the two starting
    ranges, and per (node, hand of the player acting there): the opponent's chance of being there,
    summed over its hands and weighted by its range, and the same times the chips after each action."""
    w, average, table = solver.ranges, {node.seq: solver.average(node) for node in solver.nodes}, {}

    def walk(node, hands, opp_reach):  # player 0's expected chips from here; opp_reach[p]: p's opponent's chance
        if isinstance(node, Terminal):
            if node.folder is None:
                return node.contrib[0] * solver.showdown[hands]
            return node.contrib[node.folder] * (1 if node.folder else -1)
        p = node.player
        sigma = average[node.seq][:, hands[p]]
        entry = table.setdefault((node.seq, hands[p]), [0.0, np.zeros(len(node.actions))])
        weight = w[1 - p, hands[1 - p]] * opp_reach[p]
        entry[0] += weight
        value = 0.0
        for a, child in enumerate(node.children):
            reach = list(opp_reach)
            reach[1 - p] *= sigma[a]
            chips = walk(child, hands, reach)
            entry[1][a] += weight * (chips if p == 0 else -chips)
            value += sigma[a] * chips
        return value

    total = 0.0
    for i, j in itertools.product(range(len(solver.hands)), repeat=2):
        if solver.compatible[i, j]:
            total += w[0, i] * w[1, j] * solver.deal_prob * walk(solver.root, (i, j), [1.0, 1.0])
    return total, table


@pytest.mark.parametrize("game", ["mini", "mini_nl", "c1_river"])
def test_ranges_match_a_walk_over_every_deal(game):
    """With random starting ranges (a random weight on most hands, the rest left out), the engine's
    values, EVs and opponent reaches equal a walk over every deal."""
    rng = random.Random(1)
    hands = Solver(SPOTS[game]).hands
    ranges = tuple(", ".join(f"{h}:{rng.random():.3f}" for h in hands if rng.random() < 0.8) for _ in range(2))
    solver = Solver(dataclasses.replace(SPOTS[game], ranges=ranges), method="dcfr")
    assert 0 < (solver.ranges == 0).sum() < solver.ranges.size
    solver.iterate(100)
    value, table = walk_deals(solver)
    assert solver.value(0) == pytest.approx(value, abs=1e-12)
    assert solver.value(1) == pytest.approx(-value, abs=1e-12)
    evs = solver.evs()
    for (seq, i), (weight, totals) in table.items():
        action_ev, opp_reach = evs[seq]
        player = solver.by_seq[seq].player
        assert opp_reach[i] == pytest.approx(weight / (solver.compatible[i] @ solver.ranges[1 - player]), abs=1e-12)
        if weight > 0:
            assert action_ev[:, i] == pytest.approx(totals / weight + solver.spot.blind, abs=1e-9)
        else:
            assert np.all(np.isnan(action_ev[:, i]))


def test_ev_is_undefined_where_the_opponent_never_reaches():
    solver = Solver(SPOTS["mini"])
    solver.strategy_sum[""][0] = 1  # player 1 always checks, so player 2 never faces a bet
    action_ev, opp_reach = solver.evs()["r"]
    assert np.all(np.isnan(action_ev)) and np.all(opp_reach == 0)
    assert solver.export()["nodes"]["r"]["ev"] == [None] * len(solver.hands)


@pytest.mark.parametrize("game", GAMES)
def test_terminal_sums_equal_the_matrix_products(game):
    """The O(n) sums at terminal nodes equal the products with the matrices over pairs of hands,
    up to rounding, for random reaches with some hands at 0."""
    solver = Solver(SPOTS[game])
    rng = np.random.default_rng(0)
    for _ in range(20):
        reach = rng.random(len(solver.hands)) * (rng.random(len(solver.hands)) < 0.7)
        assert np.allclose(solver._compatible_reach(reach), solver.compatible @ reach, rtol=0, atol=1e-12)
        assert np.allclose(solver._showdown_reach(reach), solver.showdown @ reach, rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", Solver.METHODS)
@pytest.mark.parametrize("game", [g for g in GAMES if g != "mini"])
def test_terminal_sums_are_the_same_engine(game, method):
    """The O(n) sums change only rounding: early on, before any drift, the strategies agree with the
    matrix products' (within 2e-14 after 10 iterations). Mini's part within a few iterations (see
    `iterating`)."""
    fast, matrix = Solver(SPOTS[game], method), MatrixSolver(SPOTS[game], method)
    fast.iterate(10)
    matrix.iterate(10)
    assert max(np.abs(fast.average(n) - matrix.average(n)).max() for n in fast.nodes) < 1e-12


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
        assert len(node["probs"]) == len(node["actions"]) == len(node["actionEv"])
        assert len(node["ev"]) == len(node["oppReach"]) == len(tree["hands"])
        for a in node["actions"]:
            assert (a["next"] in tree["nodes"]) if a["next"] else a["end"]
        assert np.allclose(np.sum(node["probs"], axis=0), 1, atol=5e-4)  # rounded to 4 decimals
