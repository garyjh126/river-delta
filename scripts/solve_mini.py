"""Reference solve of the Mini game with OpenSpiel's CFR+, exported per betting node for the viewer.

Mini: 8-card deck (ranks 2-5 in clubs and diamonds), fixed flop 3c4d5c, two hole cards each,
one limit betting round (bet 2, at most one raise). Writes build/mini.json: for each decision
node (keyed by its betting sequence, f/c/r per action), the acting player, what each seat has
put in, the legal actions, and each private hand's action probabilities.
"""

import json
import re
import sys
from pathlib import Path

import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability, expected_game_score

LOAD = (
    "universal_poker(betting=limit,numPlayers=2,numRounds=1,blind=1 1,raiseSize=2 2,"
    "firstPlayer=1 1,numSuits=2,numRanks=4,numHoleCards=2,maxRaises=2 2,"
    "numBoardCards=3 0,boardCards=3c4d5c)"
)
ITERATIONS = 5000
OUT = Path(__file__).resolve().parent.parent / "build" / "mini.json"

game = pyspiel.load_game(LOAD)
solver = pyspiel.CFRPlusSolver(game)
for _ in range(ITERATIONS):
    solver.evaluate_and_update_policy()
policy = policy_lib.pyspiel_policy_to_python_policy(game, solver.tabular_average_policy())
expl = float(exploitability.exploitability(game, policy))
values = expected_game_score.policy_value(game.new_initial_state(), [policy, policy])
print(f"exploitability {expl:.3g} chips/hand, values {values}", file=sys.stderr)

nodes = {}


def walk(state, seq, contrib):
    if state.is_terminal():
        return
    if state.is_chance_node():
        for a, _ in state.chance_outcomes():
            walk(state.child(a), seq, contrib)
        return
    p = state.current_player()
    hand = re.search(r"\[Private: (\w+)\]", state.information_state_string(p)).group(1)
    facing = contrib[1 - p] > contrib[p]
    names = {0: "fold", 1: "call" if facing else "check", 2: "raise" if facing else "bet"}
    probs = policy.action_probabilities(state, p)
    legal = state.legal_actions()
    node = nodes.setdefault(seq, {"player": p, "contrib": list(contrib), "actions": [names[a] for a in legal], "hands": {}})
    node["hands"][hand] = {names[a]: round(float(probs.get(a, 0.0)), 4) for a in legal}
    for a in legal:
        c = list(contrib)
        if a == 1:
            c[p] = c[1 - p]
        elif a == 2:
            c[p] = c[1 - p] + 2
        walk(state.child(a), seq + "fcr"[a], c)


walk(game.new_initial_state(), "", [1, 1])
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"load": LOAD, "iterations": ITERATIONS, "exploitability": expl, "nodes": nodes}))
print(OUT, file=sys.stderr)
