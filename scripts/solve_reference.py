"""Reference solves with OpenSpiel's CFR+, exported per betting node for the viewer.

    uv run python scripts/solve_reference.py mini          # ~10 s
    uv run python scripts/solve_reference.py c1            # ~30 min

Both games are one limit betting round on a fixed flop: blinds of 1 each, bet 2, at most one
raise, two hole cards each.
  mini: 8-card deck (ranks 2-5 in clubs and diamonds), flop 3c4d5c.
  c1:   24-card deck (ranks 2-7 in all four suits), flop 5c6d7h.

Writes build/<game>.json: for each decision node (keyed by its betting sequence, f/c/r per
action), the acting player, what each seat has put in, the legal actions, and each private
hand's action probabilities. Exploitability is printed at checkpoints and stored at the end.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pyspiel

LIMIT = "betting=limit,numPlayers=2,numRounds=1,blind=1 1,raiseSize=2 2,firstPlayer=1 1,maxRaises=2 2"
GAMES = {
    "mini": (f"universal_poker({LIMIT},numSuits=2,numRanks=4,numHoleCards=2,numBoardCards=3 0,boardCards=3c4d5c)", 5000),
    "c1": (f"universal_poker({LIMIT},numSuits=4,numRanks=6,numHoleCards=2,numBoardCards=3 0,boardCards=5c6d7h)", 1000),
}
CHECKPOINTS = (10, 100, 300)


def contributions(seq: str) -> list[int]:
    """What each seat has put in after the betting sequence (P1 acts first, blinds of 1, bets of 2)."""
    c = [1, 1]
    for i, a in enumerate(seq):
        p = i % 2
        c[p] = c[1 - p] + (2 if a == "r" else 0)
    return c


def export(policy) -> dict:
    nodes = {}
    for key, entries in policy.policy_table().items():
        p = int(re.search(r"\[Player: (\d)\]", key).group(1))
        hand = re.search(r"\[Private: (\w+)\]", key).group(1)
        seq = re.search(r"\[Sequences: (\w*)\]", key).group(1)
        c = contributions(seq)
        facing = c[1 - p] > c[p]
        names = {0: "fold", 1: "call" if facing else "check", 2: "raise" if facing else "bet"}
        node = nodes.setdefault(seq, {"player": p, "contrib": c, "actions": [names[a] for a, _ in entries], "hands": {}})
        node["hands"][hand] = {names[a]: round(prob, 4) for a, prob in entries}
    return dict(sorted(nodes.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", choices=GAMES)
    ap.add_argument("--iterations", type=int)
    args = ap.parse_args()
    load, iterations = GAMES[args.game]
    iterations = args.iterations or iterations

    game = pyspiel.load_game(load)
    solver = pyspiel.CFRPlusSolver(game)
    start = time.time()
    for i in range(1, iterations + 1):
        solver.evaluate_and_update_policy()
        if i in CHECKPOINTS and i < iterations:
            expl = pyspiel.exploitability(game, solver.tabular_average_policy())
            print(f"{i} iterations, {time.time() - start:.0f} s: exploitability {expl:.3g}", file=sys.stderr, flush=True)
    policy = solver.tabular_average_policy()
    expl = pyspiel.exploitability(game, policy)
    print(f"{iterations} iterations, {time.time() - start:.0f} s: exploitability {expl:.3g}", file=sys.stderr)

    out = Path(__file__).resolve().parent.parent / "build" / f"{args.game}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"load": load, "iterations": iterations, "exploitability": expl, "nodes": export(policy)}))
    print(out, file=sys.stderr)


if __name__ == "__main__":
    main()
