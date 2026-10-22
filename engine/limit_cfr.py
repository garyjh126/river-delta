"""Vector-form CFR+ for heads-up limit poker with one betting round on a fixed flop.

The first piece of our own solver. It solves the Mini and C1 games, which OpenSpiel also solves
(scripts/solve_reference.py), so every result can be checked against an independent
implementation (tests/test_limit_cfr.py).

How it works. The betting is walked as a public tree, one node per betting sequence, and each
node carries a vector over every private hand instead of one node per deal (Johanson et al.
2011). Cards enter only at the terminal nodes, through matrices over pairs of hands that zero out
pairs sharing a card. Regrets follow CFR (Zinkevich et al. 2007) with regret matching+,
alternating updates and linearly weighted averaging, as in CFR+ (Tammelin 2014; Tammelin et al.
2015). Exploitability comes from an exact best response over the same public tree.

References
  Zinkevich, Johanson, Bowling, Piccione (2007). Regret Minimization in Games with Incomplete
    Information. NIPS 2007.
  Johanson, Waugh, Bowling, Zinkevich (2011). Accelerating Best Response Calculation in Large
    Extensive Games. IJCAI 2011.
  Tammelin (2014). Solving Large Imperfect Information Games Using CFR+. arXiv:1407.5042.
  Tammelin, Burch, Johanson, Bowling (2015). Solving Heads-up Limit Texas Hold'em. IJCAI 2015.

    uv run python engine/limit_cfr.py mini c1             # writes build/<game>.engine.json
"""

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.cards import value


@dataclass(frozen=True)
class Spot:
    """One betting round of limit poker on a fixed flop. Both players post `blind`, player 0
    acts first, every bet or raise is `bet` chips, and at most `max_raises` of them are made."""

    ranks: str
    suits: str
    board: tuple[str, ...]
    iterations: int
    blind: int = 1
    bet: int = 2
    max_raises: int = 2

    @property
    def openspiel(self) -> str:
        """The same game as an OpenSpiel `universal_poker` string, for cross-checks."""
        return (
            "universal_poker(betting=limit,numPlayers=2,numRounds=1,"
            f"blind={self.blind} {self.blind},raiseSize={self.bet} {self.bet},firstPlayer=1 1,"
            f"maxRaises={self.max_raises} {self.max_raises},numSuits={len(self.suits)},"
            f"numRanks={len(self.ranks)},numHoleCards=2,numBoardCards=3 0,boardCards={''.join(self.board)})"
        )


SPOTS = {
    "mini": Spot("2345", "cd", ("3c", "4d", "5c"), iterations=5000),
    "c1": Spot("234567", "cdhs", ("5c", "6d", "7h"), iterations=1000),
}


@dataclass
class Terminal:
    folder: int | None  # the player who folded, or None for a showdown
    contrib: tuple[int, int]


@dataclass
class Node:
    seq: str  # betting so far: f fold, c check or call, r bet or raise
    player: int
    contrib: tuple[int, int]
    actions: list[str]
    children: list  # Node or Terminal, one per action


def build_tree(spot: Spot, seq: str = "", contrib: tuple[int, int] | None = None) -> Node:
    contrib = contrib or (spot.blind, spot.blind)
    p = len(seq) % 2
    facing = contrib[1 - p] > contrib[p]
    actions, children = [], []
    if facing:
        actions += ["fold", "call"]
        called = list(contrib)
        called[p] = contrib[1 - p]
        children += [Terminal(p, contrib), Terminal(None, tuple(called))]
    else:
        actions.append("check")
        children.append(Terminal(None, contrib) if seq.endswith("c") else build_tree(spot, seq + "c", contrib))
    if seq.count("r") < spot.max_raises:
        raised = list(contrib)
        raised[p] = contrib[1 - p] + spot.bet
        actions.append("raise" if facing else "bet")
        children.append(build_tree(spot, seq + "r", tuple(raised)))
    return Node(seq, p, contrib, actions, children)


def deal_hands(spot: Spot) -> list[str]:
    """Every two-card hand left once the board is out, written like OpenSpiel's private cards
    (higher card first, cards ordered by rank, then suit in the order of `spot.suits`)."""
    index = {r + s: i * len(spot.suits) + j for i, r in enumerate(spot.ranks) for j, s in enumerate(spot.suits)}
    deck = sorted((c for c in index if c not in spot.board), key=index.get, reverse=True)
    return [a + b for a, b in itertools.combinations(deck, 2)]


class Solver:
    def __init__(self, spot: Spot):
        self.spot = spot
        self.hands = deal_hands(spot)
        self.index = {h: j for j, h in enumerate(self.hands)}
        n = len(self.hands)
        cards = [{h[:2], h[2:]} for h in self.hands]
        # compatible[i, j]: hands i and j share no card, so both can be dealt at once.
        self.compatible = np.array([[not (a & b) for b in cards] for a in cards], dtype=float)
        strength = [value(list(spot.board) + [h[:2], h[2:]]) for h in self.hands]
        rank = {v: k for k, v in enumerate(sorted(set(strength)))}
        r = np.array([rank[v] for v in strength])
        # showdown[i, j]: +1 if hand i beats hand j, -1 if it loses, 0 for a tie or a clash.
        self.showdown = np.sign(r[:, None] - r[None, :]) * self.compatible
        # Every hand leaves the same number of opponent hands, so each deal has this probability.
        self.deal_prob = 1 / (n * self.compatible[0].sum())
        self.root = build_tree(spot)
        self.nodes = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            self.nodes.append(node)
            stack += [c for c in reversed(node.children) if isinstance(c, Node)]
        self.by_seq = {node.seq: node for node in self.nodes}
        self.regret = {node.seq: np.zeros((len(node.actions), n)) for node in self.nodes}
        self.strategy_sum = {node.seq: np.zeros((len(node.actions), n)) for node in self.nodes}
        self.iterations = 0

    @staticmethod
    def _normalise(weights: np.ndarray) -> np.ndarray:
        """Each column (one hand) divided by its total; uniform where the total is zero."""
        total = weights.sum(axis=0)
        return np.where(total > 0, weights / np.where(total > 0, total, 1), 1 / len(weights))

    def current(self, node: Node) -> np.ndarray:
        """Regret matching+ strategy at a node: shape (actions, hands)."""
        return self._normalise(self.regret[node.seq])

    def average(self, node: Node) -> np.ndarray:
        """The average strategy, which is what converges to equilibrium."""
        return self._normalise(self.strategy_sum[node.seq])

    def _terminal(self, node: Terminal, player: int, opp_reach: np.ndarray) -> np.ndarray:
        """Counterfactual value of each of `player`'s hands (chance probability left out)."""
        if node.folder is None:
            return node.contrib[0] * (self.showdown @ opp_reach)
        sign = -1 if node.folder == player else 1
        return sign * node.contrib[node.folder] * (self.compatible @ opp_reach)

    def _cfr(self, node, player: int, own: np.ndarray, opp: np.ndarray) -> np.ndarray:
        if isinstance(node, Terminal):
            return self._terminal(node, player, opp)
        sigma = self.current(node)
        if node.player != player:
            return sum(self._cfr(child, player, own, opp * sigma[a]) for a, child in enumerate(node.children))
        values = np.array([self._cfr(child, player, own * sigma[a], opp) for a, child in enumerate(node.children)])
        v = (sigma * values).sum(axis=0)
        self.regret[node.seq] = np.maximum(self.regret[node.seq] + values - v, 0)
        self.strategy_sum[node.seq] += self.iterations * own * sigma
        return v

    def iterate(self, iterations: int) -> None:
        n = len(self.hands)
        for _ in range(iterations):
            self.iterations += 1
            for player in (0, 1):
                self._cfr(self.root, player, np.ones(n), np.ones(n))

    def _evaluate(self, node, player: int, opp: np.ndarray, best: bool) -> np.ndarray:
        """`player`'s counterfactual values when both play the average strategy, or when
        `player` best-responds to the other's average strategy (`best`)."""
        if isinstance(node, Terminal):
            return self._terminal(node, player, opp)
        sigma = self.average(node)
        if node.player != player:
            return sum(self._evaluate(c, player, opp * sigma[a], best) for a, c in enumerate(node.children))
        values = np.array([self._evaluate(c, player, opp, best) for c in node.children])
        return values.max(axis=0) if best else (sigma * values).sum(axis=0)

    def value(self, player: int) -> float:
        """Expected chips per hand for `player` when both play the average strategy."""
        return self.deal_prob * self._evaluate(self.root, player, np.ones(len(self.hands)), False).sum()

    def best_response(self, player: int) -> float:
        """Expected chips per hand for `player` best-responding to the other's average strategy."""
        return self.deal_prob * self._evaluate(self.root, player, np.ones(len(self.hands)), True).sum()

    def exploitability(self) -> float:
        """Mean gain of the two best responses: OpenSpiel's definition for two-player zero-sum games."""
        return (self.best_response(0) + self.best_response(1)) / 2

    def export(self) -> dict:
        """The average strategy per betting sequence, in scripts/solve_reference.py's format."""
        out = {}
        for node in sorted(self.nodes, key=lambda n: n.seq):
            avg = self.average(node)
            out[node.seq] = {
                "player": node.player,
                "contrib": list(node.contrib),
                "actions": node.actions,
                "hands": {h: {a: round(float(avg[k, j]), 4) for k, a in enumerate(node.actions)} for j, h in enumerate(self.hands)},
            }
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="+", choices=SPOTS)
    ap.add_argument("--iterations", type=int, help="default: the same count as the OpenSpiel reference solve")
    args = ap.parse_args()
    for key in args.games:
        spot = SPOTS[key]
        iterations = args.iterations or spot.iterations
        solver = Solver(spot)
        start = time.time()
        solver.iterate(iterations)
        seconds = time.time() - start
        expl = solver.exploitability()
        print(f"{key}: {iterations} iterations in {seconds:.1f} s, exploitability {expl:.3g} chips/hand, "
              f"P1 value {solver.value(0):+.6f}", file=sys.stderr)
        out = Path(__file__).resolve().parent.parent / "build" / f"{key}.engine.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"solver": "engine", "load": spot.openspiel, "iterations": iterations,
                                   "seconds": round(seconds, 1), "exploitability": expl, "nodes": solver.export()}))


if __name__ == "__main__":
    main()
