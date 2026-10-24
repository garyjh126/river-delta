"""Vector-form CFR for heads-up poker with one betting round on a fixed flop.

Our solver's engine. It solves the Mini and C1 games, in limit and no-limit versions, which
OpenSpiel can also solve, so every result is checked against an independent implementation
(tests/test_cfr.py).

How it works. The betting is walked as a public tree, one node per betting sequence, and each
node carries a vector over every private hand instead of one node per deal (Johanson et al.
2011). Cards enter only at the terminal nodes, through matrices over pairs of hands that zero out
pairs sharing a card. Regrets follow CFR (Zinkevich et al. 2007) with alternating updates, in
one of two variants:

  cfr+   regret matching+ and an average weighted by iteration t (Tammelin 2014; Tammelin et
         al. 2015)
  dcfr   discounted CFR (Brown and Sandholm 2019): after each update, positive regrets are
         multiplied by t^1.5 / (t^1.5 + 1) and negative ones by 1/2, and the average is
         weighted by t^2

Exploitability comes from an exact best response over the same public tree.

References
  Zinkevich, Johanson, Bowling, Piccione (2007). Regret Minimization in Games with Incomplete
    Information. NIPS 2007.
  Johanson, Waugh, Bowling, Zinkevich (2011). Accelerating Best Response Calculation in Large
    Extensive Games. IJCAI 2011.
  Tammelin (2014). Solving Large Imperfect Information Games Using CFR+. arXiv:1407.5042.
  Tammelin, Burch, Johanson, Bowling (2015). Solving Heads-up Limit Texas Hold'em. IJCAI 2015.
  Brown, Sandholm (2019). Solving Imperfect-Information Games via Discounted Regret
    Minimization. AAAI 2019. arXiv:1809.04040.

    uv run python engine/cfr.py mini c1 c1_nl     # writes build/<game>.engine.json
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
    """One betting round on a fixed flop. Both players post `blind` and player 0 acts first.

    Limit (`stack` unset): every bet or raise is `bet` chips, and at most `max_raises` are made.
    No-limit (`stack` set): each player starts with `stack` chips and may bet or raise by each of
    `pot_sizes` (fractions of the pot after calling) while that stays below the stack, or go all-in.
    """

    ranks: str
    suits: str
    board: tuple[str, ...]
    iterations: int
    blind: int = 1
    bet: int = 2
    max_raises: int = 2
    stack: int | None = None
    pot_sizes: tuple[float, ...] = (1.0,)

    @property
    def nolimit(self) -> bool:
        return self.stack is not None

    @property
    def openspiel(self) -> str:
        """The same game as an OpenSpiel `universal_poker` string, for cross-checks."""
        cards = (f"numSuits={len(self.suits)},numRanks={len(self.ranks)},numHoleCards=2,"
                 f"numBoardCards=3 0,boardCards={''.join(self.board)}")
        common = f"numPlayers=2,numRounds=1,blind={self.blind} {self.blind},"
        if not self.nolimit:
            return (f"universal_poker(betting=limit,{common}raiseSize={self.bet} {self.bet},firstPlayer=1 1,"
                    f"maxRaises={self.max_raises} {self.max_raises},{cards})")
        if self.pot_sizes != (1.0,):
            raise ValueError("OpenSpiel's no-limit games offer only pot-size bets and all-in")
        return (f"universal_poker(betting=nolimit,{common}firstPlayer=1 1,{cards},"
                f"stack={self.stack} {self.stack},bettingAbstraction=fcpa)")


MINI = dict(ranks="2345", suits="cd", board=("3c", "4d", "5c"))
C1 = dict(ranks="234567", suits="cdhs", board=("5c", "6d", "7h"))
SPOTS = {
    "mini": Spot(**MINI, iterations=5000),
    "c1": Spot(**C1, iterations=1000),
    "mini_nl": Spot(**MINI, iterations=5000, stack=20),
    "c1_nl": Spot(**C1, iterations=1000, stack=20),
}


@dataclass
class Terminal:
    folder: int | None  # the player who folded, or None for a showdown
    contrib: tuple[int, int]


@dataclass
class Node:
    seq: str  # the betting so far, written as OpenSpiel writes it: f, c, r or r<chips raised to>
    player: int
    contrib: tuple[int, int]
    actions: list[str]  # fold, check, call, bet, raise, allin
    children: list  # Node or Terminal, one per action


def build_tree(spot: Spot, seq: str = "", player: int = 0, contrib: tuple[int, int] | None = None) -> Node:
    contrib = contrib or (spot.blind, spot.blind)
    opp = 1 - player
    facing = contrib[opp] > contrib[player]
    actions, children = [], []

    def raise_to(to: int, kind: str) -> None:
        new = list(contrib)
        new[player] = to
        actions.append(kind)
        children.append(build_tree(spot, seq + (f"r{to}" if spot.nolimit else "r"), opp, tuple(new)))

    if facing:
        actions += ["fold", "call"]
        children += [Terminal(player, contrib), Terminal(None, (contrib[opp], contrib[opp]))]
    else:
        actions.append("check")
        children.append(Terminal(None, contrib) if seq.endswith("c") else build_tree(spot, seq + "c", opp, contrib))
    kind = "raise" if facing else "bet"
    if not spot.nolimit:
        if seq.count("r") < spot.max_raises:
            raise_to(contrib[opp] + spot.bet, kind)
    else:
        for size in spot.pot_sizes:
            to = contrib[opp] + round(size * 2 * contrib[opp])
            if to < spot.stack:
                raise_to(to, kind)
        if contrib[opp] < spot.stack:
            raise_to(spot.stack, "allin")
    return Node(seq, player, contrib, actions, children)


def deal_hands(spot: Spot) -> list[str]:
    """Every two-card hand left once the board is out, written like OpenSpiel's private cards
    (higher card first, cards ordered by rank, then suit in the order of `spot.suits`)."""
    index = {r + s: i * len(spot.suits) + j for i, r in enumerate(spot.ranks) for j, s in enumerate(spot.suits)}
    deck = sorted((c for c in index if c not in spot.board), key=index.get, reverse=True)
    return [a + b for a, b in itertools.combinations(deck, 2)]


class Solver:
    METHODS = ("cfr+", "dcfr")

    def __init__(self, spot: Spot, method: str = "cfr+"):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.spot, self.method = spot, method
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
        """Regret matching: shape (actions, hands). CFR+ keeps regrets at zero or above already."""
        return self._normalise(np.maximum(self.regret[node.seq], 0))

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
        regret = self.regret[node.seq] + values - v
        t = self.iterations
        if self.method == "cfr+":
            self.regret[node.seq] = np.maximum(regret, 0)
            self.strategy_sum[node.seq] += t * own * sigma
        else:
            self.regret[node.seq] = regret * np.where(regret >= 0, t**1.5 / (t**1.5 + 1), 0.5)
            self.strategy_sum[node.seq] += t**2 * own * sigma
        return v

    def iterate(self, iterations: int) -> None:
        n = len(self.hands)
        for _ in range(iterations):
            self.iterations += 1
            for player in (0, 1):
                self._cfr(self.root, player, np.ones(n), np.ones(n))

    def _evaluate(self, node, player: int, opp: np.ndarray, best: bool, record: dict | None = None) -> np.ndarray:
        """`player`'s counterfactual values when both play the average strategy, or when
        `player` best-responds to the other's average strategy (`best`). With `record`, also keeps
        each of `player`'s nodes' action values and the opponent's reach behind each hand there."""
        if isinstance(node, Terminal):
            return self._terminal(node, player, opp)
        sigma = self.average(node)
        if node.player != player:
            return sum(self._evaluate(c, player, opp * sigma[a], best, record) for a, c in enumerate(node.children))
        values = np.array([self._evaluate(c, player, opp, best, record) for c in node.children])
        if record is not None:
            record[node.seq] = values, self.compatible @ opp
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

    def evs(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per decision node: the acting player's EV for each action and hand (actions x hands), and
        the chance that the opponent's range reaches the node given each hand.

        EV is expected chips given the hand and the node, with both playing the average strategy.
        The blinds count as dead money: EV is the chips won from the pot minus the chips put in
        after the blinds, so folding without having bet is worth 0. It is NaN where no opponent
        hand that the hand leaves possible reaches the node. The opponent's reach weights hands
        when their EVs are averaged (card removal)."""
        record = {}
        for player in (0, 1):
            self._evaluate(self.root, player, np.ones(len(self.hands)), False, record)
        dealt = self.compatible[0].sum()  # opponent hands left by any one hand
        return {seq: (values / np.where(mass > 0, mass, np.nan) + self.spot.blind, mass / dealt)
                for seq, (values, mass) in record.items()}

    def export(self) -> dict:
        """The average strategy as the viewer's tree: per node its actions (with labels and where
        each leads) and, per action, the probability for every hand in `hands`; with each hand's
        EV, its EV for each action and the opponent's reach behind it (see `evs`). EVs are null
        where that reach rounds to 0: the opponent (almost) never gets there, so they would rest on
        an unconverged sliver of range. Amounts in labels leave out the blinds, which act like
        antes: "bet 2" puts 2 more chips in."""
        def nid(node: Node) -> str:
            return node.seq or "root"

        def chips(row) -> list:
            return [None if np.isnan(x) else round(float(x), 4) for x in row]

        evs = self.evs()
        nodes = {}
        for node in sorted(self.nodes, key=lambda n: (len(n.seq), n.seq)):
            p, c = node.player, node.contrib
            pot = c[0] + c[1]
            actions = []
            for kind, child in zip(node.actions, node.children):
                if kind == "call":
                    label, size = f"call {c[1 - p] - c[p]}", 0
                elif kind in ("bet", "raise", "allin"):
                    size = child.contrib[p]
                    label = f"{'all-in' if kind == 'allin' else kind} {size - self.spot.blind}"
                else:
                    label, size = kind, 0
                if isinstance(child, Node):
                    actions.append({"kind": kind, "label": label, "size": size, "next": nid(child), "end": None})
                else:
                    end = (f"P{p + 1} folds; P{2 - p} takes the pot of {pot}" if kind == "fold"
                           else f"Showdown for a pot of {2 * child.contrib[0]}")
                    actions.append({"kind": kind, "label": label, "size": size, "next": None, "end": end})
            avg = self.average(node)
            action_ev, opp_reach = evs[node.seq]
            opp_reach = opp_reach.round(4)
            action_ev = np.where(opp_reach > 0, action_ev, np.nan)
            nodes[nid(node)] = {
                "player": p,
                "pot": pot,
                "toCall": c[1 - p] - c[p],
                "badge": [f"in {c[0]}", f"in {c[1]}"],
                "actions": actions,
                "probs": [[round(float(x), 4) for x in row] for row in avg],
                "ev": chips((avg * action_ev).sum(axis=0)),
                "actionEv": [chips(row) for row in action_ev],
                "oppReach": [float(x) for x in opp_reach],
            }
        return {"hands": self.hands, "root": "root", "nodes": nodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="+", choices=SPOTS)
    ap.add_argument("--method", choices=Solver.METHODS, default="dcfr",
                    help="default dcfr: it converges faster here (scripts/bench_engine.py)")
    ap.add_argument("--iterations", type=int, help="default: the game's own count (as in the OpenSpiel reference solves)")
    args = ap.parse_args()
    for key in args.games:
        spot = SPOTS[key]
        iterations = args.iterations or spot.iterations
        solver = Solver(spot, args.method)
        start = time.time()
        solver.iterate(iterations)
        seconds = time.time() - start
        expl = solver.exploitability()
        print(f"{key}: {args.method}, {iterations} iterations in {seconds:.1f} s, exploitability {expl:.3g} chips/hand, "
              f"P1 value {solver.value(0):+.6f}", file=sys.stderr)
        out = Path(__file__).resolve().parent.parent / "build" / f"{key}.engine.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"method": args.method, "load": spot.openspiel, "nolimit": spot.nolimit,
                                   "stack": spot.stack, "iterations": iterations, "seconds": round(seconds, 1),
                                   "exploitability": expl, "tree": solver.export()}))


if __name__ == "__main__":
    main()
