"""Vector-form CFR for heads-up poker from a fixed board (a flop, a turn or a river) over one or more
betting rounds, with a card dealt before each round after the first.

Our solver's engine. It solves the Mini and C1 games, in limit and no-limit versions, C1 on a river,
and Mini's ranks in three suits over two rounds, which OpenSpiel can also solve, so every result is
checked against an independent implementation (tests/test_cfr.py). It also solves a full-deck river
with ranges and several bet sizes, which scripts/check_river.py checks against noambrown/poker_solver.

How it works. The betting is walked as a public tree, one node per betting sequence and dealt card,
and each node carries a vector over every private hand instead of one node per deal (Johanson et
al. 2011). A dealt card is a chance node in that tree: every card is dealt in turn, the hands
holding it drop out below it, and the node's value sums its children's. Hole cards enter only at the
terminal nodes, where each hand's value sums over the opponent's hands that share no card with it.
That takes O(n) per node, not a product with a matrix over pairs of hands: a pass over the hands
sorted by strength on that board, sorted once for every showdown on it, which takes away the hands
that share a card by inclusion-exclusion (Johanson et al. 2011 again). Each player's starting range
weights the hands it is dealt: it is the reach that player's hands start the tree with. Regrets
follow CFR (Zinkevich et al. 2007) with alternating updates, in one of two variants:

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
    uv run python engine/cfr.py river --engine c++
"""

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.cards import RANKS, value
from engine.ranges import weights


@dataclass(frozen=True)
class Spot:
    """Betting on a fixed board of three cards (a flop), four (a turn) or five (a river), over
    `streets` betting rounds: each round after the first deals one more card to the board. Hands are
    the best five of the final board and two hole cards. Both players post `blind`, and player 0 acts
    first in every round.

    Limit (`stack` unset): every bet or raise is `bet` chips, and at most `max_raises` are made in
    each round. No-limit (`stack` set): each player starts with `stack` chips and may bet by each of
    `pot_sizes` (fractions of the pot after calling, rounded to whole chips with halves rounded up)
    and raise by each of `raise_sizes` (unset: the same as `pot_sizes`) while that stays below the
    stack, or go all-in. On a later street, `blind` is half the pot and `stack` adds the chips behind.

    `ranges` holds each player's starting range as range text (engine/ranges.py), or is unset for
    every hand at weight 1.
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
    raise_sizes: tuple[float, ...] | None = None
    ranges: tuple[str, str] | None = None
    streets: int = 1

    @property
    def nolimit(self) -> bool:
        return self.stack is not None

    @property
    def openspiel(self) -> str:
        """The same game as an OpenSpiel `universal_poker` string, for cross-checks."""
        if self.ranges:
            raise ValueError("OpenSpiel's universal_poker deals every hand alike, so it has no ranges")

        # One entry per round, and at least two: universal_poker reads a lone entry as a number and refuses it.
        def per_round(entry) -> str:
            return " ".join([str(entry)] * max(self.streets, 2))

        later = " ".join(["1"] * (self.streets - 1)) or "0"  # one card for each round after the first
        cards = (f"numSuits={len(self.suits)},numRanks={len(self.ranks)},numHoleCards=2,"
                 f"numBoardCards={len(self.board)} {later},boardCards={''.join(self.board)}")
        common = f"numPlayers=2,numRounds={self.streets},blind={self.blind} {self.blind},"
        first = per_round(1)
        if not self.nolimit:
            return (f"universal_poker(betting=limit,{common}raiseSize={per_round(self.bet)},firstPlayer={first},"
                    f"maxRaises={per_round(self.max_raises)},{cards})")
        if self.pot_sizes != (1.0,) or self.raise_sizes not in (None, (1.0,)):
            raise ValueError("OpenSpiel's no-limit games offer only pot-size bets and all-in")
        return (f"universal_poker(betting=nolimit,{common}firstPlayer={first},{cards},"
                f"stack={self.stack} {self.stack},bettingAbstraction=fcpa)")


MINI = dict(ranks="2345", suits="cd", board=("3c", "4d", "5c"))
C1 = dict(ranks="234567", suits="cdhs", board=("5c", "6d", "7h"))
# Mini's ranks in three suits over two rounds: a fixed flop, then a turn card dealt. OpenSpiel stores
# every deal and betting line, so this is about as big as a two-round game gets for checking against
# it: about 2 s an iteration of its CFR+, against 90 s for a 20-card deck.
MINI_TURN = dict(ranks="2345", suits="cdh", board=("3c", "4d", "5h"), streets=2)
SPOTS = {
    "mini": Spot(**MINI, iterations=5000),
    "c1": Spot(**C1, iterations=1000),
    "mini_nl": Spot(**MINI, iterations=5000, stack=20),
    "c1_nl": Spot(**C1, iterations=1000, stack=20),
    "mini_turn": Spot(**MINI_TURN, iterations=1000),
    "mini_turn_nl": Spot(**MINI_TURN, iterations=1000, stack=20),
    # C1's deck on a river where hands run from two pair through straights and flushes to four of a kind.
    "c1_river": Spot(ranks="234567", suits="cdhs", board=("2c", "4c", "5d", "7c", "7h"), iterations=1000),
    # A full-deck river, to change at will. The big blind (player 0) against the button in a
    # single-raised pot: the button bet the flop and the turn, and the big blind called both. 20 big
    # blinds in the pot, 80 behind each, bets and raises of a third, three quarters and one and a half
    # pots, and all-in. No flush is possible on this board, so the ranges ignore suits. They are
    # written by hand to fit that line, for illustration: no solve of the flop and turn made them.
    "river": Spot(ranks=RANKS, suits="cdhs", board=("Ks", "8d", "4c", "2h", "7s"), iterations=1000, blind=10,
                  stack=90, pot_sizes=(0.33, 0.75, 1.5), ranges=(
                      "KJ, KT, K9, KQo, KQs:0.5, K7s, K6s, K5s, K3s, K8s:0.5, K4s:0.5, K2s:0.5, K8o:0.5, K7o:0.5, "
                      "A8s:0.5, A8o:0.5, 98s:0.5, 87s, T8s:0.5, J8s:0.25, Q8s:0.25, 88:0.5, 44:0.5, 22:0.5, "
                      "77:0.3, 99:0.5, TT:0.5, JJ:0.5, 65s, 65o:0.5, 53s, 53o:0.25, 63s:0.5, A5s, A3s, 76s:0.5, "
                      "75s:0.5",
                      "AA, KK, QQ:0.5, JJ:0.25, 88, 44, 22, 77:0.25, AK, KQ, KJ, KT, K9s, K8s, K7s, K4s, K2s, "
                      "K6s:0.5, K5s:0.5, 65s, 53s, A5s, A3s, 63s:0.5, 76s:0.5, QJs:0.5, QTs:0.5, JTs:0.5, "
                      "T9s:0.5, 96s:0.25, AQ:0.25, AJ:0.25")),
    # The river spot's turn, K♠8♦4♣2♥, to change at will: the button bet a third of the pot on the
    # flop and the big blind called. 8 big blinds in the pot and 86 behind each (the river spot's 90),
    # so a check, a bet of three quarters and a call make the river spot's pot of 20 with 80 behind.
    # On both streets, bets of a third, three quarters and one and a half pots, raises of three
    # quarters, and all-in. No flush can come by the river, so the ranges ignore suits. They are
    # written by hand to fit the line so far, for illustration: no solve of the preflop and flop made them.
    "turn": Spot(ranks=RANKS, suits="cdhs", board=("Ks", "8d", "4c", "2h"), iterations=300, blind=4, stack=90,
                 pot_sizes=(0.33, 0.75, 1.5), raise_sizes=(0.75,), streets=2, ranges=(
                     "K2s+, K5o+, KQs:0.5, K8s:0.5, K8o:0.5, K4s:0.5, AKo:0.25, A8s, A8o, 98s, 98o:0.5, 87s, "
                     "87o:0.5, T8s, J8s, Q8s, 86s, 85s, T8o:0.25, A4s, A4o:0.5, 54s, 64s, 74s:0.5, QQ:0.5, JJ, "
                     "TT, 99, 88:0.5, 77, 66:0.5, 55:0.5, 44:0.5, 33:0.5, 22:0.5, QJs, QTs, JTs, T9s, 76s, 65s, "
                     "75s, 53s, 63s, A5s, A3s, A2s, QJo:0.5, JTo:0.25, AJs:0.5, ATs:0.5",
                     "AA, KK, QQ, JJ, TT, 99, 88, 44, 77:0.5, 66:0.5, 55:0.5, 33:0.5, 22:0.5, AK, AQ, AJ, AT, "
                     "A9s, A8, A7s, A6s, A5s, A4s, A3s, A2s, K2s+, K7o+, QJ, QT, JT, Q9s:0.5, J9s:0.5, T9s, "
                     "98s, 87s, 76s, 65s, 54s, 86s, T8s, J8s:0.5, Q8s:0.5, 64s:0.5, 75s:0.5, 53s:0.5, "
                     "63s:0.25")),
}


# The betting so far (`seq`) is written as OpenSpiel writes it, f, c, r or r<chips raised to>, with
# each round after the first behind the card dealt for it: "rc|3h|r" is a bet and a call, the 3h,
# and a bet. `dealt` holds the cards dealt so far after the fixed board.


@dataclass
class Terminal:
    folder: int | None  # the player who folded, or None for a showdown
    contrib: tuple[int, int]
    dealt: tuple[str, ...] = ()


@dataclass
class Chance:
    """The end of a round's betting before the last: the next card is dealt, one child per card."""
    seq: str
    contrib: tuple[int, int]
    dealt: tuple[str, ...]
    cards: list[str]  # the cards it can deal, in deck order
    children: list  # Node, Chance or Terminal, one per card


@dataclass
class Node:
    seq: str
    player: int
    contrib: tuple[int, int]
    actions: list[str]  # fold, check, call, bet, raise, allin
    children: list  # Node, Chance or Terminal, one per action
    dealt: tuple[str, ...] = ()


def build_tree(spot: Spot, seq: str = "", player: int = 0, contrib: tuple[int, int] | None = None,
               dealt: tuple[str, ...] = ()) -> Node:
    contrib = contrib or (spot.blind, spot.blind)
    opp = 1 - player
    facing = contrib[opp] > contrib[player]
    actions, children = [], []

    def raise_to(to: int, kind: str) -> None:
        new = list(contrib)
        new[player] = to
        actions.append(kind)
        children.append(build_tree(spot, seq + (f"r{to}" if spot.nolimit else "r"), opp, tuple(new), dealt))

    if facing:
        actions += ["fold", "call"]
        children += [Terminal(player, contrib, dealt), end_of_round(spot, seq + "c", (contrib[opp],) * 2, dealt)]
    else:
        actions.append("check")
        children.append(end_of_round(spot, seq + "c", contrib, dealt) if seq.endswith("c")
                        else build_tree(spot, seq + "c", opp, contrib, dealt))
    kind = "raise" if facing else "bet"
    if not spot.nolimit:
        if seq.split("|")[-1].count("r") < spot.max_raises:
            raise_to(contrib[opp] + spot.bet, kind)
    else:
        for size in spot.raise_sizes if facing and spot.raise_sizes is not None else spot.pot_sizes:
            # Halves round up, as noambrown/poker_solver rounds them, so both build the same tree.
            to = contrib[opp] + math.floor(size * 2 * contrib[opp] + 0.5)
            if to < spot.stack:
                raise_to(to, kind)
        if contrib[opp] < spot.stack:
            raise_to(spot.stack, "allin")
    return Node(seq, player, contrib, actions, children, dealt)


def end_of_round(spot: Spot, seq: str, contrib: tuple[int, int], dealt: tuple[str, ...]):
    """Where a round's betting ends without a fold: the showdown after the last round, or else the
    next card, then the next round's betting, or straight to the next card if both are all-in."""
    if len(dealt) + 1 == spot.streets:
        return Terminal(None, contrib, dealt)
    cards = [c for c in cards_left(spot) if c not in dealt]
    allin = spot.nolimit and contrib[0] == spot.stack
    return Chance(seq, contrib, dealt, cards, [
        end_of_round(spot, f"{seq}|{c}|", contrib, dealt + (c,)) if allin
        else build_tree(spot, f"{seq}|{c}|", 0, contrib, dealt + (c,)) for c in cards])


def cards_left(spot: Spot) -> list[str]:
    """The cards not on the board, ordered by rank, then suit in the order of `spot.suits`, highest
    first."""
    index = {r + s: i * len(spot.suits) + j for i, r in enumerate(spot.ranks) for j, s in enumerate(spot.suits)}
    return sorted((c for c in index if c not in spot.board), key=index.get, reverse=True)


def deal_hands(spot: Spot) -> list[str]:
    """Every two-card hand left once the board is out, written like OpenSpiel's private cards
    (higher card first, in deck order)."""
    return [a + b for a, b in itertools.combinations(cards_left(spot), 2)]


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
        # live[c]: 1 for each hand that doesn't hold card c, the hands left once c is dealt.
        self.live = {c: np.array([float(c not in h) for h in cards]) for c in cards_left(spot)}
        # Each hand's two cards, numbered from 0, for the sums at terminal nodes.
        deck = {c: k for k, c in enumerate(sorted({c for h in self.hands for c in (h[:2], h[2:])}))}
        self.cards = np.array([[deck[h[:2]], deck[h[2:]]] for h in self.hands])
        self.deck_size = len(deck)
        # ranges[p, i]: player p's weight on hand i, the reach its hands start the tree with.
        self.ranges = np.array([weights(r, self.hands) for r in spot.ranges]) if spot.ranges else np.ones((2, n))
        # Deal (i, j), hand i to player 0 and j to player 1, has probability ranges[0, i] * ranges[1, j] * deal_prob
        # when the two share no card.
        self.deal_prob = 1 / (self.ranges[0] @ self.compatible @ self.ranges[1])
        self.root = build_tree(spot)
        self.nodes, final = [], set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            if isinstance(node, Node):
                self.nodes.append(node)
            final |= {c.dealt for c in node.children if isinstance(c, Terminal) and c.folder is None}
            stack += [c for c in reversed(node.children) if not isinstance(c, Terminal)]
        self.by_seq = {node.seq: node for node in self.nodes}
        # Per final board (the cards dealt after the fixed board, as at a showdown), what the showdowns
        # there use instead of a matrix over pairs of hands: the hands that hold none of the dealt cards
        # from weakest to strongest, where hands as strong as the one at place k take places
        # group_start[k] up to group_end[k]. Sorted once and used by every showdown on that board
        # (Johanson et al. 2011).
        self.boards = {}
        for dealt in sorted(final):
            live, r = self._ranking(dealt)
            k = np.argsort(r, kind="stable")
            ranked = r[k]
            self.boards[dealt] = (live[k], np.searchsorted(ranked, ranked, side="left"),
                                  np.searchsorted(ranked, ranked, side="right"))
        # showdown[i, j], on one street: +1 if hand i beats hand j, -1 if it loses, 0 for a tie or a clash.
        self.showdown = self.showdown_matrix() if spot.streets == 1 else None
        self.regret = {node.seq: np.zeros((len(node.actions), n)) for node in self.nodes}
        self.strategy_sum = {node.seq: np.zeros((len(node.actions), n)) for node in self.nodes}
        self.iterations = 0

    def _ranking(self, dealt: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        """The hands that hold none of `dealt`, and the rank of each one's strength on the board with
        `dealt` added (0 for the weakest, equal for a tie)."""
        board = list(self.spot.board) + list(dealt)
        live = [i for i, h in enumerate(self.hands) if h[:2] not in dealt and h[2:] not in dealt]
        strength = [value(board + [self.hands[i][:2], self.hands[i][2:]]) for i in live]
        rank = {v: k for k, v in enumerate(sorted(set(strength)))}
        return np.array(live, dtype=int), np.array([rank[v] for v in strength], dtype=int)

    def showdown_matrix(self, dealt: tuple[str, ...] = ()) -> np.ndarray:
        """showdown[i, j] on the board with `dealt` added: +1 if hand i beats hand j, -1 if it loses,
        0 for a tie, a clash, or a hand holding a dealt card. For tests: the engine sums in O(n)."""
        live, r = self._ranking(dealt)
        out = np.zeros_like(self.compatible)
        out[np.ix_(live, live)] = np.sign(r[:, None] - r[None, :]) * self.compatible[np.ix_(live, live)]
        return out

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

    def _compatible_sums(self, x: np.ndarray) -> np.ndarray:
        """compatible @ x in O(n): x over every hand, less the hands holding either of each hand's
        cards. The hand itself holds both, so it was taken away twice: add it back once. Sums go left
        to right, as the C++ engine's do."""
        total = np.cumsum(x)[-1]
        per_card = np.bincount(self.cards.ravel(), np.repeat(x, 2), minlength=self.deck_size)
        return total - per_card[self.cards[:, 0]] - per_card[self.cards[:, 1]] + x

    def _weaker_stronger_sums(self, x: np.ndarray, order: np.ndarray, start: np.ndarray,
                              end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each place in `order` (a board's hands sorted by strength, from `boards`), x (given in
        that order) over the weaker hands and over the stronger ones, each less the hands holding one of
        that hand's cards (none holds both). Sums go from the weakest hand up and from the strongest
        down, as the C++ engine's do."""
        n, cards = len(x), self.cards[order]
        held = np.zeros((n, self.deck_size))  # held[k, c]: x[k] if the hand at place k holds card c
        held[np.arange(n), cards[:, 0]] = x
        held[np.arange(n), cards[:, 1]] = x
        none = np.zeros((1, self.deck_size))
        below = np.vstack([none, np.cumsum(held, axis=0)])  # row k: each card's x in the hands below place k
        above = np.vstack([np.cumsum(held[::-1], axis=0)[::-1], none])  # row k: in the hands at k and above
        total_below = np.concatenate([[0.0], np.cumsum(x)])
        total_above = np.concatenate([np.cumsum(x[::-1])[::-1], [0.0]])
        a, b = cards[:, 0], cards[:, 1]
        return (total_below[start] - below[start, a] - below[start, b],
                total_above[end] - above[end, a] - above[end, b])

    # Taking away can leave 1e-16 where the true sum is 0, and regret matching turns a regret of 1e-16
    # into a pure strategy. So the two sums below count the hands with reach above 0 the same way (in
    # whole numbers, exactly), and are exactly 0 where there are none.

    def _compatible_reach(self, reach: np.ndarray) -> np.ndarray:
        """compatible @ reach in O(n) (Johanson et al. 2011): the opponent's reach behind each hand."""
        return np.where(self._compatible_sums(1.0 * (reach > 0)) > 0, self._compatible_sums(reach), 0.0)

    def _showdown_reach(self, reach: np.ndarray, dealt: tuple[str, ...] = ()) -> np.ndarray:
        """showdown_matrix(dealt) @ reach in O(n) (Johanson et al. 2011), from the board's hands sorted
        by strength: for each hand, the opponent's reach with weaker hands less its reach with stronger
        ones; 0 for a hand holding a dealt card."""
        order, start, end = self.boards[dealt]
        s = reach[order]
        weaker, stronger = self._weaker_stronger_sums(s, order, start, end)
        some_weaker, some_stronger = self._weaker_stronger_sums(1.0 * (s > 0), order, start, end)
        out = np.zeros(len(reach))
        out[order] = np.where(some_weaker > 0, weaker, 0.0) - np.where(some_stronger > 0, stronger, 0.0)
        return out

    def _terminal(self, node: Terminal, player: int, opp_reach: np.ndarray) -> np.ndarray:
        """Counterfactual value of each of `player`'s hands (chance probability left out)."""
        if node.folder is None:
            return node.contrib[0] * self._showdown_reach(opp_reach, node.dealt)
        sign = -1 if node.folder == player else 1
        return sign * node.contrib[node.folder] * self._compatible_reach(opp_reach)

    def _deal(self, node: Chance, walk) -> np.ndarray:
        """A chance node in the public tree (Johanson et al. 2011): the value of each hand, summed over
        the cards dealt, from `walk(child, live)` for each card's child, where `live` marks the hands
        that don't hold the card. Below a card, the hands holding it have no reach (`walk` multiplies
        both players' reach by `live`) and get no value from it. Any two hands leave the same number of
        cards to deal, each as likely, so each card counts 1 / (cards left - 4)."""
        total = 0
        for card, child in zip(node.cards, node.children):
            live = self.live[card]
            total = total + live * walk(child, live)
        return total / (len(node.cards) - 4)

    def _cfr(self, node, player: int, own: np.ndarray, opp: np.ndarray) -> np.ndarray:
        if isinstance(node, Terminal):
            return self._terminal(node, player, opp)
        if isinstance(node, Chance):
            return self._deal(node, lambda child, live: self._cfr(child, player, own * live, opp * live))
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
        for _ in range(iterations):
            self.iterations += 1
            for player in (0, 1):
                self._cfr(self.root, player, self.ranges[player], self.ranges[1 - player])

    def _evaluate(self, node, player: int, opp: np.ndarray, best: bool, record: dict | None = None) -> np.ndarray:
        """`player`'s counterfactual values when both play the average strategy, or when
        `player` best-responds to the other's average strategy (`best`). With `record`, also keeps
        each of `player`'s nodes' action values and the opponent's reach behind each hand there."""
        if isinstance(node, Terminal):
            return self._terminal(node, player, opp)
        if isinstance(node, Chance):
            return self._deal(node, lambda child, live: self._evaluate(child, player, opp * live, best, record))
        sigma = self.average(node)
        if node.player != player:
            return sum(self._evaluate(c, player, opp * sigma[a], best, record) for a, c in enumerate(node.children))
        values = np.array([self._evaluate(c, player, opp, best, record) for c in node.children])
        if record is not None:
            record[node.seq] = values, self._compatible_reach(opp)
        return values.max(axis=0) if best else (sigma * values).sum(axis=0)

    def _expected(self, player: int, best: bool) -> float:
        values = self._evaluate(self.root, player, self.ranges[1 - player], best)
        return self.deal_prob * (self.ranges[player] * values).sum()

    def value(self, player: int) -> float:
        """Expected chips per hand for `player` when both play the average strategy."""
        return self._expected(player, False)

    def best_response(self, player: int) -> float:
        """Expected chips per hand for `player` best-responding to the other's average strategy."""
        return self._expected(player, True)

    def exploitability(self) -> float:
        """Mean gain of the two best responses: OpenSpiel's definition for two-player zero-sum games."""
        return (self.best_response(0) + self.best_response(1)) / 2

    def evs(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per decision node: the acting player's EV for each action and hand (actions x hands), and
        the chance that the opponent reaches the node given each hand (from its starting range).

        EV is expected chips given the hand, the cards dealt and the node, with both playing the
        average strategy. The blinds count as dead money: EV is the chips won from the pot minus the
        chips put in after the blinds, so folding without having bet is worth 0. It is NaN where no
        opponent hand that the hand leaves possible reaches the node, and for a hand holding a card
        dealt by then. The opponent's reach weights hands when their EVs are averaged (card removal)."""
        record = {}
        for player in (0, 1):
            self._evaluate(self.root, player, self.ranges[1 - player], False, record)
        out, start = {}, {}
        for seq, (values, mass) in record.items():
            node = self.by_seq[seq]
            live = np.prod([self.live[c] for c in node.dealt], axis=0) if node.dealt else 1.0
            key = node.player, node.dealt
            if key not in start:  # the opponent's starting range, given each of the player's hands and the cards dealt
                start[key] = live * self._compatible_reach(self.ranges[1 - node.player] * live)
            mass = live * mass
            out[seq] = (values / np.where(mass > 0, mass, np.nan) + self.spot.blind,
                        np.divide(mass, start[key], out=np.zeros_like(mass), where=start[key] > 0))
        return out

    def export(self) -> dict:
        """The average strategy as the viewer's tree: per node its actions (with labels and where
        each leads) and, per action, the probability for every hand in `hands`; with each hand's
        EV, its EV for each action and the opponent's reach behind it (see `evs`). EVs are null
        where that reach rounds to 0: the opponent (almost) never gets there, so they would rest on
        an unconverged sliver of range. Amounts in labels leave out the blinds, which act like
        antes: "bet 2" puts 2 more chips in. One street only, as the viewer shows."""
        if self.spot.streets > 1:
            raise NotImplementedError("the viewer's tree has one street")

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
        return {"hands": self.hands, "ranges": self.ranges.tolist(), "root": "root", "nodes": nodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="+", choices=SPOTS)
    ap.add_argument("--method", choices=Solver.METHODS, default="dcfr",
                    help="default dcfr: it converges faster here (scripts/bench_engine.py)")
    ap.add_argument("--iterations", type=int, help="default: the game's own count (as in the OpenSpiel reference solves)")
    ap.add_argument("--engine", choices=("python", "c++"), default="python",
                    help="c++: iterate with the C++ engine (engine/cpp), then take its average strategy here")
    args = ap.parse_args()
    for key in args.games:
        spot = SPOTS[key]
        iterations = args.iterations or spot.iterations
        solver = Solver(spot, args.method)
        if args.engine == "c++":
            from engine.native import solver as cpp_solver  # imported here: engine.native imports this module
            cpp = cpp_solver(spot, args.method)
        start = time.time()
        (cpp if args.engine == "c++" else solver).iterate(iterations)
        seconds = time.time() - start
        if args.engine == "c++":  # its average strategy, for the EVs and the export
            for i, node in enumerate(cpp.nodes):
                solver.strategy_sum[node.seq] = cpp.average(i)
        expl = solver.exploitability()
        print(f"{key}: {args.method}, {iterations} iterations in {seconds:.1f} s ({args.engine}), exploitability "
              f"{expl:.3g} chips/hand ({100 * expl / (2 * spot.blind):.3f}% of the pot), P1 value {solver.value(0):+.6f}",
              file=sys.stderr)
        try:
            load = spot.openspiel
        except ValueError:
            load = None
        out = Path(__file__).resolve().parent.parent / "build" / f"{key}.engine.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"method": args.method, "engine": args.engine, "spot": asdict(spot),
                                   "load": load, "nolimit": spot.nolimit, "stack": spot.stack, "iterations": iterations,
                                   "seconds": round(seconds, 1), "exploitability": expl,
                                   "tree": solver.export() if spot.streets == 1 else None}))


if __name__ == "__main__":
    main()
