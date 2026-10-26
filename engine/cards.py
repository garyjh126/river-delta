"""Cards and hands for the viewer's games: two hole cards plus a three-card flop or a five-card river."""

import itertools
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "shdc"
_ONE = dict(zip(RANKS, "two three four five six seven eight nine ten jack queen king ace".split()))
_MANY = dict(zip(RANKS, "twos threes fours fives sixes sevens eights nines tens jacks queens kings aces".split()))


def _rank(card: str) -> int:
    return RANKS.index(card[0]) + 2


def _straight_high(ranks: set[int]) -> int:
    """High card of the best straight these ranks contain (an ace plays high or low), else 0."""
    rs = ranks | {1} if 14 in ranks else ranks
    for hi in range(14, 4, -1):
        if all(r in rs for r in range(hi - 4, hi + 1)):
            return hi
    return 0


def value(cards: list[str]) -> tuple:
    """Comparable value of the best five of five to seven cards: (category, tie-break ranks...), 0 high
    card .. 8 straight flush."""
    if len(cards) > 5:
        return max(_five(list(five)) for five in itertools.combinations(cards, 5))
    return _five(cards)


def _five(cards: list[str]) -> tuple:
    ranks = sorted((_rank(c) for c in cards), reverse=True)
    groups = sorted(Counter(ranks).items(), key=lambda kv: (-kv[1], -kv[0]))
    shape = [n for _, n in groups]
    by_group = [r for r, _ in groups]
    flush = len({c[1] for c in cards}) == 1
    straight = _straight_high(set(ranks)) if len(groups) == 5 else 0
    if straight and flush:
        return (8, straight)
    if shape == [4, 1]:
        return (7, *by_group)
    if shape == [3, 2]:
        return (6, *by_group)
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if shape == [3, 1, 1]:
        return (3, *by_group)
    if shape == [2, 2, 1]:
        return (2, *by_group)
    if shape == [2, 1, 1, 1]:
        return (1, *by_group)
    return (0, *ranks)


def draws(cards: list[str]) -> list[str]:
    """Flush and straight draws five cards hold, for a flop with more cards to come."""
    out = []
    if max(Counter(c[1] for c in cards).values()) == 4:
        out.append("flush draw")
    ranks = {_rank(c) for c in cards}
    if not _straight_high(ranks):
        outs = sum(1 for r in range(2, 15) if r not in ranks and _straight_high(ranks | {r}))
        if outs >= 2:
            out.append("open-ended straight draw")
        elif outs == 1:
            out.append("gutshot")
    return out


def describe(cards: list[str], with_draws: bool = False) -> str:
    """Plain-words name of five cards, e.g. "two pair, kings and nines", optionally with draws."""
    v = value(cards)
    cat, a = v[0], v[1]
    one, many = (lambda r: _ONE[RANKS[r - 2]]), (lambda r: _MANY[RANKS[r - 2]])
    if cat == 8:
        text = f"straight flush, {one(a)} high"
    elif cat == 7:
        text = f"four {many(a)}"
    elif cat == 6:
        text = f"full house, {many(a)} full of {many(v[2])}"
    elif cat == 5:
        text = f"flush, {one(a)} high"
    elif cat == 4:
        text = f"straight, {one(a)} high"
    elif cat == 3:
        text = f"three {many(a)}"
    elif cat == 2:
        text = f"two pair, {many(a)} and {many(v[2])}"
    elif cat == 1:
        text = f"pair of {many(a)}"
    else:
        text = f"{one(a)} high"
    extra = draws(cards) if with_draws and cat < 4 else []
    return ", ".join([text, *extra])
