"""Ranges written the usual way, read as a weight for every two-card hand.

    "TT+, AKs, KQo:0.5, QJ, A2s+, AhKh"

Items are separated by commas. Each is a pair (QQ), a suited hand (AKs), an offsuit one (AKo),
both (AK), or one exact hand (AhKh). A "+" after a pair adds every higher pair (TT+ is TT to AA);
after any other hand it raises the second card up to one below the first (A2s+ is A2s to AKs).
":w" gives the item's hands weight w instead of 1, and a later item overrides an earlier one.
Hands the range doesn't name have weight 0.
"""

import numpy as np

from engine.cards import RANKS


def _rank(c: str) -> int:
    return RANKS.index(c)


def _item(text: str):
    """A test of whether a hand, given as (its two cards, high rank, low rank, suited), is in the item."""
    if len(text) == 4 and not text.endswith("+"):
        exact = {text[:2], text[2:]}
        return lambda cards, hi, lo, suited: cards == exact
    plus = text.endswith("+")
    body = text.removesuffix("+")
    kind = body[2:]
    top, low = sorted((_rank(body[0]), _rank(body[1])), reverse=True)
    if kind not in ("", "s", "o") or (top == low and kind):
        raise ValueError
    if top == low:
        pairs = range(top, len(RANKS)) if plus else [top]
        return lambda cards, hi, lo, suited: hi == lo and hi in pairs
    lows = range(low, top) if plus else [low]
    return lambda cards, hi, lo, suited: hi == top and lo in lows and (kind == "" or (kind == "s") == suited)


def weights(text: str, hands: list[str]) -> np.ndarray:
    """Each hand's weight in the range `text`. Hands are written as the engine writes them, e.g. "KsKh"."""
    keys = []
    for h in hands:
        hi, lo = sorted((_rank(h[0]), _rank(h[2])), reverse=True)
        keys.append(({h[:2], h[2:]}, hi, lo, h[1] == h[3]))
    out = np.zeros(len(hands))
    for item in (s.strip() for s in text.split(",")):
        if not item:
            continue
        name, _, weight = item.partition(":")
        try:
            test = _item(name.strip())
            w = float(weight) if weight else 1.0
        except (ValueError, IndexError):
            raise ValueError(f"can't read range item {item!r}") from None
        chosen = [i for i, key in enumerate(keys) if test(*key)]
        if not chosen:
            raise ValueError(f"range item {item!r} names no hand that can be dealt here")
        out[chosen] = w
    return out
