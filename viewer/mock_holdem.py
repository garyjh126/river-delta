"""Mock strategy data for a full Hold'em flop spot.

Nothing here is solved. The numbers come from a hand-strength heuristic with seeded jitter, so the
viewer can be built and judged at full size (all 1,326 starting combos, 1,176 of them possible on
this flop) before our engine exists. The page labels them as mock.

Spot: heads-up, 100 bb deep, single-raised pot (the button opens to 2.5 bb, the big blind calls),
flop Ks 9h 5h. Pot 5.5 bb with 97.5 bb behind each. Bets of 33%, 75% and 150% of the pot (the big
blind leads only the first two), one raise to 3x, and all-in. Writes build/holdem.json.

    uv run python viewer/mock_holdem.py
"""

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.cards import RANKS, SUITS, draws, value  # noqa: E402

BOARD = ["Ks", "9h", "5h"]
POT, STACK = 5.5, 97.5
SIZES = {0: (0.33, 0.75), 1: (0.33, 0.75, 1.5)}  # bet sizes as fractions of the pot: big blind, button
RAISE_TO = 3.0
SEATS = ("BB", "BTN")
OUT = Path(__file__).resolve().parent.parent / "build" / "holdem.json"

deck = [r + s for r in reversed(RANKS) for s in SUITS]
hands = [a + b for a, b in itertools.combinations(deck, 2) if a not in BOARD and b not in BOARD]
cards = [BOARD + [h[:2], h[2:]] for h in hands]
values = [value(c) for c in cards]
drawing = [draws(c) for c in cards]


def jitter(*key) -> float:
    """Deterministic pseudo-random number in [0, 1) for a key."""
    return int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:4], "big") / 2**32


def percentiles(scores) -> list[float]:
    """Each item's position among all of them, ties sharing the average: 0 = worst, 1 = best."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    out = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 / (len(order) - 1)
        i = j + 1
    return out


def preflop_score(h: str) -> float:
    hi, lo = sorted((RANKS.index(h[0]) + 2, RANKS.index(h[2]) + 2), reverse=True)
    if hi == lo:
        return 0.5 + hi / 28
    return (hi + 0.6 * lo) / 22.4 + (0.06 if h[1] == h[3] else 0) - 0.025 * max(0, hi - lo - 1)


def fade(x: float, full: float, zero: float) -> float:
    """1 at or above `full`, 0 at or below `zero`, linear between."""
    return min(1.0, max(0.0, (x - zero) / (full - zero)))


# Starting ranges (mock): the button opens about the best 45% of hands. The big blind calls about
# half its hands and re-raises most premium ones instead, so few of those are left when it calls.
pre = percentiles([preflop_score(h) for h in hands])
range0 = [
    [round(0.25 if p > 0.95 else fade(p, 0.58, 0.45), 3) for p in pre],
    [round(fade(p, 0.60, 0.50), 3) for p in pre],
]
strength = percentiles(values)

nodes = {}


def fmt(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def node(player: int, put: list[float], raised: bool, last: str | None) -> str:
    """Add the decision node where `player` acts, with `put` already in this street; return its id."""
    nid = f"n{len(nodes)}"
    nodes[nid] = None  # keep ids in creation order
    opp = 1 - player
    pot = round(POT + put[0] + put[1], 1)
    to_call = round(put[opp] - put[player], 1)
    actions = []

    def end(kind, label, size, text):
        actions.append({"kind": kind, "label": label, "size": size, "next": None, "end": text})

    def go(kind, label, size, mine, raises):
        new = list(put)
        new[player] = mine
        frac = (mine - put[player]) / pot
        actions.append({"kind": kind, "label": label, "size": size, "next": node(opp, new, raises, kind), "end": None, "frac": frac})

    if to_call == 0:
        if last == "check":
            end("check", "check", 0, f"Checks through: the turn comes, pot {fmt(pot)}")
        else:
            go("check", "check", 0, put[player], raised)
        for frac in SIZES[player]:
            amount = round(frac * pot, 1)
            go("bet", f"bet {fmt(amount)}", amount, amount, raised)
        if last == "check":
            go("allin", f"all-in {fmt(STACK)}", STACK, STACK, True)
    else:
        end("fold", "fold", 0, f"{SEATS[player]} folds; {SEATS[opp]} wins the pot of {fmt(pot)}")
        called = fmt(pot + to_call)
        text = f"All-in and called: the board runs out for a pot of {called}" if put[opp] >= STACK else f"The turn comes, pot {called}"
        end("call", f"call {fmt(to_call)}", put[opp], text)
        if put[opp] < STACK:
            to = round(RAISE_TO * put[opp], 1)
            if not raised and to < STACK / 2:
                go("raise", f"raise {fmt(to)}", to, to, True)
            go("allin", f"all-in {fmt(STACK)}", STACK, STACK, True)

    nodes[nid] = {
        "player": player,
        "pot": pot,
        "toCall": to_call,
        "faced": to_call / (pot - to_call) if to_call else 0.0,
        "badge": [f"{fmt(STACK - put[0])} bb", f"{fmt(STACK - put[1])} bb"],
        "actions": actions,
    }
    return nid


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def weights(n: dict, i: int) -> list[float]:
    """Unnormalised mock preference for each action with hand i: value bets and raises grow with
    made-hand strength (bigger sizes need stronger hands), draws bluff, weak hands fold to bets."""
    s, d = strength[i], drawing[i]
    draw = 1.0 if ("flush draw" in d or "open-ended straight draw" in d) else 0.4 if "gutshot" in d else 0.0
    out = []
    for a in n["actions"]:
        if a["kind"] == "fold":
            line = 0.25 + 0.2 * min(n["faced"], 1.5)
            w = 3 * sigmoid((line - s) / 0.05) * (1 - 0.8 * draw)
        elif a["kind"] in ("check", "call"):
            w = 1.0
        else:
            allin = a["kind"] == "allin"
            line = 0.97 if allin else 0.6 + 0.2 * min(a["frac"], 1.5)
            bluff = 0.08 * draw if allin else 0.35 * draw * min(a["frac"] + 0.3, 1.2)
            w = (1 + min(a["frac"], 1.5)) * sigmoid((s - line) / 0.04) + bluff
            if n["player"] == 0 and n["toCall"] == 0:
                w *= 0.55  # the big blind leads into the raiser less often
        out.append(w)
    return out


node(0, [0.0, 0.0], False, None)
for nid, n in nodes.items():
    probs = [[] for _ in n["actions"]]
    for i in range(len(hands)):
        key = (nid, values[i], tuple(drawing[i]))  # same made hand and draws, same mock strategy
        w = [x * (0.6 + 0.8 * jitter(*key, j)) for j, x in enumerate(weights(n, i))]
        p = [x / sum(w) for x in w]
        p = [x if x >= 0.05 else 0.0 for x in p]  # drop slivers, as a solver's output often does
        p = [round(x / sum(p), 3) for x in p]
        top = p.index(max(p))
        p[top] = round(p[top] + 1 - sum(p), 3)
        for j, x in enumerate(p):
            probs[j].append(x)
    n["probs"] = probs
    del n["faced"]
    for a in n["actions"]:
        a.pop("frac", None)

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"board": BOARD, "hands": hands, "range0": range0, "seats": SEATS, "root": "n0", "nodes": nodes}))
print(f"{len(hands)} combos, {len(nodes)} decision nodes -> {OUT}")
