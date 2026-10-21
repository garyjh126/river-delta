"""Build the strategy viewer from build/mini.json and build/c1.json (written by scripts/solve_reference.py).

Writes build/viewer.html (open it in a browser) and build/viewer.artifact.html (the same page
without the document wrapper, for publishing as a claude.ai artifact).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent.parent
DECKS = {"mini": "8-card deck", "c1": "24-card deck"}


def made(hand: str, board: list[str]) -> str:
    """The five-card hand (board plus hole cards) by the usual ranking. These decks have no ace."""
    cards = board + [hand[:2], hand[2:]]
    ranks = sorted((int(c[0]) for c in cards), reverse=True)
    groups = sorted(Counter(ranks).items(), key=lambda kv: (-kv[1], -kv[0]))
    straight = len(set(ranks)) == 5 and ranks[0] - ranks[-1] == 4
    flush = len({c[1] for c in cards}) == 1
    if straight and flush:
        return f"straight flush, {ranks[0]} high"
    if groups[0][1] == 4:
        return f"four {groups[0][0]}s"
    if groups[0][1] == 3 and groups[1][1] == 2:
        return f"full house, {groups[0][0]}s full of {groups[1][0]}s"
    if flush:
        return f"flush, {ranks[0]} high"
    if straight:
        return f"straight, {ranks[0]} high"
    if groups[0][1] == 3:
        return f"three {groups[0][0]}s"
    if groups[0][1] == 2 and groups[1][1] == 2:
        return f"two pair, {groups[0][0]}s and {groups[1][0]}s"
    if groups[0][1] == 2:
        return f"pair of {groups[0][0]}s"
    return f"{ranks[0]} high"


def game(key: str) -> dict:
    path = root / "build" / f"{key}.json"
    if not path.exists():
        sys.exit(f"{path} is missing: run `uv run python scripts/solve_reference.py {key}` first")
    solved = json.loads(path.read_text())
    board_cards = re.search(r"boardCards=(\w+)", solved["load"]).group(1)
    board = [board_cards[i : i + 2] for i in range(0, len(board_cards), 2)]
    num_ranks = int(re.search(r"numRanks=(\d+)", solved["load"]).group(1))
    return {
        "deck": DECKS[key],
        "load": solved["load"],
        "iterations": solved["iterations"],
        "exploitability": solved["exploitability"],
        "board": board,
        "ranks": [str(r) for r in range(num_ranks + 1, 1, -1)],
        "nodes": solved["nodes"],
        "made": {h: made(h, board) for h in solved["nodes"][""]["hands"]},
    }


data = {"games": {key: game(key) for key in DECKS}}
page = (root / "viewer" / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
(root / "build" / "viewer.artifact.html").write_text(page)
(root / "build" / "viewer.html").write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    f"</head><body style=\"margin:0\">{page}</body></html>"
)
for key, g in data["games"].items():
    print(f"{key}: {g['iterations']} iterations, exploitability {g['exploitability']:.3g}", file=sys.stderr)
print(root / "build" / "viewer.html", file=sys.stderr)
