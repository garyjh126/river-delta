"""Build the Mini strategy viewer from build/mini.json (written by scripts/solve_mini.py).

Writes build/mini.html (open it in a browser) and build/mini.artifact.html (the same page
without the document wrapper, for publishing as a claude.ai artifact).
"""

import json
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent.parent
solved = json.loads((root / "build" / "mini.json").read_text())
board = ["3c", "4d", "5c"]
names = {"2": "2s", "3": "3s", "4": "4s", "5": "5s"}


def made(hand: str) -> str:
    """Made hand on this board. With ranks 2-5 and two suits, no straight, flush or trips exists."""
    counts = Counter(c[0] for c in board + [hand[:2], hand[2:]])
    pairs = sorted((r for r, n in counts.items() if n == 2), reverse=True)
    assert all(n <= 2 for n in counts.values()) and pairs, hand
    if len(pairs) >= 2:
        return f"two pair, {names[pairs[0]]} and {names[pairs[1]]}"
    return f"pair of {names[pairs[0]]}"


data = {
    "load": solved["load"],
    "iterations": solved["iterations"],
    "exploitability": solved["exploitability"],
    "board": board,
    "nodes": solved["nodes"],
    "made": {h: made(h) for h in solved["nodes"][""]["hands"]},
}
page = (root / "viewer" / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
(root / "build" / "mini.artifact.html").write_text(page)
(root / "build" / "mini.html").write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    f"</head><body style=\"margin:0\">{page}</body></html>"
)
print(root / "build" / "mini.html", file=sys.stderr)
