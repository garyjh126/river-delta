"""Build the strategy viewer from the games' data in build/.

    build/<game>.engine.json   from engine/cfr.py (our solver), for mini, c1 and c1_nl
    build/holdem.json          from viewer/mock_holdem.py (mock data, not a solve)

Every game arrives as one shape: a tree of decision nodes, each with its actions and, per
action, the probability for every combo. Writes build/viewer.html (open it in a browser) and
build/viewer.artifact.html (the same page without the document wrapper, for publishing as a
claude.ai artifact).
"""

import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from engine.cards import RANKS, describe  # noqa: E402

PLAIN_SEATS = [
    {"name": "P1", "pos": "OOP", "long": "out of position"},
    {"name": "P2", "pos": "IP", "long": "in position"},
]
METHOD_NAMES = {"cfr+": "CFR+", "dcfr": "discounted CFR"}


def load(key: str, how: str) -> dict:
    path = root / "build" / f"{key}.json"
    if not path.exists():
        sys.exit(f"{path} is missing: run `{how}` first")
    return json.loads(path.read_text())


def solved_game(key: str, switch: str, deck: str, default_cell: str) -> dict:
    """A one-street game solved by engine/cfr.py."""
    s = load(f"{key}.engine", f"uv run python engine/cfr.py {key}")
    board_cards = re.search(r"boardCards=(\w+)", s["load"]).group(1)
    board = [board_cards[i : i + 2] for i in range(0, len(board_cards), 2)]
    num_ranks = int(re.search(r"numRanks=(\d+)", s["load"]).group(1))
    tree = s["tree"]
    rules = (f"No-limit, {s['stack']}-chip stacks, pot-size bets and all-in" if s["nolimit"]
             else "Limit, bet 2, one raise")
    return {
        "switch": switch,
        "deck": deck,
        "rules": rules,
        "source": (f"Our engine · {METHOD_NAMES[s['method']]} · {s['iterations']:,} iterations in {s['seconds']:.1f} s"
                   f" · exploitability {s['exploitability']:.6f} chips/hand"),
        "note": ("Strategies are the average policy of this repo's engine (engine/cfr.py). Its tests check it against "
                 "OpenSpiel on the same games: identical iterates and the same exploitability. EVs come from the same "
                 "engine, checked against a walk over every deal and betting line under OpenSpiel's rules (Mini, limit "
                 "and no-limit); they're left out where the opponent never gets to a point. Bet amounts leave out "
                 "the blinds, which act like antes. Combos that never reach a decision are shown as out of range: "
                 "the solver's numbers there are placeholders with no effect on play."),
        "code": s["load"],
        "mock": False,
        "board": board,
        "ranks": list(RANKS[:num_ranks][::-1]),
        "seats": PLAIN_SEATS,
        "defaultCell": default_cell,
        "hands": tree["hands"],
        "made": [describe(board + [h[:2], h[2:]]) for h in tree["hands"]],
        "range0": None,
        "root": tree["root"],
        "nodes": tree["nodes"],
    }


def mock_game() -> dict:
    m = load("holdem", "uv run python viewer/mock_holdem.py")
    board = m["board"]
    return {
        "switch": "Hold'em · 52 cards",
        "deck": "52-card deck",
        "rules": "BTN vs BB · single-raised pot · 100 bb",
        "source": None,
        "note": "Mock data, made by viewer/mock_holdem.py from a hand-strength heuristic with seeded jitter. No solver produced these numbers: they are here to judge the layout at full size. Starting ranges are mock too.",
        "code": None,
        "mock": True,
        "board": board,
        "ranks": list(RANKS[::-1]),
        "seats": [
            {"name": "BB", "pos": "OOP", "long": "big blind, out of position"},
            {"name": "BTN", "pos": "IP", "long": "button, in position"},
        ],
        "defaultCell": "KQs",
        "hands": m["hands"],
        "made": [describe(board + [h[:2], h[2:]], with_draws=True) for h in m["hands"]],
        "range0": m["range0"],
        "root": m["root"],
        "nodes": m["nodes"],
    }


games = {
    "mini": solved_game("mini", "Mini · 8 cards", "8-card deck", "53s"),
    "c1": solved_game("c1", "C1 · 24 cards", "24-card deck", "43s"),
    "c1_nl": solved_game("c1_nl", "C1 no-limit", "24-card deck", "43s"),
    "holdem": mock_game(),
}
data = {"order": list(games), "games": games}
page = (root / "viewer" / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
(root / "build" / "viewer.artifact.html").write_text(page)
(root / "build" / "viewer.html").write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    f"</head><body style=\"margin:0\">{page}</body></html>"
)
for key, g in games.items():
    print(f"{key}: {len(g['hands'])} combos, {len(g['nodes'])} decision nodes, {g['source'] or 'mock'}", file=sys.stderr)
print(f"{root / 'build' / 'viewer.html'} ({len(page) / 1e6:.2f} MB)", file=sys.stderr)
