"""Build the strategy viewer from the games' data in build/.

    build/<game>.engine.json   from engine/cfr.py (our solver), for mini, c1, c1_nl and river
    build/holdem.json          from viewer/mock_holdem.py (mock data, not a solve)

Every game arrives as one shape: a tree of decision nodes, each with its actions and, per
action, the probability for every combo. Writes build/viewer.html (open it in a browser) and
build/viewer.artifact.html (the same page without the document wrapper, for publishing as a
claude.ai artifact). With --public, writes build/public/index.html instead: the solved games only,
without the mock one, for the project's GitHub Pages site.
"""

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from engine.cards import RANKS, describe  # noqa: E402

PLAIN_SEATS = [
    {"name": "P1", "pos": "OOP", "long": "out of position"},
    {"name": "P2", "pos": "IP", "long": "in position"},
]
HOLDEM_SEATS = [
    {"name": "BB", "pos": "OOP", "long": "big blind, out of position"},
    {"name": "BTN", "pos": "IP", "long": "button, in position"},
]
METHOD_NAMES = {"cfr+": "CFR+", "dcfr": "discounted CFR"}


def load(key: str, how: str) -> dict:
    path = root / "build" / f"{key}.json"
    if not path.exists():
        sys.exit(f"{path} is missing: run `{how}` first")
    return json.loads(path.read_text())


def solved_game(key: str, switch: str, deck: str, default_cell: str) -> dict:
    """A one-street game solved by engine/cfr.py."""
    how = f"uv run python engine/cfr.py {key}" + (" --engine c++" if key == "river" else "")
    s = load(f"{key}.engine", how)
    spot, tree = s["spot"], s["tree"]
    board = spot["board"]
    source = (f"Our engine{' in C++' if s['engine'] == 'c++' else ''} · {METHOD_NAMES[s['method']]} · "
              f"{s['iterations']:,} iterations in {s['seconds']:.1f} s · exploitability {s['exploitability']:.6f} chips/hand")
    game = {
        "switch": switch,
        "deck": deck,
        "street": "River" if len(board) == 5 else "Flop",
        "rules": (f"No-limit, {s['stack']}-chip stacks, pot-size bets and all-in" if s["nolimit"]
                  else "Limit, bet 2, one raise"),
        "source": source,
        "note": ("Strategies are the average policy of this repo's engine (engine/cfr.py). Its tests check it against "
                 "OpenSpiel on the same games: identical iterates and the same exploitability. EVs come from the same "
                 "engine, checked against a walk over every deal and betting line under OpenSpiel's rules (Mini, limit "
                 "and no-limit); they're left out where the opponent never gets to a point. Bet amounts leave out "
                 "the blinds, which act like antes. Combos that never reach a decision are shown as out of range: "
                 "the solver's numbers there are placeholders with no effect on play."),
        "code": s["load"],
        "mock": False,
        "board": board,
        "ranks": list(spot["ranks"][::-1]),
        "seats": PLAIN_SEATS,
        "defaultCell": default_cell,
        "hands": tree["hands"],
        "made": [describe(board + [h[:2], h[2:]]) for h in tree["hands"]],
        "range0": tree["ranges"] if spot["ranges"] else None,
        "root": tree["root"],
        "nodes": tree["nodes"],
    }
    if key == "river":
        pot, behind = 2 * spot["blind"], spot["stack"] - spot["blind"]
        sizes = ", ".join(f"{100 * x:g}%" for x in spot["pot_sizes"])
        game.update({
            "rules": f"BB vs BTN · pot {pot} bb, {behind} bb behind · bets and raises of {sizes} of the pot, all-in",
            "source": f"{source} ({100 * s['exploitability'] / pot:.3f}% of the pot)",
            "note": ("Strategies are the average policy of this repo's engine, iterated in C++ (engine/cpp). "
                     "scripts/check_river.py checks this spot against noambrown/poker_solver, an independent river "
                     "solver: the same betting tree, and our best response measures its strategy's exploitability "
                     "as its own does. EVs come from the same engine; they're left out where the opponent never gets "
                     f"to a point. The starting ranges are written by hand for illustration, not from a solve of the "
                     f"flop and turn. Bet amounts are chips added on the river; the {pot} bb already in the pot count "
                     "as dead money. Combos that never reach a decision are shown as out of range: the solver's "
                     "numbers there are placeholders with no effect on play."),
            "code": f"BB: {spot['ranges'][0]}  ·  BTN: {spot['ranges'][1]}",
            "seats": HOLDEM_SEATS,
        })
    return game


def mock_game() -> dict:
    m = load("holdem", "uv run python viewer/mock_holdem.py")
    board = m["board"]
    return {
        "switch": "Hold'em · 52 cards",
        "deck": "52-card deck",
        "street": "Flop",
        "rules": "BTN vs BB · single-raised pot · 100 bb",
        "source": None,
        "note": "Mock data, made by viewer/mock_holdem.py from a hand-strength heuristic with seeded jitter. No solver produced these numbers: they are here to judge the layout at full size. Starting ranges are mock too.",
        "code": None,
        "mock": True,
        "board": board,
        "ranks": list(RANKS[::-1]),
        "seats": HOLDEM_SEATS,
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
    "river": solved_game("river", "River · 52 cards", "52-card deck", "KJo"),
}
public = "--public" in sys.argv[1:]
if not public:
    games["holdem"] = mock_game()
data = {"order": list(games), "games": games}
page = (root / "viewer" / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f"</head><body style=\"margin:0\">{page}</body></html>")
if public:
    out = root / "build" / "public" / "index.html"
    out.parent.mkdir(exist_ok=True)
else:
    out = root / "build" / "viewer.html"
    (root / "build" / "viewer.artifact.html").write_text(page)
out.write_text(document)
for key, g in games.items():
    print(f"{key}: {len(g['hands'])} combos, {len(g['nodes'])} decision nodes, {g['source'] or 'mock'}", file=sys.stderr)
print(f"{out} ({len(page) / 1e6:.2f} MB)", file=sys.stderr)
