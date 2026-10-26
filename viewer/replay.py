"""Record a run of the C++ engine step by step and build the replay page from it.

    uv run python viewer/replay.py                            # Mini, CFR+, 10 iterations
    uv run python viewer/replay.py --iterations 30 --method dcfr

Each iteration walks the betting tree once per player, as Solver::cfr in engine/cpp/cfr.cpp does,
and Solver::trace records every step of those walks: reaching a decision node, ending the hand,
and leaving the node with its values and updated regrets. The page (viewer/replay_template.html)
steps through them like a debugger, with breakpoints on nodes. Writes build/replay.html. Numbers
are rounded to 6 significant digits.
"""

import argparse
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from engine.cards import describe  # noqa: E402
from engine.cfr import SPOTS  # noqa: E402
from engine.native import solver  # noqa: E402


def rounded(xs) -> list[float]:
    return [float(f"{x:.6g}") for x in xs]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--method", choices=("cfr+", "dcfr"), default="cfr+")
    args = parser.parse_args()

    spot = SPOTS["mini"]
    s = solver(spot, args.method)
    steps, exploitability = [], []
    for _ in range(args.iterations):
        for step in s.trace(1):
            steps.append({
                "kind": step.kind.name, "t": step.iteration, "p": step.player, "node": step.node, "a": step.action,
                "own": rounded(step.own), "opp": rounded(step.opp), "strategy": rounded(step.strategy),
                "values": rounded(step.values), "value": rounded(step.value), "regret": rounded(step.regret),
                "sum": rounded(step.strategy_sum),
            })
        exploitability.append(s.exploitability())

    board = list(spot.board)
    data = {
        "game": "Mini · 8 cards",
        "rules": "Limit, blinds 1 each, bet 2, one raise",
        "method": args.method,
        "iterations": args.iterations,
        "board": board,
        "hands": s.hands,
        "made": [describe(board + [h[:2], h[2:]]) for h in s.hands],
        "showdown": s.showdown.astype(int).tolist(),
        "nodes": [{"seq": n.seq, "player": n.player, "contrib": list(n.contrib), "actions": n.actions,
                   "children": [{"node": c.node, "folder": c.folder, "contrib": list(c.contrib)} for c in n.children]}
                  for n in s.nodes],
        "steps": steps,
        "exploitability": exploitability,
    }
    page = (root / "viewer" / "replay_template.html").read_text().replace(
        "/*DATA*/null", json.dumps(data, separators=(",", ":")))
    out = root / "build" / "replay.html"
    out.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        f'</head><body style="margin:0">{page}</body></html>'
    )
    print(f"{len(steps):,} steps over {args.iterations} iterations ({args.method}), exploitability "
          f"{exploitability[-1]:.6f} chips per hand at the end -> {out} ({len(page) / 1e6:.2f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
