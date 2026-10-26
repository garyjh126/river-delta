"""The river spot against noambrown/poker_solver (MIT), an independent river solver.

    uv run python scripts/check_river.py     # ~1 min; fetches and builds poker_solver into build/ the first time

1. Hands poker_solver the same subgame (board, pot, stacks, bet sizes, both ranges), built at a
   pinned commit with double-precision regrets.
2. Checks that it builds the same betting tree: every decision node, with the same actions and
   amounts.
3. Runs its discounted CFR, dumps its average strategy, and measures that strategy with our best
   response: our exploitability of its strategy must equal its own report of it.
4. Solves the spot with our C++ engine for as many iterations, and compares the two strategies'
   exploitability, the value each gives the first player, and the time taken.

Amounts are in chips (big blinds here). Both solvers run on one thread.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.cfr import SPOTS, Node, Solver  # noqa: E402
from engine.native import solver as cpp_solver  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO, COMMIT = "https://github.com/noambrown/poker_solver.git", "6a10442877ffc8fd28af93e16e279b9bbdd97b2a"
CHECKOUT = ROOT / "build" / "poker_solver"
BINARY = CHECKOUT / "cpp" / "build" / "river_solver_optimized"


def build() -> None:
    if not CHECKOUT.exists():
        subprocess.run(["git", "clone", "-q", REPO, CHECKOUT], check=True)
    subprocess.run(["git", "-C", CHECKOUT, "checkout", "-q", COMMIT], check=True)
    env = {k: v for k, v in os.environ.items() if k not in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS")}
    subprocess.run(["cmake", "-S", CHECKOUT / "cpp", "-B", CHECKOUT / "cpp" / "build", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCFR_USE_DOUBLE=ON"], check=True, stdout=subprocess.DEVNULL, env=env)
    subprocess.run(["cmake", "--build", CHECKOUT / "cpp" / "build", "-j"], check=True, stdout=subprocess.DEVNULL)


def tokens(node: Node) -> list[str]:
    """poker_solver's name for each of the node's actions: c (check or call), f, b<chips added> for a
    bet, r<chips added beyond the call> for a raise."""
    p, c = node.player, node.contrib
    out = []
    for kind, child in zip(node.actions, node.children):
        if kind in ("check", "call", "fold"):
            out.append("f" if kind == "fold" else "c")
        else:
            to = child.contrib[p]
            out.append(f"b{to - c[p]}" if c[0] == c[1] else f"r{to - c[1 - p]}")
    return out


def keys(solver: Solver) -> dict[str, Node]:
    """poker_solver's key for each decision node: its actions so far, joined by "/", or "root"."""
    out = {}

    def walk(node: Node, path: list[str]) -> None:
        out["/".join(path) or "root"] = node
        for token, child in zip(tokens(node), node.children):
            if isinstance(child, Node):
                walk(child, path + [token])

    walk(solver.root, [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=SPOTS["river"].iterations)
    args = ap.parse_args()
    spot = SPOTS["river"]
    build()

    ours = Solver(spot, "dcfr")
    config = {
        "board": list(spot.board), "pot": 2 * spot.blind, "stack": spot.stack - spot.blind,
        "bet_sizes": list(spot.pot_sizes), "include_all_in": True, "max_raises": 1000,
        "players": [{"hands": [h for h, w in zip(ours.hands, ours.ranges[p]) if w > 0],
                     "weights": [float(w) for w in ours.ranges[p] if w > 0]} for p in (0, 1)],
    }
    work = ROOT / "build" / "river_check"
    work.mkdir(parents=True, exist_ok=True)
    (work / "subgame.json").write_text(json.dumps(config))
    start = time.time()
    run = subprocess.run([BINARY, "--config", work / "subgame.json", "--algo", "dcfr", "--iters", str(args.iterations),
                          "--dump-strategy", work / "strategy.json"], check=True, capture_output=True, text=True)
    their_seconds = time.time() - start
    internal, terminal = map(int, re.search(r"internal=(\d+) terminal=(\d+)", run.stdout).groups())
    their_expl = float(re.search(r"Exploitability \(chips\): ([\d.e-]+)", run.stdout).group(1))

    # 2. The same tree.
    ends = sum(not isinstance(c, Node) for n in ours.nodes for c in n.children)
    assert (internal, terminal) == (len(ours.nodes), ends), (internal, terminal, len(ours.nodes), ends)
    dump = json.loads((work / "strategy.json").read_text())["players"]
    by_key = keys(ours)
    assert set(by_key) == set(dump[0]["profile"]) | set(dump[1]["profile"])
    for key, node in by_key.items():
        assert sorted(dump[node.player]["profile"][key]["actions"]) == sorted(tokens(node)), key

    # 3. Their strategy, measured by our best response.
    index = {frozenset((h[:2], h[2:])): i for i, h in enumerate(ours.hands)}
    theirs = Solver(spot)
    for key, node in by_key.items():
        entry, player = dump[node.player]["profile"][key], dump[node.player]
        column = {a: k for k, a in enumerate(entry["actions"])}
        for h, probs in zip(player["hands"], entry["strategy"]):
            i = index[frozenset((h[:2], h[2:]))]
            for a, token in enumerate(tokens(node)):
                theirs.strategy_sum[node.seq][a, i] = probs[column[token]]
    measured = theirs.exploitability()

    # 4. Our own solve.
    start = time.time()
    cpp = cpp_solver(spot, "dcfr")
    cpp.iterate(args.iterations)
    our_seconds = time.time() - start
    for i, node in enumerate(cpp.nodes):
        ours.strategy_sum[node.seq] = cpp.average(i)

    pot, mine = 2 * spot.blind, ours.exploitability()
    print(f"Same tree: {internal} decision nodes and {terminal} ends, every action and amount the same.")
    print(f"{args.iterations:,} iterations of discounted CFR each, one thread:\n")
    print("| | poker_solver | ours |")
    print("|---|---:|---:|")
    print(f"| exploitability by poker_solver's best response (chips) | {their_expl:.6f} | |")
    print(f"| exploitability by our best response (chips) | {measured:.6f} | {mine:.6f} |")
    print(f"| the same, % of the pot | {100 * measured / pot:.4f}% | {100 * mine / pot:.4f}% |")
    print(f"| value to the big blind (chips) | {theirs.value(0):+.6f} | {ours.value(0):+.6f} |")
    print(f"| seconds | {their_seconds:.2f} | {our_seconds:.2f} |")
    assert abs(measured - their_expl) < 2e-6, (measured, their_expl)  # it prints six decimals
    # Each value is within twice its exploitability of the game's value.
    assert abs(theirs.value(0) - ours.value(0)) <= 2 * (measured + ours.exploitability())


if __name__ == "__main__":
    main()
