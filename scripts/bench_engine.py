"""Exploitability against iterations and time for the engine's CFR+ and discounted CFR.

    uv run python scripts/bench_engine.py                 # C1 limit and no-limit, up to 3,000 iterations

Prints one Markdown table per game. Times count only the solver's iterations, not the
exploitability checks between them. Exploitability is in chips per hand (blinds are 1 chip).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.cfr import SPOTS, Solver  # noqa: E402

CHECKPOINTS = (10, 30, 100, 300, 1000, 3000)


def run(game: str, method: str, iterations: int) -> list[tuple[int, float, float]]:
    solver = Solver(SPOTS[game], method)
    rows, spent = [], 0.0
    for checkpoint in (c for c in CHECKPOINTS if c <= iterations):
        start = time.perf_counter()
        solver.iterate(checkpoint - solver.iterations)
        spent += time.perf_counter() - start
        rows.append((checkpoint, spent, solver.exploitability()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", choices=SPOTS, default=["c1", "c1_nl"])
    ap.add_argument("--iterations", type=int, default=3000)
    args = ap.parse_args()
    for game in args.games:
        plus, dcfr = (run(game, method, args.iterations) for method in ("cfr+", "dcfr"))
        print(f"\n{game}: exploitability in chips per hand (seconds of solving)\n")
        print("| iterations | CFR+ | discounted CFR | CFR+ / discounted |")
        print("|---:|---:|---:|---:|")
        for (t, s1, e1), (_, s2, e2) in zip(plus, dcfr):
            print(f"| {t:,} | {e1:.2e} ({s1:.2f} s) | {e2:.2e} ({s2:.2f} s) | {e1 / e2:.1f}x |")


if __name__ == "__main__":
    main()
