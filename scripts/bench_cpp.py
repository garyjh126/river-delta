"""The C++ engine against the Python engine: seconds for the same iterations of the same method.

    uv run python scripts/bench_cpp.py            # Mini, C1 and C1 no-limit, both methods (~2 min)

Prints a Markdown table. Times count only the iterations (the best of 3 fresh solves), not building
the solver or checking exploitability. Both engines run on one thread.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.cfr import SPOTS, Solver  # noqa: E402
from engine.native import solver as cpp_solver  # noqa: E402


def seconds(make, iterations: int, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        solver = make()
        start = time.perf_counter()
        solver.iterate(iterations)
        best = min(best, time.perf_counter() - start)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", choices=SPOTS, default=["mini", "c1", "c1_nl"])
    ap.add_argument("--scale", type=int, default=10, help="run this many times each game's own iteration count")
    args = ap.parse_args()
    print("| game | method | iterations | Python | C++ | C++ is faster by |")
    print("|---|---|---:|---:|---:|---:|")
    for game in args.games:
        spot = SPOTS[game]
        iterations = spot.iterations * args.scale
        for method in Solver.METHODS:
            py = seconds(lambda: Solver(spot, method), iterations)
            cpp = seconds(lambda: cpp_solver(spot, method), iterations)
            print(f"| {game} | {method} | {iterations:,} | {py:.2f} s | {cpp:.3f} s | {py / cpp:.0f}x |", flush=True)


if __name__ == "__main__":
    main()
