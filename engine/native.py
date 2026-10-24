"""The C++ engine (engine/cpp) from Python: builds it with CMake if its sources changed, then imports it.

    from engine.native import solver
    s = solver(SPOTS["c1"])
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pybind11

ROOT = Path(__file__).resolve().parent.parent
SOURCE, BUILD = ROOT / "engine" / "cpp", ROOT / "build" / "cpp"


def load():
    """The cfr_cpp module, rebuilt first (a no-op when nothing changed)."""
    if not (BUILD / "CMakeCache.txt").exists():
        # Leave out compiler and linker flags from the shell, so the system toolchain links alone.
        env = {k: v for k, v in os.environ.items() if k not in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS")}
        subprocess.run(["cmake", "-S", SOURCE, "-B", BUILD, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                        f"-DPython_EXECUTABLE={sys.executable}", f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"],
                       check=True, stdout=subprocess.DEVNULL, env=env)
    subprocess.run(["cmake", "--build", BUILD], check=True, stdout=subprocess.DEVNULL)
    if str(BUILD) not in sys.path:
        sys.path.insert(0, str(BUILD))
    return importlib.import_module("cfr_cpp")


def solver(spot, method: str = "cfr+"):
    """A C++ solver for an engine.cfr.Spot."""
    cpp = load()
    return cpp.Solver(cpp.Spot(ranks=spot.ranks, suits=spot.suits, board=list(spot.board), blind=spot.blind,
                               bet=spot.bet, max_raises=spot.max_raises, stack=spot.stack,
                               pot_sizes=list(spot.pot_sizes)), method)
