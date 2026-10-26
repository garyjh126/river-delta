// A command-line run of the C++ engine, for stepping through it in a debugger (.vscode/launch.json):
// solves one of engine/cfr.py's games and prints its exploitability.
//
//     cfr_debug [game] [iterations] [method]    # defaults: mini 1 cfr+
#include "cfr.hpp"

#include <cstdio>
#include <map>
#include <string>

int main(int argc, char** argv) {
    // engine/cfr.py's SPOTS, but for the full-deck river, whose ranges are read in Python.
    const cfr::Spot mini{"2345", "cd", {"3c", "4d", "5c"}};
    const cfr::Spot c1{"234567", "cdhs", {"5c", "6d", "7h"}};
    const cfr::Spot c1_river{"234567", "cdhs", {"2c", "4c", "5d", "7c", "7h"}};
    auto nolimit = [](cfr::Spot spot) {
        spot.stack = 20;
        return spot;
    };
    const std::map<std::string, cfr::Spot> spots = {
        {"mini", mini}, {"c1", c1}, {"mini_nl", nolimit(mini)}, {"c1_nl", nolimit(c1)}, {"c1_river", c1_river}};

    const std::string game = argc > 1 ? argv[1] : "mini";
    const int iterations = argc > 2 ? std::stoi(argv[2]) : 1;
    const std::string method = argc > 3 ? argv[3] : "cfr+";

    cfr::Solver solver(spots.at(game), method);
    solver.iterate(iterations);
    std::printf("%s, %s, %d iterations: %zu hands, %zu decision nodes, exploitability %.6e chips per hand\n",
                game.c_str(), method.c_str(), solver.iterations(), solver.hands().size(), solver.nodes().size(),
                solver.exploitability());
}
