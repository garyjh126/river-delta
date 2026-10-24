// Vector-form CFR for heads-up poker with one betting round on a fixed flop: a port of
// engine/cfr.py, which stays the answer key (tests/test_cpp.py). Same method, same references.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace cfr {

// One betting round on a fixed flop, as engine/cfr.py's Spot. Both players post `blind` and
// player 0 acts first. Limit when `stack` is unset, no-limit when it is set.
struct Spot {
    std::string ranks;
    std::string suits;
    std::vector<std::string> board;
    int blind = 1;
    int bet = 2;
    int max_raises = 2;
    std::optional<int> stack;
    std::vector<double> pot_sizes = {1.0};

    bool nolimit() const { return stack.has_value(); }
};

// Where an action leads: the next decision node, or the end of the hand.
struct Child {
    int node = -1;  // index into Solver::nodes(), or -1 at the end of the hand
    int folder = -1;  // at the end: the player who folded, or -1 for a showdown
    std::array<int, 2> contrib{};  // at the end: chips each player has put in
};

struct Node {
    std::string seq;  // the betting so far, written as OpenSpiel writes it
    int player = 0;
    std::array<int, 2> contrib{};
    std::vector<std::string> actions;  // fold, check, call, bet, raise, allin
    std::vector<Child> children;  // one per action
};

class Solver {
public:
    explicit Solver(Spot spot);

    const Spot& spot() const { return spot_; }
    const std::vector<std::string>& hands() const { return hands_; }
    const std::vector<Node>& nodes() const { return nodes_; }  // nodes()[0] is the root
    // compatible[i * n + j]: 1 if hands i and j share no card, so both can be dealt at once.
    const std::vector<double>& compatible() const { return compatible_; }
    // showdown[i * n + j]: +1 if hand i beats hand j, -1 if it loses, 0 for a tie or a clash.
    const std::vector<double>& showdown() const { return showdown_; }
    double deal_prob() const { return deal_prob_; }

private:
    int build_tree(const std::string& seq, int player, std::array<int, 2> contrib);

    Spot spot_;
    std::vector<std::string> hands_;
    int n_ = 0;
    std::vector<double> compatible_, showdown_;
    double deal_prob_ = 0;  // every hand leaves the same number of opponent hands
    std::vector<Node> nodes_;
};

// Every two-card hand left once the board is out, in engine/cfr.py's order.
std::vector<std::string> deal_hands(const Spot& spot);

}  // namespace cfr
