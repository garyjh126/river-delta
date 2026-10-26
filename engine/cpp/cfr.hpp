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

// One step of an iteration, recorded by Solver::trace for replaying a run (viewer/replay.py).
// Vectors are over hands; (actions x hands) ones are row-major, as strategies are.
struct Step {
    enum class Kind { visit, end, update };  // reach a decision node, end the hand, leave the node
    Kind kind = Kind::visit;
    int iteration = 0;
    int player = 0;  // the player whose regrets this pass updates
    int node = 0;  // the decision node; at an end, the node whose action ended the hand
    int action = -1;  // at an end: that action
    std::vector<double> own, opp;  // reach of the updating player's hands and the opponent's, here
    std::vector<double> strategy;  // visit and update: the current strategy at the node
    std::vector<double> values;  // end: each hand's value; update: each action's (actions x hands)
    std::vector<double> value;  // update: the node's value, the strategy's mix of the actions' values
    std::vector<double> regret, strategy_sum;  // update at the player's own node: after the update
};

// CFR with alternating updates (Zinkevich et al. 2007) over the public tree, with a vector over
// every hand at each node (Johanson et al. 2011), in one of two variants:
//   cfr+   regret matching+ and an average weighted by iteration t (Tammelin 2014)
//   dcfr   discounted CFR (Brown and Sandholm 2019): positive regrets times t^1.5 / (t^1.5 + 1)
//          and negative ones times 1/2 after each update, and the average weighted by t^2
// Strategies are (actions x hands) row-major: entry [a * n + h] is hand h's chance of action a.
class Solver {
public:
    explicit Solver(Spot spot, const std::string& method = "cfr+");

    const Spot& spot() const { return spot_; }
    const std::vector<std::string>& hands() const { return hands_; }
    const std::vector<Node>& nodes() const { return nodes_; }  // nodes()[0] is the root
    // compatible[i * n + j]: 1 if hands i and j share no card, so both can be dealt at once.
    const std::vector<double>& compatible() const { return compatible_; }
    // showdown[i * n + j]: +1 if hand i beats hand j, -1 if it loses, 0 for a tie or a clash.
    const std::vector<double>& showdown() const { return showdown_; }
    double deal_prob() const { return deal_prob_; }
    int iterations() const { return iterations_; }

    // Regret matching. CFR+ keeps regrets at zero or above already.
    std::vector<double> current(int node) const;
    // The average strategy, which is what converges to equilibrium.
    std::vector<double> average(int node) const;
    void iterate(int iterations);
    // Runs `iterations` more iterations, as iterate does, and returns every step they took.
    std::vector<Step> trace(int iterations);

    // Expected chips per hand for `player` when both play the average strategy.
    double value(int player) const;
    // Expected chips per hand for `player` best-responding to the other's average strategy.
    double best_response(int player) const;
    // Mean gain of the two best responses: OpenSpiel's definition for two-player zero-sum games.
    double exploitability() const;

private:
    int build_tree(const std::string& seq, int player, std::array<int, 2> contrib);
    // Each hand's column divided by its total; uniform where the total is zero.
    std::vector<double> normalise(std::vector<double> weights) const;
    // Counterfactual value of each of `player`'s hands (chance probability left out).
    std::vector<double> terminal(const Child& end, int player, const std::vector<double>& opp_reach) const;
    std::vector<double> cfr(const Child& at, int player, const std::vector<double>& own,
                            const std::vector<double>& opp);
    // `player`'s counterfactual values when both play the average strategy, or when `player`
    // best-responds to the other's average strategy (`best`).
    std::vector<double> evaluate(const Child& at, int player, const std::vector<double>& opp, bool best) const;
    double expected(int player, bool best) const;

    Spot spot_;
    std::vector<std::string> hands_;
    int n_ = 0;
    std::vector<double> compatible_, showdown_;
    std::vector<double> compatible_columns_, showdown_columns_;  // the same, column by column
    double deal_prob_ = 0;  // every hand leaves the same number of opponent hands
    std::vector<Node> nodes_;
    bool dcfr_ = false;
    std::vector<size_t> offset_;  // where each node's rows start in regret_ and strategy_sum_
    std::vector<double> regret_, strategy_sum_;
    int iterations_ = 0;
    std::vector<Step>* steps_ = nullptr;  // where cfr() records its steps while trace() runs
};

// Every two-card hand left once the board is out, in engine/cfr.py's order.
std::vector<std::string> deal_hands(const Spot& spot);

}  // namespace cfr
