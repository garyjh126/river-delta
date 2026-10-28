// Vector-form CFR for heads-up poker from a fixed board over one or more betting rounds: a port of
// engine/cfr.py, which stays the answer key (tests/test_cpp.py). Same method, same references.
#pragma once

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cfr {

// Betting on a fixed board (a flop, a turn or a river) over `streets` rounds, each round after the
// first dealing one more card, as engine/cfr.py's Spot. Both players post `blind` and player 0 acts
// first in every round. Limit when `stack` is unset, no-limit when it is set.
struct Spot {
    std::string ranks;
    std::string suits;
    std::vector<std::string> board;
    int blind = 1;
    int bet = 2;
    int max_raises = 2;
    std::optional<int> stack;
    std::vector<double> pot_sizes = {1.0};
    std::optional<std::vector<double>> raise_sizes;  // no-limit raises; unset, the same as pot_sizes
    // Each player's starting range, as a weight on every hand in deal_hands order; empty for every
    // hand at weight 1.
    std::vector<std::vector<double>> ranges;
    int streets = 1;

    bool nolimit() const { return stack.has_value(); }
};

// Where an action or a dealt card leads: the next decision node, a chance node, or the end of the hand.
struct Child {
    int node = -1;  // index into Solver::nodes(), or -1
    int chance = -1;  // index into Solver::chances(), or -1
    int folder = -1;  // at the end: the player who folded, or -1 for a showdown
    std::array<int, 2> contrib{};  // at the end: chips each player has put in
    int board = -1;  // at a showdown: index into Solver::boards()

    bool end() const { return node < 0 && chance < 0; }
};

// The betting so far (`seq`) is written as OpenSpiel writes it, with each round after the first
// behind the card dealt for it ("rc|3h|r"); `dealt` holds the cards dealt after the fixed board.
struct Node {
    std::string seq;
    int player = 0;
    std::array<int, 2> contrib{};
    std::vector<std::string> actions;  // fold, check, call, bet, raise, allin
    std::vector<Child> children;  // one per action
    std::vector<std::string> dealt;
};

// The end of a round's betting before the last: the next card is dealt, one child per card.
struct Chance {
    std::string seq;
    std::array<int, 2> contrib{};
    std::vector<std::string> dealt;
    std::vector<std::string> cards;  // the cards it can deal, in deck order
    std::vector<Child> children;  // one per card
};

// A final board (the cards dealt after the fixed board, as at a showdown): the hands that hold none
// of them from weakest to strongest, where hands as strong as the one at place k take places
// group_start[k] up to group_end[k]. Sorted once, for every showdown on that board.
struct Board {
    std::vector<std::string> dealt;
    std::vector<int> order, group_start, group_end;
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
// every hand at each node and a chance node for each dealt card (Johanson et al. 2011), and terminal
// nodes in O(n) over hands sorted by strength (the same paper), in one of two variants:
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
    const std::vector<Chance>& chances() const { return chances_; }
    const std::vector<Board>& boards() const { return boards_; }
    // compatible[i * n + j]: 1 if hands i and j share no card, so both can be dealt at once.
    const std::vector<double>& compatible() const { return compatible_; }
    // showdown[i * n + j], on one street (empty on more): +1 if hand i beats hand j, -1 if it loses,
    // 0 for a tie or a clash.
    const std::vector<double>& showdown() const { return showdown_; }
    // ranges()[p][i]: player p's weight on hand i, the reach its hands start the tree with.
    const std::array<std::vector<double>, 2>& ranges() const { return ranges_; }
    // Deal (i, j) has probability ranges()[0][i] * ranges()[1][j] * deal_prob() when the two share no card.
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
    using Walk = std::function<std::vector<double>(const Child&, const std::vector<double>&)>;

    int build_tree(const std::string& seq, int player, std::array<int, 2> contrib,
                   const std::vector<std::string>& dealt);
    // Where a round's betting ends without a fold: a showdown after the last round, or else a chance
    // node dealing the next card, as engine/cfr.py's end_of_round.
    Child end_of_round(const std::string& seq, std::array<int, 2> contrib, const std::vector<std::string>& dealt);
    Board make_board(const std::vector<std::string>& dealt) const;
    // Each hand's column divided by its total; uniform where the total is zero.
    std::vector<double> normalise(std::vector<double> weights) const;
    // compatible() times `reach`, in O(n): engine/cfr.py's _compatible_reach, adding up in the same order.
    std::vector<double> compatible_reach(const std::vector<double>& reach) const;
    // The showdown matrix on boards()[board] times `reach`, in O(n): engine/cfr.py's _showdown_reach,
    // adding up in the same order.
    std::vector<double> showdown_reach(const std::vector<double>& reach, int board) const;
    // Counterfactual value of each of `player`'s hands (chance probability left out).
    std::vector<double> terminal(const Child& end, int player, const std::vector<double>& opp_reach) const;
    // A chance node: each hand's value summed over the cards dealt, from `walk(child, live)` for each
    // card, as engine/cfr.py's _deal.
    std::vector<double> deal(const Chance& chance, const Walk& walk) const;
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
    // What the terminal nodes use instead: each hand's two cards, numbered from 0, with boards_.
    std::vector<std::array<int, 2>> cards_;
    int deck_size_ = 0;
    // live_[c][h]: 1 if hand h doesn't hold card c, so it is still possible once c is dealt.
    std::map<std::string, std::vector<double>> live_;
    std::array<std::vector<double>, 2> ranges_;
    double deal_prob_ = 0;
    std::vector<Node> nodes_;
    std::vector<Chance> chances_;
    std::vector<Board> boards_;
    std::map<std::vector<std::string>, int> board_index_;  // where each final board is in boards_
    bool dcfr_ = false;
    std::vector<size_t> offset_;  // where each node's rows start in regret_ and strategy_sum_
    std::vector<double> regret_, strategy_sum_;
    int iterations_ = 0;
    std::vector<Step>* steps_ = nullptr;  // where cfr() records its steps while trace() runs
};

// The cards not on the board, highest first, in engine/cfr.py's order.
std::vector<std::string> cards_left(const Spot& spot);
// Every two-card hand left once the board is out, in engine/cfr.py's order.
std::vector<std::string> deal_hands(const Spot& spot);

}  // namespace cfr
