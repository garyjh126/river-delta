#include "cfr.hpp"

#include "cards.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace cfr {

std::vector<std::string> deal_hands(const Spot& spot) {
    // Cards ordered by rank, then suit in the order of `spot.suits`, highest first; each hand
    // is written higher card first, as OpenSpiel writes private cards.
    std::vector<std::string> deck;
    for (int i = static_cast<int>(spot.ranks.size()) - 1; i >= 0; --i) {
        for (int j = static_cast<int>(spot.suits.size()) - 1; j >= 0; --j) {
            std::string card{spot.ranks[i], spot.suits[j]};
            if (std::find(spot.board.begin(), spot.board.end(), card) == spot.board.end()) {
                deck.push_back(card);
            }
        }
    }
    std::vector<std::string> hands;
    for (size_t a = 0; a < deck.size(); ++a) {
        for (size_t b = a + 1; b < deck.size(); ++b) {
            hands.push_back(deck[a] + deck[b]);
        }
    }
    return hands;
}

Solver::Solver(Spot spot, const std::string& method) : spot_(std::move(spot)) {
    if (method != "cfr+" && method != "dcfr") {
        throw std::invalid_argument("method must be cfr+ or dcfr");
    }
    dcfr_ = method == "dcfr";
    hands_ = deal_hands(spot_);
    n_ = static_cast<int>(hands_.size());
    std::vector<int> strength;
    for (const auto& h : hands_) {
        std::vector<std::string> cards = spot_.board;
        cards.push_back(h.substr(0, 2));
        cards.push_back(h.substr(2, 2));
        strength.push_back(cards::value(cards));
    }
    compatible_.resize(n_ * n_);
    showdown_.resize(n_ * n_);
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            const auto &a = hands_[i], &b = hands_[j];
            const bool clash = a.compare(0, 2, b, 0, 2) == 0 || a.compare(0, 2, b, 2, 2) == 0 ||
                               a.compare(2, 2, b, 0, 2) == 0 || a.compare(2, 2, b, 2, 2) == 0;
            compatible_[i * n_ + j] = clash ? 0.0 : 1.0;
            const double sign = (strength[i] > strength[j]) - (strength[i] < strength[j]);
            showdown_[i * n_ + j] = sign * compatible_[i * n_ + j];
        }
    }
    if (!spot_.ranges.empty() && spot_.ranges.size() != 2) {
        throw std::invalid_argument("ranges must be empty or hold one range for each player");
    }
    for (int p : {0, 1}) {
        ranges_[p] = spot_.ranges.empty() ? std::vector<double>(n_, 1.0) : spot_.ranges[p];
        if (static_cast<int>(ranges_[p].size()) != n_) {
            throw std::invalid_argument("each range must hold one weight per hand");
        }
    }
    double deals = 0;
    for (int i = 0; i < n_; ++i) {
        for (int j = 0; j < n_; ++j) {
            deals += ranges_[0][i] * compatible_[i * n_ + j] * ranges_[1][j];
        }
    }
    deal_prob_ = 1 / deals;
    std::map<std::string, int> deck;  // ordered, so cards are numbered as engine/cfr.py numbers them
    for (const auto& h : hands_) {
        deck[h.substr(0, 2)] = 0;
        deck[h.substr(2, 2)] = 0;
    }
    for (auto& [card, number] : deck) {
        number = deck_size_++;
    }
    for (const auto& h : hands_) {
        cards_.push_back({deck[h.substr(0, 2)], deck[h.substr(2, 2)]});
    }
    order_.resize(n_);
    std::iota(order_.begin(), order_.end(), 0);
    std::stable_sort(order_.begin(), order_.end(), [&](int a, int b) { return strength[a] < strength[b]; });
    group_start_.resize(n_);
    group_end_.resize(n_);
    for (int k = 0; k < n_;) {
        int end = k;
        while (end < n_ && strength[order_[end]] == strength[order_[k]]) {
            ++end;
        }
        for (int m = k; m < end; ++m) {
            group_start_[m] = k;
            group_end_[m] = end;
        }
        k = end;
    }
    build_tree("", 0, {spot_.blind, spot_.blind});
    size_t size = 0;
    for (const auto& node : nodes_) {
        offset_.push_back(size);
        size += node.actions.size() * n_;
    }
    regret_.assign(size, 0.0);
    strategy_sum_.assign(size, 0.0);
}

int Solver::build_tree(const std::string& seq, int player, std::array<int, 2> contrib) {
    const int index = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{seq, player, contrib, {}, {}});
    const int opp = 1 - player;
    const bool facing = contrib[opp] > contrib[player];
    std::vector<std::string> actions;
    std::vector<Child> children;

    auto raise_to = [&](int to, const std::string& kind) {
        std::array<int, 2> next = contrib;
        next[player] = to;
        actions.push_back(kind);
        const std::string step = spot_.nolimit() ? "r" + std::to_string(to) : "r";
        children.push_back(Child{build_tree(seq + step, opp, next)});
    };

    if (facing) {
        actions.insert(actions.end(), {"fold", "call"});
        children.push_back(Child{-1, player, contrib});
        children.push_back(Child{-1, -1, {contrib[opp], contrib[opp]}});
    } else {
        actions.push_back("check");
        if (!seq.empty() && seq.back() == 'c') {
            children.push_back(Child{-1, -1, contrib});
        } else {
            children.push_back(Child{build_tree(seq + "c", opp, contrib)});
        }
    }
    const std::string kind = facing ? "raise" : "bet";
    if (!spot_.nolimit()) {
        if (std::count(seq.begin(), seq.end(), 'r') < spot_.max_raises) {
            raise_to(contrib[opp] + spot_.bet, kind);
        }
    } else {
        for (double size : spot_.pot_sizes) {
            // Halves round up, as engine/cfr.py rounds them.
            const int to = contrib[opp] + static_cast<int>(std::floor(size * 2 * contrib[opp] + 0.5));
            if (to < *spot_.stack) {
                raise_to(to, kind);
            }
        }
        if (contrib[opp] < *spot_.stack) {
            raise_to(*spot_.stack, "allin");
        }
    }
    // Children were pushed after this node, so look it up again rather than hold a reference.
    nodes_[index].actions = std::move(actions);
    nodes_[index].children = std::move(children);
    return index;
}

std::vector<double> Solver::normalise(std::vector<double> weights) const {
    const size_t actions = weights.size() / n_;
    for (int h = 0; h < n_; ++h) {
        double total = 0;
        for (size_t a = 0; a < actions; ++a) {
            total += weights[a * n_ + h];
        }
        for (size_t a = 0; a < actions; ++a) {
            weights[a * n_ + h] = total > 0 ? weights[a * n_ + h] / total : 1.0 / actions;
        }
    }
    return weights;
}

std::vector<double> Solver::current(int node) const {
    const auto first = regret_.begin() + offset_[node];
    std::vector<double> positive(first, first + nodes_[node].actions.size() * n_);
    for (double& r : positive) {
        r = std::max(r, 0.0);
    }
    return normalise(std::move(positive));
}

std::vector<double> Solver::average(int node) const {
    const auto first = strategy_sum_.begin() + offset_[node];
    return normalise(std::vector<double>(first, first + nodes_[node].actions.size() * n_));
}

// Taking away can leave 1e-16 where the true sum is 0, and regret matching turns a regret of 1e-16
// into a pure strategy. So both sums below also count the hands with reach above 0, the same way,
// and are exactly 0 where there are none.

std::vector<double> Solver::compatible_reach(const std::vector<double>& reach) const {
    double total = 0;
    int count = 0;
    std::vector<double> per_card(deck_size_, 0.0);
    std::vector<int> per_card_count(deck_size_, 0);
    for (int h = 0; h < n_; ++h) {
        const int some = reach[h] > 0;
        total += reach[h];
        count += some;
        for (int c : cards_[h]) {
            per_card[c] += reach[h];
            per_card_count[c] += some;
        }
    }
    // Every hand, less those holding either card; the hand itself holds both, so add it back once.
    std::vector<double> out(n_);
    for (int h = 0; h < n_; ++h) {
        const auto [a, b] = cards_[h];
        const bool some = count - per_card_count[a] - per_card_count[b] + (reach[h] > 0) > 0;
        out[h] = some ? total - per_card[a] - per_card[b] + reach[h] : 0.0;
    }
    return out;
}

std::vector<double> Solver::showdown_reach(const std::vector<double>& reach) const {
    std::vector<double> out(n_);
    std::vector<double> per_card(deck_size_);
    std::vector<int> per_card_count(deck_size_);
    double total;
    int count;
    auto start = [&] {
        total = 0;
        count = 0;
        std::fill(per_card.begin(), per_card.end(), 0.0);
        std::fill(per_card_count.begin(), per_card_count.end(), 0);
    };
    auto add = [&](int h) {
        const int some = reach[h] > 0;
        total += reach[h];
        count += some;
        for (int c : cards_[h]) {
            per_card[c] += reach[h];
            per_card_count[c] += some;
        }
    };
    // The reach so far, less the hands holding one of hand h's cards (none holds both).
    auto others = [&](int h) {
        const auto [a, b] = cards_[h];
        return count - per_card_count[a] - per_card_count[b] > 0 ? total - per_card[a] - per_card[b] : 0.0;
    };
    // The weaker hands, from the weakest up: at each group of equally strong hands, the running sums
    // hold every hand below it.
    start();
    for (int k = 0; k < n_; k = group_end_[k]) {
        for (int m = k; m < group_end_[k]; ++m) {
            out[order_[m]] = others(order_[m]);
        }
        for (int m = k; m < group_end_[k]; ++m) {
            add(order_[m]);
        }
    }
    // Less the stronger hands, from the strongest down.
    start();
    for (int k = n_ - 1; k >= 0; k = group_start_[k] - 1) {
        for (int m = group_start_[k]; m <= k; ++m) {
            out[order_[m]] -= others(order_[m]);
        }
        for (int m = k; m >= group_start_[k]; --m) {
            add(order_[m]);
        }
    }
    return out;
}

std::vector<double> Solver::terminal(const Child& end, int player, const std::vector<double>& opp_reach) const {
    std::vector<double> out;
    double scale;
    if (end.folder < 0) {
        out = showdown_reach(opp_reach);
        scale = end.contrib[0];
    } else {
        out = compatible_reach(opp_reach);
        scale = (end.folder == player ? -1 : 1) * end.contrib[end.folder];
    }
    for (double& x : out) {
        x = scale * x;
    }
    return out;
}

std::vector<double> Solver::cfr(const Child& at, int player, const std::vector<double>& own,
                                const std::vector<double>& opp) {
    if (at.node < 0) {
        return terminal(at, player, opp);
    }
    const Node& node = nodes_[at.node];
    const size_t actions = node.actions.size();
    const std::vector<double> sigma = current(at.node);
    if (steps_) {
        steps_->push_back({.kind = Step::Kind::visit, .iteration = iterations_, .player = player, .node = at.node,
                           .own = own, .opp = opp, .strategy = sigma});
    }
    std::vector<double> reach(n_);
    if (node.player != player) {
        std::vector<double> total(n_, 0.0);
        std::vector<double> values;  // kept for the trace only
        for (size_t a = 0; a < actions; ++a) {
            for (int h = 0; h < n_; ++h) {
                reach[h] = opp[h] * sigma[a * n_ + h];
            }
            const std::vector<double> v = cfr(node.children[a], player, own, reach);
            for (int h = 0; h < n_; ++h) {
                total[h] += v[h];
            }
            if (steps_) {
                if (node.children[a].node < 0) {
                    steps_->push_back({.kind = Step::Kind::end, .iteration = iterations_, .player = player,
                                       .node = at.node, .action = static_cast<int>(a), .own = own, .opp = reach,
                                       .values = v});
                }
                values.insert(values.end(), v.begin(), v.end());
            }
        }
        if (steps_) {
            steps_->push_back({.kind = Step::Kind::update, .iteration = iterations_, .player = player,
                               .node = at.node, .own = own, .opp = opp, .strategy = sigma, .values = values,
                               .value = total});
        }
        return total;
    }
    std::vector<double> values(actions * n_);
    for (size_t a = 0; a < actions; ++a) {
        for (int h = 0; h < n_; ++h) {
            reach[h] = own[h] * sigma[a * n_ + h];
        }
        const std::vector<double> v = cfr(node.children[a], player, reach, opp);
        std::copy(v.begin(), v.end(), values.begin() + a * n_);
        if (steps_ && node.children[a].node < 0) {
            steps_->push_back({.kind = Step::Kind::end, .iteration = iterations_, .player = player, .node = at.node,
                               .action = static_cast<int>(a), .own = reach, .opp = opp, .values = v});
        }
    }
    std::vector<double> v(n_, 0.0);
    for (size_t a = 0; a < actions; ++a) {
        for (int h = 0; h < n_; ++h) {
            v[h] += sigma[a * n_ + h] * values[a * n_ + h];
        }
    }
    // Same order of operations as engine/cfr.py, so each step rounds as numpy's does.
    const double t = iterations_;
    const double up = std::pow(t, 1.5) / (std::pow(t, 1.5) + 1);
    const double weight = dcfr_ ? t * t : t;
    double* regret = regret_.data() + offset_[at.node];
    double* strategy_sum = strategy_sum_.data() + offset_[at.node];
    for (size_t a = 0; a < actions; ++a) {
        for (int h = 0; h < n_; ++h) {
            const size_t k = a * n_ + h;
            const double r = regret[k] + values[k] - v[h];
            regret[k] = dcfr_ ? r * (r >= 0 ? up : 0.5) : std::max(r, 0.0);
            strategy_sum[k] += weight * own[h] * sigma[k];
        }
    }
    if (steps_) {
        const size_t size = actions * n_;
        steps_->push_back({.kind = Step::Kind::update, .iteration = iterations_, .player = player, .node = at.node,
                           .own = own, .opp = opp, .strategy = sigma, .values = values, .value = v,
                           .regret = std::vector<double>(regret, regret + size),
                           .strategy_sum = std::vector<double>(strategy_sum, strategy_sum + size)});
    }
    return v;
}

void Solver::iterate(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        ++iterations_;
        for (int player : {0, 1}) {
            cfr(Child{0}, player, ranges_[player], ranges_[1 - player]);
        }
    }
}

std::vector<Step> Solver::trace(int iterations) {
    std::vector<Step> steps;
    steps_ = &steps;
    iterate(iterations);
    steps_ = nullptr;
    return steps;
}

std::vector<double> Solver::evaluate(const Child& at, int player, const std::vector<double>& opp,
                                     bool best) const {
    if (at.node < 0) {
        return terminal(at, player, opp);
    }
    const Node& node = nodes_[at.node];
    const std::vector<double> sigma = average(at.node);
    if (node.player != player) {
        std::vector<double> total(n_, 0.0), reach(n_);
        for (size_t a = 0; a < node.actions.size(); ++a) {
            for (int h = 0; h < n_; ++h) {
                reach[h] = opp[h] * sigma[a * n_ + h];
            }
            const std::vector<double> v = evaluate(node.children[a], player, reach, best);
            for (int h = 0; h < n_; ++h) {
                total[h] += v[h];
            }
        }
        return total;
    }
    std::vector<double> out(n_, best ? -std::numeric_limits<double>::infinity() : 0.0);
    for (size_t a = 0; a < node.actions.size(); ++a) {
        const std::vector<double> v = evaluate(node.children[a], player, opp, best);
        for (int h = 0; h < n_; ++h) {
            out[h] = best ? std::max(out[h], v[h]) : out[h] + sigma[a * n_ + h] * v[h];
        }
    }
    return out;
}

double Solver::expected(int player, bool best) const {
    const std::vector<double> v = evaluate(Child{0}, player, ranges_[1 - player], best);
    double total = 0;
    for (int h = 0; h < n_; ++h) {
        total += ranges_[player][h] * v[h];
    }
    return deal_prob_ * total;
}

double Solver::value(int player) const { return expected(player, false); }

double Solver::best_response(int player) const { return expected(player, true); }

double Solver::exploitability() const { return (best_response(0) + best_response(1)) / 2; }

}  // namespace cfr
