#include "cfr.hpp"

#include "cards.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cfr {

namespace {

// out[i] = sum over j of m[i * n + j] * v[j]
std::vector<double> times(const std::vector<double>& m, const std::vector<double>& v) {
    const size_t n = v.size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0;
        for (size_t j = 0; j < n; ++j) {
            sum += m[i * n + j] * v[j];
        }
        out[i] = sum;
    }
    return out;
}

}  // namespace

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
        std::vector<std::string> five = spot_.board;
        five.push_back(h.substr(0, 2));
        five.push_back(h.substr(2, 2));
        strength.push_back(cards::value(five));
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
    double opponents = 0;
    for (int j = 0; j < n_; ++j) {
        opponents += compatible_[j];
    }
    deal_prob_ = 1 / (n_ * opponents);
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
            // nearbyint rounds halves to even, as Python's round does.
            const int to = contrib[opp] + static_cast<int>(std::nearbyint(size * 2 * contrib[opp]));
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

std::vector<double> Solver::terminal(const Child& end, int player, const std::vector<double>& opp_reach) const {
    std::vector<double> out;
    double scale;
    if (end.folder < 0) {
        out = times(showdown_, opp_reach);
        scale = end.contrib[0];
    } else {
        out = times(compatible_, opp_reach);
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
    std::vector<double> reach(n_);
    if (node.player != player) {
        std::vector<double> total(n_, 0.0);
        for (size_t a = 0; a < actions; ++a) {
            for (int h = 0; h < n_; ++h) {
                reach[h] = opp[h] * sigma[a * n_ + h];
            }
            const std::vector<double> v = cfr(node.children[a], player, own, reach);
            for (int h = 0; h < n_; ++h) {
                total[h] += v[h];
            }
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
    return v;
}

void Solver::iterate(int iterations) {
    const std::vector<double> ones(n_, 1.0);
    for (int i = 0; i < iterations; ++i) {
        ++iterations_;
        for (int player : {0, 1}) {
            cfr(Child{0}, player, ones, ones);
        }
    }
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
    double total = 0;
    for (double v : evaluate(Child{0}, player, std::vector<double>(n_, 1.0), best)) {
        total += v;
    }
    return deal_prob_ * total;
}

double Solver::value(int player) const { return expected(player, false); }

double Solver::best_response(int player) const { return expected(player, true); }

double Solver::exploitability() const { return (best_response(0) + best_response(1)) / 2; }

}  // namespace cfr
