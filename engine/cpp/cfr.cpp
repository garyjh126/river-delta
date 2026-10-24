#include "cfr.hpp"

#include <algorithm>
#include <cmath>

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

Solver::Solver(Spot spot) : spot_(std::move(spot)) {
    hands_ = deal_hands(spot_);
    build_tree("", 0, {spot_.blind, spot_.blind});
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

}  // namespace cfr
