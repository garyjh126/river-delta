#include "cards.hpp"

#include <algorithm>
#include <array>
#include <string_view>
#include <utility>

namespace cards {

namespace {

int rank_of(const std::string& card) {
    return static_cast<int>(std::string_view("23456789TJQKA").find(card[0])) + 2;
}

// High card of the best straight these ranks contain (an ace plays high or low), else 0.
int straight_high(const std::array<int, 15>& count) {
    for (int hi = 14; hi >= 5; --hi) {
        bool all = true;
        for (int r = hi - 4; r <= hi; ++r) {
            all = all && count[r == 1 ? 14 : r] > 0;
        }
        if (all) {
            return hi;
        }
    }
    return 0;
}

int pack(int category, const std::vector<int>& ranks) {
    int v = category;
    for (size_t i = 0; i < 5; ++i) {
        v = v << 4 | (i < ranks.size() ? ranks[i] : 0);
    }
    return v;
}

}  // namespace

int value(const std::vector<std::string>& cards) {
    std::array<int, 15> count{};
    std::vector<int> ranks;  // high to low
    for (const auto& card : cards) {
        ++count[rank_of(card)];
        ranks.push_back(rank_of(card));
    }
    std::sort(ranks.rbegin(), ranks.rend());
    // Groups of equal rank, larger groups first, then higher ranks.
    std::vector<std::pair<int, int>> groups;  // (size, rank)
    for (int r = 14; r >= 2; --r) {
        if (count[r]) {
            groups.emplace_back(count[r], r);
        }
    }
    std::stable_sort(groups.begin(), groups.end(), [](auto a, auto b) { return a.first > b.first; });
    std::vector<int> shape, by_group;
    for (auto [size, rank] : groups) {
        shape.push_back(size);
        by_group.push_back(rank);
    }
    const bool flush = std::all_of(cards.begin(), cards.end(), [&](const auto& c) { return c[1] == cards[0][1]; });
    const int straight = groups.size() == 5 ? straight_high(count) : 0;
    using S = std::vector<int>;
    if (straight && flush) return pack(8, {straight});
    if (shape == S{4, 1}) return pack(7, by_group);
    if (shape == S{3, 2}) return pack(6, by_group);
    if (flush) return pack(5, ranks);
    if (straight) return pack(4, {straight});
    if (shape == S{3, 1, 1}) return pack(3, by_group);
    if (shape == S{2, 2, 1}) return pack(2, by_group);
    if (shape == S{2, 1, 1, 1}) return pack(1, by_group);
    return pack(0, ranks);
}

}  // namespace cards
