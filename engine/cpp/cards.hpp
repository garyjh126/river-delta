// Five-card hand values, as engine/cards.py's `value`.
#pragma once

#include <string>
#include <vector>

namespace cards {

// Comparable value of exactly five cards such as "Ah": the category (0 high card .. 8 straight
// flush) and then its tie-break ranks, four bits each, so a larger value is a better hand.
int value(const std::vector<std::string>& cards);

}  // namespace cards
