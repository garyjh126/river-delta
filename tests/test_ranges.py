import pytest

from engine.cards import RANKS
from engine.cfr import Spot, deal_hands
from engine.ranges import weights

DECK = Spot(ranks=RANKS, suits="cdhs", board=(), iterations=0)
RIVER = Spot(ranks=RANKS, suits="cdhs", board=("Ks", "8d", "4c", "2h", "7s"), iterations=0)


def combos(text: str, spot: Spot = DECK) -> float:
    return weights(text, deal_hands(spot)).sum()


def test_combo_counts():
    assert combos("AA") == 6
    assert combos("AKs") == 4
    assert combos("AKo") == 12
    assert combos("AK") == combos("KA") == 16
    assert combos("TT+") == 5 * 6
    assert combos("A2s+") == 12 * 4
    assert combos("K9o+") == 4 * 12  # K9o, KTo, KJo, KQo
    assert combos("AhKh") == combos("KhAh") == 1
    assert combos("22+, A2+, K2+") == 13 * 6 + 12 * 16 + 11 * 16


def test_weights_and_overrides():
    hands = deal_hands(DECK)
    w = weights("AK:0.5, AKs", hands)
    assert w.sum() == 12 * 0.5 + 4
    assert w[hands.index("AsKs")] == 1 and w[hands.index("AsKh")] == 0.5
    assert combos("QQ, QQ:0") == 0


def test_board_blocks_hands():
    assert combos("KK", RIVER) == 3
    assert combos("K8s", RIVER) == 2  # K8 of clubs and of hearts
    assert len(deal_hands(RIVER)) == 47 * 46 // 2


@pytest.mark.parametrize("text", ["AX", "AAs", "AKx", "AK:lots", "Ks8d"])
def test_bad_items_raise(text):
    with pytest.raises(ValueError):
        weights(text, deal_hands(RIVER))
