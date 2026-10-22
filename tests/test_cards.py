from engine.cards import describe, draws, value


def test_categories_rank_in_order():
    hands = [
        ["9h", "Th", "Jh", "Qh", "Kh"],  # straight flush
        ["9c", "9d", "9h", "9s", "2c"],  # four of a kind
        ["9c", "9d", "9h", "2s", "2c"],  # full house
        ["2h", "7h", "9h", "Jh", "Kh"],  # flush
        ["9c", "Td", "Jh", "Qs", "Kc"],  # straight
        ["9c", "9d", "9h", "2s", "3c"],  # three of a kind
        ["9c", "9d", "2h", "2s", "3c"],  # two pair
        ["9c", "9d", "2h", "4s", "3c"],  # pair
        ["9c", "Kd", "2h", "4s", "3c"],  # high card
    ]
    values = [value(h) for h in hands]
    assert [v[0] for v in values] == [8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert values == sorted(values, reverse=True)


def test_ace_plays_low_and_high_in_straights():
    assert value(["Ah", "2c", "3d", "4s", "5h"]) == (4, 5)
    assert value(["Ts", "Jh", "Qd", "Kc", "Ah"]) == (4, 14)
    assert value(["Ts", "Jh", "Qd", "Kc", "Ah"]) > value(["9c", "Td", "Jh", "Qs", "Kc"]) > value(["Ah", "2c", "3d", "4s", "5h"])


def test_kickers_break_ties():
    assert value(["Kc", "Kd", "Ah", "4s", "3c"]) > value(["Kh", "Ks", "Qh", "4c", "3d"])
    assert value(["Kc", "Kd", "9h", "9s", "Ac"]) > value(["Kh", "Ks", "9c", "9d", "Qc"])
    assert value(["Kc", "Kd", "9h", "9s", "Ac"]) == value(["Kh", "Ks", "9c", "9d", "Ad"])


def test_describe_and_draws_on_a_flop():
    board = ["Ks", "9h", "5h"]
    assert describe(board + ["Ah", "Kh"], with_draws=True) == "pair of kings, flush draw"
    assert describe(board + ["Kc", "9c"]) == "two pair, kings and nines"
    assert describe(board + ["7h", "6h"], with_draws=True) == "king high, flush draw, gutshot"
    assert draws(["8c", "7d", "6h", "2s", "Kc"]) == []  # three to a straight is no draw
    assert draws(["8c", "7d", "6h", "5s", "Kc"]) == ["open-ended straight draw"]
