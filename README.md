# hu-solver (working name)

A heads-up poker solver, at its very start. The engine doesn't exist yet. What's here is the
strategy viewer it will feed: a page that walks a game's betting tree and shows how often each
hand takes each action at every decision.

## The viewer

The viewer covers three games, each on a fixed flop with two hole cards per player:

* **Mini**: an 8-card deck (ranks 2 to 5 in clubs and diamonds), flop 3♣4♦5♣, 10 possible hands.
  One limit betting round (blinds of 1, bet 2, at most one raise).
* **C1**: a 24-card deck (ranks 2 to 7 in all four suits), flop 5♣6♦7♥, 210 possible hands. Same
  betting as Mini.
* **Hold'em**: the full 52-card deck, flop K♠9♥5♥, 1,176 possible hands (of the 1,326 starting
  combos, 150 hold a board card). Button against big blind in a single-raised pot, 100 bb deep,
  no-limit flop betting: bets of 33%, 75% and 150% of the pot, one raise, all-in.

Until our own engine exists, Mini and C1 are solved with OpenSpiel's CFR+. **The Hold'em numbers
are mock data** from a hand-strength heuristic (`viewer/mock_holdem.py`), there only to judge the
layout at full size; the page labels them as mock.

```bash
uv run python scripts/solve_reference.py mini   # ~10 s  -> build/mini.json
uv run python scripts/solve_reference.py c1     # ~30 min -> build/c1.json
uv run python viewer/mock_holdem.py             # ~1 s   -> build/holdem.json (mock)
uv run python viewer/build.py                   # -> build/viewer.html, open it in a browser
```

`build/viewer.artifact.html` is the same page without the document wrapper, for publishing as a
claude.ai artifact.

## Licence

Not chosen yet. Until it is, all rights are reserved.
