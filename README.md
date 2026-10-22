# hu-solver (working name)

A heads-up poker solver, at its very start. What's here is the first slice of the engine and the
strategy viewer it feeds: a page that walks a game's betting tree and shows how often each hand
takes each action at every decision.

## The engine

`engine/limit_cfr.py` solves one-street limit games on a fixed flop with CFR+. It walks the
public betting tree once per update and carries a vector over every private hand at each node,
instead of visiting every deal separately; card removal enters only at the end of the hand,
through matrices over pairs of hands. The method follows published work: CFR (Zinkevich et al.
2007), CFR+ (Tammelin 2014; Tammelin et al. 2015) and public-tree best response (Johanson et al.
2011). Full references are in the module's docstring.

```bash
uv run python engine/limit_cfr.py mini c1   # -> build/mini.engine.json, build/c1.engine.json
uv run pytest                               # ~30 s: the engine against OpenSpiel, and the hand evaluator
```

It is checked against OpenSpiel, an independent implementation of the same games: the same
infostates and showdown results, the same CFR+ iterates, and the same exploitability for our
strategy whether our best response or OpenSpiel's computes it. On C1 it runs 1,000 iterations in
a fraction of a second; OpenSpiel's CFR+ takes about half an hour for the same 1,000
(`scripts/solve_reference.py` prints both runs' timings).

## The viewer

The viewer covers three games, each on a fixed flop with two hole cards per player:

* **Mini**: an 8-card deck (ranks 2 to 5 in clubs and diamonds), flop 3♣4♦5♣, 10 possible hands.
  One limit betting round (blinds of 1, bet 2, at most one raise).
* **C1**: a 24-card deck (ranks 2 to 7 in all four suits), flop 5♣6♦7♥, 210 possible hands. Same
  betting as Mini.
* **Hold'em**: the full 52-card deck, flop K♠9♥5♥, 1,176 possible hands (of the 1,326 starting
  combos, 150 hold a board card). Button against big blind in a single-raised pot, 100 bb deep,
  no-limit flop betting: bets of 33%, 75% and 150% of the pot, one raise, all-in.

Mini and C1 show our engine's strategies. **The Hold'em numbers are mock data** from a
hand-strength heuristic (`viewer/mock_holdem.py`), there only to judge the layout at full size;
the page labels them as mock.

```bash
uv run python engine/limit_cfr.py mini c1       # ~1 s   -> build/mini.engine.json, build/c1.engine.json
uv run python viewer/mock_holdem.py             # ~1 s   -> build/holdem.json (mock)
uv run python viewer/build.py                   # -> build/viewer.html, open it in a browser
```

OpenSpiel's own solves, kept for comparison, come from `scripts/solve_reference.py mini` (~10 s)
and `scripts/solve_reference.py c1` (~30 min).

`build/viewer.artifact.html` is the same page without the document wrapper, for publishing as a
claude.ai artifact.

## Licence

Not chosen yet. Until it is, all rights are reserved.
