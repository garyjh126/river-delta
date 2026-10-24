# hu-solver (working name)

A heads-up poker solver, at its very start. What's here is the first slice of the engine and the
strategy viewer it feeds: a page that walks a game's betting tree and shows how often each hand
takes each action at every decision, and what each hand and action is worth there.

## The engine

`engine/cfr.py` solves one-street games on a fixed flop, limit or no-limit. It walks the public
betting tree once per update and carries a vector over every private hand at each node, instead
of visiting every deal separately; card removal enters only at the end of the hand, through
matrices over pairs of hands. It has two update rules, both from published work: CFR+ (Tammelin
2014; Tammelin et al. 2015) and discounted CFR (Brown and Sandholm 2019), on top of CFR
(Zinkevich et al. 2007) and public-tree best response (Johanson et al. 2011). Full references are
in the module's docstring.

```bash
uv run python engine/cfr.py mini c1 c1_nl      # discounted CFR -> build/<game>.engine.json
uv run python engine/cfr.py c1 --method cfr+   # CFR+ instead
uv run python scripts/bench_engine.py          # CFR+ against discounted CFR on C1, limit and no-limit
uv run pytest                                  # ~1.5 min: the engine against OpenSpiel, and the hand evaluator
```

It is checked against OpenSpiel, an independent implementation of the same games, in both betting
types: the same infostates and showdown results, the same iterates as OpenSpiel's CFR+ and
discounted CFR, and the same exploitability for our strategy whether our best response or
OpenSpiel's computes it. On C1 it runs 1,000 iterations in a fraction of a second; OpenSpiel's
CFR+ takes about half an hour for the same 1,000 (`scripts/solve_reference.py` prints its timings).

Discounted CFR is the default because it converges faster on these games. From
`scripts/bench_engine.py` (exploitability in chips per hand; times vary by machine):

| C1, limit | CFR+ | discounted CFR |
|---:|---:|---:|
| 100 iterations | 4.05e-03 | 9.77e-04 |
| 300 iterations | 5.39e-04 | 5.23e-05 |
| 1,000 iterations | 6.14e-05 | 1.30e-05 |
| 3,000 iterations | 1.44e-05 | 4.98e-06 |

| C1, no-limit | CFR+ | discounted CFR |
|---:|---:|---:|
| 100 iterations | 1.53e-02 | 3.43e-03 |
| 300 iterations | 3.01e-03 | 5.74e-04 |
| 1,000 iterations | 5.12e-04 | 9.78e-05 |
| 3,000 iterations | 9.90e-05 | 2.78e-05 |

Both cost the same per iteration. The lead is not universal: on Mini, CFR+ is slightly ahead after
5,000 iterations (1.6e-05 against 2.2e-05, from `engine/cfr.py mini` with each method).

## The viewer

The viewer covers four games, each on a fixed flop with two hole cards per player:

* **Mini**: an 8-card deck (ranks 2 to 5 in clubs and diamonds), flop 3♣4♦5♣, 10 possible hands.
  One limit betting round (blinds of 1, bet 2, at most one raise).
* **C1**: a 24-card deck (ranks 2 to 7 in all four suits), flop 5♣6♦7♥, 210 possible hands. Same
  betting as Mini.
* **C1 no-limit**: the same cards with 20-chip stacks, pot-size bets and raises, and all-in.
* **Hold'em**: the full 52-card deck, flop K♠9♥5♥, 1,176 possible hands (of the 1,326 starting
  combos, 150 hold a board card). Button against big blind in a single-raised pot, 100 bb deep,
  no-limit flop betting: bets of 33%, 75% and 150% of the pot, one raise, all-in.

Mini and both C1 games show our engine's strategies. Their EV tab shows each hand's expected value
at every decision, and each action's, in chips: what the hand wins from the pot by the end, minus
the chips it puts in after the blinds, which count as dead money. The EVs come from the engine
(`Solver.evs`); the tests check them against a walk over every deal and betting line under
OpenSpiel's rules. **The Hold'em numbers are mock data** from a
hand-strength heuristic (`viewer/mock_holdem.py`), there only to judge the layout at full size;
the page labels them as mock.

```bash
uv run python engine/cfr.py mini c1 c1_nl       # ~2 s   -> build/<game>.engine.json
uv run python viewer/mock_holdem.py             # ~1 s   -> build/holdem.json (mock)
uv run python viewer/build.py                   # -> build/viewer.html, open it in a browser
```

OpenSpiel's own solves, kept for comparison, come from `scripts/solve_reference.py mini` (~10 s)
and `scripts/solve_reference.py c1` (~30 min).

`build/viewer.artifact.html` is the same page without the document wrapper, for publishing as a
claude.ai artifact.

## Licence

Not chosen yet. Until it is, all rights are reserved.
