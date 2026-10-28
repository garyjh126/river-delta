# river-delta

A heads-up poker solver, at its very start. What's here is the first slice of the engine and the
strategy viewer it feeds: a page that walks a game's betting tree and shows how often each hand
takes each action at every decision, and what each hand and action is worth there.

## The engine

`engine/cfr.py` solves heads-up spots from a fixed board, a flop, a turn or a river, over one or more
streets, limit or no-limit, from each player's starting range. It walks the public betting tree once
per update and carries a vector over every private hand at each node, instead of visiting every
deal separately. Each card dealt between streets is a chance node in that tree: every card is dealt
in turn, and the hands holding it drop out below it. Card removal between the two hands enters only
at the end of the hand, in one pass over the hands sorted by strength on the final board, sorted
once for every showdown on it (Johanson et al. 2011). It has two update rules, both from published
work: CFR+ (Tammelin 2014; Tammelin et al. 2015) and discounted CFR (Brown and Sandholm 2019), on
top of CFR (Zinkevich et al. 2007) and public-tree best response (Johanson et al. 2011). Full
references are in the module's docstring.

```bash
uv run python engine/cfr.py mini c1 c1_nl      # discounted CFR -> build/<game>.engine.json
uv run python engine/cfr.py c1 --method cfr+   # CFR+ instead
uv run python scripts/bench_engine.py          # CFR+ against discounted CFR on C1, limit and no-limit
uv run pytest                                  # ~9 min: the engine against OpenSpiel, the C++ engine against it, the hand evaluator
```

It is checked against OpenSpiel, an independent implementation of the same games, in both betting
types: the same infostates and showdown results, the same iterates as OpenSpiel's CFR+ and
discounted CFR, and the same exploitability for our strategy whether our best response or
OpenSpiel's computes it. On C1 it runs 1,000 iterations in about 1.4 seconds; OpenSpiel's
CFR+ takes about half an hour for the same 1,000 (`scripts/solve_reference.py` prints its timings).

Over two streets the check runs on `mini_turn` and `mini_turn_nl`: Mini's ranks in three suits (12
cards), the flop 3♣4♦5♥ fixed and the turn dealt, limit and no-limit. OpenSpiel stores every deal
and betting line, so that is about as big as it can check (about 2 s an iteration of its CFR+; a
20-card deck takes 90 s). The same infostates, showdowns on every turn, exploitability and EVs. From
zero regrets the iterates part from OpenSpiel's at the second iteration: early strategies are often
pure, so two actions are often worth exactly the same, and which way 5e-15 of rounding goes picks a
strategy. So both start from the same random regrets, where nothing ties, and give the same CFR+
and discounted CFR iterates to 1e-9 (3e-15 seen). OpenSpiel's public cards are sorted, which loses
the order of two dealt cards, so three streets are checked against a walk over every deal instead.

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
| 3,000 iterations | 9.90e-05 | 1.17e-04 |

Both cost the same per iteration. The lead is not steady: on C1 no-limit, discounted CFR's
exploitability rises between 2,000 and 3,000 iterations and ends behind CFR+'s at 3,000. On Mini it
leads after 5,000 iterations (1.1e-05 against 3.1e-05, from `engine/cfr.py mini` with each method).

## The C++ engine

`engine/cpp/` is the same engine in C++, step for step: the same method, with no new features (the
EVs and the viewer's export stay in Python). It is a Python module built with pybind11;
`engine/native.py` builds it with CMake and Ninja when a source has changed, then imports it. It
needs CMake, Ninja and a C++20 compiler.

```python
from engine.cfr import SPOTS
from engine.native import solver

s = solver(SPOTS["c1"], "dcfr")   # same games and methods as engine/cfr.py
s.iterate(1000)
s.exploitability()                # 1.296e-05 chips/hand
```

The Python engine is its answer key. `tests/test_cpp.py` checks, on all four games, C1's river with
and without random ranges, the full-deck river, the two-street games (one with random ranges), a
three-street game and the full-deck turn: the same hands, betting tree and showdown results, with
the hands sorted the same way on every final board; the same hand ranking on 92,504 five-card hands
and 10,000 seven-card ones; and, for both CFR+ and discounted CFR, the same average strategy bit for
bit after each of the first 10 iterations and at checkpoints up to each game's full count, with the
same values, best responses and exploitability to within 1e-12 chips per hand. The full-deck turn
takes the Python engine 27 s an iteration, so its iterates are checked for 2 iterations with one
bet size and one raise size; its full tree is checked on its own.

Bit for bit holds because both engines add up every sum in the same order. At the end of the hand
they don't multiply by matrices over pairs of hands: they make one pass over the hands sorted by
strength and take away the hands that share a card (Johanson et al. 2011), which is O(n) instead
of O(n²). `tests/test_cfr.py` checks those sums against the matrix products. Taking away can leave
1e-16 where the true sum is 0, and regret matching turns a regret of 1e-16 into a pure strategy, so
both engines also count the hands behind each sum and make it exactly 0 where there are none.

The order of a sum matters on Mini: its regrets tie exactly, and CFR+ floors them at 0, so the way
1e-16 of rounding goes picks a strategy. With discounted CFR, two orders end Mini no-limit's 5,000
iterations at different near-equilibrium strategies (exploitability 1.1e-05 against 1.5e-05, same
game value), and on Mini the O(n) sums part from OpenSpiel's iterates within a few iterations. So
the tests check Mini's iterates against OpenSpiel with the matrix products (a test-only subclass,
which rounds as OpenSpiel's sums over deals do), and every other game's with the engine itself.

Speed, from `scripts/bench_cpp.py` (one thread each, ten times each game's own iteration count;
both methods take the same time):

| game | iterations | Python | C++ | C++ is faster by |
|---|---:|---:|---:|---:|
| Mini | 50,000 | 26.4 s | 0.22 s | 122x |
| C1 | 10,000 | 13.7 s | 0.35 s | 39x |
| C1 no-limit | 10,000 | 30.3 s | 0.80 s | 38x |

The pass over sorted hands made the C++ engine 3.7 times faster on C1 (1.3 s before, when its time
went into products over every pair of its 210 hands) and 19 times faster on a full-deck river with
1,081 hands (42 ms an iteration before, 2.2 ms after). Mini, with 10 hands, got a little slower
(0.15 s before). The Python engine got 4 to 5 times slower: numpy makes the same pass as a dozen
small array operations, twice for the counts, where one product over pairs of hands was a single
compiled call. It is the answer key, not the engine to run.

## A full-deck river

`SPOTS["river"]` in `engine/cfr.py` is a river on the full 52-card deck, there to change at will:
the big blind against the button in a single-raised pot, after the big blind called a bet on the
flop and one on the turn. The board is K♠8♦4♣2♥7♠, with 20 big blinds in the pot and 80 behind each,
and bets and raises of a third, three quarters and one and a half pots, plus all-in: 1,081 hands and
60 decision nodes. Each player's starting range is range text (`engine/ranges.py`: items such as
`TT+`, `A2s+`, `KQo:0.5`, `AhKh`), written by hand to fit that line. Nothing solved the flop and
turn to make them, so they are for illustration.

```bash
uv run python engine/cfr.py river --engine c++   # ~3 s -> build/river.engine.json
uv run python scripts/check_river.py            # ~10 s; fetches and builds noambrown/poker_solver into build/ the first time
```

`--engine c++` iterates with the C++ engine, then hands its average strategy to the Python engine
for the EVs and the export. `scripts/check_river.py` checks the spot against noambrown/poker_solver
(MIT), an independent river solver: it builds the same tree, every action and amount the same, and
our best response puts its strategy's exploitability where its own does, which checks our hand
ranking, card removal, ranges, payoffs and best response against an outside implementation. Bet
sizes round halves up, as poker_solver rounds them, so that the trees match. From the script, 1,000
iterations of discounted CFR each:

| | poker_solver | ours |
|---|---:|---:|
| exploitability by poker_solver's best response (chips) | 0.001742 | |
| exploitability by our best response (chips) | 0.001742 | 0.002001 |
| the same, % of the pot | 0.0087% | 0.0100% |
| value to the big blind (chips) | -0.704113 | -0.704279 |

poker_solver's discounted CFR discounts before each update rather than after, so the two runs differ
even in exact arithmetic.

## A full-deck turn

`SPOTS["turn"]` is the river spot's turn, K♠8♦4♣2♥, there to change at will: the button bet a third
of the pot on the flop and the big blind called, leaving 8 big blinds in the pot and 86 behind each,
so a check, a bet of three quarters of the pot and a call make the river spot's pot of 20 with 80
behind. On both streets the players may bet a third, three quarters or one and a half pots, raise by
three quarters, or go all-in: 14,052 decision nodes (36 on the turn, the rest across the 48 rivers)
and 1,128 hands. No flush can come by the river, so the ranges ignore suits; they are written by
hand for the line so far, for illustration.

```bash
uv run python engine/cfr.py turn --engine c++   # ~3.5 min -> build/turn.engine.json (no viewer tree yet)
```

The C++ engine on one thread, discounted CFR, using 0.70 GB at its peak (exploitability as a share
of the 8 bb pot):

| iterations | seconds | exploitability |
|---:|---:|---:|
| 100 | 52 | 1.94% |
| 200 | 107 | 0.51% |
| 300 | 162 | 0.24% |
| 500 | 272 | 0.14% |
| 1,000 | 548 | 0.034% |

## The viewer

The viewer covers five games, each on a fixed board with two hole cards per player:

* **Mini**: an 8-card deck (ranks 2 to 5 in clubs and diamonds), flop 3♣4♦5♣, 10 possible hands.
  One limit betting round (blinds of 1, bet 2, at most one raise).
* **C1**: a 24-card deck (ranks 2 to 7 in all four suits), flop 5♣6♦7♥, 210 possible hands. Same
  betting as Mini.
* **C1 no-limit**: the same cards with 20-chip stacks, pot-size bets and raises, and all-in.
* **River**: the full-deck river spot above, board K♠8♦4♣2♥7♠, 1,081 possible hands, of which each
  player's starting range holds about 125 combos; the grid greys out the rest.
* **Hold'em**: the full 52-card deck, flop K♠9♥5♥, 1,176 possible hands (of the 1,326 starting
  combos, 150 hold a board card). Button against big blind in a single-raised pot, 100 bb deep,
  no-limit flop betting: bets of 33%, 75% and 150% of the pot, one raise, all-in.

Mini, both C1 games and the river show our engine's strategies. Their EV tab shows each hand's
expected value at every decision, and each action's, in chips: what the hand wins from the pot by
the end, minus the chips it puts in after the blinds (on the river, after the pot it starts with),
which count as dead money. The EVs come from the engine (`Solver.evs`); the tests check them
against a walk over every deal and betting line under OpenSpiel's rules, and with ranges against a
walk over every deal weighted by them. **The Hold'em numbers are mock data** from a
hand-strength heuristic (`viewer/mock_holdem.py`), there only to judge the layout at full size;
the page labels them as mock.

```bash
uv run python engine/cfr.py mini c1 c1_nl       # ~8 s   -> build/<game>.engine.json
uv run python engine/cfr.py river --engine c++  # ~3 s   -> build/river.engine.json
uv run python viewer/mock_holdem.py             # ~1 s   -> build/holdem.json (mock)
uv run python viewer/build.py                   # -> build/viewer.html, open it in a browser
```

OpenSpiel's own solves, kept for comparison, come from `scripts/solve_reference.py mini` (~10 s)
and `scripts/solve_reference.py c1` (~30 min).

`build/viewer.artifact.html` is the same page without the document wrapper, for publishing as a
claude.ai artifact.

## Stepping through CFR

Two ways to watch the C++ engine work, both on Mini.

**The replay page.** `Solver::trace` runs iterations like `iterate` and records every step of
every walk down the betting tree: reaching a decision node, ending the hand, and leaving the node
with its values and updated regrets. `viewer/replay.py` records a run and builds a page that steps
through it like a debugger: step, step over, step out, continue to breakpoints set on the tree,
a call stack, and a table of every hand's reach, strategy, values and regrets at each step.
`tests/test_cpp.py` checks that a recorded run ends where an unrecorded one does, with the Python
engine's regrets bit for bit.

```bash
uv run python viewer/replay.py                             # Mini, CFR+, 10 iterations -> build/replay.html
uv run python viewer/replay.py --iterations 30 --method dcfr
```

**A debugger.** `engine/cpp/debug_main.cpp` builds `cfr_debug`, which solves one game from the
command line (`cfr_debug [game] [iterations] [method]`). The `debug` preset in
`engine/cpp/CMakePresets.json` builds it without optimisation into `build/cpp-debug`
(`cmake --preset debug && cmake --build --preset debug`, run in `engine/cpp`). In VS Code, the
"C++ engine: Mini, 1 iteration" launch configuration builds it and runs it under LLDB (CodeLLDB
extension).

## Papers

`docs/references.md` lists the papers behind each technique, with links.

## Licence

MIT; see `LICENSE`.
