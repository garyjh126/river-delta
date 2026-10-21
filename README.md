# hu-solver (working name)

A heads-up poker solver, at its very start. The engine doesn't exist yet. What's here is the
strategy viewer it will feed: a page that walks a game's betting tree and shows how often each
hand takes each action at every decision.

## The viewer

The viewer covers two tiny Hold'em games. Both have a fixed flop, two hole cards each, and one
limit betting round (blinds of 1, bet 2, at most one raise):

* **Mini**: an 8-card deck (ranks 2 to 5 in clubs and diamonds), flop 3♣4♦5♣, 10 possible hands.
* **C1**: a 24-card deck (ranks 2 to 7 in all four suits), flop 5♣6♦7♥, 210 possible hands.

Until our own engine exists, the strategies come from OpenSpiel's CFR+.

```bash
uv run python scripts/solve_reference.py mini   # ~10 s  -> build/mini.json
uv run python scripts/solve_reference.py c1     # ~30 min -> build/c1.json
uv run python viewer/build.py                   # -> build/viewer.html, open it in a browser
```

`build/viewer.artifact.html` is the same page without the document wrapper, for publishing as a
claude.ai artifact.

## Licence

Not chosen yet. Until it is, all rights are reserved.
