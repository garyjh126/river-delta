# hu-solver (working name)

A heads-up poker solver, at its very start. The engine doesn't exist yet. What's here is the
strategy viewer it will feed: a page that walks a game's betting tree and shows how often each
hand takes each action at every decision.

## The viewer

The first viewer covers Mini, a tiny Hold'em game: an 8-card deck (ranks 2 to 5 in clubs and
diamonds), a fixed flop of 3♣4♦5♣, two hole cards each, and one limit betting round (bet 2, at
most one raise). Until our own engine exists, its strategy comes from OpenSpiel's CFR+.

```bash
uv run python scripts/solve_mini.py   # CFR+ reference solve -> build/mini.json
uv run python viewer/build.py         # viewer page -> build/mini.html, open it in a browser
```

`build/mini.artifact.html` is the same page without the document wrapper, for publishing as a
claude.ai artifact.

## Licence

Not chosen yet. Until it is, all rights are reserved.
