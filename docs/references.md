# References

The papers this solver is built from, or will be. Every technique in the code cites one of
them. Copies for reading go in `docs/papers/` under the file names below. Git ignores that folder,
because the copyright belongs to the publishers or authors, so fetch them from the links.

## In the engine now

**Zinkevich, Johanson, Bowling, Piccione (2007).** Regret Minimization in Games with Incomplete
Information. NIPS 2007. [PDF](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf) ·
`zinkevich2007-cfr.pdf`
CFR itself: regret at each infostate, regret matching, and why the average strategy converges to
equilibrium in two-player zero-sum games. `Solver::cfr` in `engine/cpp/cfr.cpp` and `engine/cfr.py`.

**Johanson, Waugh, Bowling, Zinkevich (2011).** Accelerating Best Response Calculation in Large
Extensive Games. IJCAI 2011.
[PDF](https://poker.cs.ualberta.ca/publications/ijcai2011_accelerated_best_response.pdf) ·
`johanson2011-accelerating-best-response.pdf`
Walking the public betting tree once with a vector over every hand, which the whole engine does,
and the best response behind our exploitability. Its public tree has a chance node for each public
card, whose value is the sum of its children's; that is how the engine deals the turn and river
(`deal` in `engine/cpp/cfr.cpp`, `_deal` in `engine/cfr.py`). Its section on terminal nodes is how
the engine settles showdowns and folds: one pass over the hands sorted by strength, with the hands
that share a card taken away by inclusion-exclusion, O(n) instead of O(n²) (`showdown_reach` and
`compatible_reach` in `engine/cpp/cfr.cpp`, `_showdown_reach` and `_compatible_reach` in
`engine/cfr.py`), with the hands sorted once per final board for every showdown on it, as it
suggests (`boards`). Its section on parallel computation, solving the subtrees below a chance node
at once, is the way to threads.

**Tammelin (2014).** Solving Large Imperfect Information Games Using CFR+.
[arXiv:1407.5042](https://arxiv.org/abs/1407.5042) · `tammelin2014-cfr-plus.pdf`
CFR+: regrets floored at zero, and an average weighted by iteration. The `cfr+` method.

**Tammelin, Burch, Johanson, Bowling (2015).** Solving Heads-up Limit Texas Hold'em. IJCAI 2015.
[PDF](https://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf) ·
`tammelin2015-solving-hulhe.pdf`
CFR+ with alternating updates, as the engine runs it, and the engineering used to solve
heads-up limit Hold'em with it.

**Brown, Sandholm (2019).** Solving Imperfect-Information Games via Discounted Regret
Minimization. AAAI 2019. [arXiv:1809.04040](https://arxiv.org/abs/1809.04040) ·
`brown2019-discounted-cfr.pdf`
Discounted CFR, the engine's default method (`dcfr`).

## Next on the roadmap

**Johanson, Bard, Lanctot, Gibson, Bowling (2012).** Efficient Nash Equilibrium Approximation
through Monte Carlo Counterfactual Regret Minimization. AAMAS 2012.
[PDF](https://poker.cs.ualberta.ca/publications/AAMAS12-pcs.pdf) ·
`johanson2012-public-chance-sampling.pdf`
Public chance sampling: deal the public cards (turn, river) by sampling while keeping the vectors
over private hands. The engine deals every card instead, which is the same walk without the
sampling (exact iterates, checkable against OpenSpiel); sampling is a speed-up for later.

**Waugh (2013).** A Fast and Optimal Hand Isomorphism Algorithm. AAAI 2013 Workshop on Computer
Poker and Imperfect Information.
[PDF](https://www.cs.cmu.edu/~waugh/publications/isomorphism13.pdf) ·
`waugh2013-hand-isomorphism.pdf`
Treating boards and hands that differ only by a swap of suits as one. Needed once boards are no
longer fixed.

**Johanson, Burch, Valenzano, Bowling (2013).** Evaluating State-Space Abstractions in
Extensive-Form Games. AAMAS 2013.
[PDF](https://poker.cs.ualberta.ca/publications/AAMAS13-abstraction.pdf) ·
`johanson2013-evaluating-abstractions.pdf`
How to compare card abstractions (bucketing) by how exploitable their strategies are in the real
game, and how the common bucketing methods compare.

**Ganzfried, Sandholm (2014).** Potential-Aware Imperfect-Recall Abstraction with Earth Mover's
Distance in Imperfect-Information Games. AAAI 2014.
[PDF](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf) ·
`ganzfried2014-potential-aware-abstraction.pdf`
Bucketing hands by how their strength can change on later cards, not only by their strength now.

## Solving part of the game at a time

**Burch, Johanson, Bowling (2014).** Solving Imperfect Information Games Using Decomposition.
AAAI 2014. [PDF](https://poker.cs.ualberta.ca/publications/aaai2014-cfrd.pdf) ·
`burch2014-cfr-d.pdf`
CFR-D: solve the start of the game, then each later part separately, and re-solve parts on
demand without becoming exploitable.

**Moravčík et al. (2017).** DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit
Poker. Science 356. [arXiv:1701.01724](https://arxiv.org/abs/1701.01724) ·
`moravcik2017-deepstack.pdf`
Continual re-solving: solve a few actions ahead at each decision, with a learned estimate of each
hand's value beyond that.

## Background

**Bowling, Burch, Johanson, Tammelin (2015).** Heads-up Limit Hold'em Poker is Solved. Science 347.
[Summary page](https://poker.cs.ualberta.ca/15science.html). Paywalled, so not in `docs/papers/`;
Tammelin et al. (2015) above is its technical companion.

**Johanson (2016).** Robust Strategies and Counter-Strategies: From Superhuman to Optimal Play.
PhD thesis, University of Alberta.
[PDF](https://poker.cs.ualberta.ca/publications/2016-johanson-phd-thesis.pdf) · `johanson2016-phd.pdf`
The Alberta methods in one place: CFR and its variants, public-tree best response, abstraction.

**Burch (2017).** Time and Space: Why Imperfect Information Games are Hard. PhD thesis,
University of Alberta.
[PDF](https://poker.cs.ualberta.ca/publications/Burch_Neil_E_201712_PhD.pdf) · `burch2017-phd.pdf`
CFR+ and decomposition in depth, with what makes these games costly to solve.

## Three or more players

**Szafron, Gibson, Sturtevant (2013).** A Parameterized Family of Equilibrium Profiles for
Three-Player Kuhn Poker. AAMAS 2013.
[PDF](https://poker.cs.ualberta.ca/publications/AAMAS13-3pkuhn.pdf) ·
`szafron2013-three-player-kuhn.pdf`
Exact equilibria for three-player Kuhn poker (a four-card deck): a game with a known answer to
test a multiplayer solver against.

**Gibson (2014).** Regret Minimization in Games and the Development of Champion Multiplayer
Computer Poker-Playing Agents. PhD thesis, University of Alberta.
[PDF](https://poker.cs.ualberta.ca/publications/gibson.phd.pdf) · `gibson2014-phd-multiplayer.pdf`
What CFR does and doesn't guarantee with three or more players, and the three-player agents
built with it.

**Brown, Sandholm (2019).** Superhuman AI for Multiplayer Poker. Science 365.
[DOI 10.1126/science.aay2400](https://doi.org/10.1126/science.aay2400). Paywalled, so not in
`docs/papers/`.
Six-player no-limit Hold'em, solved with a sampled form of CFR plus search during play.

## Code

**OpenSpiel** (Apache-2.0), [github.com/google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel):
the independent implementation the tests check the engine against.

**noambrown/poker_solver** (MIT), [github.com/noambrown/poker_solver](https://github.com/noambrown/poker_solver):
a river-only solver with a permissive licence, the cross-check for the full-deck river spot
(`scripts/check_river.py`).
