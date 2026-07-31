# Yellowstone value data V2 design — 2026-07-26

## Goal

Build a new, replayable value-learning dataset that:

- represents one-card and two-card turns correctly;
- removes irrelevant within-turn card order;
- canonicalizes every supported symmetry exactly;
- learns refill choices without exposing hidden information to the deployed model;
- supports future feature changes without collecting the games again.

The first V2 model is trained from scratch. Legacy data and weights are not
mixed into it. The legacy canonical model is used only as the generation-0
teacher.

## Value point and history

The value record keeps the perspective of the player whose candidate is being
evaluated. The candidate turn itself is not appended to its history. Its
result is already represented by the resulting state.

History contains the three completed turns immediately before the evaluated
turn. Each history slot contains:

- `turn_present`;
- relative player index;
- `play_count` (`1` or `2`);
- the actual public 3x3 frame at turn start, with a missing mask;
- the final selected 3x3 frame;
- the total number of cards on the board at turn start, including stacked
  cards;
- explicit net movement magnitudes `abs(frame_dx)` and `abs(frame_dy)`;
- up to two cards, each with its own `card_present`;
- total score change for the complete turn;
- total negative-card increase for the complete turn;
- whether settlement occurred while resolving the turn;
- refill result:
  - `not_offered`;
  - `none`;
  - `deck`;
  - `negative_cards`.

The cards inside one turn are unordered. After spatial, rank, and color
canonicalization, they are sorted by canonical color and rank. Score and
negative-card changes are turn totals and are not attached to individual
cards.

An absent history turn is different from a real zero-delta turn:
`turn_present=0` is mandatory for padding. A one-card turn uses
`turn_present=1`, `play_count=1`, and `card2_present=0`.

The start frame is the previous completed turn's final selected frame. It is
missing at the start of a game rather than inferred from the set of frames
that happen to contain the initial board. Only net start-to-final movement is
encoded; an intermediate first-placement frame in a two-card turn is not a
model feature.

The candidate currently being evaluated is not appended to the three-turn
history. It has a separate frame feature block containing its start frame
(or a missing mask), final frame, start-of-turn board card count, and explicit
absolute x/y movement. This deliberately permits two candidates with the same
resulting `GameState` to receive different values. The expected benefit is to
make frame movement a strong, explicit strategic feature; the acknowledged
risk is action leakage and the reintroduction of same-result value splits.

For speed, equivalent placement frames continue to collapse to the first
frame found by the existing deterministic action enumeration. The model
therefore does not search every frame-movement variant. This retains the
existing speedup but introduces a known enumeration-order bias, including a
possible left/right or up/down data-distribution bias. This is an explicit
speed-over-purity decision and should be revisited if frame features prove
important in ablation tests.

Settlement does not imply that the board itself was reset. The per-turn
settlement flag instead marks the public round/deck/negative-card boundary so
the model need not infer that discontinuity indirectly.

## Strict canonicalization

Supported game symmetries are:

- horizontal reflection;
- vertical reflection together with rank inversion
  (`1<->7`, `2<->6`, `3<->5`);
- all 24 color permutations.

There are at most 96 transforms. V2 uses a strict residual-candidate method:

1. Compare the complete color-independent board representation vertically.
2. If the board is tied vertically, compare the viewer hand's rank-count
   vector from rank 1 through rank 7 against its reverse.
3. Keep both vertical candidates only if still tied.
4. Compare the complete board occupancy grid horizontally.
5. Keep both horizontal candidates only if still tied.
6. Resolve colors in stages:
   - board;
   - viewer hand;
   - history and negative-card observation.
7. Keep only unresolved tied color permutations.
8. Compare the remaining transformed, model-visible semantic records
   lexicographically and select the minimum.

Thus the common case constructs one final tensor. Full 96-way comparison is
reserved for a fully unresolved state.

Canonicalization may use only information available to the record's viewer.
It must not use opponent hands, the real hidden deck composition/order, or
the real hidden remainder of an opponent's negative-card pile.

Required tests:

- all 96 transforms of the same visible record collapse to one byte-identical
  input;
- canonicalization is idempotent;
- equivalent green-then-blue and blue-then-green results collapse;
- one-card history and absent history remain distinct;
- unresolved ties are deterministic;
- every transform preserves legal transitions.

## Negative-card information

The raw replay retains the complete simulator state. The model encoder exposes
only legal viewer information.

For the viewer:

- exact color-by-rank negative-card counts: `4 * 7 = 28` values.

For each opponent:

- expected count by rank: 7 values;
- expected count by color: 4 values;
- an `exact/estimated` flag.

Before an opponent has recovered cards from their negative pile after a
settlement, their values are exact. After a hidden random recovery, public
expected counts are scaled by the known remaining fraction. Later publicly
received negative cards increment the corresponding rank and color
expectations. The initial implementation does not perform Bayesian updates
from later revealed hand cards.

The full event log is retained so alternative encodings can be generated
later.

## Refill-conditioned values

V2 represents a refill method selected but its random result not yet resolved:

- `no_pending`;
- `none`;
- `deck`;
- `negative_cards`.

The model estimates expected final win probability from the public state and
this pending refill source. This lets deployment compare refill choices
without seeing the hidden deck.

Generation 0 may use the real hidden deck as privileged teacher information,
but the deployed/student encoder must never receive it. Every raw record must
identify the teacher checkpoint and whether privileged teacher evaluation was
used.

## Generation-0 collection policy

Generation 0 collects 200,000 four-player games.

- Teacher checkpoint:
  `models/win_value_canonical_old_plus_6h_001.pt`.
- The teacher stays frozen for all 200,000 games.
- All four players use the same hybrid policy.
- The normal policy is heuristic.
- At a turn start with more than two hand cards, use heuristic behavior.
- At a turn start with at most two hand cards, construct three categories and
  find the highest-valued candidate inside each available category:
  1. finish the turn while retaining at least one hand card;
  2. empty the hand and refill from the deck;
  3. empty the hand and refill from negative cards.

For generation 0, deck and negative-card refill candidates use 20 random
outcomes and the legacy teacher's mean estimate. Actual deck composition is
allowed only for this privileged teacher calculation.

Category selection probabilities are:

- three available: best `60%`, each other category `20%`;
- two available: best `80%`, other category `20%`;
- one available: `100%`.

The mass of an unavailable category is added to the best available category.
Exploration happens between category winners, not between all candidates
inside one category.

Teacher estimates are audit metadata, not terminal truth labels. Actual game
results provide the learning target.

## Frame policy

Category 1, retain hand:

- use the normal minimum-negative-card frame representatives.

Category 2, deck refill:

- use minimum-negative-card frame representatives.

Category 3, negative-card refill:

- if the player started the turn with at most five negative cards, enumerate
  every legal frame sequence;
- keep candidates with distinct resulting negative-card contents;
- deduplicate candidates with identical final game states;
- a candidate that finishes with fewer than six negative cards is unavailable
  and receives value zero without sampling;
- if the player started with at least six negative cards, use the normal
  minimum-negative-card frame representatives.

No additional full-frame audit is included in the initial implementation.

## Generation 1 and later

Train the first V2 model from scratch using only the new 200,000-game dataset.
Do not resume legacy weights and do not mix legacy tensors.

Promote the model only after:

- schema, replay, visibility, and canonicalization tests pass;
- same-seed practical evaluations are complete for every seat;
- results are compared with heuristic and canonical old +6h baselines;
- no material regression or seat-specific failure is found.

After promotion:

- use the V2 model directly for category and refill-source values;
- do not perform the 20-sample legacy-teacher average;
- collect 50,000 games with a frozen teacher;
- retrain and evaluate the next generation;
- retain each generation checkpoint, data range, model hash, and metrics.

Final evaluation uses zero exploration.

## Replay storage

Raw data is stored as compressed, sharded, replayable per-game event logs.
Each game includes:

- schema and rules version;
- initial seed and sufficient initial hidden state for deterministic replay;
- every action and selected frame;
- refill source and realized random result;
- low-hand decision candidates, teacher estimates, category probabilities,
  and selected candidate;
- teacher checkpoint path, hash, and generation;
- final winner set.

Derived model tensors are separate artifacts. Training/validation/test splits
are by game ID, never by individual state.

Completed shards survive interruption and are never rewritten in place.

## Mandatory 10-game preflight

Before starting the 200,000-game collection:

1. run exactly 10 games through the complete V2 replay writer and policy;
2. verify deterministic replay and derived input generation;
3. report:
   - compressed bytes per game;
   - projected raw size for 200,000 games;
   - projected derived-tensor size;
   - seconds per game and projected collection duration;
   - low-hand decision count;
   - candidates per category;
   - time spent in 20-sample refill evaluation.

If storage or time is abnormal, stop before the full collection and revise the
format or batching.

### Latest preflight result (2026-07-27)

The frame-feature schema was tested with the same generation-0 teacher and
seed `20260726`:

- games: 10;
- derived value records: 822;
- low-hand decisions: 40;
- category candidates:
  - retain hand: 69;
  - deck refill: 111;
  - negative-card refill: 310;
- collection time: 2.158 seconds (`0.2158` seconds/game);
- time inside 20-sample refill evaluation: 0.971 seconds;
- compressed replay: 24,954 bytes (`2,495.4` bytes/game);
- compressed derived tensors: 89,043 bytes (`8,904.3` bytes/game).

Linear projections for 200,000 games:

- compressed replay: about 499.1 MB decimal (476 MiB);
- compressed derived tensors: about 1.781 GB decimal (1.66 GiB);
- collection duration: about 43,159 seconds (12.0 hours);
- low-hand decisions: about 800,000.

These are small-sample linear projections, not capacity guarantees. Long-run
shards must continue recording observed bytes/game and seconds/game in the
restart manifest.

## Known risks

- The 197,800-game generation-0 dataset uses the former rule that delayed
  settlement when a refill drew the deck to exactly zero while still filling
  the hand to six. Runtime rules were corrected on 2026-07-27; that dataset
  and its first model are intentionally not regenerated.
- The legacy teacher evaluates some post-refill states outside its training
  distribution.
- The generation-0 teacher uses privileged real-deck composition.
- Twenty samples add selection noise.
- A public student cannot reproduce hidden-state-specific teacher decisions;
  it learns their public-state average.
- The V2 model has a larger context and must be trained from scratch.
- Candidate frame movement intentionally permits different values for equal
  resulting game states and can learn action-policy leakage.
- Equivalent frames use the first enumerated representative, so frame
  features inherit deterministic enumeration-order bias.
- The first promotion threshold is finalized after the canonical old +6h
  baseline evaluation finishes.
