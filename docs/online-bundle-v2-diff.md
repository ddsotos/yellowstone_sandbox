# Changes from the previous online bundle

## Summary

| Area | Previous bundle | V2 preview |
|---|---|---|
| Context size | 81 | 300 |
| History | Last 2 placements | Last 3 completed turns |
| One-card turn | Placement-shaped history | Explicit one-card completed turn |
| Two-card order | Order remains visible | Cards sorted after canonicalization |
| Frame history | Not modeled | Start/end frame and net x/y movement |
| Candidate frame | Not modeled | Start/end frame and net x/y movement |
| Board density | Current board | Also start-of-turn total card count |
| Negative cards | Limited aggregate information | Own exact 4x7; opponent public marginals |
| Refill choice | Mostly evaluated after sampled refill | Explicit pending refill source |
| Canonicalization | `fast_lr_ud_color_v1` | Strict residual comparison over up to 96 symmetries |
| Dataset | Derived tensors | Replayable raw games plus derived tensors |
| First teacher | Heuristic/legacy value model | Canonical old +6h, with 20 refill samples |
| Future refill evaluation | Sample average | Direct V2 expected value after validation |

## History semantics

V2 stores the three completed turns before the candidate. Each turn contains:

- relative player;
- one or two unordered cards;
- start and final 3x3 frames;
- start-of-turn board card count;
- absolute frame movement in x and y;
- total score and negative-card deltas;
- refill result and settlement marker.

The current candidate is not appended to history. Its frame movement is a
separate feature block. Candidate movement intentionally permits different
predictions for otherwise equal resulting game states.

## Canonicalization

Every model-visible record is canonicalized using:

- horizontal reflection;
- vertical reflection together with rank inversion `1 <-> 7`;
- all 24 color permutations.

Board, hand, three-turn history, frames, own negative cards, opponent public
marginals, and candidate features participate in the final deterministic
comparison. Ten real collected positions were each tested under all 96
transformations; all 960 transformed inputs matched exactly.

Raw replay files are intentionally not canonicalized. Conversion and live V2
inference both call the same `canonical_tensors_v2` implementation.

## Public-information changes

The viewer's negative pile uses exact 4-color by 7-rank counts. Each opponent
uses:

- seven expected rank counts;
- four expected color counts;
- an exact/estimated flag.

After an opponent secretly recovers cards from their negative pile, public
expectations are scaled by the known remaining fraction. Hidden card identity
is never exposed to the deployed model.

## Refill learning

Generation 0 uses the legacy model and 20 actual refill samples to rank deck
and negative-card refill candidates. This is a bootstrap mechanism only.

The V2 record represents the refill choice before its random result with
`pending_refill_source`. Once validated, future V2 self-play should evaluate
that expected value directly rather than repeat 20 sampled refills.

## Known intentional compromises

- Equivalent zero-loss or equal-loss frames keep the first enumerated
  representative. This preserves speed but leaves enumeration-order bias.
- Candidate frame movement can produce different predictions for equal final
  game states. This is intentional but can also learn action-policy leakage.
- The first V2 model scored 26.03% in a same-seed 1,000-game seat-0 evaluation,
  below the canonical old +6h model's 29.08%. Every-seat evaluation is not yet
  complete, so the V2 model remains a developer preview.

## Practical evaluation correction (2026-07-27)

The first V2 evaluator initially exposed `RefillSource.NONE` after the hand
became empty. That was outside the generation-0 collection policy: the intended
three categories were retaining cards, deck refill, and negative-card refill.
The V2 adapter now excludes the unsupported empty-hand/no-refill action.

Corrected seat-0 results against three heuristic opponents, seed `20260725`:

| Model | Games | Win rate |
|---|---:|---:|
| V2 generation 0, 197,800 games | 1,000 | 26.03% |
| Canonical old | 1,000 | 28.23% |
| Canonical old +6h | 1,000 | 29.08% |

The corrected raw result is bundled under `evaluations/`.

## Runtime rule correction (2026-07-27)

A deck refill now settles immediately whenever the draw leaves the deck at
exactly zero, even when the player successfully reaches six cards. The former
runtime delayed settlement until a later refill failed to reach six cards.

The 197,800-game generation-0 dataset and its first V2 model retain the former
rule and are not regenerated. Raw replays carry their rules version and remain
replayable through the compatibility path. New games use
`yellowstone-python-2026-07-27-empty-deck-settlement`.

## API migration

Previous:

```python
TorchWinValueEstimator -> ValueRecord -> 81-value context
```

V2:

```python
TorchWinValueEstimatorV2 -> ValueRecordV2 -> strict canonical 300-value context
```

The online application must additionally retain:

- `CompletedTurnTracker`;
- `PublicNegativeKnowledgeTracker`;
- candidate start/final frame facts;
- `PendingRefillSource`.

Use `replay_v2.records_from_replay` as the current semantic reference. A
dedicated online candidate adapter is still required before drop-in deployment.

`models/win_value.pt` remains the V1 canonical comparison model so existing
V1 tests and audit tools continue to run. Only `models/win_value_v2.pt`, when
present, is the new-schema checkpoint.
