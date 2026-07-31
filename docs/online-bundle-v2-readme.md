# Yellowstone V2 online bundle preview

This is a side-by-side preview of the new Yellowstone value-learning pipeline.
It does not replace the previous `online_bundle`.

Read `CHANGES_FROM_PREVIOUS_BUNDLE.md` first. It lists the input, model,
canonicalization, replay, and integration differences.

## Included

- current game rules and candidate generation;
- strict V2 canonicalization and the 300-value public-information context;
- completed-turn and public negative-card knowledge trackers;
- replay-to-V2-record conversion;
- V2 model architecture, training, and inference code;
- V2 tests and design documents;
- the previous canonical old +6h checkpoint as a comparison baseline.

## Model status

See `MODEL_STATUS.json`.

The first V2 checkpoint was trained from 197,800 generation-0 games and has
completed a same-seed 1,000-game practical evaluation in seat 0. It scored
26.03% against three heuristic opponents, below the 29.08% canonical old +6h
comparison model. This bundle therefore remains a developer preview and must
not silently replace the previous production model.

When present, the V2 checkpoint is stored as:

```text
models/win_value_v2.pt
```

The comparison-only legacy checkpoint is:

```text
models/legacy_canonical_old_plus_6h.pt
```

For backward-compatible V1 tests and tools, the same legacy checkpoint is
also copied to `models/win_value.pt`. That filename does **not** contain a V2
model.

## Installation and tests

```powershell
python -m pip install -e ".[value]"
python -m pytest -q
```

## V2 inference primitive

```python
from yellowstone.value_v2 import TorchWinValueEstimatorV2

estimator = TorchWinValueEstimatorV2("models/win_value_v2.pt")
probability = estimator(value_record_v2)
```

`value_record_v2` must contain viewer-safe state, the three completed turns
before the candidate, candidate frame movement, public negative-card
knowledge, and the pending refill source. The estimator performs strict
canonicalization internally.

## Integration warning

`TorchWinValueEstimatorV2` is not a drop-in replacement for
`TorchWinValueEstimator`. The old `choose_highest_value_turn` path constructs
the old 81-value record. `yellowstone.value_evaluation_v2` now provides the
executable V2 candidate adapter and maintains both public trackers.

The adapter intentionally excludes `RefillSource.NONE` after the hand becomes
empty. Generation 0 did not train that action and the intended three-way
decision was: retain a non-empty hand, empty the hand and refill from the deck,
or empty the hand and recover negative cards. Including the untrained fourth
choice caused invalidly optimistic predictions and catastrophic play.
