# Yellowstone online integration bundle

This directory contains a portable snapshot for implementing an online
Yellowstone game:

- `src/yellowstone/`: game rules, candidate generation, value-input encoding,
  canonicalization, and model inference
- `models/win_value.pt`: selected deployable checkpoint
- `docs/`: rules, model lineage, optimization notes, and canonicalization design
- `tests/`: copied rule and inference tests
- `MODEL_STATUS.json`: exact checkpoint lineage and canonicalization status
- `MANIFEST.sha256`: SHA-256 hashes for copied source, tests, and documentation

## Installation

Python 3.13 is the current tested version.

```powershell
python -m pip install -e ".[value]"
python -m pytest -q
```

## Model use

```python
from yellowstone.value_policy import (
    TorchWinValueEstimator,
    choose_highest_value_turn,
)

estimator = TorchWinValueEstimator("models/win_value.pt")
turn = choose_highest_value_turn(
    state,
    estimator,
    history=recent_placements,
    prune_negative_card_increase_above=8,
    approximate_new_color_neighbor_limit=True,
)
```

`TorchWinValueEstimator` reads checkpoint metadata. Canonical checkpoints apply
`fast_lr_ud_color_v1` automatically before CNN inference; legacy checkpoints do
not. The online caller must not canonicalize the real `GameState` or transform
the selected action back. Candidate actions remain in the original game frame;
only candidate value-model inputs are canonicalized.

The p/q pruning thresholds used by the current evaluation setup are:

- `negative_cards + loss_score >= 10`: maximum increase 4
- otherwise: maximum increase 8

The caller is responsible for choosing the appropriate threshold at each turn.

## Refreshing the model

From the source `rl_bundle` directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\export_online_bundle.ps1
```

The exporter prefers the newest canonical checkpoint and falls back to
`win_value_6h_plus_8h_001.pt` while canonical training is incomplete.

The source workspace may also run `wait_and_refresh_online_bundle.ps1`; when
present, it replaces the bundled fallback automatically after the final
canonical checkpoint is completed.
