# Yellowstone RL Bundle

This directory is a self-contained, minimal Python package for Yellowstone Park
reinforcement-learning research. It contains the rule engine, fixed action space,
observations, heuristic opponent, rendering, serialization, and reference spatial
encoders, plus local experiment scripts, datasets, checkpoints, and evaluation
artifacts. It deliberately excludes LLM clients.

## Setup

From this directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

Optional tensor and CNN tests require:

```powershell
.venv\Scripts\python -m pip install -e ".[cnn]"
```

## Heuristic win-value pilot

Collect the initial 10,000-game, all-heuristic dataset and train the
player-perspective win-value CNN:

```powershell
.venv\Scripts\python -m pip install -e ".[value]"
.venv\Scripts\python -m yellowstone.value_learning --games 10000 --output data/heuristic_value_data
.venv\Scripts\python -m yellowstone.train_value --data data/heuristic_value_data --checkpoint models/win_value.pt
```

The chunk archives contain one record for every completed player turn.  It keeps
only public board/player counts, the evaluating player's own hand, and the
two latest public card placements.  Train/validation/test are split by game,
not by individual record.  `yellowstone.value_policy.choose_highest_value_turn`
evaluates legal one-card-end and two-card-refill-boundary candidates once a
trained estimator is supplied.

## Project layout

- `src/`: Yellowstone engine and learning code
- `tests/`: automated tests
- `scripts/`: collection, training, evaluation, and maintenance scripts
- `data/`: generated learning datasets and dataset manifests
- `models/`: trained checkpoints
- `results/evaluations/`: win-rate and model-comparison results
- `results/benchmarks/`: speed and pruning benchmarks
- `results/audits/`: saved strategy-audit cases
- `results/smoke/`: smoke-test artifacts
- `logs/`: runtime logs
- `docs/`: progress notes and design documents

The authoritative C/D storage, archive, restore, and experiment naming rules are
in `docs/storage-layout.md`.

The package uses `src/` layout. Running `pytest` from `rl_bundle/` tests only the
copied runtime and the reference modules.

## Included APIs

- `yellowstone.game`: immutable `GameState -> legal_actions -> apply_action` rules.
- `yellowstone.action_space`: fixed placement/end/refill action indexes and masks.
- `yellowstone.env`: learner player 0 against deterministic heuristic NPCs.
- `yellowstone.board_tensor`: spatial board and global-feature encoders.
- `yellowstone.symmetry`: color permutation, horizontal reflection, and
  rank-inverting vertical reflection.
- `yellowstone.cnn`: optional PyTorch policy/value reference network.

Vertical reflection changes every rank `r` to `8-r` together with board/frame y.
Quarter-turn rotations remain unsupported. Placement action indices are transformed
through the selected card after the transformed hand is sorted; callers must not copy
hand indexes directly.

## Provenance

`MANIFEST.sha256` lists each source, documentation, test, and configuration file
(except the manifest itself) with its SHA-256 digest. Runtime and baseline tests
were copied from the repository at bundle creation; `board_tensor.py`,
`symmetry.py`, `cnn.py`, and their tests are new reference implementations.
