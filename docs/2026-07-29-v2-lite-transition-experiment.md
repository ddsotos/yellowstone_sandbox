# V2-lite transition experiment — 2026-07-29

## Purpose

Test whether V2's larger context and indirect frame-movement features explain
its weaker policy win rate. The experiment reuses the 197,800 generation-0
replays and trains a new model from random initialization for one epoch.

## Inputs

The spatial input has 58 channels:

- 29 channels for the state after the candidate turn;
- 29 signed channels for `after - before`.

Each 29-channel board contains 28 color/rank stack-count channels and one total
stack-count channel. The pre-play board is exactly recoverable as
`after - delta`. The same canonical transform is applied to both components.

The compact context has 138 values:

| block | values |
|---|---:|
| resulting own hand | 36 |
| resulting player summaries | 12 |
| resulting turn/deck state | 13 |
| pending refill source | 4 |
| last two completed turns, without frame coordinates | 50 |
| own negative pile rank/color marginals | 11 |
| pre-play player summaries | 12 |
| total | 138 |

Opponent negative-pile estimates, candidate frame coordinates, and historical
frame coordinates are excluded. Own negative cards retain seven rank counts
and four color counts; the exactness bit is omitted because the viewer always
knows their own pile exactly.

## Controls

- source: `data/v2_generation0_200k_frame_features` (197,800 completed games);
- split/training seed: `20260726`;
- random initialization, one epoch, batch 256, learning rate `1e-3`;
- evaluation seed: `20260725`;
- adaptive p/q pruning, approximate new-color neighbors, traditional refill;
- four seats, 1,000 games per seat;
- report test Brier/logloss, win rate, and one-card turn rate.

## Outputs

- tensors: `data/v2_lite_transition_generation0_197800_tensors/`;
- checkpoint:
  `models/win_value_v2_lite_transition_generation0_197800_epoch001.pt`;
- status: `results/evaluations/v2_lite_transition.status.json`;
- final comparison:
  `results/evaluations/v2_lite_transition_generation0_197800_epoch001.md`.

Conversion is shard-restartable. The background pipeline waits for the V1
history 2x2 evaluation process, if it is still running, before beginning the
CPU-intensive conversion.

