# Action delta pre-play privileged audit

`yellowstone.evaluate_action_delta` can optionally report a turn-start win
probability beside the action-delta score:

```powershell
python -m yellowstone.evaluate_action_delta `
  --delta-checkpoint models/action_delta_continuous_epoch001_pct030.pt `
  --games 1000 --seed 20260725 --player-index 0 `
  --pre-play-mode privileged_audit `
  --pre-play-checkpoint models/privileged_state_generation0_197800_epoch002.pt
```

The audit reports:

- `mean_pre_play_win_probability`;
- `mean_predicted_delta`;
- `mean_combined_post_play_win_probability`, calculated as
  `clip(pre_play_probability + predicted_delta, 0, 1)`;
- the number and rate of turns where the sum required clipping.

This mode uses the exact private hands of all opponents. It is an input-contract
audit only. Its probabilities must not be reported as deployable model results,
used in the browser's normal analysis mode, or included in official policy
comparisons. The audit does not use the pre-play probability for action
selection, so enabling it does not change the action-delta policy.
