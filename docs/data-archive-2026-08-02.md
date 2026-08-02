# Data Archive Notes 2026-08-02

This note records which Yellowstone replay and tensor datasets are archived on
`D:\codex-backup`, and which datasets intentionally remain on `C:` because a
background job still references them.

## Archived On D

### `D:\codex-backup\yellow_3_legacy_training_data_2026-07-27`

Older heuristic replay and derived training data from the pre-2026-07-29
experiments. These are not part of the current active 6-hour snapshot training
or the variant heuristic training.

Approximate size at the 2026-08-02 check: 15.365 GiB.

### `D:\codex-backup\yellow_3_derived_training_data_2026-07-29`

Derived tensors from older generation0, historyfix, history3, and preflight
experiments. These are historical comparison artifacts and are not the current
canonical board_columns_v1 or 6-hour snapshot tensors.

Approximate size at the 2026-08-02 check: 4.301 GiB.

### `D:\codex-backup\yellow_3_unused_game_data_2026-08-02`

Unused replay or tensor datasets moved off `C:` on 2026-08-02 to recover local
workspace space. This archive also contains
`v2_heuristic_safe_counts_rank_color_snapshot_20260802_040724`, an empty failed
or superseded snapshot directory that was moved from `rl_bundle\data`.

Approximate size at the 2026-08-02 check before moving the empty directory:
11.398 GiB.

## Kept On C Because Active Jobs Reference Them

The following datasets should not be moved while their status JSON still shows
active work or a live PID.

### 6-hour heuristic snapshot pipeline

Status:
`rl_bundle\results\evaluations\v2_heuristic_safe_counts_rank_color_6h_snapshot_training.status.json`

Kept datasets:

- `data\v2_heuristic_safe_counts_rank_color_20260801`
- `data\v2_heuristic_safe_counts_rank_color_snapshot_20260802_081716`
- `data\v2_heuristic_safe_counts_rank_color_snapshot_20260802_081716_canonical`
- `data\v2_heuristic_safe_counts_rank_color_snapshot_20260802_081716_board_columns_v1`
- `data\v2_heuristic_safe_counts_rank_color_snapshot_20260802_081716_board_columns_v2`
- `data\v2_heuristic_safe_counts_rank_color_snapshot_20260802_081716_preplay_board_columns`

Meaning:

- `v2_heuristic_safe_counts_rank_color_20260801` is the 702,700-game raw replay
  source collected by the heuristic safe-counts rank/color policy.
- `...snapshot_20260802_081716` is the fixed snapshot created from completed
  shards of that replay source.
- `...canonical` is the canonical Original V1 tensor conversion.
- `...board_columns_v1` is the `[1,7,3]` board-columns V1 tensor conversion.
- `...board_columns_v2` is the V2 public-context tensor conversion with the
  board reduced to `[1,7,3]` board columns.
- `...preplay_board_columns` is the pre-play tensor conversion with the board
  reduced to `[1,7,3]` board columns.

### Variant heuristic board_columns_v1 training

Status:
`rl_bundle\results\evaluations\v2_variant_board5_hand6_oneoff_tiered_board_columns_v1_training.status.json`

Kept datasets:

- `data\v2_variant_board5_hand6_oneoff_tiered_heuristic4_20260802`
- `data\v2_variant_board5_hand6_oneoff_tiered_heuristic4_20260802_part2`
- `data\v2_variant_board5_hand6_oneoff_tiered_heuristic4_300000_snapshot`
- `data\v2_variant_board5_hand6_oneoff_tiered_heuristic4_300000_snapshot_canonical`

Meaning:

- `...heuristic4_20260802` is the first 100,000-game raw replay set for the
  board>=5, hand=6, one-off tiered variant heuristic.
- `...heuristic4_20260802_part2` is the second 200,000-game raw replay set for
  the same variant heuristic.
- `...300000_snapshot` is the combined fixed 300,000-game snapshot used for
  variant board_columns_v1 training.
- `...300000_snapshot_canonical` is the canonical intermediate conversion for
  that 300,000-game variant snapshot.

### Continuous variant heuristic collection

Status:
`rl_bundle\results\collections\v2_variant_board5_hand6_oneoff_tiered_heuristic4_continuous_20260802.status.json`

Kept dataset:

- `data\v2_variant_board5_hand6_oneoff_tiered_heuristic4_continuous_20260802`

Meaning:

- Continuous raw replay collection for the board>=5, hand=6, one-off tiered
  variant heuristic. It is intended to keep running until a stop instruction is
  given.

## Move Rule

Do not move any dataset named in a running status JSON as `source`, `data`, or
current conversion output. Once the relevant pipeline has completed and no live
PID references the dataset, derived tensors and raw replay that are not needed
for near-term comparison can be moved under `D:\codex-backup`.

## Continuous Variant Chunk Training

The continuous variant replay collection can be trained in retained 200,000-game
increments with:

```powershell
.\scripts\run_continuous_variant_chunk_training.ps1
```

Status:
`rl_bundle\results\evaluations\v2_variant_board5_hand6_oneoff_tiered_continuous_chunk_training.status.json`

Stop file:
`rl_bundle\results\evaluations\v2_variant_board5_hand6_oneoff_tiered_continuous_chunk_training.stop`

For each cumulative threshold, the pipeline writes a fixed snapshot and derived
tensors:

- `data\v2_variant_board5_hand6_oneoff_tiered_continuous_chunk_training_0200000_snapshot`
- `data\v2_variant_board5_hand6_oneoff_tiered_continuous_chunk_training_0200000_canonical`
- `data\v2_variant_board5_hand6_oneoff_tiered_continuous_chunk_training_0200000_board_columns_v1`
- `data\v2_variant_board5_hand6_oneoff_tiered_continuous_chunk_training_0200000_preplay_board_columns`

The next thresholds use the same naming pattern with `0400000`, `0600000`, and
so on. These datasets are intentionally retained so each 200,000-game training
point can be inspected, reused, or archived later.

For each threshold, three training jobs are run with at most two concurrent
training processes:

- `board_columns_v1_scratch`: random initialization on that threshold's retained
  board-columns tensor.
- `board_columns_v1_finetune`: first threshold starts from the 6-hour
  board_columns_v1 checkpoint; later thresholds continue from the previous
  threshold's finetuned checkpoint when available.
- `preplay_board_columns`: random initialization on the retained pre-play
  board-columns tensor.

When no new 200,000-game threshold is available, the watcher waits one hour and
checks the continuous collection manifest again.
