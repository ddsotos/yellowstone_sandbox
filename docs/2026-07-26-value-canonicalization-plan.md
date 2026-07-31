# 2026-07-26 Yellowstone value-input canonicalization

## Goal

Reduce equivalent board representations before training and inference without
increasing CNN evaluations.

## Fast canonicalization

`fast_lr_ud_color_v1` applies these rules in order:

1. Mirror x when the visual left half has more cards than the right half.
2. Invert y and every rank (`1↔7`, `2↔6`, `3↔5`) when ranks 5-7 contain more
   cards than ranks 1-3.
3. Rename board colors from visual left to right as blue, red, green, yellow.
4. Order colors absent from the board by the viewing player's per-rank hand
   counts, comparing rank 1 through rank 7. Exact ties keep absolute color order.

The transform uses only the public board/history and the viewing player's hand.
It does not inspect the deck or opponents' hands when choosing the transform.

Ties intentionally keep the current orientation. This favors speed over perfect
canonicalization. In the initial 100-record audit, 94 records collapsed all 96
color/horizontal/vertical variants to one input; five produced two forms and one
produced three.

## Model lineage

| Checkpoint | One-pass training data |
| --- | --- |
| `win_value_canonical_old_001.pt` | `part_0..660000`, from scratch |
| `win_value_canonical_old_plus_6h_001.pt` | `part_660100..960000`, resumed |
| `win_value_canonical_old_plus_6h_plus_8h_001.pt` | `part_966400..1466300`, resumed |

`part_960100..966300` is deliberately excluded to match the selected historical
increments. Each checkpoint records `input_canonicalization=fast_lr_ud_color_v1`,
so value inference canonicalizes candidate turn-end records automatically.

## Validation

- Symmetry tests cover legal actions and state transitions.
- Complete heuristic games test rank-inverting vertical reflection through
  refills, settlements, scoring, and game end.
- Tensor tests cover idempotence, requested direction/color order, conversion
  resumability, and a non-tied 96-transform orbit.
- `audit_value_canonicalization` reports approximate orbit collapse on sampled
  heuristic records.
