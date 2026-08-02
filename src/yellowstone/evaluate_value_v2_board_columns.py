"""Evaluate a V2 board-columns checkpoint against heuristic opponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yellowstone.value_board_columns_v2 import (
    CANONICALIZATION_BOARD_COLUMNS_V2,
    HISTORY_SEMANTICS_BOARD_COLUMNS_V2,
    VALUE_SCHEMA_BOARD_COLUMNS_V2,
    TorchWinValueEstimatorV2BoardColumns,
)
from yellowstone.value_evaluation_v2 import evaluate_v2_value_player


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_v2_value_player(
        TorchWinValueEstimatorV2BoardColumns(args.checkpoint),
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
    )
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "fractional_wins": result["wins"],
            "evaluated_player_one_card_turns": result["one_card_turns"],
            "evaluated_player_two_card_turns": result["turns"] - result["one_card_turns"],
            "evaluated_player_one_card_turn_rate": (
                result["one_card_turns"] / result["turns"] if result["turns"] else 0.0
            ),
            "value_schema": VALUE_SCHEMA_BOARD_COLUMNS_V2,
            "input_canonicalization": CANONICALIZATION_BOARD_COLUMNS_V2,
            "history_semantics": HISTORY_SEMANTICS_BOARD_COLUMNS_V2,
            "opponent_private_inputs": False,
        }
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
