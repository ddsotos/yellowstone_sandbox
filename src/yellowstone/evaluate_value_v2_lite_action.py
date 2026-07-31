"""Evaluate V2-lite terminal value with explicit action cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from yellowstone.value_evaluation_v2_lite import (
    evaluate_v2_lite_value_player,
)
from yellowstone.value_v2_lite_action import (
    CANONICALIZATION_V2_LITE_ACTION,
    HISTORY_SEMANTICS_V2_LITE_ACTION,
    VALUE_SCHEMA_V2_LITE_ACTION,
    TorchWinValueEstimatorV2LiteAction,
)


def evaluate_v2_lite_action(
    checkpoint_path: str | Path,
    *,
    games: int,
    seed: int,
    player_index: int,
) -> dict:
    started = perf_counter()
    estimator = TorchWinValueEstimatorV2LiteAction(checkpoint_path)
    result = evaluate_v2_lite_value_player(
        estimator,
        games=games,
        seed=seed,
        player_index=player_index,
    )
    return {
        **result,
        "fractional_wins": result["wins"],
        "evaluated_player_one_card_turns": result["one_card_turns"],
        "evaluated_player_two_card_turns": result["two_card_turns"],
        "evaluated_player_one_card_turn_rate": result[
            "one_card_turn_rate"
        ],
        "value_schema": VALUE_SCHEMA_V2_LITE_ACTION,
        "input_canonicalization": CANONICALIZATION_V2_LITE_ACTION,
        "history_semantics": HISTORY_SEMANTICS_V2_LITE_ACTION,
        "opponent_private_inputs": False,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_v2_lite_action(
        args.checkpoint,
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
