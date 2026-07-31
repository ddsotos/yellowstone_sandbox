"""Summarize the four-seat V2-lite plus action-card experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from yellowstone.value_v2_lite_action import (
    CANONICALIZATION_V2_LITE_ACTION,
    HISTORY_SEMANTICS_V2_LITE_ACTION,
    VALUE_CONTEXT_SIZE_V2_LITE_ACTION,
    VALUE_SCHEMA_V2_LITE_ACTION,
)


def summarize_v2_lite_action(
    *,
    checkpoint_path: Path,
    evaluation_directory: Path,
    output_path: Path,
    games_per_seat: int,
    seed: int,
    timings_path: Path,
    reference_path: Path | None = None,
) -> dict:
    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    expected = {
        "value_schema": VALUE_SCHEMA_V2_LITE_ACTION,
        "input_canonicalization": CANONICALIZATION_V2_LITE_ACTION,
        "history_semantics": HISTORY_SEMANTICS_V2_LITE_ACTION,
        "context_size": VALUE_CONTEXT_SIZE_V2_LITE_ACTION,
        "opponent_private_inputs": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"V2-lite-action checkpoint differs at {key}")
    seats = []
    for player_index in range(4):
        path = evaluation_directory / (
            "v2_lite_action_new_88966_epoch001_"
            f"{games_per_seat}_seed{seed}_p{player_index}.json"
        )
        row = json.loads(path.read_text(encoding="utf-8-sig"))
        if (
            int(row["games"]) != games_per_seat
            or int(row["player_index"]) != player_index
            or int(row["seed"]) != seed
        ):
            raise ValueError(f"incompatible V2-lite-action evaluation: {path}")
        for key, value in expected.items():
            if key == "context_size":
                continue
            if row.get(key) != value:
                raise ValueError(f"evaluation differs at {key}: {path}")
        seats.append(row)
    total_games = sum(int(row["games"]) for row in seats)
    total_wins = sum(float(row["fractional_wins"]) for row in seats)
    total_one = sum(
        int(row["evaluated_player_one_card_turns"]) for row in seats
    )
    total_two = sum(
        int(row["evaluated_player_two_card_turns"]) for row in seats
    )
    timings = (
        json.loads(timings_path.read_text(encoding="utf-8-sig"))
        if timings_path.exists()
        else {}
    )
    reference = None
    if reference_path is not None and reference_path.exists():
        source = json.loads(reference_path.read_text(encoding="utf-8-sig"))
        reference_row = next(
            row for row in source["rows"] if int(row["source_games"]) == 88966
        )
        split = source["split_audit"]
        if (
            checkpoint["validation_game_ids_sha256"]
            != split["validation_game_ids_sha256"]
            or checkpoint["test_game_ids_sha256"]
            != split["test_game_ids_sha256"]
        ):
            raise ValueError("V1 and V2-lite-action split hashes differ")
        reference = {
            "model": "Original V1 new 88966 epoch001",
            "all_seats_win_rate": reference_row["all_seats_win_rate"],
            "test_brier": reference_row["test_brier"],
            "test_log_loss": reference_row["test_log_loss"],
        }
    result = {
        "status": "complete",
        "model": "V2-lite transition plus unordered action cards",
        "checkpoint": str(checkpoint_path),
        "training_games": int(checkpoint["training_games"]),
        "training_records": int(checkpoint["training_records"]),
        "epochs": int(checkpoint["epochs"]),
        "context_size": int(checkpoint["context_size"]),
        "board_channels": int(checkpoint["board_channels"]),
        "training_seed": int(checkpoint["seed"]),
        "evaluation_seed": seed,
        "games_per_seat": games_per_seat,
        "metrics": checkpoint["metrics"],
        "seats": seats,
        "all_seats_games": total_games,
        "all_seats_fractional_wins": total_wins,
        "all_seats_win_rate": total_wins / total_games,
        "all_seats_one_card_turns": total_one,
        "all_seats_two_card_turns": total_two,
        "all_seats_one_card_turn_rate": total_one / (total_one + total_two),
        "reference": reference,
        "timings_seconds": timings,
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V2-lite + explicit action cards evaluation",
        "",
        "| seat | win rate | one-card rate | one | two |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in seats:
        lines.append(
            f"| {row['player_index']} | {row['win_rate']:.3%} | "
            f"{row['evaluated_player_one_card_turn_rate']:.3%} | "
            f"{row['evaluated_player_one_card_turns']} | "
            f"{row['evaluated_player_two_card_turns']} |"
        )
    lines.extend(
        [
            f"| all | {result['all_seats_win_rate']:.3%} | "
            f"{result['all_seats_one_card_turn_rate']:.3%} | "
            f"{total_one} | {total_two} |",
            "",
            f"- test Brier: {checkpoint['metrics']['test_all_brier']:.6f}",
            f"- test logloss: {checkpoint['metrics']['test_all_log_loss']:.6f}",
            "- context: 138 V2-lite values + 12 unordered action-card values",
            "- This is a regular public-input four-seat evaluation.",
        ]
    )
    if reference is not None:
        lines.append(
            "- Original V1 new 88966 all-seat reference: "
            f"{reference['all_seats_win_rate']:.3%}"
        )
    output_path.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    payload = summarize_v2_lite_action(
        checkpoint_path=args.checkpoint,
        evaluation_directory=args.evaluation_directory,
        output_path=args.output,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        timings_path=args.timings,
        reference_path=args.reference,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
