"""Summarize the matched-input Original V1 Conv2 versus Conv3 experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def summarize(
    *,
    checkpoint_path: Path,
    baseline_path: Path,
    baseline_comparison_path: Path | None,
    evaluation_directory: Path,
    timings_path: Path,
    output_path: Path,
    games_per_seat: int,
    seed: int,
) -> dict[str, object]:
    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    expected_contract = {
        "value_schema": "yellowstone.value.v1",
        "history_semantics": "rolling_last_two_placements",
        "input_canonicalization": "fast_lr_ud_color_v1",
        "model_architecture": "yellowstone.win_value.v1.conv3_64_fc128",
        "convolution_layers": 3,
        "hidden_channels": 64,
        "hidden_size": 128,
    }
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected_contract.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Conv3 checkpoint contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_row = next(
        row
        for row in baseline["rows"]
        if row["checkpoint_model"] == "original"
        and row["history_semantics_match"]
    )
    baseline_checkpoint = torch.load(
        baseline["models"]["original"],
        map_location="cpu",
        weights_only=False,
    )
    baseline_training_seconds = None
    if (
        baseline_comparison_path is not None
        and baseline_comparison_path.exists()
    ):
        baseline_comparison = json.loads(
            baseline_comparison_path.read_text(encoding="utf-8")
        )
        baseline_condition = next(
            (
                row
                for row in baseline_comparison["conditions"]
                if row["key"] == "v1_original_epoch002"
            ),
            None,
        )
        if baseline_condition is not None:
            baseline_training_seconds = float(
                baseline_condition["training_seconds"]
            )
    seats: list[dict[str, object]] = []
    for player_index in range(4):
        path = evaluation_directory / (
            "v1_original_conv3_generation0_197800_epoch002_"
            f"{games_per_seat}_seed{seed}_p{player_index}.json"
        )
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        if int(evaluation["games"]) != games_per_seat:
            raise ValueError(f"incomplete evaluation: {path}")
        seats.append(
            {
                "player_index": player_index,
                "games": int(evaluation["games"]),
                "fractional_wins": float(evaluation["fractional_wins"]),
                "win_rate": float(evaluation["win_rate"]),
                "one_card_turns": int(
                    evaluation["evaluated_player_one_card_turns"]
                ),
                "two_card_turns": int(
                    evaluation["evaluated_player_two_card_turns"]
                ),
                "one_card_turn_rate": float(
                    evaluation["evaluated_player_one_card_turn_rate"]
                ),
                "elapsed_seconds": float(evaluation["elapsed_seconds"]),
                "result": str(path),
            }
        )
    total_games = sum(int(row["games"]) for row in seats)
    total_wins = sum(float(row["fractional_wins"]) for row in seats)
    total_one = sum(int(row["one_card_turns"]) for row in seats)
    total_two = sum(int(row["two_card_turns"]) for row in seats)
    conv3_rate = total_wins / total_games
    baseline_rate = float(baseline_row["all_seats_win_rate"])
    timings = (
        json.loads(timings_path.read_text(encoding="utf-8"))
        if timings_path.exists()
        else {}
    )
    payload = {
        "status": "complete",
        "experiment": "v1_original_conv2_vs_conv3_matched_input",
        "games_per_seat": games_per_seat,
        "evaluation_seed": seed,
        "training_seed": int(checkpoint["seed"]),
        "training_games": int(checkpoint["training_games"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_contract": expected_contract,
        "metrics": checkpoint["metrics"],
        "timings_seconds": timings,
        "baseline": {
            "checkpoint": baseline["models"]["original"],
            "all_seats_games": int(baseline_row["all_seats_games"]),
            "all_seats_win_rate": baseline_rate,
            "all_seats_one_card_turn_rate": float(
                baseline_row["all_seats_one_card_turn_rate"]
            ),
            "metrics": baseline_checkpoint.get("metrics"),
            "training_seconds": baseline_training_seconds,
            "source": str(baseline_path),
        },
        "conv3": {
            "seats": seats,
            "all_seats_games": total_games,
            "all_seats_fractional_wins": total_wins,
            "all_seats_win_rate": conv3_rate,
            "all_seats_one_card_turns": total_one,
            "all_seats_two_card_turns": total_two,
            "all_seats_one_card_turn_rate": (
                total_one / (total_one + total_two)
            ),
        },
        "conv3_minus_conv2_win_rate": conv3_rate - baseline_rate,
        "automatic_model_replacement": False,
    }
    atomic_write(
        output_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Original V1 Conv2 vs Conv3 matched-input comparison",
        "",
        "| Model | Seat 0 | Seat 1 | Seat 2 | Seat 3 | All seats | One-card rate |",
        "|:--|--:|--:|--:|--:|--:|--:|",
        (
            "| Conv2 baseline | "
            + " | ".join(
                f"{float(row['win_rate']):.3%}"
                for row in baseline_row["seats"]
            )
            + f" | {baseline_rate:.3%} | "
            f"{float(baseline_row['all_seats_one_card_turn_rate']):.3%} |"
        ),
        (
            "| Conv3 | "
            + " | ".join(
                f"{float(row['win_rate']):.3%}" for row in seats
            )
            + f" | {conv3_rate:.3%} | "
            f"{payload['conv3']['all_seats_one_card_turn_rate']:.3%} |"
        ),
        "",
        f"- Conv3 - Conv2: `{conv3_rate - baseline_rate:+.3%}`",
        "- This comparison uses matched rolling-history input contracts.",
        "- No model is replaced automatically.",
    ]
    atomic_write(markdown_path, "\n".join(lines) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-comparison", type=Path)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    summarize(
        checkpoint_path=args.checkpoint,
        baseline_path=args.baseline,
        baseline_comparison_path=args.baseline_comparison,
        evaluation_directory=args.evaluation_directory,
        timings_path=args.timings,
        output_path=args.output,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
