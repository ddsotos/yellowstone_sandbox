"""Summarize V2-lite offline metrics, win rates, and one-card rates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def summarize(
    *,
    checkpoint_path: Path,
    evaluation_directory: Path,
    output_path: Path,
    games_per_seat: int,
    seed: int,
    timings_path: Path,
) -> dict[str, object]:
    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    seats = []
    for player_index in range(4):
        path = evaluation_directory / (
            "v2_lite_transition_generation0_197800_epoch001_"
            f"{games_per_seat}_seed{seed}_p{player_index}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["games"]) != games_per_seat:
            raise ValueError(f"unexpected game count: {path}")
        seats.append(payload)
    total_games = sum(int(row["games"]) for row in seats)
    total_wins = sum(float(row["wins"]) for row in seats)
    total_one = sum(int(row["one_card_turns"]) for row in seats)
    total_turns = sum(int(row["turns"]) for row in seats)
    timings = (
        json.loads(timings_path.read_text(encoding="utf-8-sig"))
        if timings_path.exists()
        else {}
    )
    result = {
        "model": "V2-lite transition",
        "checkpoint": str(checkpoint_path),
        "training_games": 197800,
        "epochs": 1,
        "context_size": int(checkpoint["context_size"]),
        "board_channels": 58,
        "training_seed": int(checkpoint["seed"]),
        "evaluation_seed": seed,
        "games_per_seat": games_per_seat,
        "metrics": checkpoint["metrics"],
        "seats": seats,
        "all_seats_win_rate": total_wins / total_games,
        "all_seats_one_card_turn_rate": total_one / total_turns,
        "timings_seconds": timings,
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V2-lite transition evaluation",
        "",
        "| seat | win rate | one-card rate | one | two |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in seats:
        lines.append(
            f"| {row['player_index']} | {100 * row['win_rate']:.3f}% "
            f"| {100 * row['one_card_turn_rate']:.3f}% "
            f"| {row['one_card_turns']} | {row['two_card_turns']} |"
        )
    lines.extend(
        [
            f"| all | {100 * result['all_seats_win_rate']:.3f}% "
            f"| {100 * result['all_seats_one_card_turn_rate']:.3f}% "
            f"| {total_one} | {total_turns - total_one} |",
            "",
            f"- test Brier: {checkpoint['metrics']['test_brier']:.6f}",
            f"- test logloss: {checkpoint['metrics']['test_log_loss']:.6f}",
            "- context: 138 values",
            "- board: after 29 channels + signed delta 29 channels",
        ]
    )
    output_path.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V2-lite")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--timings", type=Path, required=True)
    args = parser.parse_args()
    summarize(
        checkpoint_path=args.checkpoint,
        evaluation_directory=args.evaluation_directory,
        output_path=args.output,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        timings_path=args.timings,
    )


if __name__ == "__main__":
    main()
