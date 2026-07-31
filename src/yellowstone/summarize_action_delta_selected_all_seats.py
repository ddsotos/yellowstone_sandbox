"""Summarize four-seat screens for selected action-delta milestones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_selected_all_seats(
    training_summary: str | Path,
    evaluation_directory: str | Path,
    output: str | Path,
    *,
    percentages: tuple[int, ...],
    games_per_seat: int,
    seed: int,
) -> dict:
    source = json.loads(
        Path(training_summary).read_text(encoding="utf-8-sig")
    )
    milestone_by_percent = {
        int(row["percent"]): row for row in source["milestones"]
    }
    evaluation_root = Path(evaluation_directory)
    results = []
    for percent in percentages:
        milestone = milestone_by_percent.get(percent)
        if milestone is None:
            raise ValueError(f"missing training milestone: {percent}")
        seats = []
        for player_index in range(4):
            path = evaluation_root / (
                f"action_delta_milestone_pct{percent:03d}_{games_per_seat}_"
                f"seed{seed}_p{player_index}.json"
            )
            row = json.loads(path.read_text(encoding="utf-8-sig"))
            if (
                int(row["games"]) != games_per_seat
                or int(row["seed"]) != seed
                or int(row["player_index"]) != player_index
            ):
                raise ValueError(f"incompatible milestone evaluation: {path}")
            seats.append(row)
        total_games = sum(int(row["games"]) for row in seats)
        total_wins = sum(float(row["fractional_wins"]) for row in seats)
        total_one = sum(
            int(row["evaluated_player_one_card_turns"]) for row in seats
        )
        total_two = sum(
            int(row["evaluated_player_two_card_turns"]) for row in seats
        )
        results.append(
            {
                "percent": percent,
                "checkpoint": milestone["checkpoint"],
                "processed_train_records": milestone[
                    "processed_train_records"
                ],
                "actual_fraction": milestone["actual_fraction"],
                "metrics": milestone["metrics"],
                "seats": seats,
                "all_seats_games": total_games,
                "all_seats_fractional_wins": total_wins,
                "all_seats_win_rate": total_wins / total_games,
                "all_seats_one_card_turns": total_one,
                "all_seats_two_card_turns": total_two,
                "all_seats_one_card_turn_rate": (
                    total_one / (total_one + total_two)
                ),
            }
        )
    payload = {
        "status": "complete",
        "experiment": "action_delta_selected_milestones_all_seats",
        "evaluation_seed": seed,
        "games_per_seat": games_per_seat,
        "same_conditions_all_seats": True,
        "training": source["training"],
        "milestones": results,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Action-delta selected milestones: four-seat screen",
        "",
        f"Seed {seed}, {games_per_seat} games/seat.",
        "",
        "| train | seat | win rate | one-card rate | one | two |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for row in result["seats"]:
            lines.append(
                f"| {result['percent']}% | {row['player_index']} | "
                f"{row['win_rate']:.3%} | "
                f"{row['evaluated_player_one_card_turn_rate']:.3%} | "
                f"{row['evaluated_player_one_card_turns']} | "
                f"{row['evaluated_player_two_card_turns']} |"
            )
        lines.append(
            f"| {result['percent']}% | all | "
            f"{result['all_seats_win_rate']:.3%} | "
            f"{result['all_seats_one_card_turn_rate']:.3%} | "
            f"{result['all_seats_one_card_turns']} | "
            f"{result['all_seats_two_card_turns']} |"
        )
    output_path.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentages", default="30,100")
    parser.add_argument("--games-per-seat", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    result = summarize_selected_all_seats(
        args.training_summary,
        args.evaluation_directory,
        args.output,
        percentages=tuple(
            int(value.strip())
            for value in args.percentages.split(",")
            if value.strip()
        ),
        games_per_seat=args.games_per_seat,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
