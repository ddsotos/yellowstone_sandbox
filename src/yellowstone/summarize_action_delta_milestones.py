"""Summarize seat-0 evaluations of continuous action-delta checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_action_delta_milestones(
    training_summary: str | Path,
    evaluation_directory: str | Path,
    output: str | Path,
    *,
    games: int,
    seed: int,
    player_index: int = 0,
) -> dict:
    training_path = Path(training_summary)
    training = json.loads(training_path.read_text(encoding="utf-8-sig"))
    evaluation_root = Path(evaluation_directory)
    rows = []
    for milestone in training["milestones"]:
        percent = int(milestone["percent"])
        evaluation_path = evaluation_root / (
            f"action_delta_milestone_pct{percent:03d}_{games}_"
            f"seed{seed}_p{player_index}.json"
        )
        evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8-sig")
        )
        if (
            int(evaluation["games"]) != games
            or int(evaluation["seed"]) != seed
            or int(evaluation["player_index"]) != player_index
        ):
            raise ValueError(f"incompatible milestone evaluation: {evaluation_path}")
        rows.append(
            {
                **milestone,
                "evaluation": evaluation,
            }
        )
    payload = {
        "status": "complete",
        "experiment": "action_delta_continuous_epoch001_milestones",
        "official_four_seat_evaluation": False,
        "screen_player_index": player_index,
        "games_per_checkpoint": games,
        "evaluation_seed": seed,
        "training": {
            key: training[key]
            for key in (
                "snapshot",
                "snapshot_sha256",
                "total_records",
                "total_train_records",
                "seed",
                "batch_size",
                "learning_rate",
                "epochs",
            )
        },
        "milestones": rows,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Action-delta continuous-training milestone screen",
        "",
        (
            f"Seat {player_index}, {games} games/checkpoint, seed {seed}. "
            "This is not an official four-seat evaluation."
        ),
        "",
        "| train | records | test MAE | test RMSE | win rate | one-card rate | one | two | seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        evaluation = row["evaluation"]
        metrics = row["metrics"]
        lines.append(
            f"| {row['percent']}% | {row['processed_train_records']} | "
            f"{metrics['test_all_mae']:.6f} | "
            f"{metrics['test_all_rmse']:.6f} | "
            f"{evaluation['win_rate']:.3%} | "
            f"{evaluation['evaluated_player_one_card_turn_rate']:.3%} | "
            f"{evaluation['evaluated_player_one_card_turns']} | "
            f"{evaluation['evaluated_player_two_card_turns']} | "
            f"{evaluation['elapsed_seconds']:.1f} |"
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
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--player-index", type=int, default=0)
    args = parser.parse_args()
    payload = summarize_action_delta_milestones(
        args.training_summary,
        args.evaluation_directory,
        args.output,
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
