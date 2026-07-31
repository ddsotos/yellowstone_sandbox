"""Summarize the four-seat public action-delta screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(
    *,
    proposer_selection: Path,
    critic_selection: Path,
    delta_checkpoint: Path,
    evaluation_directory: Path,
    output: Path,
    games_per_seat: int,
    seed: int,
) -> dict:
    import torch

    proposer = json.loads(proposer_selection.read_text(encoding="utf-8-sig"))
    critic = json.loads(critic_selection.read_text(encoding="utf-8-sig"))
    checkpoint = torch.load(
        delta_checkpoint, map_location="cpu", weights_only=False
    )
    seats = []
    for player in range(4):
        path = evaluation_directory / (
            f"action_delta_{games_per_seat}_seed{seed}_p{player}.json"
        )
        row = json.loads(path.read_text(encoding="utf-8-sig"))
        if int(row["games"]) != games_per_seat:
            raise ValueError(f"incomplete action-delta evaluation: {path}")
        seats.append(row)
    total_games = sum(int(row["games"]) for row in seats)
    total_wins = sum(float(row["fractional_wins"]) for row in seats)
    total_one = sum(int(row["evaluated_player_one_card_turns"]) for row in seats)
    total_two = sum(int(row["evaluated_player_two_card_turns"]) for row in seats)
    payload = {
        "status": "complete",
        "experiment": "privileged_state_public_action_delta_v1",
        "official_policy_input_privileged": False,
        "critic_is_privileged_audit_only": True,
        "proposer": proposer,
        "critic": critic,
        "delta_checkpoint": str(delta_checkpoint),
        "delta_metrics": checkpoint["metrics"],
        "evaluation_seed": seed,
        "games_per_seat": games_per_seat,
        "seats": seats,
        "all_seats_games": total_games,
        "all_seats_fractional_wins": total_wins,
        "all_seats_win_rate": total_wins / total_games,
        "all_seats_one_card_turns": total_one,
        "all_seats_two_card_turns": total_two,
        "all_seats_one_card_turn_rate": total_one / (total_one + total_two),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Privileged state / public action-delta evaluation",
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
            (
                f"| all | {payload['all_seats_win_rate']:.3%} | "
                f"{payload['all_seats_one_card_turn_rate']:.3%} | "
                f"{total_one} | {total_two} |"
            ),
            "",
            "- The four-output state critic uses privileged hands and is audit-only.",
            "- The deployed action-delta model contains no opponent-private inputs.",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposer-selection", type=Path, required=True)
    parser.add_argument("--critic-selection", type=Path, required=True)
    parser.add_argument("--delta-checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                proposer_selection=args.proposer_selection,
                critic_selection=args.critic_selection,
                delta_checkpoint=args.delta_checkpoint,
                evaluation_directory=args.evaluation_directory,
                output=args.output,
                games_per_seat=args.games_per_seat,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
