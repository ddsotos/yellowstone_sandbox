"""Summarize the Original/Historyfix checkpoint by history-input cross."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = (
    ("original", "rolling"),
    ("original", "turn_local"),
    ("historyfix", "rolling"),
    ("historyfix", "turn_local"),
)
MATCHED_CONDITIONS = {
    ("original", "rolling"),
    ("historyfix", "turn_local"),
}


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def summarize(
    *,
    evaluation_directory: Path,
    output: Path,
    games_per_seat: int,
    seed: int,
) -> dict[str, object]:
    rows = []
    for model, history in CONDITIONS:
        seats = []
        total_games = 0
        total_wins = 0.0
        total_one = 0
        total_two = 0
        for player_index in range(4):
            path = evaluation_directory / (
                f"v1_history_cross_{model}_{history}_epoch002_"
                f"{games_per_seat}_seed{seed}_p{player_index}.json"
            )
            result = _read(path)
            if int(result["games"]) != games_per_seat:
                raise ValueError(f"game count differs: {path}")
            one = int(result["evaluated_player_one_card_turns"])
            two = int(result["evaluated_player_two_card_turns"])
            rate = float(result["evaluated_player_one_card_turn_rate"])
            if one + two <= 0 or abs(rate - one / (one + two)) > 1e-12:
                raise ValueError(f"one-card turn facts differ: {path}")
            games = int(result["games"])
            wins = float(result["wins"])
            total_games += games
            total_wins += wins
            total_one += one
            total_two += two
            seats.append(
                {
                    "player_index": player_index,
                    "games": games,
                    "wins": wins,
                    "win_rate": float(result["win_rate"]),
                    "one_card_turns": one,
                    "two_card_turns": two,
                    "one_card_turn_rate": rate,
                    "result": path.as_posix(),
                }
            )
        rows.append(
            {
                "checkpoint_model": model,
                "inference_history": history,
                "history_semantics_match": (
                    (model, history) in MATCHED_CONDITIONS
                ),
                "seats": seats,
                "all_seats_games": total_games,
                "all_seats_wins": total_wins,
                "all_seats_win_rate": total_wins / total_games,
                "all_seats_one_card_turns": total_one,
                "all_seats_two_card_turns": total_two,
                "all_seats_one_card_turn_rate": (
                    total_one / (total_one + total_two)
                ),
            }
        )
    matched_rows = [
        row for row in rows if row["history_semantics_match"]
    ]
    mismatch_rows = [
        row for row in rows if not row["history_semantics_match"]
    ]
    payload: dict[str, object] = {
        "status": "complete",
        "experiment": "v1_history_semantics_matched_evaluation",
        "result_policy": (
            "Only checkpoint/inference pairs with matching history semantics "
            "are model-performance results."
        ),
        "games_per_seat": games_per_seat,
        "seed": seed,
        "adaptive_pq_pruning": True,
        "approximate_new_color_neighbors": True,
        "traditional_refill": True,
        "models": {
            "original": (
                "models/win_value_v1_original_generation0_"
                "197800_epoch002.pt"
            ),
            "historyfix": (
                "models/win_value_v1_historyfix_generation0_"
                "197800_epoch002.pt"
            ),
        },
        "rows": matched_rows,
        "excluded_mismatched_conditions": [
            {
                "checkpoint_model": row["checkpoint_model"],
                "inference_history": row["inference_history"],
                "reason": "training and inference history semantics differ",
            }
            for row in mismatch_rows
        ],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(payload, output.with_suffix(".md"))
    canonical_output = output.with_name("v1_history_matched_evaluation.json")
    canonical_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(payload, canonical_output.with_suffix(".md"))

    audit_payload: dict[str, object] = {
        "status": "complete",
        "experiment": "v1_history_semantics_mismatch_input_contract_audit",
        "warning": (
            "Deliberately mismatched input-contract results. Do not use as "
            "model win rates or in model comparisons."
        ),
        "games_per_seat": games_per_seat,
        "seed": seed,
        "rows": mismatch_rows,
    }
    audit_output = output.with_name("v1_history_mismatch_input_audit.json")
    audit_output.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_markdown(payload: dict[str, object], path: Path) -> None:
    lines = [
        "# V1 checkpoint × inference history 2×2 cross",
        "",
        "| Checkpoint | Inference history | Seat 0 | Seat 1 | "
        "Seat 2 | Seat 3 | All seats | One-card rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['checkpoint_model']} | {row['inference_history']} | "
            + " | ".join(
                f"{seat['win_rate']:.3%}" for seat in row["seats"]
            )
            + f" | {row['all_seats_win_rate']:.3%} | "
            f"{row['all_seats_one_card_turn_rate']:.3%} |"
        )
    lines.extend(
        [
            "",
            "One-card rate is evaluated-player one-card PLAY turns divided by "
            "all evaluated-player completed PLAY turns.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    result = summarize(
        evaluation_directory=args.evaluation_directory,
        output=args.output,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
