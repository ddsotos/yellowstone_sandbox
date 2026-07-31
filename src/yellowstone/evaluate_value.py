"""Command-line evaluation of one learned value player versus heuristics."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import fmean
from time import monotonic

from yellowstone.serialization import action_to_dict, game_state_to_dict
from yellowstone.value_evaluation import evaluate_value_player
from yellowstone.value_policy import TorchWinValueEstimator

HISTORYFIX_VALUE_SCHEMA = "yellowstone.value.v1_historyfix"
HISTORYFIX_HISTORY_SEMANTICS = (
    "evaluated_turn_only_one_card_zero_padded"
)
ORIGINAL_VALUE_SCHEMA = "yellowstone.value.v1"
ORIGINAL_HISTORY_SEMANTICS = "rolling_last_two_placements"
CANONICALIZATION = "fast_lr_ud_color_v1"


def validate_checkpoint_contract(
    checkpoint_path: Path,
    *,
    current_turn_history_only: bool,
) -> dict[str, object]:
    """Hard-fail unless checkpoint metadata matches inference history."""
    import torch
    from yellowstone.cnn import win_value_architecture_from_checkpoint

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    expected = {
        "value_schema": (
            HISTORYFIX_VALUE_SCHEMA
            if current_turn_history_only
            else ORIGINAL_VALUE_SCHEMA
        ),
        "history_semantics": (
            HISTORYFIX_HISTORY_SEMANTICS
            if current_turn_history_only
            else ORIGINAL_HISTORY_SEMANTICS
        ),
        "input_canonicalization": CANONICALIZATION,
    }
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "checkpoint/inference input contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        **expected,
        **win_value_architecture_from_checkpoint(checkpoint),
        "inference_history": (
            "turn_local"
            if current_turn_history_only
            else "rolling"
        ),
    }


def evaluation_payload(
    result: object,
    *,
    boost_percent: float,
    checkpoint_contract: dict[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    selections = result.selections
    raw = [selection.predicted_win_probability for selection in selections]
    adjusted = [selection.selection_score for selection in selections]

    def stats(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "mean": fmean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "games": result.games,
        "fractional_wins": result.wins,
        "wins": result.wins,
        "win_rate": result.win_rate,
        "one_card_win_probability_boost_percent": boost_percent,
        "evaluated_player_one_card_turns": (
            result.evaluated_player_one_card_turns
        ),
        "evaluated_player_two_card_turns": (
            result.evaluated_player_two_card_turns
        ),
        "evaluated_player_one_card_turn_rate": (
            result.evaluated_player_one_card_turn_rate
        ),
        "selected_raw_win_probability": stats(raw),
        "selected_adjusted_score": stats(adjusted),
        "policy_fingerprint": result.policy_fingerprint,
        "all_one_card_candidates_saturated": (
            result.all_one_card_candidates_saturated
        ),
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_contract": checkpoint_contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Yellowstone win-value player")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--duration-hours", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument(
        "--prune-negative-above",
        type=int,
        help="activate only with a safe two-card witness; keep candidates with increase <= this value",
    )
    parser.add_argument("--adaptive-pq-pruning", action="store_true")
    parser.add_argument("--approximate-new-color-neighbors", action="store_true")
    parser.add_argument(
        "--current-turn-history-only",
        action="store_true",
        help="use one/two placements from the evaluated candidate only",
    )
    parser.add_argument(
        "--one-card-win-probability-boost-percent",
        type=float,
        default=0.0,
        help="relative percent boost applied only to one-card candidate scores",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration_hours is not None and args.duration_hours <= 0:
        parser.error("--duration-hours must be positive")
    if args.adaptive_pq_pruning and args.prune_negative_above is not None:
        parser.error("choose either --adaptive-pq-pruning or --prune-negative-above")
    if (
        not isfinite(args.one_card_win_probability_boost_percent)
        or args.one_card_win_probability_boost_percent < 0
    ):
        parser.error(
            "--one-card-win-probability-boost-percent must be finite and non-negative"
        )
    checkpoint_contract = validate_checkpoint_contract(
        args.checkpoint,
        current_turn_history_only=args.current_turn_history_only,
    )
    started = monotonic()
    result = evaluate_value_player(
        TorchWinValueEstimator(str(args.checkpoint)),
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
        prune_negative_card_increase_above=args.prune_negative_above,
        adaptive_pq_pruning=args.adaptive_pq_pruning,
        approximate_new_color_neighbor_limit=args.approximate_new_color_neighbors,
        current_turn_history_only=args.current_turn_history_only,
        one_card_win_probability_boost_percent=(
            args.one_card_win_probability_boost_percent
        ),
        duration_seconds=(args.duration_hours * 3600 if args.duration_hours else None),
    )
    top10 = {}
    for profile in ("p", "q"):
        selected = sorted(
            (record for record in result.pruning_records if record[0] == profile),
            key=lambda record: record[2].negative_card_increase,
            reverse=True,
        )[:10]
        top10[profile] = [
            {
                "negative_card_increase": selection.negative_card_increase,
                "predicted_win_probability": selection.predicted_win_probability,
                "selection_score": selection.selection_score,
                "actions": [action_to_dict(action) for action in selection.candidate.actions],
                "complete_state": game_state_to_dict(state),
            }
            for _, state, selection in selected
        ]
    payload = evaluation_payload(
        result,
        boost_percent=args.one_card_win_probability_boost_percent,
        checkpoint_contract=checkpoint_contract,
        elapsed_seconds=monotonic() - started,
    )
    payload["pruning_top10"] = top10
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
