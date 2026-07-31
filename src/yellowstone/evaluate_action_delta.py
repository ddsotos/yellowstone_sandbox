"""Evaluate one public action-delta player against heuristic opponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Literal

from yellowstone.action_delta import (
    ActionDeltaPlayer,
    TorchActionDeltaEstimator,
)
from yellowstone.bots import HeuristicBot
from yellowstone.game import (
    apply_known_legal_action,
    create_initial_state,
    legal_actions,
)
from yellowstone.privileged_state import (
    PrivilegedStateRecord,
    TorchPrivilegedStateEstimator,
)
from yellowstone.types import Phase, PlaceCardAction


PrePlayMode = Literal["none", "privileged_audit"]


def combined_post_play_probability(
    pre_play_probability: float,
    predicted_delta: float,
) -> float:
    return min(1.0, max(0.0, pre_play_probability + predicted_delta))


def evaluate_action_delta(
    delta_checkpoint: str | Path,
    *,
    games: int,
    seed: int,
    player_index: int,
    pre_play_mode: PrePlayMode = "none",
    pre_play_checkpoint: str | Path | None = None,
) -> dict:
    started = perf_counter()
    seeds = Random(seed)
    estimator = TorchActionDeltaEstimator(delta_checkpoint)
    if pre_play_mode == "privileged_audit":
        if pre_play_checkpoint is None:
            raise ValueError(
                "privileged_audit requires a pre-play checkpoint"
            )
        pre_play_estimator = TorchPrivilegedStateEstimator(
            pre_play_checkpoint
        )
    elif pre_play_mode == "none":
        if pre_play_checkpoint is not None:
            raise ValueError(
                "pre-play checkpoint requires privileged_audit mode"
            )
        pre_play_estimator = None
    else:
        raise ValueError(f"unknown pre-play mode: {pre_play_mode}")
    wins = 0.0
    turns = one = candidate_sum = 0
    delta_sum = 0.0
    pre_play_sum = combined_sum = 0.0
    combined_clipped = 0
    for _ in range(games):
        state = create_initial_state(4, seed=seeds.randrange(2**63))
        rng = Random(seeds.randrange(2**63))
        player = ActionDeltaPlayer(player_index, None, estimator)
        heuristic = HeuristicBot()
        while state.phase != Phase.GAME_OVER:
            pending_pre_play_probability = None
            choices_before = len(player.choices)
            if (
                pre_play_estimator is not None
                and state.current_player_index == player_index
                and state.phase == Phase.PLAY
                and not player.remaining_plan
                and any(
                    isinstance(action, PlaceCardAction)
                    for action in legal_actions(state)
                )
            ):
                values = pre_play_estimator(
                    PrivilegedStateRecord(
                        game_id=-1,
                        state=state,
                        history=tuple(player.history),
                        target=(0.0, 0.0, 0.0, 0.0),
                    )
                )
                pending_pre_play_probability = values[0]
            action = (
                player.choose_action(state)
                if state.current_player_index == player_index
                else heuristic.choose_action(state)
            )
            if action is None:
                raise RuntimeError("evaluation policy stopped")
            before = state
            state = apply_known_legal_action(state, action, rng=rng)
            player.observe(before, action, state)
            if len(player.choices) > choices_before:
                if pending_pre_play_probability is None:
                    if pre_play_estimator is not None:
                        raise AssertionError(
                            "missing privileged pre-play probability"
                        )
                else:
                    predicted_delta = (
                        player.choices[-1].predicted_win_probability
                    )
                    unbounded = (
                        pending_pre_play_probability + predicted_delta
                    )
                    pre_play_sum += pending_pre_play_probability
                    combined_sum += combined_post_play_probability(
                        pending_pre_play_probability,
                        predicted_delta,
                    )
                    combined_clipped += int(
                        unbounded < 0.0 or unbounded > 1.0
                    )
        for choice in player.choices:
            turns += 1
            one += int(
                sum(
                    isinstance(action, PlaceCardAction)
                    for action in choice.actions
                )
                == 1
            )
            candidate_sum += choice.candidate_count
            delta_sum += choice.predicted_win_probability
        if player_index in state.winners:
            wins += 1.0 / len(state.winners)
    result = {
        "games": games,
        "fractional_wins": wins,
        "win_rate": wins / games,
        "seed": seed,
        "player_index": player_index,
        "turns": turns,
        "evaluated_player_one_card_turns": one,
        "evaluated_player_two_card_turns": turns - one,
        "evaluated_player_one_card_turn_rate": one / turns if turns else 0.0,
        "mean_candidate_count": candidate_sum / turns if turns else 0.0,
        "mean_predicted_delta": delta_sum / turns if turns else 0.0,
        "candidate_source": "all_retained_turn_end_candidates",
        "adaptive_pq_pruning": True,
        "approximate_new_color_neighbor_limit": True,
        "pre_play_mode": pre_play_mode,
        "elapsed_seconds": perf_counter() - started,
    }
    if pre_play_estimator is not None:
        result["pre_play_audit"] = {
            "audit_only": True,
            "privileged_inputs": True,
            "uses_opponent_private_hands": True,
            "checkpoint": str(pre_play_checkpoint),
            "value_schema": "yellowstone.value.privileged-state.v1",
            "history_semantics": (
                "rolling_last_two_placements_before_turn"
            ),
            "turns": turns,
            "mean_pre_play_win_probability": (
                pre_play_sum / turns if turns else 0.0
            ),
            "mean_combined_post_play_win_probability": (
                combined_sum / turns if turns else 0.0
            ),
            "combined_probability_clipped_turns": combined_clipped,
            "combined_probability_clipped_rate": (
                combined_clipped / turns if turns else 0.0
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposer-checkpoint",
        type=Path,
        help="deprecated compatibility option; ignored in all-candidate mode",
    )
    parser.add_argument("--delta-checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, required=True)
    parser.add_argument(
        "--pre-play-mode",
        choices=("none", "privileged_audit"),
        default="none",
    )
    parser.add_argument("--pre-play-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_action_delta(
        args.delta_checkpoint,
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
        pre_play_mode=args.pre_play_mode,
        pre_play_checkpoint=args.pre_play_checkpoint,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
