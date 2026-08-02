"""Evaluate one V1 value player against three exploratory NPCs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from time import monotonic

from yellowstone.evaluate_value import evaluation_payload, validate_checkpoint_contract
from yellowstone.exploratory_collection import (
    ExploratoryValueNpc,
    choose_exploratory_refill,
)
from yellowstone.fast_value_npc import _append_history
from yellowstone.game import apply_known_legal_action, create_initial_state, legal_actions
from yellowstone.serialization import action_to_dict, game_state_to_dict
from yellowstone.types import Phase, PlaceCardAction
from yellowstone.value_evaluation import ValueTurnPlayer
from yellowstone.value_learning import HISTORY_SIZE, RecentPlacement
from yellowstone.value_policy import TorchWinValueEstimator


def evaluate_value_player_vs_explore(
    value_checkpoint: Path,
    explore_checkpoint: Path,
    *,
    games: int,
    seed: int,
    player_index: int,
    adaptive_pq_pruning: bool,
    approximate_new_color_neighbor_limit: bool,
    current_turn_history_only: bool,
    lazy_single_pass: bool,
) -> object:
    """Return the same EvaluationResult shape as heuristic-opponent evaluation."""
    from yellowstone.value_evaluation import EvaluationResult

    if games <= 0:
        raise ValueError("games must be positive")
    seeds = Random(seed)
    wins = 0.0
    one_card_turns = 0
    two_card_turns = 0
    pruning_records = []
    selections = []
    estimator = TorchWinValueEstimator(str(value_checkpoint))

    for _ in range(games):
        state = create_initial_state(4, seed=seeds.randrange(2**63))
        gameplay_rng = Random(seeds.randrange(2**63))
        explore_rng = Random(seeds.randrange(2**63))
        value_player = ValueTurnPlayer(
            player_index,
            estimator,
            adaptive_pq_pruning=adaptive_pq_pruning,
            approximate_new_color_neighbor_limit=(
                approximate_new_color_neighbor_limit
            ),
            current_turn_history_only=current_turn_history_only,
        )
        explore_npc = ExploratoryValueNpc(
            explore_checkpoint,
            lazy_single_pass=lazy_single_pass,
        )
        history: list[RecentPlacement] = []
        planned = []
        planned_player = None

        while state.phase != Phase.GAME_OVER:
            acting_player = state.current_player_index
            if acting_player == player_index:
                action = value_player.choose_action(state)
            elif planned:
                if acting_player != planned_player:
                    raise AssertionError("planned exploratory player changed")
                action = planned.pop(0)
            elif (
                state.phase == Phase.PLAY
                and state.cards_played_this_turn == 0
                and any(isinstance(item, PlaceCardAction) for item in legal_actions(state))
            ):
                choice = explore_npc.choose_turn(
                    state,
                    tuple(history[-HISTORY_SIZE:]),
                    rng=explore_rng,
                )
                planned = list(choice.actions)
                planned_player = acting_player
                action = planned.pop(0)
            else:
                action, _ = choose_exploratory_refill(state, rng=explore_rng)

            if action not in legal_actions(state):
                raise RuntimeError(f"policy selected illegal action: {action!r}")
            before = state
            state = apply_known_legal_action(state, action, rng=gameplay_rng)
            if before.current_player_index == player_index:
                if (
                    action.__class__.__name__ == "EndTurnAction"
                    and before.cards_played_this_turn == 1
                ):
                    one_card_turns += 1
                elif (
                    isinstance(action, PlaceCardAction)
                    and before.cards_played_this_turn == 1
                ):
                    two_card_turns += 1
            value_player.observe(before, action, state)
            _append_history(history, before, action, state)
            if (
                before.current_player_index != state.current_player_index
                or state.phase == Phase.GAME_OVER
            ):
                planned = []
                planned_player = None

        if player_index in state.winners:
            wins += 1.0 / len(state.winners)
        pruning_records.extend(value_player.pruning_records)
        selections.extend(value_player.selections)

    return EvaluationResult(
        games=games,
        wins=wins,
        pruning_records=tuple(pruning_records),
        evaluated_player_one_card_turns=one_card_turns,
        evaluated_player_two_card_turns=two_card_turns,
        selections=tuple(selections),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--explore-checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--adaptive-pq-pruning", action="store_true")
    parser.add_argument("--approximate-new-color-neighbors", action="store_true")
    parser.add_argument("--current-turn-history-only", action="store_true")
    parser.add_argument("--no-lazy-single-pass", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_contract = validate_checkpoint_contract(
        args.checkpoint,
        current_turn_history_only=args.current_turn_history_only,
    )
    started = monotonic()
    result = evaluate_value_player_vs_explore(
        args.checkpoint,
        args.explore_checkpoint,
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
        adaptive_pq_pruning=args.adaptive_pq_pruning,
        approximate_new_color_neighbor_limit=args.approximate_new_color_neighbors,
        current_turn_history_only=args.current_turn_history_only,
        lazy_single_pass=not args.no_lazy_single_pass,
    )
    payload = evaluation_payload(
        result,
        boost_percent=0.0,
        checkpoint_contract=checkpoint_contract,
        elapsed_seconds=monotonic() - started,
    )
    payload["opponent_policy"] = "exploratory_value_npc"
    payload["opponent_checkpoint"] = str(args.explore_checkpoint)
    payload["opponent_lazy_single_pass"] = not args.no_lazy_single_pass
    payload["pruning_top10"] = {
        profile: [
            {
                "negative_card_increase": selection.negative_card_increase,
                "predicted_win_probability": selection.predicted_win_probability,
                "selection_score": selection.selection_score,
                "actions": [
                    action_to_dict(action)
                    for action in selection.candidate.actions
                ],
                "complete_state": game_state_to_dict(state),
            }
            for _, state, selection in sorted(
                (
                    record
                    for record in result.pruning_records
                    if record[0] == profile
                ),
                key=lambda record: record[2].negative_card_increase,
                reverse=True,
            )[:10]
        ]
        for profile in ("p", "q")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
