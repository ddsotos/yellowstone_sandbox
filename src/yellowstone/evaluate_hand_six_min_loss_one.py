"""Evaluate a board-five hand-six one-off heuristic against base heuristic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from time import monotonic

from yellowstone.bots import (
    FixedFrameHandSixOneOffMinLossOneCardBot,
    HeuristicBot,
    ONE_OFF_BUCKET_PROBABILITIES,
    _board_card_count,
)
from yellowstone.game import apply_known_legal_action, create_initial_state, legal_actions
from yellowstone.types import EndTurnAction, Phase, PlaceCardAction

POLICY_NAME = "board_five_hand_six_one_off_tiered_min_loss_one_card"


def evaluate_hand_six_min_loss_one(
    *,
    games: int,
    seed: int,
    player_index: int,
) -> dict[str, object]:
    """Return seat win rate for the variant policy vs three heuristics."""
    if games <= 0:
        raise ValueError("games must be positive")
    if not 0 <= player_index < 4:
        raise ValueError("player_index must be in 0..3")

    seed_rng = Random(seed)
    wins = 0.0
    one_card_turns = 0
    two_card_turns = 0
    hand_six_turn_starts = 0
    hand_six_board_five_turn_starts = 0
    hand_six_one_card_turns = 0
    hand_six_branch_eligible_turns = 0
    hand_six_branch_taken_turns = 0
    eligible_by_one_off = {"0": 0, "1": 0, "2_plus": 0}
    taken_by_one_off = {"0": 0, "1": 0, "2_plus": 0}
    started = monotonic()

    for _ in range(games):
        state = create_initial_state(4, seed=seed_rng.randrange(2**63))
        gameplay_rng = Random(seed_rng.randrange(2**63))
        variant = FixedFrameHandSixOneOffMinLossOneCardBot(
            rng=Random(seed_rng.randrange(2**63)),
        )
        heuristic = HeuristicBot()

        while state.phase != Phase.GAME_OVER:
            before = state
            if before.current_player_index == player_index:
                policy = variant
            else:
                policy = heuristic
            action = policy.choose_action(before)
            if action is None:
                raise RuntimeError("policy returned no action before game end")
            if action not in legal_actions(before):
                raise RuntimeError(f"policy selected illegal action: {action!r}")
            state = apply_known_legal_action(before, action, rng=gameplay_rng)

            if before.current_player_index == player_index:
                branch = variant.last_branch
                if branch is not None:
                    hand_six_branch_eligible_turns += 1
                    eligible_by_one_off[branch.bucket] += 1
                if (
                    before.phase == Phase.PLAY
                    and before.cards_played_this_turn == 0
                    and isinstance(action, PlaceCardAction)
                    and len(before.players[player_index].hand) == 6
                ):
                    hand_six_turn_starts += 1
                    if _board_card_count(before) >= 5:
                        hand_six_board_five_turn_starts += 1
                if (
                    isinstance(action, EndTurnAction)
                    and before.cards_played_this_turn == 1
                ):
                    one_card_turns += 1
                    if len(before.players[player_index].hand) == 5:
                        hand_six_one_card_turns += 1
                elif (
                    isinstance(action, PlaceCardAction)
                    and before.cards_played_this_turn == 1
                ):
                    two_card_turns += 1
                if branch is not None and branch.taken:
                    hand_six_branch_taken_turns += 1
                    taken_by_one_off[branch.bucket] += 1

        if player_index in state.winners:
            wins += 1.0 / len(state.winners)

    completed_turns = one_card_turns + two_card_turns
    return {
        "policy": POLICY_NAME,
        "opponent_policy": "heuristic",
        "games": games,
        "seed": seed,
        "player_index": player_index,
        "one_off_bucket_probabilities": ONE_OFF_BUCKET_PROBABILITIES,
        "wins": wins,
        "fractional_wins": wins,
        "win_rate": wins / games,
        "evaluated_player_one_card_turns": one_card_turns,
        "evaluated_player_two_card_turns": two_card_turns,
        "evaluated_player_one_card_turn_rate": (
            one_card_turns / completed_turns if completed_turns else 0.0
        ),
        "evaluated_player_hand_six_turn_starts": hand_six_turn_starts,
        "evaluated_player_hand_six_board_five_turn_starts": (
            hand_six_board_five_turn_starts
        ),
        "evaluated_player_hand_six_one_card_turns": hand_six_one_card_turns,
        "evaluated_player_hand_six_one_card_turn_rate": (
            hand_six_one_card_turns / hand_six_turn_starts
            if hand_six_turn_starts
            else 0.0
        ),
        "board_card_count_minimum": 5,
        "hand_six_branch_eligible_turns": hand_six_branch_eligible_turns,
        "hand_six_branch_taken_turns": hand_six_branch_taken_turns,
        "hand_six_branch_taken_rate": (
            hand_six_branch_taken_turns / hand_six_branch_eligible_turns
            if hand_six_branch_eligible_turns
            else 0.0
        ),
        "hand_six_branch_eligible_by_one_off": eligible_by_one_off,
        "hand_six_branch_taken_by_one_off": taken_by_one_off,
        "hand_six_branch_taken_rate_by_one_off": {
            bucket: (
                taken_by_one_off[bucket] / eligible_by_one_off[bucket]
                if eligible_by_one_off[bucket]
                else 0.0
            )
            for bucket in eligible_by_one_off
        },
        "elapsed_seconds": monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = evaluate_hand_six_min_loss_one(
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
