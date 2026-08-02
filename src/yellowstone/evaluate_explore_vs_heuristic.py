"""Evaluate one exploratory NPC against three heuristic players."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from random import Random
from time import monotonic

from yellowstone.bots import HeuristicBot
from yellowstone.exploratory_collection import (
    ExploratoryValueNpc,
    choose_exploratory_refill,
)
from yellowstone.fast_value_npc import _append_history
from yellowstone.game import apply_known_legal_action, create_initial_state, legal_actions
from yellowstone.types import EndTurnAction, Phase, PlaceCardAction
from yellowstone.value_learning import HISTORY_SIZE, RecentPlacement


def evaluate_explore_vs_heuristic(
    checkpoint: Path,
    *,
    games: int,
    seed: int,
    player_index: int,
    lazy_single_pass: bool,
) -> dict[str, object]:
    if games <= 0:
        raise ValueError("games must be positive")
    seeds = Random(seed)
    wins = 0.0
    one_card_turns = 0
    two_card_turns = 0
    selection_modes: Counter[str] = Counter()
    refill_sources: Counter[str] = Counter()
    heuristic = HeuristicBot()
    started = monotonic()

    for _ in range(games):
        state = create_initial_state(4, seed=seeds.randrange(2**63))
        gameplay_rng = Random(seeds.randrange(2**63))
        explore_rng = Random(seeds.randrange(2**63))
        npc = ExploratoryValueNpc(checkpoint, lazy_single_pass=lazy_single_pass)
        history: list[RecentPlacement] = []
        planned = []

        while state.phase != Phase.GAME_OVER:
            if state.current_player_index == player_index:
                if planned:
                    action = planned.pop(0)
                elif (
                    state.phase == Phase.PLAY
                    and state.cards_played_this_turn == 0
                    and any(
                        isinstance(item, PlaceCardAction)
                        for item in legal_actions(state)
                    )
                ):
                    choice = npc.choose_turn(
                        state,
                        tuple(history[-HISTORY_SIZE:]),
                        rng=explore_rng,
                    )
                    selection_modes[choice.selection_mode] += 1
                    planned = list(choice.actions)
                    action = planned.pop(0)
                else:
                    action, audit = choose_exploratory_refill(
                        state, rng=explore_rng
                    )
                    refill_sources[str(audit["selected_source"])] += 1
            else:
                action = heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("policy returned no action before game end")
            if action not in legal_actions(state):
                raise RuntimeError(f"policy selected illegal action: {action!r}")
            before = state
            state = apply_known_legal_action(state, action, rng=gameplay_rng)
            if before.current_player_index == player_index:
                if (
                    isinstance(action, EndTurnAction)
                    and before.cards_played_this_turn == 1
                ):
                    one_card_turns += 1
                elif (
                    isinstance(action, PlaceCardAction)
                    and before.cards_played_this_turn == 1
                ):
                    two_card_turns += 1
            _append_history(history, before, action, state)
            if (
                before.current_player_index != state.current_player_index
                or state.phase == Phase.GAME_OVER
            ):
                planned = []

        if player_index in state.winners:
            wins += 1.0 / len(state.winners)

    completed_turns = one_card_turns + two_card_turns
    return {
        "games": games,
        "wins": wins,
        "fractional_wins": wins,
        "win_rate": wins / games,
        "player_index": player_index,
        "checkpoint": str(checkpoint),
        "opponent_policy": "heuristic",
        "explore_lazy_single_pass": lazy_single_pass,
        "evaluated_player_one_card_turns": one_card_turns,
        "evaluated_player_two_card_turns": two_card_turns,
        "evaluated_player_one_card_turn_rate": (
            one_card_turns / completed_turns if completed_turns else 0.0
        ),
        "selection_modes": dict(sorted(selection_modes.items())),
        "refill_sources": dict(sorted(refill_sources.items())),
        "elapsed_seconds": monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--no-lazy-single-pass", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = evaluate_explore_vs_heuristic(
        args.checkpoint,
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
        lazy_single_pass=not args.no_lazy_single_pass,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
