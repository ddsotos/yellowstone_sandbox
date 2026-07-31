"""Evaluate the V1 three-prior-turn history model against heuristics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from yellowstone.bots import HeuristicBot
from yellowstone.game import (
    apply_known_legal_action,
    create_initial_state,
    legal_actions,
)
from yellowstone.types import Action, GameState, Phase, PlaceCardAction
from yellowstone.value_history3 import (
    CANONICALIZATION_HISTORY3,
    VALUE_CONTEXT_SIZE_HISTORY3,
    VALUE_SCHEMA_HISTORY3,
    History3Tracker,
    ValueRecordHistory3,
    board_tensor_history3,
    context_tensor_history3,
)
from yellowstone.value_policy import (
    _has_zero_negative_two_card_witness,
    enumerate_turn_end_candidates,
)


class TorchHistory3Estimator:
    def __init__(self, checkpoint_path: str | Path) -> None:
        import torch

        from yellowstone.cnn import build_win_value_net

        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if payload.get("value_schema") != VALUE_SCHEMA_HISTORY3:
            raise ValueError("checkpoint is not a V1 history3 model")
        if payload.get("context_size") != VALUE_CONTEXT_SIZE_HISTORY3:
            raise ValueError("history3 checkpoint context size differs")
        if (
            payload.get("input_canonicalization")
            != CANONICALIZATION_HISTORY3
        ):
            raise ValueError("history3 checkpoint canonicalization differs")
        self._torch = torch
        self._model = build_win_value_net(
            context_size=VALUE_CONTEXT_SIZE_HISTORY3
        )
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()

    def estimate_many(
        self, records: tuple[ValueRecordHistory3, ...]
    ) -> tuple[float, ...]:
        import numpy as np

        from yellowstone.value_canonicalization import (
            canonicalize_value_tensors,
        )

        if not records:
            return ()
        board = np.stack(
            [board_tensor_history3(record) for record in records]
        )
        context = np.stack(
            [context_tensor_history3(record) for record in records]
        )
        board, context = canonicalize_value_tensors(board, context)
        with self._torch.no_grad():
            values = self._torch.sigmoid(
                self._model(
                    self._torch.from_numpy(board),
                    self._torch.from_numpy(context),
                )
            ).tolist()
        return tuple(float(value) for value in values)


@dataclass(slots=True)
class History3ValuePlayer:
    player_index: int
    estimator: TorchHistory3Estimator
    adaptive_pq_pruning: bool = True
    approximate_new_color_neighbor_limit: bool = True
    history: History3Tracker = field(default_factory=History3Tracker)
    _remaining_plan: list[Action] = field(default_factory=list)
    _heuristic: HeuristicBot = field(default_factory=HeuristicBot)

    def choose_action(self, state: GameState) -> Action:
        if state.current_player_index != self.player_index:
            raise ValueError("history3 player may act only on its own turn")
        actions = legal_actions(state)
        if self._remaining_plan:
            action = self._remaining_plan.pop(0)
            if action not in actions:
                raise RuntimeError("history3 plan became illegal")
            return action
        if state.phase == Phase.REFILL:
            action = self._heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no legal refill")
            return action
        if state.phase != Phase.PLAY:
            raise RuntimeError(f"unexpected phase: {state.phase}")
        if not any(isinstance(action, PlaceCardAction) for action in actions):
            action = self._heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no legal fallback action")
            return action
        if state.cards_played_this_turn != 0:
            raise RuntimeError("history3 plan missing during active turn")

        player = state.players[self.player_index]
        limit = None
        if self.adaptive_pq_pruning:
            limit = (
                4
                if len(player.negative_cards) + player.loss_score >= 10
                else 8
            )
        pruning_limit = (
            limit
            if limit is not None
            and _has_zero_negative_two_card_witness(state)
            else None
        )
        candidates = enumerate_turn_end_candidates(
            state,
            history=(),
            max_negative_card_increase=pruning_limit,
            approximate_new_color_neighbor_limit=(
                self.approximate_new_color_neighbor_limit
            ),
        )
        records = tuple(
            ValueRecordHistory3(
                game_id=-1,
                perspective_player_index=self.player_index,
                state=candidate.record.state,
                history_before_turn=self.history.snapshot(),
                target=0.0,
            )
            for candidate in candidates
        )
        scores = self.estimator.estimate_many(records)
        best = max(range(len(candidates)), key=lambda index: scores[index])
        self._remaining_plan = list(candidates[best].actions)
        return self.choose_action(state)

    def observe(
        self, before: GameState, action: Action, after: GameState
    ) -> None:
        self.history.observe(before, action, after)


def evaluate_history3(
    checkpoint: str | Path,
    *,
    games: int,
    seed: int,
    player_index: int,
) -> dict[str, object]:
    if games <= 0:
        raise ValueError("games must be positive")
    estimator = TorchHistory3Estimator(checkpoint)
    seeds = Random(seed)
    wins = 0.0
    for _ in range(games):
        state = create_initial_state(4, seed=seeds.randrange(2**63))
        rng = Random(seeds.randrange(2**63))
        player = History3ValuePlayer(player_index, estimator)
        heuristic = HeuristicBot()
        while state.phase != Phase.GAME_OVER:
            action = (
                player.choose_action(state)
                if state.current_player_index == player_index
                else heuristic.choose_action(state)
            )
            if action is None:
                raise RuntimeError("evaluation policy returned no action")
            before = state
            state = apply_known_legal_action(state, action, rng=rng)
            player.observe(before, action, state)
        if player_index in state.winners:
            wins += 1.0 / len(state.winners)
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "seed": seed,
        "player_index": player_index,
        "value_schema": VALUE_SCHEMA_HISTORY3,
        "history_semantics": "three_prior_completed_turns_two_slots_each",
        "adaptive_pq_pruning": True,
        "approximate_new_color_neighbors": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_history3(
        args.checkpoint,
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
