"""Evaluate a transition-aware V2-lite player against heuristic players."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Callable

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state, legal_actions
from yellowstone.types import Action, GameState, Phase, PlaceCardAction, RefillAction
from yellowstone.value_policy import (
    _has_zero_negative_two_card_witness,
    enumerate_turn_end_candidates,
)
from yellowstone.value_v2 import CompletedTurnTracker, PendingRefillSource
from yellowstone.value_v2_lite import (
    VALUE_SCHEMA_V2_LITE,
    ValueRecordV2Lite,
    canonical_tensors_v2_lite,
)


@dataclass(frozen=True, slots=True)
class V2LiteTurnChoice:
    actions: tuple[Action, ...]
    record: ValueRecordV2Lite
    predicted_win_probability: float
    candidate_count: int


def choose_v2_lite_turn(
    state: GameState,
    estimate: Callable[[ValueRecordV2Lite], float],
    *,
    history: CompletedTurnTracker,
    prune_negative_card_increase_above: int | None,
    approximate_new_color_neighbor_limit: bool,
) -> V2LiteTurnChoice:
    pruning_limit = (
        prune_negative_card_increase_above
        if prune_negative_card_increase_above is not None
        and _has_zero_negative_two_card_witness(state)
        else None
    )
    candidates = enumerate_turn_end_candidates(
        state,
        max_negative_card_increase=pruning_limit,
        approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
        collapse_equivalent_frames=True,
    )
    if not candidates:
        raise ValueError("no V2-lite completed-turn candidates are legal")
    player_index = state.current_player_index
    records: list[ValueRecordV2Lite] = []
    plans: list[tuple[Action, ...]] = []
    for candidate in candidates:
        pending_actions = tuple(
            action
            for action in legal_actions(candidate.record.state)
            if isinstance(action, RefillAction)
            and action.source.value != PendingRefillSource.NONE.value
        )
        if candidate.record.state.phase == Phase.REFILL:
            if not pending_actions:
                raise AssertionError("refill-boundary candidate lacks actions")
            variants = tuple(
                (
                    PendingRefillSource(action.source.value),
                    (*candidate.actions, action),
                )
                for action in pending_actions
            )
        else:
            variants = ((PendingRefillSource.NO_PENDING, candidate.actions),)
        for pending, plan in variants:
            records.append(
                ValueRecordV2Lite(
                    game_id=-1,
                    perspective_player_index=player_index,
                    state_before_turn=state,
                    state=candidate.record.state,
                    history_before_turn=history.snapshot()[-2:],
                    pending_refill_source=pending,
                    target=0.0,
                )
            )
            plans.append(plan)
    estimate_many = getattr(estimate, "estimate_many", None)
    scores = (
        tuple(float(value) for value in estimate_many(tuple(records)))
        if estimate_many is not None
        else tuple(float(estimate(record)) for record in records)
    )
    best = max(range(len(records)), key=lambda index: scores[index])
    return V2LiteTurnChoice(
        actions=plans[best],
        record=records[best],
        predicted_win_probability=scores[best],
        candidate_count=len(records),
    )


@dataclass(slots=True)
class V2LiteTurnPlayer:
    player_index: int
    estimate: Callable[[ValueRecordV2Lite], float]
    adaptive_pq_pruning: bool = True
    approximate_new_color_neighbor_limit: bool = True
    history: CompletedTurnTracker = field(default_factory=CompletedTurnTracker)
    remaining_plan: list[Action] = field(default_factory=list)
    heuristic: HeuristicBot = field(default_factory=HeuristicBot)
    choices: list[V2LiteTurnChoice] = field(default_factory=list)

    def choose_action(self, state: GameState) -> Action:
        if state.current_player_index != self.player_index:
            raise ValueError("V2-lite player may act only on its own turn")
        actions = legal_actions(state)
        if self.remaining_plan:
            action = self.remaining_plan.pop(0)
            if action not in actions:
                raise RuntimeError("selected V2-lite plan became illegal")
            return action
        if state.phase == Phase.REFILL:
            action = self.heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no legal V2-lite fallback refill")
            return action
        if not any(isinstance(action, PlaceCardAction) for action in actions):
            action = self.heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no legal V2-lite fallback action")
            return action
        player = state.players[self.player_index]
        pruning_limit = None
        if self.adaptive_pq_pruning:
            pruning_limit = (
                4
                if len(player.negative_cards) + player.loss_score >= 10
                else 8
            )
        choice = choose_v2_lite_turn(
            state,
            self.estimate,
            history=self.history,
            prune_negative_card_increase_above=pruning_limit,
            approximate_new_color_neighbor_limit=(
                self.approximate_new_color_neighbor_limit
            ),
        )
        self.choices.append(choice)
        self.remaining_plan = list(choice.actions)
        return self.choose_action(state)

    def observe(self, before: GameState, action: Action, after: GameState) -> None:
        self.history.observe(before, action, after)


class TorchWinValueEstimatorV2Lite:
    def __init__(self, checkpoint_path: str | Path) -> None:
        import torch

        from yellowstone.cnn import build_win_value_net_v2_lite

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("value_schema") != VALUE_SCHEMA_V2_LITE:
            raise ValueError("checkpoint is not a Yellowstone V2-lite model")
        self.torch = torch
        self.model = build_win_value_net_v2_lite()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def __call__(self, record: ValueRecordV2Lite) -> float:
        return self.estimate_many((record,))[0]

    def estimate_many(
        self, records: tuple[ValueRecordV2Lite, ...]
    ) -> tuple[float, ...]:
        import numpy as np

        if not records:
            return ()
        tensors = [canonical_tensors_v2_lite(record) for record in records]
        boards = np.stack([item[0] for item in tensors])
        contexts = np.stack([item[1] for item in tensors])
        with self.torch.no_grad():
            values = self.torch.sigmoid(
                self.model(
                    self.torch.from_numpy(boards),
                    self.torch.from_numpy(contexts),
                )
            ).tolist()
        return tuple(float(value) for value in values)


def evaluate_v2_lite_value_player(
    estimate: Callable[[ValueRecordV2Lite], float],
    *,
    games: int,
    seed: int,
    player_index: int = 0,
) -> dict[str, object]:
    seeds = Random(seed)
    wins = 0.0
    turns = 0
    one_card_turns = 0
    probability_sum = 0.0
    candidate_sum = 0
    for _ in range(games):
        state = create_initial_state(4, seed=seeds.randrange(2**63))
        rng = Random(seeds.randrange(2**63))
        player = V2LiteTurnPlayer(player_index=player_index, estimate=estimate)
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
        for choice in player.choices:
            turns += 1
            one_card_turns += int(
                sum(
                    isinstance(action, PlaceCardAction)
                    for action in choice.actions
                )
                == 1
            )
            probability_sum += choice.predicted_win_probability
            candidate_sum += choice.candidate_count
        if player_index in state.winners:
            wins += 1.0 / len(state.winners)
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "seed": seed,
        "player_index": player_index,
        "turns": turns,
        "one_card_turns": one_card_turns,
        "two_card_turns": turns - one_card_turns,
        "one_card_turn_rate": one_card_turns / turns if turns else 0.0,
        "mean_predicted_win_probability": probability_sum / turns,
        "mean_candidate_count": candidate_sum / turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V2-lite")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_v2_lite_value_player(
        TorchWinValueEstimatorV2Lite(args.checkpoint),
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()

