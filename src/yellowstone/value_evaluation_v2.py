"""Practical V2 value-model evaluation against heuristic opponents."""

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
from yellowstone.value_v2 import (
    CandidateFrameContext,
    CompletedTurnTracker,
    PendingRefillSource,
    PublicNegativeKnowledge,
    PublicNegativeKnowledgeTracker,
    TorchWinValueEstimatorV2,
    ValueRecordV2,
)


@dataclass(frozen=True, slots=True)
class V2TurnChoice:
    actions: tuple[Action, ...]
    record: ValueRecordV2
    predicted_win_probability: float
    candidate_count: int
    pruning_active: bool


def choose_v2_turn(
    state: GameState,
    estimate: Callable[[ValueRecordV2], float],
    *,
    history: CompletedTurnTracker,
    negative_knowledge: PublicNegativeKnowledge,
    prune_negative_card_increase_above: int | None,
    approximate_new_color_neighbor_limit: bool,
) -> V2TurnChoice:
    """Choose one complete turn, including a pending refill decision."""
    pruning_limit = (
        prune_negative_card_increase_above
        if prune_negative_card_increase_above is not None
        and _has_zero_negative_two_card_witness(state)
        else None
    )
    base_candidates = enumerate_turn_end_candidates(
        state,
        max_negative_card_increase=pruning_limit,
        approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
        collapse_equivalent_frames=True,
    )
    if not base_candidates:
        raise ValueError("no V2 completed-turn candidates are legal")

    player_index = state.current_player_index
    start_board_card_count = sum(len(stack) for stack in state.board.values())
    records: list[ValueRecordV2] = []
    plans: list[tuple[Action, ...]] = []
    for candidate in base_candidates:
        placements = tuple(
            action
            for action in candidate.actions
            if isinstance(action, PlaceCardAction)
        )
        if not placements:
            raise AssertionError("V2 turn candidate has no placement")
        candidate_frame = CandidateFrameContext(
            start_frame=history.current_frame,
            end_frame=placements[-1].frame,
            start_board_card_count=start_board_card_count,
        )
        pending_actions = tuple(
            action
            for action in legal_actions(candidate.record.state)
            if isinstance(action, RefillAction)
            and action.source.value != PendingRefillSource.NONE.value
        )
        variants: tuple[tuple[PendingRefillSource, tuple[Action, ...]], ...]
        if candidate.record.state.phase == Phase.REFILL:
            if not pending_actions:
                raise AssertionError("refill-boundary candidate lacks refill actions")
            variants = tuple(
                (
                    PendingRefillSource(action.source.value),
                    (*candidate.actions, action),
                )
                for action in pending_actions
            )
        else:
            variants = (
                (PendingRefillSource.NO_PENDING, candidate.actions),
            )
        for pending_source, plan in variants:
            records.append(
                ValueRecordV2(
                    game_id=-1,
                    perspective_player_index=player_index,
                    state=candidate.record.state,
                    history_before_turn=history.snapshot(),
                    candidate_frame=candidate_frame,
                    negative_knowledge=negative_knowledge,
                    pending_refill_source=pending_source,
                    target=0.0,
                )
            )
            plans.append(plan)

    estimate_many = getattr(estimate, "estimate_many", None)
    if estimate_many is not None:
        scores = tuple(float(value) for value in estimate_many(tuple(records)))
    else:
        scores = tuple(float(estimate(record)) for record in records)
    best_index = max(range(len(records)), key=lambda index: scores[index])
    return V2TurnChoice(
        actions=plans[best_index],
        record=records[best_index],
        predicted_win_probability=scores[best_index],
        candidate_count=len(records),
        pruning_active=pruning_limit is not None,
    )


@dataclass(slots=True)
class V2TurnPlayer:
    player_index: int
    estimate: Callable[[ValueRecordV2], float]
    adaptive_pq_pruning: bool = True
    approximate_new_color_neighbor_limit: bool = True
    history: CompletedTurnTracker = field(default_factory=CompletedTurnTracker)
    knowledge: PublicNegativeKnowledgeTracker = field(
        default_factory=lambda: PublicNegativeKnowledgeTracker(4)
    )
    _remaining_plan: list[Action] = field(default_factory=list)
    _heuristic: HeuristicBot = field(default_factory=HeuristicBot)
    choices: list[V2TurnChoice] = field(default_factory=list)

    def choose_action(self, state: GameState) -> Action:
        if state.current_player_index != self.player_index:
            raise ValueError("V2 value player may act only on its own turn")
        actions = legal_actions(state)
        if self._remaining_plan:
            action = self._remaining_plan.pop(0)
            if action not in actions:
                raise RuntimeError("selected V2 turn plan became illegal")
            return action
        if state.phase == Phase.REFILL:
            action = self._heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no legal V2 fallback refill")
            return action
        if state.phase != Phase.PLAY:
            raise RuntimeError(f"unexpected phase: {state.phase}")
        if not any(isinstance(action, PlaceCardAction) for action in actions):
            action = self._heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no legal V2 fallback play action")
            return action
        if state.cards_played_this_turn != 0:
            raise RuntimeError("V2 plan is missing during an active turn")
        player = state.players[self.player_index]
        pruning_limit = None
        if self.adaptive_pq_pruning:
            pruning_limit = (
                4
                if len(player.negative_cards) + player.loss_score >= 10
                else 8
            )
        choice = choose_v2_turn(
            state,
            self.estimate,
            history=self.history,
            negative_knowledge=self.knowledge.snapshot(),
            prune_negative_card_increase_above=pruning_limit,
            approximate_new_color_neighbor_limit=(
                self.approximate_new_color_neighbor_limit
            ),
        )
        self.choices.append(choice)
        self._remaining_plan = list(choice.actions)
        return self.choose_action(state)

    def observe(self, before: GameState, action: Action, after: GameState) -> None:
        self.history.observe(before, action, after)
        self.knowledge.observe(before, action, after)


def evaluate_v2_value_player(
    estimate: Callable[[ValueRecordV2], float],
    *,
    games: int,
    seed: int,
    player_index: int = 0,
    adaptive_pq_pruning: bool = True,
    approximate_new_color_neighbor_limit: bool = True,
) -> dict[str, object]:
    """Evaluate one V2 player against three heuristic players."""
    if games <= 0:
        raise ValueError("games must be positive")
    seeds = Random(seed)
    wins = 0.0
    turns = 0
    one_card_turns = 0
    refill_choices = {
        PendingRefillSource.NO_PENDING.value: 0,
        PendingRefillSource.NONE.value: 0,
        PendingRefillSource.DECK.value: 0,
        PendingRefillSource.NEGATIVE_CARDS.value: 0,
    }
    predicted_probability_sum = 0.0
    candidate_count_sum = 0
    for _ in range(games):
        state = create_initial_state(4, seed=seeds.randrange(2**63))
        rng = Random(seeds.randrange(2**63))
        value_player = V2TurnPlayer(
            player_index=player_index,
            estimate=estimate,
            adaptive_pq_pruning=adaptive_pq_pruning,
            approximate_new_color_neighbor_limit=(
                approximate_new_color_neighbor_limit
            ),
        )
        heuristic = HeuristicBot()
        while state.phase != Phase.GAME_OVER:
            action = (
                value_player.choose_action(state)
                if state.current_player_index == player_index
                else heuristic.choose_action(state)
            )
            if action is None:
                raise RuntimeError("V2 evaluation policy returned no action")
            before = state
            state = apply_known_legal_action(state, action, rng=rng)
            value_player.observe(before, action, state)
        for choice in value_player.choices:
            turns += 1
            one_card_turns += int(
                sum(
                    isinstance(action, PlaceCardAction)
                    for action in choice.actions
                )
                == 1
            )
            refill_choices[choice.record.pending_refill_source.value] += 1
            predicted_probability_sum += choice.predicted_win_probability
            candidate_count_sum += choice.candidate_count
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
        "refill_choices": refill_choices,
        "mean_predicted_win_probability": (
            predicted_probability_sum / turns if turns else 0.0
        ),
        "mean_candidate_count": candidate_count_sum / turns if turns else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a V2 value model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    estimator = TorchWinValueEstimatorV2(str(args.checkpoint))
    result = evaluate_v2_value_player(
        estimator,
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
