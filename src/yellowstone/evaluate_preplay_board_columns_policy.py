"""Evaluate a pre-play board-columns state-value policy against heuristics."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from time import monotonic

from yellowstone.bots import HeuristicBot
from yellowstone.cnn import build_win_value_net
from yellowstone.game import apply_known_legal_action, create_initial_state, legal_actions
from yellowstone.types import Action, EndTurnAction, GameState, Phase, PlaceCardAction, RefillAction
from yellowstone.value_board_columns_v2 import (
    CANONICALIZATION_PREPLAY_BOARD_COLUMNS,
    PREPLAY_BOARD_COLUMNS_CONTEXT_SIZE,
    VALUE_SCHEMA_PREPLAY_BOARD_COLUMNS,
    board_columns_from_canonical_board,
    board_columns_v2_metadata,
)
from yellowstone.value_policy import enumerate_turn_end_candidates
from yellowstone.value_v2 import CompletedTurnTracker, PendingRefillSource
from yellowstone.value_v2_lite import ValueRecordV2Lite, canonical_tensors_v2_lite


CHECKPOINT = Path(
    "models/v2_heuristic_safe_counts_rank_color_6h_snapshot_training_"
    "preplay_board_columns_epoch001.pt"
)


class PreplayBoardColumnsEstimator:
    def __init__(self, checkpoint_path: Path) -> None:
        import torch

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        expected = board_columns_v2_metadata(preplay=True)
        mismatches = {
            key: {"expected": value, "actual": checkpoint.get(key)}
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint contract mismatch: {mismatches}")
        self.torch = torch
        self.contract = {
            **expected,
            "context_size": PREPLAY_BOARD_COLUMNS_CONTEXT_SIZE,
            "checkpoint": str(checkpoint_path),
        }
        self.model = build_win_value_net(
            context_size=PREPLAY_BOARD_COLUMNS_CONTEXT_SIZE,
            board_channels=1,
            board_height=7,
            board_width=3,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def estimate_many(
        self, states: list[GameState], viewers: list[int], histories: list[tuple]
    ) -> list[float]:
        import numpy as np

        if not states:
            return []
        encoded = [
            canonical_tensors_v2_lite(
                ValueRecordV2Lite(
                    game_id=-1,
                    perspective_player_index=viewer,
                    state_before_turn=state,
                    state=state,
                    history_before_turn=history[-2:],
                    pending_refill_source=PendingRefillSource.NO_PENDING,
                    target=0.0,
                )
            )
            for state, viewer, history in zip(states, viewers, histories, strict=True)
        ]
        board, context, _ = board_columns_from_canonical_board(
            np.stack([item[0][:29] for item in encoded]),
            np.stack([item[1] for item in encoded]),
        )
        with self.torch.no_grad():
            values = self.torch.sigmoid(
                self.model(
                    self.torch.from_numpy(board),
                    self.torch.from_numpy(context),
                )
            ).tolist()
        return [float(value) for value in values]


@dataclass(slots=True)
class CandidatePlan:
    actions: tuple[Action, ...]
    score: float
    one_card: bool
    refill_source: str | None


@dataclass(slots=True)
class PreplayBoardColumnsPlayer:
    player_index: int
    estimator: PreplayBoardColumnsEstimator
    rng: Random
    tracker: CompletedTurnTracker = field(default_factory=CompletedTurnTracker)
    heuristic: HeuristicBot = field(default_factory=HeuristicBot)
    remaining: list[Action] = field(default_factory=list)
    selections: list[CandidatePlan] = field(default_factory=list)
    refill_choices: dict[str, int] = field(default_factory=dict)

    def choose_action(self, state: GameState) -> Action:
        actions = legal_actions(state)
        if state.current_player_index != self.player_index:
            action = self.heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("heuristic returned no action")
            return action
        if state.phase == Phase.REFILL or not any(
            isinstance(action, PlaceCardAction) for action in actions
        ):
            return self._choose_refill_or_heuristic(state, actions)
        if state.phase != Phase.PLAY:
            raise RuntimeError(f"unexpected phase: {state.phase}")
        if state.cards_played_this_turn == 0:
            plan = self._select_turn_plan(state)
            self.selections.append(plan)
            self.remaining = list(plan.actions)
        if not self.remaining:
            raise RuntimeError("missing preplay plan")
        action = self.remaining.pop(0)
        if action not in actions:
            raise RuntimeError("selected preplay plan became illegal")
        return action

    def observe(self, before: GameState, action: Action, after: GameState) -> None:
        self.tracker.observe(before, action, after)

    def _choose_refill_or_heuristic(
        self, state: GameState, actions: tuple[Action, ...]
    ) -> Action:
        refills = [action for action in actions if isinstance(action, RefillAction)]
        if not refills:
            action = self.heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no legal fallback action")
            return action
        scored = self._score_refill_actions(state, tuple(refills))
        best = max(scored, key=lambda row: row[0])
        source = best[1].source.value
        self.refill_choices[source] = self.refill_choices.get(source, 0) + 1
        return best[1]

    def _select_turn_plan(self, state: GameState) -> CandidatePlan:
        candidates = enumerate_turn_end_candidates(
            state,
            history=(),
            approximate_new_color_neighbor_limit=True,
        )
        options: list[tuple[tuple[Action, ...], bool, str | None, GameState, CompletedTurnTracker]] = []
        for candidate in candidates:
            after, tracker = _apply_actions_with_tracker(
                state, candidate.actions, self.tracker, self.rng
            )
            refill_source = None
            if after.phase == Phase.REFILL:
                refills = tuple(
                    action
                    for action in legal_actions(after)
                    if isinstance(action, RefillAction)
                )
                for refill_action in refills:
                    sampled_rng = Random(self.rng.randrange(2**63))
                    refill_after, refill_tracker = _apply_actions_with_tracker(
                        after, (refill_action,), tracker, sampled_rng
                    )
                    options.append(
                        (
                            (*candidate.actions, refill_action),
                            False,
                            refill_action.source.value,
                            refill_after,
                            refill_tracker,
                        )
                    )
                continue
            else:
                actions = candidate.actions
            options.append(
                (
                    tuple(actions),
                    any(isinstance(action, EndTurnAction) for action in actions),
                    refill_source,
                    after,
                    tracker,
                )
            )
        if not options:
            raise RuntimeError("no preplay turn candidates")
        scores = self._score_states(
            [row[3] for row in options],
            [row[4] for row in options],
        )
        scored = [
            CandidatePlan(
                actions=tuple(actions),
                score=score,
                one_card=one_card,
                refill_source=refill_source,
            )
            for (actions, one_card, refill_source, _, _), score in zip(
                options, scores, strict=True
            )
        ]
        best = max(scored, key=lambda row: row.score)
        if best.refill_source is not None:
            self.refill_choices[best.refill_source] = (
                self.refill_choices.get(best.refill_source, 0) + 1
            )
        return best

    def _score_refill_actions(
        self, state: GameState, refills: tuple[RefillAction, ...]
    ) -> list[tuple[float, RefillAction, GameState, CompletedTurnTracker]]:
        rows = []
        for index, action in enumerate(refills):
            sampled_rng = Random(self.rng.randrange(2**63) + index)
            after, tracker = _apply_actions_with_tracker(
                state, (action,), self.tracker, sampled_rng
            )
            rows.append((0.0, action, after, tracker))
        scores = self._score_states(
            [row[2] for row in rows],
            [row[3] for row in rows],
        )
        rows = [
            (score, action, after, tracker)
            for score, (_, action, after, tracker) in zip(scores, rows, strict=True)
        ]
        return rows

    def _score_states(
        self, states: list[GameState], trackers: list[CompletedTurnTracker]
    ) -> list[float]:
        scores: list[float | None] = []
        nonterminal_states: list[GameState] = []
        nonterminal_histories: list[tuple] = []
        nonterminal_indexes: list[int] = []
        for index, (state, tracker) in enumerate(zip(states, trackers, strict=True)):
            if state.phase == Phase.GAME_OVER:
                scores.append(
                    1.0 / len(state.winners)
                    if self.player_index in state.winners and state.winners
                    else 0.0
                )
            else:
                scores.append(None)
                nonterminal_indexes.append(index)
                nonterminal_states.append(state)
                nonterminal_histories.append(tracker.snapshot())
        estimated = self.estimator.estimate_many(
            nonterminal_states,
            [self.player_index] * len(nonterminal_states),
            nonterminal_histories,
        )
        for index, score in zip(nonterminal_indexes, estimated, strict=True):
            scores[index] = score
        return [float(score) for score in scores]


def _apply_actions_with_tracker(
    state: GameState,
    actions: tuple[Action, ...],
    tracker: CompletedTurnTracker,
    rng: Random,
) -> tuple[GameState, CompletedTurnTracker]:
    current = state
    next_tracker = copy.deepcopy(tracker)
    for action in actions:
        before = current
        current = apply_known_legal_action(current, action, rng=rng)
        next_tracker.observe(before, action, current)
    return current, next_tracker


def evaluate_policy(
    *,
    checkpoint: Path,
    games: int,
    seed: int,
    player_index: int,
) -> dict[str, object]:
    started = monotonic()
    estimator = PreplayBoardColumnsEstimator(checkpoint)
    seeds = Random(seed)
    wins = 0.0
    one_card_turns = 0
    two_card_turns = 0
    refill_choices: dict[str, int] = {}
    selected_scores: list[float] = []
    completed = 0
    while completed < games:
        state = create_initial_state(4, seed=seeds.randrange(2**63))
        game_rng = Random(seeds.randrange(2**63))
        policy_rng = Random(seeds.randrange(2**63))
        player = PreplayBoardColumnsPlayer(player_index, estimator, policy_rng)
        while state.phase != Phase.GAME_OVER:
            action = player.choose_action(state)
            before = state
            state = apply_known_legal_action(state, action, rng=game_rng)
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
            player.observe(before, action, state)
        if player_index in state.winners:
            wins += 1.0 / len(state.winners)
        for key, value in player.refill_choices.items():
            refill_choices[key] = refill_choices.get(key, 0) + value
        selected_scores.extend(plan.score for plan in player.selections)
        completed += 1
    return {
        "games": completed,
        "wins": wins,
        "fractional_wins": wins,
        "win_rate": wins / completed,
        "player_index": player_index,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_contract": estimator.contract,
        "evaluated_player_one_card_turns": one_card_turns,
        "evaluated_player_two_card_turns": two_card_turns,
        "evaluated_player_one_card_turn_rate": (
            one_card_turns / (one_card_turns + two_card_turns)
            if one_card_turns + two_card_turns
            else 0.0
        ),
        "refill_choices": refill_choices,
        "selected_score": {
            "count": len(selected_scores),
            "mean": sum(selected_scores) / len(selected_scores)
            if selected_scores
            else None,
            "min": min(selected_scores) if selected_scores else None,
            "max": max(selected_scores) if selected_scores else None,
        },
        "negative_refill_evaluation": "single random sampled refill hand per candidate score",
        "elapsed_seconds": monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_policy(
        checkpoint=args.checkpoint,
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
