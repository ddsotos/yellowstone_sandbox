"""Play and measure one learned turn-value player against heuristic opponents."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from random import Random
from time import monotonic
from typing import Callable

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state, legal_actions
from yellowstone.types import (
    Action,
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
)
from yellowstone.value_learning import HISTORY_SIZE, RecentPlacement, ValueRecord
from yellowstone.value_policy import TurnSelection, select_highest_value_turn


@dataclass(slots=True)
class ValueTurnPlayer:
    """One player that commits to the highest-value completed-turn candidate."""

    player_index: int
    estimate: Callable[[ValueRecord], float]
    prune_negative_card_increase_above: int | None = None
    adaptive_pq_pruning: bool = False
    approximate_new_color_neighbor_limit: bool = False
    current_turn_history_only: bool = False
    one_card_win_probability_boost_percent: float = 0.0
    history: list[RecentPlacement] = field(default_factory=list)
    board_center_frame_history: list[tuple[int, int]] = field(default_factory=list)
    board_center_chain_states: list[GameState] = field(default_factory=list)
    pruning_records: list[tuple[str, GameState, TurnSelection]] = field(default_factory=list)
    selections: list[TurnSelection] = field(default_factory=list)
    _remaining_plan: list[Action] = field(default_factory=list)
    _heuristic: HeuristicBot = field(default_factory=HeuristicBot)

    def choose_action(self, state: GameState) -> Action:
        if state.current_player_index != self.player_index:
            raise ValueError("value player may act only on its own turn")
        actions = legal_actions(state)
        if state.phase == Phase.REFILL:
            action = self._heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no legal refill action")
            return action
        if state.phase != Phase.PLAY:
            raise RuntimeError(f"unexpected phase: {state.phase}")
        if not any(isinstance(action, PlaceCardAction) for action in actions):
            action = self._heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no legal play-phase action")
            return action
        if state.cards_played_this_turn == 0:
            profile = ""
            limit = self.prune_negative_card_increase_above
            if self.adaptive_pq_pruning:
                player = state.players[self.player_index]
                profile, limit = (
                    ("p", 4)
                    if len(player.negative_cards) + player.loss_score >= 10
                    else ("q", 8)
                )
            selection = select_highest_value_turn(
                state,
                self.estimate,
                history=(
                    ()
                    if self.current_turn_history_only
                    else tuple(self.history)
                ),
                board_center_frame_history=tuple(self.board_center_frame_history),
                board_center_chain_states=tuple(self.board_center_chain_states),
                prune_negative_card_increase_above=limit,
                approximate_new_color_neighbor_limit=self.approximate_new_color_neighbor_limit,
                one_card_win_probability_boost_percent=(
                    self.one_card_win_probability_boost_percent
                ),
            )
            self.selections.append(selection)
            if profile and selection.pruning_active:
                self.pruning_records.append((profile, state, selection))
            self._remaining_plan = list(selection.candidate.actions)
        if not self._remaining_plan:
            raise RuntimeError("missing selected turn plan")
        action = self._remaining_plan.pop(0)
        if action not in actions:
            raise RuntimeError("selected turn plan became illegal")
        return action

    def observe(self, before: GameState, action: Action, after: GameState) -> None:
        """Record each public placement, including opponent placements."""
        if (
            before.current_player_index == self.player_index
            and isinstance(action, EndTurnAction)
            and before.cards_played_this_turn == 1
        ):
            from yellowstone.value_board_centered import board_center_frame_origin

            try:
                self.board_center_frame_history.append(
                    board_center_frame_origin(after)
                )
                del self.board_center_frame_history[:-HISTORY_SIZE]
                self.board_center_chain_states.append(after)
                del self.board_center_chain_states[:-12]
            except ValueError:
                pass
        if not isinstance(action, PlaceCardAction):
            return
        player_index = before.current_player_index
        card = before.players[player_index].hand[action.hand_index]
        self.history.append(
            RecentPlacement(
                player_index=player_index,
                card=card,
                score_delta=(
                    before.players[player_index].loss_score
                    - after.players[player_index].loss_score
                ),
                negative_card_delta=(
                    len(after.players[player_index].negative_cards)
                    - len(before.players[player_index].negative_cards)
                ),
            )
        )
        del self.history[:-HISTORY_SIZE]
        if before.current_player_index == self.player_index and (
            before.cards_played_this_turn == 1
        ):
            from yellowstone.value_board_centered import board_center_frame_origin

            try:
                self.board_center_frame_history.append(
                    board_center_frame_origin(after)
                )
                del self.board_center_frame_history[:-HISTORY_SIZE]
                self.board_center_chain_states.append(after)
                del self.board_center_chain_states[:-12]
            except ValueError:
                pass


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    games: int
    wins: float
    pruning_records: tuple[tuple[str, GameState, TurnSelection], ...] = ()
    evaluated_player_one_card_turns: int = 0
    evaluated_player_two_card_turns: int = 0
    selections: tuple[TurnSelection, ...] = ()

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    @property
    def evaluated_player_one_card_turn_rate(self) -> float:
        completed_turns = (
            self.evaluated_player_one_card_turns
            + self.evaluated_player_two_card_turns
        )
        return (
            self.evaluated_player_one_card_turns / completed_turns
            if completed_turns
            else 0.0
        )

    @property
    def policy_fingerprint(self) -> str:
        decisions = [
            [
                (
                    type(action).__name__,
                    getattr(action, "hand_index", None),
                    getattr(getattr(action, "position", None), "x", None),
                    getattr(getattr(action, "position", None), "y", None),
                    getattr(getattr(action, "frame", None), "x", None),
                    getattr(getattr(action, "frame", None), "y", None),
                )
                for action in selection.candidate.actions
            ]
            for selection in self.selections
        ]
        encoded = json.dumps(decisions, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def all_one_card_candidates_saturated(self) -> bool:
        relevant = [
            selection
            for selection in self.selections
            if selection.one_card_candidate_count
        ]
        return bool(relevant) and all(
            selection.all_one_card_candidates_saturated
            for selection in relevant
        )


def evaluate_value_player(
    estimate: Callable[[ValueRecord], float],
    *,
    games: int,
    seed: int = 0,
    player_index: int = 0,
    prune_negative_card_increase_above: int | None = None,
    adaptive_pq_pruning: bool = False,
    approximate_new_color_neighbor_limit: bool = False,
    current_turn_history_only: bool = False,
    one_card_win_probability_boost_percent: float = 0.0,
    duration_seconds: float | None = None,
) -> EvaluationResult:
    """Return fractional win rate for one learned player vs three heuristics."""
    if games <= 0:
        raise ValueError("games must be positive")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    seeds = Random(seed)
    wins = 0.0
    one_card_turns = 0
    two_card_turns = 0
    records: list[tuple[str, GameState, TurnSelection]] = []
    selections: list[TurnSelection] = []
    deadline = monotonic() + duration_seconds if duration_seconds is not None else None
    completed = 0
    while completed < games and (deadline is None or monotonic() < deadline):
        state = create_initial_state(4, seed=seeds.randrange(2**63))
        rng = Random(seeds.randrange(2**63))
        value_player = ValueTurnPlayer(
            player_index,
            estimate,
            prune_negative_card_increase_above=prune_negative_card_increase_above,
            adaptive_pq_pruning=adaptive_pq_pruning,
            approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
            current_turn_history_only=current_turn_history_only,
            one_card_win_probability_boost_percent=(
                one_card_win_probability_boost_percent
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
                raise RuntimeError("policy returned no action before game end")
            before = state
            state = apply_known_legal_action(state, action, rng=rng)
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
            value_player.observe(before, action, state)
        if player_index in state.winners:
            wins += 1.0 / len(state.winners)
        records.extend(value_player.pruning_records)
        selections.extend(value_player.selections)
        completed += 1
    return EvaluationResult(
        games=completed,
        wins=wins,
        pruning_records=tuple(records),
        evaluated_player_one_card_turns=one_card_turns,
        evaluated_player_two_card_turns=two_card_turns,
        selections=tuple(selections),
    )
