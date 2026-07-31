"""Deterministic bot policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Protocol

from yellowstone.game import (
    apply_known_legal_action,
    frame_positions,
    legal_actions,
    occupied_count_in_frame,
)
from yellowstone.types import (
    Action,
    Board,
    Card,
    Color,
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
    RefillAction,
    RefillSource,
)


class BotPolicy(Protocol):
    """Interface for deterministic or learned action policies."""

    def choose_action(self, state: GameState) -> Action | None:
        """Choose one legal action for the given state."""


@dataclass(frozen=True, slots=True)
class HeuristicBot:
    """Deterministic heuristic policy."""

    def choose_action(self, state: GameState) -> Action | None:
        """Choose an action using the documented heuristic rules."""
        return choose_heuristic_action(state)


@dataclass(slots=True)
class RandomBot:
    """Random policy for baseline evaluation."""

    rng: Random = field(default_factory=Random)

    def choose_action(self, state: GameState) -> Action | None:
        """Choose a random legal action."""
        actions = legal_actions(state)
        if not actions:
            return None
        return self.rng.choice(actions)


@dataclass(slots=True)
class ExploratoryHeuristicBot:
    """Heuristic policy that sometimes stops after one card for value data."""

    rng: Random = field(default_factory=Random)
    one_card_probabilities: dict[int, float] = field(
        default_factory=lambda: {
            6: 0.35,
            5: 0.25,
            4: 0.15,
        }
    )
    _forced_one_card_players: set[int] = field(default_factory=set, init=False)

    def choose_action(self, state: GameState) -> Action | None:
        """Choose heuristic actions while broadening turn-start hand counts."""
        if state.phase != Phase.PLAY:
            return choose_heuristic_action(state)

        player_index = state.current_player_index
        if state.cards_played_this_turn == 1:
            if player_index in self._forced_one_card_players:
                self._forced_one_card_players.remove(player_index)
                end_turn = _first_action(legal_actions(state), EndTurnAction)
                if end_turn is not None:
                    return end_turn
            return choose_heuristic_action(state)

        if state.cards_played_this_turn != 0:
            return choose_heuristic_action(state)

        player = state.players[player_index]
        probability = self.one_card_probabilities.get(len(player.hand), 0.0)
        if probability <= 0.0 or self.rng.random() >= probability:
            return choose_heuristic_action(state)

        first_action = _choose_no_damage_two_card_first_action(state)
        if first_action is None:
            return choose_heuristic_action(state)

        self._forced_one_card_players.add(player_index)
        return first_action


@dataclass(frozen=True, slots=True)
class _PlacementCandidate:
    action: PlaceCardAction
    sort_key: tuple[int, ...]


def choose_heuristic_action(state: GameState) -> Action | None:
    """Choose one deterministic heuristic action."""
    actions = legal_actions(state)
    if not actions:
        return None
    if state.phase == Phase.GAME_OVER:
        return None
    if state.phase == Phase.REFILL:
        return _choose_refill_action(state, actions)
    if state.phase != Phase.PLAY:
        return None

    # A settlement may pass play to a player with an empty hand.  The rule
    # engine exposes refill actions at that turn start; treat them exactly as
    # the normal refill phase instead of reporting no move.
    refill_actions = tuple(action for action in actions if isinstance(action, RefillAction))
    if refill_actions:
        return _choose_refill_action(state, refill_actions)

    place_actions = _place_actions(actions)
    if not place_actions:
        end_turn = _first_action(actions, EndTurnAction)
        return end_turn

    if state.cards_played_this_turn == 0:
        no_damage_first = _choose_no_damage_first_action(state, place_actions)
        if no_damage_first is not None:
            return no_damage_first
        return min(place_actions, key=lambda action: placement_sort_key(state, action))

    no_damage_second = _no_damage_candidates(
        state,
        place_actions,
        _player_negative_count(state),
    )
    if no_damage_second:
        return min(no_damage_second, key=lambda candidate: candidate.sort_key).action

    positive_actions = [
        action for action in place_actions if _loss_score_delta(state, action) > 0
    ]
    if positive_actions:
        return min(positive_actions, key=lambda action: placement_sort_key(state, action))

    if _first_action(actions, EndTurnAction) is not None:
        return _first_action(actions, EndTurnAction)
    return min(place_actions, key=lambda action: placement_sort_key(state, action))


def placement_sort_key(state: GameState, action: PlaceCardAction) -> tuple[int, ...]:
    """Return a deterministic sort key for one placement action.

    Smaller tuples are better. The order mirrors docs/heuristic-bot-design.md.
    """
    player = state.players[state.current_player_index]
    card = player.hand[action.hand_index]
    return (
        _negative_card_delta(state, action),
        -_loss_score_delta(state, action),
        -abs(card.rank_index - 3),
        0 if action.position in state.board else 1,
        -_same_color_remaining_count(player.hand, action.hand_index, card.color),
        -_board_color_count(state.board, card.color),
        action.hand_index,
        action.position.x,
        action.position.y,
        action.frame.x,
        action.frame.y,
    )


def _choose_refill_action(state: GameState, actions: tuple[Action, ...]) -> Action | None:
    player = state.players[state.current_player_index]
    if not player.hand and len(player.negative_cards) >= 6:
        negative_refill = RefillAction(RefillSource.NEGATIVE_CARDS)
        if negative_refill in actions:
            return negative_refill
    deck_refill = RefillAction(RefillSource.DECK)
    if deck_refill in actions:
        return deck_refill
    return min(actions, key=_action_tie_break_key)


def _choose_no_damage_first_action(
    state: GameState,
    first_actions: tuple[PlaceCardAction, ...],
) -> PlaceCardAction | None:
    first_candidates = _no_damage_candidates(
        state,
        first_actions,
        _player_negative_count(state),
    )
    if not first_candidates:
        return None
    return min(first_candidates, key=lambda candidate: candidate.sort_key).action


def _choose_no_damage_two_card_first_action(state: GameState) -> PlaceCardAction | None:
    actions = legal_actions(state)
    first_actions = _place_actions(actions)
    first_candidates = _no_damage_candidates(
        state,
        first_actions,
        _player_negative_count(state),
    )
    ordered_first_actions = tuple(
        candidate.action
        for candidate in sorted(first_candidates, key=lambda c: c.sort_key)
    )
    for first_action in ordered_first_actions:
        after_first = apply_known_legal_action(state, first_action)
        second_actions = _place_actions(legal_actions(after_first))
        second_candidates = _no_damage_candidates(
            after_first,
            second_actions,
            _player_negative_count(after_first),
        )
        if second_candidates:
            return first_action
    return None


def _no_damage_candidates(
    state: GameState,
    actions: tuple[PlaceCardAction, ...],
    target_negative_count: int,
) -> tuple[_PlacementCandidate, ...]:
    candidates: list[_PlacementCandidate] = []
    for action in actions:
        negative_count_after_action = (
            _player_negative_count(state) + _negative_card_delta(state, action)
        )
        if negative_count_after_action != target_negative_count:
            continue
        candidates.append(
            _PlacementCandidate(
                action=action,
                sort_key=placement_sort_key(state, action),
            )
        )
    return tuple(candidates)


def _negative_card_delta(state: GameState, action: PlaceCardAction) -> int:
    frame_cells = frame_positions(action.frame)
    return sum(
        len(stack)
        for position, stack in state.board.items()
        if position not in frame_cells
    )


def _loss_score_delta(state: GameState, action: PlaceCardAction) -> int:
    occupied_before = occupied_count_in_frame(state.board, action.frame)
    was_occupied = action.position in state.board
    occupied_after = occupied_before if was_occupied else occupied_before + 1
    if was_occupied:
        return 0
    if occupied_before < 8 <= occupied_after:
        return 1
    if occupied_before < 9 <= occupied_after:
        return 3
    return 0


def _player_negative_count(state: GameState) -> int:
    return len(state.players[state.current_player_index].negative_cards)


def _same_color_remaining_count(
    hand: tuple[Card, ...],
    played_index: int,
    color: Color,
) -> int:
    return sum(
        1
        for index, card in enumerate(hand)
        if index != played_index and card.color == color
    )


def _board_color_count(board: Board, color: Color) -> int:
    return sum(
        1
        for stack in board.values()
        for card in stack
        if card.color == color
    )


def _place_actions(actions: tuple[Action, ...]) -> tuple[PlaceCardAction, ...]:
    return tuple(action for action in actions if isinstance(action, PlaceCardAction))


def _first_action[T: Action](
    actions: tuple[Action, ...],
    action_type: type[T],
) -> T | None:
    return next((action for action in actions if isinstance(action, action_type)), None)


def _action_tie_break_key(action: Action) -> tuple[int, ...]:
    if isinstance(action, PlaceCardAction):
        return (
            0,
            action.hand_index,
            action.position.x,
            action.position.y,
            action.frame.x,
            action.frame.y,
        )
    if isinstance(action, EndTurnAction):
        return (1,)
    if isinstance(action, RefillAction):
        refill_order = {
            RefillSource.NEGATIVE_CARDS: 0,
            RefillSource.DECK: 1,
            RefillSource.NONE: 2,
        }
        return (2, refill_order[action.source])
    return (3,)
