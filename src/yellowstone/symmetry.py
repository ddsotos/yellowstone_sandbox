"""Yellowstone color, horizontal, and rank-inverting vertical symmetries."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from yellowstone.game import sort_hand
from yellowstone.types import (
    BOARD_SIZE,
    FRAME_SIZE,
    Action,
    Board,
    Card,
    Color,
    EndTurnAction,
    Frame,
    GameState,
    PlaceCardAction,
    PlayerState,
    Position,
    RefillAction,
)


ColorMap = Mapping[Color, Color]


def identity_color_map() -> dict[Color, Color]:
    """Return the identity color permutation."""
    return {color: color for color in Color}


def validate_color_permutation(color_map: ColorMap) -> None:
    """Validate that every game color is mapped exactly once."""
    if set(color_map) != set(Color) or set(color_map.values()) != set(Color):
        raise ValueError("color_map must be a bijection over all Color values")


def transform_state(
    state: GameState,
    *,
    color_map: ColorMap | None = None,
    horizontal_reflection: bool = False,
    vertical_reflection: bool = False,
) -> GameState:
    """Transform colors and optionally mirror x and rank/y."""
    mapping = identity_color_map() if color_map is None else color_map
    validate_color_permutation(mapping)
    players = tuple(
        replace(
            player,
            hand=sort_hand(
                _transform_card(card, mapping, vertical_reflection)
                for card in player.hand
            ),
            negative_cards=tuple(
                _transform_card(card, mapping, vertical_reflection)
                for card in player.negative_cards
            ),
        )
        for player in state.players
    )
    board: Board = {
        _transform_position(
            position,
            horizontal_reflection=horizontal_reflection,
            vertical_reflection=vertical_reflection,
        ): tuple(
            _transform_card(card, mapping, vertical_reflection) for card in stack
        )
        for position, stack in state.board.items()
    }
    return replace(
        state,
        players=players,
        board=board,
        deck=tuple(
            _transform_card(card, mapping, vertical_reflection) for card in state.deck
        ),
    )


def transform_action(
    state: GameState,
    action: Action,
    *,
    color_map: ColorMap | None = None,
    horizontal_reflection: bool = False,
    vertical_reflection: bool = False,
) -> Action:
    """Transform an action, resolving its placement index in the sorted new hand."""
    mapping = identity_color_map() if color_map is None else color_map
    validate_color_permutation(mapping)
    if isinstance(action, (EndTurnAction, RefillAction)):
        return action
    player = state.players[state.current_player_index]
    if not 0 <= action.hand_index < len(player.hand):
        raise ValueError(f"hand index is not available: {action.hand_index}")
    original_card = player.hand[action.hand_index]
    transformed_card = _transform_card(original_card, mapping, vertical_reflection)
    occurrence = sum(card == original_card for card in player.hand[: action.hand_index])
    transformed_hand = sort_hand(
        _transform_card(card, mapping, vertical_reflection) for card in player.hand
    )
    transformed_index = _index_of_occurrence(transformed_hand, transformed_card, occurrence)
    return PlaceCardAction(
        hand_index=transformed_index,
        position=_transform_position(
            action.position,
            horizontal_reflection=horizontal_reflection,
            vertical_reflection=vertical_reflection,
        ),
        frame=_transform_frame(
            action.frame,
            horizontal_reflection=horizontal_reflection,
            vertical_reflection=vertical_reflection,
        ),
    )


def rotate_90(*_args: object, **_kwargs: object) -> None:
    """Reject a non-symmetry explicitly instead of silently transforming ranks."""
    raise ValueError("90-degree rotation is not a Yellowstone symmetry")


def _transform_card(
    card: Card, color_map: ColorMap, vertical_reflection: bool
) -> Card:
    return Card(
        color=color_map[card.color],
        rank_index=BOARD_SIZE - 1 - card.rank_index
        if vertical_reflection
        else card.rank_index,
    )


def _transform_position(
    position: Position,
    *,
    horizontal_reflection: bool,
    vertical_reflection: bool,
) -> Position:
    return Position(
        x=BOARD_SIZE - 1 - position.x if horizontal_reflection else position.x,
        y=BOARD_SIZE - 1 - position.y if vertical_reflection else position.y,
    )


def _transform_frame(
    frame: Frame,
    *,
    horizontal_reflection: bool,
    vertical_reflection: bool,
) -> Frame:
    return Frame(
        x=BOARD_SIZE - FRAME_SIZE - frame.x if horizontal_reflection else frame.x,
        y=BOARD_SIZE - FRAME_SIZE - frame.y if vertical_reflection else frame.y,
    )


def _index_of_occurrence(cards: tuple[Card, ...], target: Card, occurrence: int) -> int:
    matches = [index for index, card in enumerate(cards) if card == target]
    try:
        return matches[occurrence]
    except IndexError as error:
        raise AssertionError("transformed card disappeared from hand") from error
