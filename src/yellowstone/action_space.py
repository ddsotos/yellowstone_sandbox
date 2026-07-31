"""Fixed action-index mapping for reinforcement learning."""

from __future__ import annotations

from yellowstone.game import legal_actions
from yellowstone.types import (
    BOARD_SIZE,
    FRAME_SIZE,
    HAND_SIZE,
    Action,
    EndTurnAction,
    Frame,
    GameState,
    PlaceCardAction,
    Position,
    RefillAction,
    RefillSource,
)


FRAME_AXIS_SIZE = BOARD_SIZE - FRAME_SIZE + 1
FRAME_COUNT = FRAME_AXIS_SIZE * FRAME_AXIS_SIZE
PLACEMENT_ACTION_COUNT = HAND_SIZE * BOARD_SIZE * FRAME_COUNT
END_TURN_ACTION_INDEX = PLACEMENT_ACTION_COUNT
REFILL_ACTION_START_INDEX = END_TURN_ACTION_INDEX + 1
REFILL_SOURCE_ORDER = (
    RefillSource.DECK,
    RefillSource.NEGATIVE_CARDS,
    RefillSource.NONE,
)
ACTION_SPACE_SIZE = REFILL_ACTION_START_INDEX + len(REFILL_SOURCE_ORDER)


def action_to_index(action: Action) -> int:
    """Encode an action as a fixed action index."""
    if isinstance(action, PlaceCardAction):
        _validate_place_action_for_index(action)
        frame_index = action.frame.y * FRAME_AXIS_SIZE + action.frame.x
        return (
            action.hand_index * BOARD_SIZE * FRAME_COUNT
            + action.position.x * FRAME_COUNT
            + frame_index
        )
    if isinstance(action, EndTurnAction):
        return END_TURN_ACTION_INDEX
    if isinstance(action, RefillAction):
        return REFILL_ACTION_START_INDEX + REFILL_SOURCE_ORDER.index(action.source)
    raise TypeError(f"unsupported action: {action!r}")


def action_from_index(index: int, state: GameState) -> Action:
    """Decode an action index using the current state where needed."""
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise ValueError(f"action index out of range: {index}")
    if index < PLACEMENT_ACTION_COUNT:
        hand_index, remainder = divmod(index, BOARD_SIZE * FRAME_COUNT)
        x, frame_index = divmod(remainder, FRAME_COUNT)
        frame_y, frame_x = divmod(frame_index, FRAME_AXIS_SIZE)
        player = state.players[state.current_player_index]
        if hand_index >= len(player.hand):
            raise ValueError(f"hand index is not available: {hand_index}")
        return PlaceCardAction(
            hand_index=hand_index,
            position=Position(x=x, y=player.hand[hand_index].rank_index),
            frame=Frame(x=frame_x, y=frame_y),
        )
    if index == END_TURN_ACTION_INDEX:
        return EndTurnAction()
    source_index = index - REFILL_ACTION_START_INDEX
    return RefillAction(REFILL_SOURCE_ORDER[source_index])


def legal_action_indices(state: GameState) -> tuple[int, ...]:
    """Return legal action indexes for the current state."""
    return tuple(action_to_index(action) for action in legal_actions(state))


def legal_action_mask(state: GameState) -> tuple[bool, ...]:
    """Return a fixed-size legal action mask."""
    legal_indexes = set(legal_action_indices(state))
    return tuple(index in legal_indexes for index in range(ACTION_SPACE_SIZE))


def action_space_metadata() -> dict[str, int]:
    """Return action-space sizes for downstream agents and tests."""
    return {
        "action_space_size": ACTION_SPACE_SIZE,
        "placement_action_count": PLACEMENT_ACTION_COUNT,
        "end_turn_action_index": END_TURN_ACTION_INDEX,
        "refill_action_start_index": REFILL_ACTION_START_INDEX,
    }


def _validate_place_action_for_index(action: PlaceCardAction) -> None:
    if not 0 <= action.hand_index < HAND_SIZE:
        raise ValueError(f"hand index out of range: {action.hand_index}")
    if not 0 <= action.position.x < BOARD_SIZE:
        raise ValueError(f"position x out of range: {action.position.x}")
    if not 0 <= action.position.y < BOARD_SIZE:
        raise ValueError(f"position y out of range: {action.position.y}")
    if not 0 <= action.frame.x < FRAME_AXIS_SIZE:
        raise ValueError(f"frame x out of range: {action.frame.x}")
    if not 0 <= action.frame.y < FRAME_AXIS_SIZE:
        raise ValueError(f"frame y out of range: {action.frame.y}")
