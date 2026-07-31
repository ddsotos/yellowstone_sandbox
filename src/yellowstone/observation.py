"""Fixed-length numeric observation encoding for reinforcement learning."""

from __future__ import annotations

from yellowstone.heuristic_turn_plan import (
    heuristic_played_rank_features,
    heuristic_turn_features,
)
from yellowstone.types import (
    FRAME_SIZE,
    HAND_SIZE,
    Card,
    Color,
    GameState,
    Position,
)


COLOR_ORDER = (Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW)
RELATIVE_COLOR_FEATURE_SIZE = len(COLOR_ORDER)
HAND_SLOT_FEATURE_SIZE = 1 + RELATIVE_COLOR_FEATURE_SIZE + 1
OBSERVED_PLAYER_COUNT = 4
PLAYER_FEATURE_SIZE = 3
BOARD_RANK_COUNT_FEATURE_SIZE = 7
BOARD_LEFT_COLUMN_FEATURE_SIZE = 1
BOARD_COLUMN_COUNT_FEATURE_SIZE = FRAME_SIZE
BOARD_OBSERVATION_SIZE = (
    BOARD_RANK_COUNT_FEATURE_SIZE
    + BOARD_LEFT_COLUMN_FEATURE_SIZE
    + BOARD_COLUMN_COUNT_FEATURE_SIZE
)
HAND_OBSERVATION_SIZE = HAND_SIZE * HAND_SLOT_FEATURE_SIZE
PLAYERS_OBSERVATION_SIZE = OBSERVED_PLAYER_COUNT * PLAYER_FEATURE_SIZE
OPPONENT_LAST_TURN_OBSERVATION_SIZE = OBSERVED_PLAYER_COUNT - 1
HEURISTIC_BONUS_OBSERVATION_SIZE = 2
HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE = 2
HEURISTIC_PLAYED_RANK_OBSERVATION_SIZE = 3
SCALAR_OBSERVATION_SIZE = 2
OBSERVATION_SIZE = (
    BOARD_OBSERVATION_SIZE
    + HAND_OBSERVATION_SIZE
    + PLAYERS_OBSERVATION_SIZE
    + OPPONENT_LAST_TURN_OBSERVATION_SIZE
    + HEURISTIC_BONUS_OBSERVATION_SIZE
    + HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE
    + HEURISTIC_PLAYED_RANK_OBSERVATION_SIZE
    + SCALAR_OBSERVATION_SIZE
)


def state_to_observation(
    state: GameState,
    *,
    heuristic_bonuses: tuple[int, int] | None = None,
    heuristic_negative_deltas: tuple[int, int] | None = None,
    heuristic_played_ranks: tuple[int, int, int] | None = None,
) -> tuple[int, ...]:
    """Encode a state as a fixed-length integer tuple."""
    values: list[int] = []
    values.extend(_board_features(state))
    values.extend(_current_hand_features(state))
    values.extend(_player_features(state))
    values.extend(_opponent_last_turn_play_count_features(state))
    if heuristic_bonuses is None or heuristic_negative_deltas is None:
        turn_features = heuristic_turn_features(state)
        if heuristic_bonuses is None:
            heuristic_bonuses = turn_features[:2]
        if heuristic_negative_deltas is None:
            heuristic_negative_deltas = turn_features[2:]
    if heuristic_played_ranks is None:
        heuristic_played_ranks = heuristic_played_rank_features(state)
    values.extend(heuristic_bonuses)
    values.extend(heuristic_negative_deltas)
    values.extend(heuristic_played_ranks)
    values.extend(
        [
            _deck_bucket(len(state.deck)),
            state.settlement_count,
        ]
    )
    if len(values) != OBSERVATION_SIZE:
        raise AssertionError(f"unexpected observation size: {len(values)}")
    return tuple(values)


def observation_metadata() -> dict[str, int]:
    """Return feature sizes for downstream encoders and tests."""
    return {
        "observation_size": OBSERVATION_SIZE,
        "board_size": BOARD_OBSERVATION_SIZE,
        "hand_size": HAND_OBSERVATION_SIZE,
        "players_size": PLAYERS_OBSERVATION_SIZE,
        "opponent_last_turn_play_counts_size": (
            OPPONENT_LAST_TURN_OBSERVATION_SIZE
        ),
        "heuristic_bonus_size": HEURISTIC_BONUS_OBSERVATION_SIZE,
        "heuristic_negative_delta_size": (
            HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE
        ),
        "heuristic_played_rank_size": HEURISTIC_PLAYED_RANK_OBSERVATION_SIZE,
        "scalar_size": SCALAR_OBSERVATION_SIZE,
    }


def _board_features(state: GameState) -> list[int]:
    left_x = _board_left_column(state)
    rank_counts = [0] * BOARD_RANK_COUNT_FEATURE_SIZE
    column_counts = [0] * BOARD_COLUMN_COUNT_FEATURE_SIZE
    for position, stack in state.board.items():
        if 0 <= position.y < len(rank_counts):
            rank_counts[position.y] += len(stack)
        relative_x = position.x - left_x
        if 0 <= relative_x < len(column_counts):
            column_counts[relative_x] += len(stack)
    return [*rank_counts, left_x, *column_counts]


def _board_anchor(state: GameState) -> Position:
    if not state.board:
        return Position(x=0, y=0)
    min_x = min(position.x for position in state.board)
    min_y = min(position.y for position in state.board)
    return Position(x=min_x, y=min_y)


def _board_left_column(state: GameState) -> int:
    if not state.board:
        return 0
    return min(position.x for position in state.board)


def _current_hand_features(state: GameState) -> list[int]:
    values: list[int] = []
    player = state.players[state.current_player_index]
    relative_color_indices = _relative_color_indices(state)
    for index in range(HAND_SIZE):
        if index < len(player.hand):
            card = player.hand[index]
            values.append(1)
            values.extend(
                _one_hot(
                    relative_color_indices[card.color],
                    RELATIVE_COLOR_FEATURE_SIZE,
                )
            )
            values.append(card.rank_index)
        else:
            values.extend([0] * HAND_SLOT_FEATURE_SIZE)
    return values


def _player_features(state: GameState) -> list[int]:
    values: list[int] = []
    for index in range(OBSERVED_PLAYER_COUNT):
        if index < len(state.players):
            player = state.players[index]
            values.extend(
                [
                    len(player.hand),
                    len(player.negative_cards),
                    player.loss_score,
                ]
            )
        else:
            values.extend([0] * PLAYER_FEATURE_SIZE)
    return values


def _opponent_last_turn_play_count_features(state: GameState) -> list[int]:
    counts = _last_turn_play_counts(state)
    return [
        counts[(state.current_player_index + offset) % len(state.players)]
        for offset in range(1, min(len(state.players), OBSERVED_PLAYER_COUNT))
    ]


def _last_turn_play_counts(state: GameState) -> tuple[int, ...]:
    if len(state.last_turn_play_counts) == len(state.players):
        return state.last_turn_play_counts
    return tuple(0 for _ in state.players)


def _deck_bucket(deck_count: int) -> int:
    if deck_count == 0:
        return 0
    if deck_count <= HAND_SIZE:
        return 1
    if deck_count <= HAND_SIZE * 3:
        return 2
    return 3


def _relative_color_indices(state: GameState) -> dict[Color, int]:
    anchor = _board_anchor(state)
    assigned: dict[Color, int] = {}
    used_indices: set[int] = set()

    for relative_index, x in enumerate(
        range(anchor.x + FRAME_SIZE - 1, anchor.x - 1, -1)
    ):
        for color in _colors_in_column(state, x, anchor.y):
            assigned[color] = relative_index
            used_indices.add(relative_index)

    player = state.players[state.current_player_index]
    for card in player.hand:
        if card.color in assigned:
            continue
        relative_index = _first_unused_relative_color_index(used_indices)
        assigned[card.color] = relative_index
        used_indices.add(relative_index)

    for color in COLOR_ORDER:
        if color in assigned:
            continue
        relative_index = _first_unused_relative_color_index(used_indices)
        assigned[color] = relative_index
        used_indices.add(relative_index)
    return assigned


def _colors_in_column(state: GameState, x: int, min_y: int) -> tuple[Color, ...]:
    colors = {
        card.color
        for y in range(min_y, min_y + FRAME_SIZE)
        for card in state.board.get(Position(x=x, y=y), ())
    }
    return tuple(color for color in COLOR_ORDER if color in colors)


def _first_unused_relative_color_index(used_indices: set[int]) -> int:
    for index in range(RELATIVE_COLOR_FEATURE_SIZE):
        if index not in used_indices:
            return index
    raise AssertionError("relative color index exhausted")


def _one_hot(index: int, size: int) -> list[int]:
    return [1 if current == index else 0 for current in range(size)]
