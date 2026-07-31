"""Fast approximate canonicalization for win-value model inputs.

The transform intentionally favors speed over complete orbit collapse:

1. mirror horizontally when the visual left half has more cards;
2. invert ranks/y when ranks 5-7 have more cards than ranks 1-3;
3. rename board colors from visual left to right as blue, red, green, yellow;
4. order colors absent from the board by the viewer's low-to-high hand counts.

Ties at each step keep the current orientation or absolute color order.
"""

from __future__ import annotations

from dataclasses import dataclass

from yellowstone.value_learning import COLOR_ORDER, RANK_BOARD_CHANNELS


CANONICALIZATION_NAME = "fast_lr_ud_color_v1"
_BOARD_SIZE = 7
_COLOR_COUNT = 4
_COLOR_RANK_CHANNELS = _COLOR_COUNT * _BOARD_SIZE
_HAND_SIZE = 6
_HAND_FEATURES = 6
_HAND_CONTEXT_SIZE = _HAND_SIZE * _HAND_FEATURES
_HISTORY_OFFSET = _HAND_CONTEXT_SIZE + 12 + 9
_HISTORY_FEATURES = 12

# Encoded COLOR_ORDER is red, blue, green, yellow.  The canonical board order
# requested by the user is blue, red, green, yellow.
_CANONICAL_COLOR_SEQUENCE = (1, 0, 2, 3)
# game.sort_hand compares Color.value strings: blue, green, red, yellow.
_COLOR_SORT_PRIORITY = (2, 0, 1, 3)


@dataclass(frozen=True, slots=True)
class CanonicalizationStats:
    records: int
    horizontal_reflections: int
    vertical_reflections: int


def canonicalize_value_tensors(board, context):
    """Return canonicalized copies of batched board/context NumPy arrays."""
    canonical_board, canonical_context, _ = canonicalize_value_tensors_with_stats(
        board, context
    )
    return canonical_board, canonical_context


def canonicalize_value_tensors_with_stats(board, context):
    """Canonicalize encoded records and return aggregate reflection counts."""
    import numpy as np

    board_array = np.asarray(board)
    context_array = np.asarray(context)
    if board_array.ndim != 4 or board_array.shape[1:] != (
        RANK_BOARD_CHANNELS,
        _BOARD_SIZE,
        _BOARD_SIZE,
    ):
        raise ValueError(f"unexpected board shape: {board_array.shape}")
    if (
        context_array.ndim != 2
        or context_array.shape[1] < _HISTORY_OFFSET
        or (context_array.shape[1] - _HISTORY_OFFSET) % _HISTORY_FEATURES
    ):
        raise ValueError(f"unexpected context shape: {context_array.shape}")
    history_size = (
        context_array.shape[1] - _HISTORY_OFFSET
    ) // _HISTORY_FEATURES
    if len(board_array) != len(context_array):
        raise ValueError("board and context batch sizes differ")

    result_board = np.array(board_array, copy=True)
    result_context = np.array(context_array, copy=True)
    record_count = len(result_board)
    if record_count == 0:
        return (
            result_board,
            result_context,
            CanonicalizationStats(0, 0, 0),
        )

    occupancy = result_board[:, -1]
    right_count = occupancy[:, :, :3].sum(axis=(1, 2))
    left_count = occupancy[:, :, 4:].sum(axis=(1, 2))
    horizontal = left_count > right_count
    if horizontal.any():
        result_board[horizontal] = result_board[horizontal, :, :, ::-1]

    occupancy = result_board[:, -1]
    low_rank_count = occupancy[:, :3, :].sum(axis=(1, 2))
    high_rank_count = occupancy[:, 4:, :].sum(axis=(1, 2))
    vertical = high_rank_count > low_rank_count
    if vertical.any():
        color_rank = result_board[:, :_COLOR_RANK_CHANNELS].reshape(
            record_count,
            _COLOR_COUNT,
            _BOARD_SIZE,
            _BOARD_SIZE,
            _BOARD_SIZE,
        )
        color_rank[vertical] = color_rank[vertical, :, ::-1, ::-1, :]
        result_board[vertical, -1] = result_board[vertical, -1, ::-1, :]

        hand = result_context[:, :_HAND_CONTEXT_SIZE].reshape(
            record_count, _HAND_SIZE, _HAND_FEATURES
        )
        valid_hand = hand[:, :, 0] > 0.5
        hand_rank = hand[:, :, 5]
        hand_vertical = vertical[:, None] & valid_hand
        normalized_ranks = np.asarray(
            [rank / 6 for rank in range(_BOARD_SIZE)], dtype=result_context.dtype
        )
        reflected_hand_rank = 6 - np.rint(hand_rank[hand_vertical] * 6).astype(int)
        hand_rank[hand_vertical] = normalized_ranks[reflected_hand_rank]
        for history_index in range(history_size):
            offset = _HISTORY_OFFSET + history_index * _HISTORY_FEATURES
            history_present = result_context[:, offset] > 0.5
            history_rank = result_context[:, offset + 9]
            mask = vertical & history_present
            reflected_history_rank = (
                6 - np.rint(history_rank[mask] * 6).astype(int)
            )
            history_rank[mask] = normalized_ranks[reflected_history_rank]

    _canonicalize_colors_in_place(
        result_board, result_context, history_size=history_size
    )
    stats = CanonicalizationStats(
        records=record_count,
        horizontal_reflections=int(horizontal.sum()),
        vertical_reflections=int(vertical.sum()),
    )
    return result_board, result_context, stats


def _canonicalize_colors_in_place(
    board, context, *, history_size: int
) -> None:
    import numpy as np

    record_count = len(board)
    rows = np.arange(record_count)
    color_rank = board[:, :_COLOR_RANK_CHANNELS].reshape(
        record_count,
        _COLOR_COUNT,
        _BOARD_SIZE,
        _BOARD_SIZE,
        _BOARD_SIZE,
    )
    color_columns = color_rank.sum(axis=(2, 3))
    present = color_columns.sum(axis=2) > 0
    column_index = color_columns.argmax(axis=2)

    hand = context[:, :_HAND_CONTEXT_SIZE].reshape(
        record_count, _HAND_SIZE, _HAND_FEATURES
    )
    hand_present = hand[:, :, 0] > 0.5
    hand_color = hand[:, :, 1:5].argmax(axis=2)
    hand_rank = np.rint(hand[:, :, 5] * 6).astype(np.int8)
    color_ids = np.arange(_COLOR_COUNT)
    rank_ids = np.arange(_BOARD_SIZE)
    hand_counts = (
        hand_present[:, :, None, None]
        & (hand_color[:, :, None, None] == color_ids[None, None, :, None])
        & (hand_rank[:, :, None, None] == rank_ids[None, None, None, :])
    ).sum(axis=1)
    rank_weights = (7 ** np.arange(6, -1, -1)).astype(np.int64)
    hand_signature = (hand_counts * rank_weights[None, None, :]).sum(axis=2)
    max_signature = int(6 * rank_weights[0])

    # Visual left is the high-x side because (0, 0) is the upper-right cell.
    present_key = _BOARD_SIZE - 1 - column_index
    absent_key = (
        1_000_000
        + (max_signature - hand_signature) * _COLOR_COUNT
        + color_ids[None, :]
    )
    color_key = np.where(present, present_key, absent_key)
    ordered_old_colors = np.argsort(color_key, axis=1, stable=True)

    old_to_new = np.empty((record_count, _COLOR_COUNT), dtype=np.int8)
    old_to_new[rows[:, None], ordered_old_colors] = np.asarray(
        _CANONICAL_COLOR_SEQUENCE, dtype=np.int8
    )[None, :]

    original_color_rank = color_rank.copy()
    for old_color in range(_COLOR_COUNT):
        color_rank[rows, old_to_new[:, old_color]] = original_color_rank[:, old_color]

    mapped_hand_color = old_to_new[rows[:, None], hand_color]
    hand[:, :, 1:5] = 0.0
    hand[rows[:, None], np.arange(_HAND_SIZE)[None, :], mapped_hand_color + 1] = (
        hand_present
    )
    sort_priority = np.asarray(_COLOR_SORT_PRIORITY, dtype=np.int8)[mapped_hand_color]
    hand_sort_key = np.where(
        hand_present,
        hand_rank.astype(np.int16) * _COLOR_COUNT + sort_priority,
        10_000,
    )
    hand_order = np.argsort(hand_sort_key, axis=1, stable=True)
    hand[:] = np.take_along_axis(hand, hand_order[:, :, None], axis=1)

    for history_index in range(history_size):
        offset = _HISTORY_OFFSET + history_index * _HISTORY_FEATURES
        history_present = context[:, offset] > 0.5
        history_color = context[:, offset + 5 : offset + 9].argmax(axis=1)
        mapped_history_color = old_to_new[rows, history_color]
        context[:, offset + 5 : offset + 9] = 0.0
        context[rows, offset + 5 + mapped_history_color] = history_present
