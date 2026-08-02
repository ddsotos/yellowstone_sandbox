"""Compact Original V1 tensors with a rank-by-column board."""

from __future__ import annotations

from dataclasses import dataclass

from yellowstone.value_learning import HISTORY_SIZE, VALUE_CONTEXT_SIZE


CANONICALIZATION_BOARD_COLUMNS_V1 = "board_columns_v1_history_none"
BOARD_COLUMNS_CHANNELS = 1
BOARD_COLUMNS_HEIGHT = 7
BOARD_COLUMNS_WIDTH = 3
BOARD_COLUMNS_LEFT_MARGIN_CLASSES = 5
BOARD_COLUMNS_CONTEXT_SIZE = (
    VALUE_CONTEXT_SIZE - HISTORY_SIZE * 12 + BOARD_COLUMNS_LEFT_MARGIN_CLASSES
)


@dataclass(frozen=True, slots=True)
class BoardColumnsStats:
    records: int = 0
    left_margin_0: int = 0
    left_margin_1: int = 0
    left_margin_2: int = 0
    left_margin_3: int = 0
    left_margin_4: int = 0


def board_columns_metadata() -> dict[str, object]:
    return {
        "input_canonicalization": CANONICALIZATION_BOARD_COLUMNS_V1,
        "board_shape": [
            BOARD_COLUMNS_CHANNELS,
            BOARD_COLUMNS_HEIGHT,
            BOARD_COLUMNS_WIDTH,
        ],
        "board_columns_cell": "card_count",
        "board_columns_rows": "rank_1_to_7_after_fast_lr_ud_color_v1",
        "board_columns_columns": "occupied_span_left_justified_after_fast_lr_ud_color_v1",
        "board_columns_left_margin": "5-class one-hot for empty columns to the left before left-justify",
        "history_semantics": "none",
        "removed_context_features": "rolling_last_two_placements",
    }


def board_columns_from_canonical_v1_tensors(board, context):
    """Convert fast-canonical Original V1 tensors to [1,7,3] board columns."""
    import numpy as np

    if board.ndim != 4 or board.shape[1:] != (29, 7, 7):
        raise ValueError(f"expected board shape [N,29,7,7], got {board.shape}")
    if context.ndim != 2 or context.shape[1] != VALUE_CONTEXT_SIZE:
        raise ValueError(f"expected context shape [N,{VALUE_CONTEXT_SIZE}], got {context.shape}")

    compact_board = np.zeros(
        (board.shape[0], BOARD_COLUMNS_CHANNELS, BOARD_COLUMNS_HEIGHT, BOARD_COLUMNS_WIDTH),
        dtype=np.float32,
    )
    compact_context = np.zeros(
        (context.shape[0], BOARD_COLUMNS_CONTEXT_SIZE),
        dtype=np.float32,
    )
    compact_context[:, : VALUE_CONTEXT_SIZE - HISTORY_SIZE * 12] = context[
        :, : VALUE_CONTEXT_SIZE - HISTORY_SIZE * 12
    ]
    margins = [0] * BOARD_COLUMNS_LEFT_MARGIN_CLASSES
    occupancy = board[:, -1, :, :]
    for index in range(board.shape[0]):
        columns = np.flatnonzero(occupancy[index].sum(axis=0) > 0)
        if len(columns) == 0:
            raise ValueError(f"record {index} has an empty board")
        left = int(columns.min())
        right = int(columns.max())
        width = right - left + 1
        if width > BOARD_COLUMNS_WIDTH:
            raise ValueError(f"record {index} occupied width {width} exceeds 3")
        if not 0 <= left < BOARD_COLUMNS_LEFT_MARGIN_CLASSES:
            raise ValueError(f"record {index} left margin out of range: {left}")
        compact_board[index, 0, :, :width] = occupancy[index, :, left : right + 1]
        compact_context[index, -BOARD_COLUMNS_LEFT_MARGIN_CLASSES + left] = 1.0
        margins[left] += 1
    return compact_board, compact_context, BoardColumnsStats(
        records=int(board.shape[0]),
        left_margin_0=margins[0],
        left_margin_1=margins[1],
        left_margin_2=margins[2],
        left_margin_3=margins[3],
        left_margin_4=margins[4],
    )
