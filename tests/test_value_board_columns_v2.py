import numpy as np

from yellowstone.value_board_columns_v2 import (
    BOARD_COLUMNS_V2_CONTEXT_SIZE,
    PREPLAY_BOARD_COLUMNS_CONTEXT_SIZE,
    VALUE_CONTEXT_SIZE_V2,
    VALUE_CONTEXT_SIZE_V2_LITE,
    board_columns_from_canonical_board,
    board_columns_v2_metadata,
)


def test_v2_board_columns_shape_and_metadata() -> None:
    board = np.zeros((1, 29, 7, 7), dtype=np.float32)
    context = np.zeros((1, VALUE_CONTEXT_SIZE_V2), dtype=np.float32)
    board[0, -1, 1, 1] = 1
    board[0, -1, 6, 3] = 2

    compact_board, compact_context, stats = board_columns_from_canonical_board(
        board, context
    )

    assert compact_board.shape == (1, 1, 7, 3)
    assert compact_context.shape == (1, BOARD_COLUMNS_V2_CONTEXT_SIZE)
    assert compact_board[0, 0, 1, 0] == 1
    assert compact_board[0, 0, 6, 2] == 2
    assert compact_context[0, -7:].tolist() == [0, 1, 0, 0, 0, 0, 0]
    assert stats.left_margin_counts == (0, 1, 0, 0, 0, 0, 0)
    assert board_columns_v2_metadata(preplay=False)["board_shape"] == [1, 7, 3]


def test_preplay_board_columns_context_size() -> None:
    board = np.zeros((1, 29, 7, 7), dtype=np.float32)
    context = np.zeros((1, VALUE_CONTEXT_SIZE_V2_LITE), dtype=np.float32)
    board[0, -1, 2, 2] = 1

    _, compact_context, _ = board_columns_from_canonical_board(board, context)

    assert compact_context.shape == (1, PREPLAY_BOARD_COLUMNS_CONTEXT_SIZE)
    assert board_columns_v2_metadata(preplay=True)["opponent_private_inputs"] is False
