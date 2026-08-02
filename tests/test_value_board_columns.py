import numpy as np

from yellowstone.value_board_columns import (
    BOARD_COLUMNS_CONTEXT_SIZE,
    board_columns_from_canonical_v1_tensors,
)
from yellowstone.value_learning import VALUE_CONTEXT_SIZE


def test_board_columns_left_justifies_and_strips_history():
    board = np.zeros((1, 29, 7, 7), dtype=np.float32)
    context = np.arange(VALUE_CONTEXT_SIZE, dtype=np.float32).reshape(1, -1)
    board[0, -1, 2, 2] = 1
    board[0, -1, 4, 4] = 2

    compact_board, compact_context, stats = board_columns_from_canonical_v1_tensors(
        board, context
    )

    assert compact_board.shape == (1, 1, 7, 3)
    assert compact_context.shape == (1, BOARD_COLUMNS_CONTEXT_SIZE)
    assert compact_board[0, 0, 2, 0] == 1
    assert compact_board[0, 0, 4, 2] == 2
    assert compact_context[0, :57].tolist() == context[0, :57].tolist()
    assert compact_context[0, -5:].tolist() == [0, 0, 1, 0, 0]
    assert stats.records == 1
    assert stats.left_margin_2 == 1


def test_board_columns_rejects_width_over_three():
    board = np.zeros((1, 29, 7, 7), dtype=np.float32)
    context = np.zeros((1, VALUE_CONTEXT_SIZE), dtype=np.float32)
    board[0, -1, 0, 0] = 1
    board[0, -1, 0, 3] = 1

    try:
        board_columns_from_canonical_v1_tensors(board, context)
    except ValueError as error:
        assert "exceeds 3" in str(error)
    else:
        raise AssertionError("expected width validation error")
