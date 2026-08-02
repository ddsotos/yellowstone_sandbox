"""Compact V2 public tensors with a rank-by-column board."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yellowstone.cnn import build_win_value_net
from yellowstone.value_v2 import VALUE_CONTEXT_SIZE_V2
from yellowstone.value_v2_lite import VALUE_CONTEXT_SIZE_V2_LITE


CANONICALIZATION_BOARD_COLUMNS_V2 = "board_columns_v2"
VALUE_SCHEMA_BOARD_COLUMNS_V2 = "yellowstone.value.v2-board-columns.v1"
HISTORY_SEMANTICS_BOARD_COLUMNS_V2 = "rolling_last_three_completed_turns_v2"
CANONICALIZATION_PREPLAY_BOARD_COLUMNS = "preplay_board_columns_v1"
VALUE_SCHEMA_PREPLAY_BOARD_COLUMNS = "yellowstone.value.preplay-board-columns.v1"
HISTORY_SEMANTICS_PREPLAY_BOARD_COLUMNS = "last_two_completed_turns_before_turn"
BOARD_COLUMNS_CHANNELS = 1
BOARD_COLUMNS_HEIGHT = 7
BOARD_COLUMNS_WIDTH = 3
BOARD_COLUMNS_LEFT_MARGIN_CLASSES = 7
BOARD_COLUMNS_V2_CONTEXT_SIZE = VALUE_CONTEXT_SIZE_V2 + BOARD_COLUMNS_LEFT_MARGIN_CLASSES
PREPLAY_BOARD_COLUMNS_CONTEXT_SIZE = (
    VALUE_CONTEXT_SIZE_V2_LITE + BOARD_COLUMNS_LEFT_MARGIN_CLASSES
)


@dataclass(frozen=True, slots=True)
class BoardColumnsV2Stats:
    records: int = 0
    left_margin_counts: tuple[int, ...] = (0, 0, 0, 0, 0)


def board_columns_v2_metadata(
    *,
    preplay: bool = False,
) -> dict[str, object]:
    return {
        "value_schema": (
            VALUE_SCHEMA_PREPLAY_BOARD_COLUMNS
            if preplay
            else VALUE_SCHEMA_BOARD_COLUMNS_V2
        ),
        "input_canonicalization": (
            CANONICALIZATION_PREPLAY_BOARD_COLUMNS
            if preplay
            else CANONICALIZATION_BOARD_COLUMNS_V2
        ),
        "base_input_canonicalization": (
            "strict_residual_v2_lite_transition_preplay_state"
            if preplay
            else "strict_residual_v2"
        ),
        "history_semantics": (
            HISTORY_SEMANTICS_PREPLAY_BOARD_COLUMNS
            if preplay
            else HISTORY_SEMANTICS_BOARD_COLUMNS_V2
        ),
        "board_shape": [
            BOARD_COLUMNS_CHANNELS,
            BOARD_COLUMNS_HEIGHT,
            BOARD_COLUMNS_WIDTH,
        ],
        "board_columns_cell": "card_count",
        "board_columns_rows": "rank_1_to_7_after_v2_residual_canonicalization",
        "board_columns_columns": "occupied_span_left_justified_after_v2_residual_canonicalization",
        "board_columns_left_margin": "7-class one-hot for empty columns to the left before left-justify",
        "opponent_private_inputs": False,
    }


def board_columns_from_canonical_board(board, context):
    """Convert a canonical V2-style [N,C,7,7] board to [N,1,7,3]."""
    import numpy as np

    if board.ndim != 4 or board.shape[2:] != (7, 7):
        raise ValueError(f"expected board shape [N,C,7,7], got {board.shape}")
    if context.ndim != 2 or context.shape[0] != board.shape[0]:
        raise ValueError("board/context record counts differ")
    compact_board = np.zeros(
        (board.shape[0], BOARD_COLUMNS_CHANNELS, BOARD_COLUMNS_HEIGHT, BOARD_COLUMNS_WIDTH),
        dtype=np.float32,
    )
    compact_context = np.zeros(
        (context.shape[0], context.shape[1] + BOARD_COLUMNS_LEFT_MARGIN_CLASSES),
        dtype=np.float32,
    )
    compact_context[:, : context.shape[1]] = context
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
        compact_context[index, context.shape[1] + left] = 1.0
        margins[left] += 1
    return compact_board, compact_context, BoardColumnsV2Stats(
        records=int(board.shape[0]),
        left_margin_counts=tuple(margins),
    )


class TorchWinValueEstimatorV2BoardColumns:
    """Batched V2 board-columns inference adapter."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        import torch

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        expected = board_columns_v2_metadata(preplay=False)
        mismatches = {
            key: {"expected": value, "actual": checkpoint.get(key)}
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(f"V2 board-columns checkpoint differs: {mismatches}")
        self.torch = torch
        self.model = build_win_value_net(
            context_size=BOARD_COLUMNS_V2_CONTEXT_SIZE,
            board_channels=1,
            board_height=7,
            board_width=3,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def __call__(self, record) -> float:
        return self.estimate_many((record,))[0]

    def estimate_many(self, records: tuple[object, ...]) -> tuple[float, ...]:
        import numpy as np

        from yellowstone.value_v2 import canonical_tensors_v2

        encoded = [canonical_tensors_v2(record) for record in records]
        board, context, _ = board_columns_from_canonical_board(
            np.stack([item[0] for item in encoded]),
            np.stack([item[1] for item in encoded]),
        )
        with self.torch.no_grad():
            values = self.torch.sigmoid(
                self.model(
                    self.torch.from_numpy(board),
                    self.torch.from_numpy(context),
                )
            ).tolist()
        return tuple(float(value) for value in values)
