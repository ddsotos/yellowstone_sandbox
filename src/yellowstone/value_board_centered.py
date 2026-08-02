"""V2-aligned board-centered win-value V1 encoding.

This encoder keeps the Original V1 public/own feature set and rolling history
semantics, but chooses spatial and color orientation with the same residual
canonicalization policy used by V2: enumerate visible symmetries and pick the
lexicographically smallest public board, using viewer hand ranks as a vertical
tie-breaker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from yellowstone.types import Card, Color, GameState
from yellowstone.value_learning import (
    COLOR_ORDER,
    HAND_SIZE,
    HISTORY_SIZE,
    RANK_BOARD_CHANNELS,
    VALUE_CONTEXT_SIZE,
    ValueRecord,
)


BOARD_CENTERED_V1 = "board_centered_v1"
BOARD_CENTERED_V1_HISTORY_NONE = "board_centered_v1_history_none"
BOARD_CENTERED_V1_HISTORY_OWN_FRAME_DELTA_2CYCLE = (
    "board_centered_v1_history_own_frame_delta_2cycle"
)
BOARD_CENTERED_V1_HISTORY_V1 = "board_centered_v1_history_v1"
BOARD_CENTERED_V1_HISTORY_TURN_LOCAL = "board_centered_v1_history_turn_local"
BOARD_CENTERED_V1_CHAIN_HISTORY = "bcenter_v1_chain_history"
BOARD_CENTERED_V1_CANONICALIZATIONS = (
    BOARD_CENTERED_V1,
    BOARD_CENTERED_V1_HISTORY_NONE,
    BOARD_CENTERED_V1_HISTORY_OWN_FRAME_DELTA_2CYCLE,
    BOARD_CENTERED_V1_HISTORY_V1,
    BOARD_CENTERED_V1_HISTORY_TURN_LOCAL,
    BOARD_CENTERED_V1_CHAIN_HISTORY,
)
BOARD_CENTERED_BOARD_CHANNELS = 1
BOARD_CENTERED_BOARD_SIZE = 3
ANCHOR_RANK_MIN = 4
ANCHOR_RANK_MAX = 7
RANK_DELTA_MIN = -6
RANK_DELTA_MAX = 3
RANK_DELTA_CLASSES = RANK_DELTA_MAX - RANK_DELTA_MIN + 1
MARGIN_CLASSES = 4
EMPTY_STATE_CLASSES = 4
FRAME_DELTA_MIN = -4
FRAME_DELTA_MAX = 4
FRAME_DELTA_CLASSES = FRAME_DELTA_MAX - FRAME_DELTA_MIN + 1

_BOARD_SIZE = 7
_COLOR_COUNT = 4
_COLOR_RANK_CHANNELS = _COLOR_COUNT * _BOARD_SIZE
_HISTORY_FEATURES = 12

# anchor4 + right-margin4 + top-margin4 + column-empty4 + row-empty4,
# then V1 context with each rank scalar replaced by a 10-class delta one-hot.
BOARD_CENTERED_CONTEXT_PREFIX_SIZE = (
    (ANCHOR_RANK_MAX - ANCHOR_RANK_MIN + 1)
    + MARGIN_CLASSES * 2
    + EMPTY_STATE_CLASSES * 2
)
BOARD_CENTERED_V1_CONTEXT_SIZE = (
    BOARD_CENTERED_CONTEXT_PREFIX_SIZE
    + VALUE_CONTEXT_SIZE
    + (HAND_SIZE + HISTORY_SIZE) * (RANK_DELTA_CLASSES - 1)
)
BOARD_CENTERED_CHAIN_HISTORY_FEATURES = 8
BOARD_CENTERED_V1_CHAIN_CONTEXT_SIZE = (
    BOARD_CENTERED_CONTEXT_PREFIX_SIZE
    + HAND_SIZE * (1 + _COLOR_COUNT + RANK_DELTA_CLASSES)
    + 12
    + 9
    + BOARD_CENTERED_CHAIN_HISTORY_FEATURES
)


@dataclass(frozen=True, slots=True)
class BoardCenteredStats:
    records: int
    min_anchor_rank: int
    max_anchor_rank: int
    min_rank_delta: int
    max_rank_delta: int
    left_margin_counts: tuple[int, ...]
    top_margin_counts: tuple[int, ...]
    column_empty_state_counts: tuple[int, ...]
    row_empty_state_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BoardCenteredTransformV1:
    vertical_reflection: bool
    horizontal_reflection: bool
    old_to_new_color: tuple[int, ...]


def board_centered_metadata(canonicalization: str) -> dict[str, object]:
    if canonicalization not in BOARD_CENTERED_V1_CANONICALIZATIONS:
        raise ValueError(f"unsupported board-centered canonicalization: {canonicalization}")
    history_variant = _history_variant(canonicalization)
    metadata = {
        "input_canonicalization": canonicalization,
        "base_input_canonicalization": "strict_residual_v2_aligned_v1",
        "spatial_canonicalization": "v2_residual_lexicographic_board",
        "color_canonicalization": "v2_residual_color_signature_v1_visible",
        "board_shape": [BOARD_CENTERED_BOARD_CHANNELS, 3, 3],
        "board_cell_values": [0, 1, 2],
        "anchor_rank_range": [ANCHOR_RANK_MIN, ANCHOR_RANK_MAX],
        "anchor_rank_classes": 4,
        "rank_delta_range": [RANK_DELTA_MIN, RANK_DELTA_MAX],
        "rank_delta_classes": RANK_DELTA_CLASSES,
        "board_center_history_variant": history_variant,
        "board_center_history_slots": 4 if history_variant == "chain_history" else 2,
        "board_center_frame_delta_range": [FRAME_DELTA_MIN, FRAME_DELTA_MAX],
        "board_center_frame_delta_classes": FRAME_DELTA_CLASSES,
        "margin_range": [0, 3],
        "margin_classes": MARGIN_CLASSES,
        "margin_semantics": {
            "top_margin": "empty rows above the transformed board frame",
            "left_margin": "empty columns left of the transformed board frame",
        },
        "column_empty_state": {
            "classes": 4,
            "bit0": "second_column_empty",
            "bit1": "third_column_empty",
        },
        "row_empty_state": {
            "classes": 4,
            "bit0": "second_row_empty",
            "bit1": "third_row_empty",
        },
    }
    if history_variant == "chain_history":
        metadata.update(
            {
                "board_center_chain_history_features": [
                    "play_after_vs_play_before_color_layout_changed",
                    "play_after_vs_play_before_top_rank_delta",
                    "play_before_vs_4_turns_prior_after_color_layout_changed",
                    "play_before_vs_4_turns_prior_after_top_rank_delta",
                    "4_turns_prior_after_vs_8_turns_prior_after_color_layout_changed",
                    "4_turns_prior_after_vs_8_turns_prior_after_top_rank_delta",
                    "8_turns_prior_after_vs_12_turns_prior_after_color_layout_changed",
                    "8_turns_prior_after_vs_12_turns_prior_after_top_rank_delta",
                ],
                "board_center_chain_history_missing_padding": (
                    "missing prior public states are encoded as color_changed=0, top_rank_delta=0"
                ),
                "board_center_chain_history_rank_delta_range": [-6, 6],
            }
        )
    return metadata


def board_center_record_for_player(record: ValueRecord):
    """Return V2-aligned board-centered tensors for one Original V1 record."""
    board, context, _ = board_center_records_with_stats((record,))
    return board[0], context[0]


def board_center_frame_origin(state: GameState) -> tuple[int, int]:
    """Return the raw top-left 3x3 b-center frame origin for a public board."""
    occupied = tuple(state.board)
    if not occupied:
        raise ValueError("cannot center an empty board")
    frame_x = max(position.x for position in occupied) - 2
    frame_y = max(position.y for position in occupied) - 2
    if frame_x < 0 or frame_y < 0:
        raise ValueError(
            f"board-centered frame is outside the board: x={frame_x}, y={frame_y}"
        )
    if (
        min(position.x for position in occupied) < frame_x
        or min(position.y for position in occupied) < frame_y
    ):
        raise ValueError("board does not fit b-center 3x3 frame")
    return frame_x, frame_y


def board_center_records_with_stats(
    records: tuple[ValueRecord, ...],
    *,
    canonicalization: str = BOARD_CENTERED_V1,
) -> tuple[np.ndarray, np.ndarray, BoardCenteredStats]:
    """Encode Original V1 records with V2-aligned residual canonicalization."""
    if canonicalization not in BOARD_CENTERED_V1_CANONICALIZATIONS:
        raise ValueError(f"unsupported board-centered canonicalization: {canonicalization}")
    if not records:
        raise ValueError("at least one record is required")
    boards: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    anchors: list[int] = []
    deltas: list[int] = []
    left_margins = [0] * MARGIN_CLASSES
    top_margins = [0] * MARGIN_CLASSES
    column_states = [0] * EMPTY_STATE_CLASSES
    row_states = [0] * EMPTY_STATE_CLASSES
    for row, record in enumerate(records):
        board, context, audit = _encode_record(
            record,
            row,
            history_variant=_history_variant(canonicalization),
        )
        boards.append(board)
        contexts.append(context)
        anchors.append(audit["anchor_rank"])
        deltas.extend(audit["rank_deltas"])
        left_margins[audit["left_margin"]] += 1
        top_margins[audit["top_margin"]] += 1
        column_states[audit["column_empty_state"]] += 1
        row_states[audit["row_empty_state"]] += 1
    return (
        np.stack(boards),
        np.stack(contexts),
        BoardCenteredStats(
            records=len(records),
            min_anchor_rank=min(anchors),
            max_anchor_rank=max(anchors),
            min_rank_delta=min(deltas) if deltas else 0,
            max_rank_delta=max(deltas) if deltas else 0,
            left_margin_counts=tuple(left_margins),
            top_margin_counts=tuple(top_margins),
            column_empty_state_counts=tuple(column_states),
            row_empty_state_counts=tuple(row_states),
        ),
    )


def _encode_record(
    record: ValueRecord, row: int, *, history_variant: str
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    transform = _residual_transform_v1(record)
    totals = np.zeros((_BOARD_SIZE, _BOARD_SIZE), dtype=np.float32)
    for position, stack in record.state.board.items():
        x = 6 - position.x if transform.horizontal_reflection else position.x
        y = 6 - position.y if transform.vertical_reflection else position.y
        totals[y, x] += len(stack)
    occupied_y, occupied_x = np.nonzero(totals > 0)
    if len(occupied_x) == 0:
        raise ValueError(f"record {row} has an empty board")
    left_edge = int(occupied_x.max())
    anchor_index = int(occupied_y.max())
    frame_x = left_edge - 2
    frame_y = anchor_index - 2
    anchor_rank = anchor_index + 1
    left_margin = _BOARD_SIZE - 1 - left_edge
    top_margin = _BOARD_SIZE - 1 - anchor_index
    _validate_class(anchor_rank, ANCHOR_RANK_MIN, ANCHOR_RANK_MAX, "anchor_rank", row)
    _validate_class(left_margin, 0, 3, "left_margin", row)
    _validate_class(top_margin, 0, 3, "top_margin", row)
    if frame_x < 0 or frame_y < 0:
        raise ValueError(
            f"record {row} board-centered frame is outside the board: "
            f"x={frame_x}, y={frame_y}"
        )
    if int(occupied_x.min()) < frame_x or int(occupied_y.min()) < frame_y:
        raise ValueError(f"record {row} board does not fit b-center 3x3 frame")
    window = totals[frame_y : frame_y + 3, frame_x : frame_x + 3]
    if not np.all((window == 0) | (window == 1) | (window == 2)):
        raise ValueError(f"record {row} has a b-center cell count outside 0/1/2")
    board = window[::-1, ::-1][None, ...].astype(np.float32)
    column_empty_state = (
        (1 if np.all(board[0, :, 1] == 0) else 0)
        | (2 if np.all(board[0, :, 2] == 0) else 0)
    )
    row_empty_state = (
        (1 if np.all(board[0, 1, :] == 0) else 0)
        | (2 if np.all(board[0, 2, :] == 0) else 0)
    )

    viewer = record.perspective_player_index
    values: list[float] = []
    rank_deltas: list[int] = []
    values.extend(_one_hot(anchor_rank - ANCHOR_RANK_MIN, 4))
    values.extend(_one_hot(left_margin, MARGIN_CLASSES))
    values.extend(_one_hot(top_margin, MARGIN_CLASSES))
    values.extend(_one_hot(column_empty_state, EMPTY_STATE_CLASSES))
    values.extend(_one_hot(row_empty_state, EMPTY_STATE_CLASSES))

    hand = sorted(
        (
            transform.old_to_new_color[COLOR_ORDER.index(card.color)],
            _transformed_rank(card, transform),
        )
        for card in record.state.players[viewer].hand
    )
    for slot in range(HAND_SIZE):
        if slot < len(hand):
            color, rank = hand[slot]
            delta = rank - anchor_index
            rank_deltas.append(delta)
            values.extend([1.0, *_one_hot(color, _COLOR_COUNT), *_rank_delta_one_hot(delta, row)])
        else:
            values.extend([0.0] * (1 + _COLOR_COUNT + RANK_DELTA_CLASSES))

    state = record.state
    for offset in range(4):
        player = state.players[(viewer + offset) % 4]
        values.extend(
            [
                player.loss_score / 35,
                len(player.hand) / HAND_SIZE,
                len(player.negative_cards) / 56,
            ]
        )
    values.extend(_one_hot((state.current_player_index - viewer) % 4, 4))
    from yellowstone.types import Phase

    values.extend(_one_hot((Phase.PLAY, Phase.REFILL, Phase.GAME_OVER).index(state.phase), 3))
    values.extend([state.cards_played_this_turn / 2, state.settlement_count / 10])
    _extend_history_values(
        values,
        rank_deltas,
        record=record,
        row=row,
        viewer=viewer,
        transform=transform,
        anchor_index=anchor_index,
        frame_x=frame_x,
        frame_y=frame_y,
        history_variant=history_variant,
    )
    if len(values) != _context_size_for_history_variant(history_variant):
        raise AssertionError(
            f"unexpected b-center context size: {len(values)}"
        )
    return (
        board,
        np.asarray(values, dtype=np.float32),
        {
            "anchor_rank": anchor_rank,
            "rank_deltas": rank_deltas,
            "left_margin": left_margin,
            "top_margin": top_margin,
            "column_empty_state": column_empty_state,
            "row_empty_state": row_empty_state,
        },
    )


_HISTORY_FEATURES_REPLACED = 1 + 4 + 4 + RANK_DELTA_CLASSES + 2
_FRAME_DELTA_HISTORY_FEATURES = 1 + FRAME_DELTA_CLASSES * 2 + 2


def _history_variant(canonicalization: str) -> str:
    if canonicalization in (BOARD_CENTERED_V1, BOARD_CENTERED_V1_HISTORY_V1):
        return "v1"
    if canonicalization == BOARD_CENTERED_V1_HISTORY_NONE:
        return "none"
    if canonicalization == BOARD_CENTERED_V1_HISTORY_TURN_LOCAL:
        return "turn_local"
    if canonicalization == BOARD_CENTERED_V1_HISTORY_OWN_FRAME_DELTA_2CYCLE:
        return "own_frame_delta_2cycle"
    if canonicalization == BOARD_CENTERED_V1_CHAIN_HISTORY:
        return "chain_history"
    raise ValueError(f"unsupported board-centered canonicalization: {canonicalization}")


def _extend_history_values(
    values: list[float],
    rank_deltas: list[int],
    *,
    record: ValueRecord,
    row: int,
    viewer: int,
    transform: BoardCenteredTransformV1,
    anchor_index: int,
    frame_x: int,
    frame_y: int,
    history_variant: str,
) -> None:
    if history_variant == "none":
        values.extend([0.0] * (HISTORY_SIZE * _HISTORY_FEATURES_REPLACED))
        return
    if history_variant == "own_frame_delta_2cycle":
        history = record.board_center_frame_history[-HISTORY_SIZE:]
        missing = HISTORY_SIZE - len(history)
        values.extend([0.0] * (missing * _FRAME_DELTA_HISTORY_FEATURES))
        for raw_prior_x, raw_prior_y in history:
            prior_x = 6 - (raw_prior_x + 2) if transform.horizontal_reflection else raw_prior_x
            prior_y = 6 - (raw_prior_y + 2) if transform.vertical_reflection else raw_prior_y
            delta_x = frame_x - prior_x
            delta_y = frame_y - prior_y
            values.extend(
                [
                    1.0,
                    *_frame_delta_one_hot(delta_y, row),
                    *_frame_delta_one_hot(delta_x, row),
                    0.0,
                    0.0,
                ]
            )
        return
    if history_variant == "chain_history":
        _extend_chain_history_values(
            values,
            record=record,
            transform=transform,
            current_state=record.state,
        )
        return
    if history_variant not in ("v1", "turn_local"):
        raise ValueError(f"unsupported b-center history variant: {history_variant}")
    for placement in record.history[-HISTORY_SIZE:]:
        rank = _transformed_rank(placement.card, transform)
        delta = rank - anchor_index
        rank_deltas.append(delta)
        values.extend(
            [
                1.0,
                *_one_hot((placement.player_index - viewer) % 4, 4),
                *_one_hot(
                    transform.old_to_new_color[COLOR_ORDER.index(placement.card.color)],
                    _COLOR_COUNT,
                ),
                *_rank_delta_one_hot(delta, row),
                placement.score_delta / 3,
                placement.negative_card_delta / 9,
            ]
        )
    missing = HISTORY_SIZE - len(record.history[-HISTORY_SIZE:])
    values.extend([0.0] * (missing * _HISTORY_FEATURES_REPLACED))


def _context_size_for_history_variant(history_variant: str) -> int:
    return (
        BOARD_CENTERED_V1_CHAIN_CONTEXT_SIZE
        if history_variant == "chain_history"
        else BOARD_CENTERED_V1_CONTEXT_SIZE
    )


def _extend_chain_history_values(
    values: list[float],
    *,
    record: ValueRecord,
    transform: BoardCenteredTransformV1,
    current_state: GameState,
) -> None:
    states = tuple(record.board_center_chain_states)
    pairs = (
        (current_state, states[-1] if len(states) >= 1 else None),
        (states[-1] if len(states) >= 1 else None, states[-4] if len(states) >= 4 else None),
        (states[-4] if len(states) >= 4 else None, states[-8] if len(states) >= 8 else None),
        (states[-8] if len(states) >= 8 else None, states[-12] if len(states) >= 12 else None),
    )
    for after_state, before_state in pairs:
        if after_state is None or before_state is None:
            values.extend([0.0, 0.0])
            continue
        after_colors, after_top = _chain_state_signature(after_state, transform)
        before_colors, before_top = _chain_state_signature(before_state, transform)
        values.extend(
            [
                1.0 if after_colors != before_colors else 0.0,
                (after_top - before_top) / 6,
            ]
        )


def _chain_state_signature(
    state: GameState,
    transform: BoardCenteredTransformV1,
) -> tuple[tuple[tuple[int, int, int], ...], int]:
    colors: list[tuple[int, int, int]] = []
    top_rank = 0
    for position, stack in state.board.items():
        x = 6 - position.x if transform.horizontal_reflection else position.x
        y = 6 - position.y if transform.vertical_reflection else position.y
        top_rank = max(top_rank, y)
        for card in stack:
            colors.append(
                (
                    x,
                    y,
                    transform.old_to_new_color[COLOR_ORDER.index(card.color)],
                )
            )
    return tuple(sorted(colors)), top_rank


def _residual_transform_v1(record: ValueRecord) -> BoardCenteredTransformV1:
    vertical_candidates = [False, True]
    vertical_board_keys = {
        vertical: min(
            _occupancy_key(record.state, vertical, horizontal)
            for horizontal in (False, True)
        )
        for vertical in vertical_candidates
    }
    minimum = min(vertical_board_keys.values())
    vertical_candidates = [
        value for value in vertical_candidates if vertical_board_keys[value] == minimum
    ]
    if len(vertical_candidates) > 1:
        hand_keys = {
            vertical: _hand_rank_key(record, vertical)
            for vertical in vertical_candidates
        }
        minimum_hand = min(hand_keys.values())
        vertical_candidates = [
            value for value in vertical_candidates if hand_keys[value] == minimum_hand
        ]
    spatial_keys = {
        (vertical, horizontal): _occupancy_key(record.state, vertical, horizontal)
        for vertical in vertical_candidates
        for horizontal in (False, True)
    }
    minimum_spatial = min(spatial_keys.values())
    spatial = [
        pair for pair, key in spatial_keys.items() if key == minimum_spatial
    ]
    result: list[BoardCenteredTransformV1] = []
    for vertical, horizontal in spatial:
        signatures = [
            _color_signature(record, old_color, vertical, horizontal)
            for old_color in range(_COLOR_COUNT)
        ]
        ordered = sorted(range(_COLOR_COUNT), key=lambda index: signatures[index])
        mapping = [0] * _COLOR_COUNT
        for new_color, old_color in enumerate(ordered):
            mapping[old_color] = new_color
        result.append(
            BoardCenteredTransformV1(
                vertical_reflection=vertical,
                horizontal_reflection=horizontal,
                old_to_new_color=tuple(mapping),
            )
        )
    return min(
        result,
        key=lambda transform: (
            transform.vertical_reflection,
            transform.horizontal_reflection,
            transform.old_to_new_color,
        ),
    )


def _occupancy_key(
    state: GameState, vertical: bool, horizontal: bool
) -> tuple[int, ...]:
    cells = [[0] * _BOARD_SIZE for _ in range(_BOARD_SIZE)]
    for position, stack in state.board.items():
        x = 6 - position.x if horizontal else position.x
        y = 6 - position.y if vertical else position.y
        cells[y][x] += len(stack)
    return tuple(value for row in cells for value in row)


def _hand_rank_key(record: ValueRecord, vertical: bool) -> tuple[int, ...]:
    counts = [0] * _BOARD_SIZE
    for card in record.state.players[record.perspective_player_index].hand:
        rank = 6 - card.rank_index if vertical else card.rank_index
        counts[rank] += 1
    return tuple(counts)


def _color_signature(
    record: ValueRecord,
    old_color: int,
    vertical: bool,
    horizontal: bool,
) -> tuple[float, ...]:
    color = COLOR_ORDER[old_color]
    board = [0.0] * (_BOARD_SIZE * _BOARD_SIZE)
    for position, stack in record.state.board.items():
        x = 6 - position.x if horizontal else position.x
        y = 6 - position.y if vertical else position.y
        board[y * _BOARD_SIZE + x] += sum(card.color == color for card in stack)
    viewer = record.perspective_player_index
    hand = [0.0] * _BOARD_SIZE
    for card in record.state.players[viewer].hand:
        if card.color == color:
            hand[6 - card.rank_index if vertical else card.rank_index] += 1.0
    history: list[float] = []
    missing = HISTORY_SIZE - len(record.history[-HISTORY_SIZE:])
    history.extend([0.0] * (missing * _BOARD_SIZE))
    for placement in record.history[-HISTORY_SIZE:]:
        counts = [0.0] * _BOARD_SIZE
        if placement.card.color == color:
            counts[6 - placement.card.rank_index if vertical else placement.card.rank_index] += 1.0
        history.extend(counts)
    return tuple(board + hand + history)


def _transformed_rank(card: Card, transform: BoardCenteredTransformV1) -> int:
    return 6 - card.rank_index if transform.vertical_reflection else card.rank_index


def board_center_value_tensors(
    board: np.ndarray,
    context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return board-centered V1 tensors, raising on contract violations."""
    centered_board, centered_context, _ = board_center_value_tensors_with_stats(
        board, context
    )
    return centered_board, centered_context


def board_center_value_tensors_with_stats(
    board: np.ndarray,
    context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, BoardCenteredStats]:
    """Return board-centered V1 tensors and audit counters."""
    if board.ndim != 4 or board.shape[1:] != (
        RANK_BOARD_CHANNELS,
        _BOARD_SIZE,
        _BOARD_SIZE,
    ):
        raise ValueError(f"expected board shape [N,29,7,7], got {board.shape}")
    if context.ndim != 2 or context.shape[1] != VALUE_CONTEXT_SIZE:
        raise ValueError(f"expected context shape [N,81], got {context.shape}")
    if board.shape[0] != context.shape[0]:
        raise ValueError("board/context record counts differ")

    out_board = np.zeros(
        (
            board.shape[0],
            BOARD_CENTERED_BOARD_CHANNELS,
            BOARD_CENTERED_BOARD_SIZE,
            BOARD_CENTERED_BOARD_SIZE,
        ),
        dtype=np.float32,
    )
    out_context = np.zeros(
        (context.shape[0], BOARD_CENTERED_V1_CONTEXT_SIZE),
        dtype=np.float32,
    )
    anchors: list[int] = []
    deltas: list[int] = []
    right_margins = [0] * MARGIN_CLASSES
    top_margins = [0] * MARGIN_CLASSES
    column_states = [0] * EMPTY_STATE_CLASSES
    row_states = [0] * EMPTY_STATE_CLASSES

    totals = board[:, -1]
    color_rank = board[:, :_COLOR_RANK_CHANNELS].reshape(
        board.shape[0], _COLOR_COUNT, _BOARD_SIZE, _BOARD_SIZE, _BOARD_SIZE
    )
    for row in range(board.shape[0]):
        occupied_y, occupied_x = np.nonzero(totals[row] > 0)
        if len(occupied_x) == 0:
            raise ValueError(f"record {row} has an empty board")
        right_edge = int(occupied_x.min())
        anchor_index = int(occupied_y.max())
        frame_x = right_edge
        frame_y = anchor_index - 2
        anchor_rank = anchor_index + 1
        right_margin = right_edge
        top_margin = _BOARD_SIZE - 1 - anchor_index
        _validate_class(anchor_rank, ANCHOR_RANK_MIN, ANCHOR_RANK_MAX, "anchor_rank", row)
        _validate_class(right_margin, 0, 3, "right_margin", row)
        _validate_class(top_margin, 0, 3, "top_margin", row)
        if frame_x < 0 or frame_x + 2 >= _BOARD_SIZE or frame_y < 0:
            raise ValueError(
                f"record {row} board-centered frame is outside the board: "
                f"x={frame_x}, y={frame_y}"
            )
        if (
            int(occupied_x.max()) > frame_x + 2
            or int(occupied_y.min()) < frame_y
        ):
            raise ValueError(f"record {row} board does not fit b-center 3x3 frame")

        window = totals[row, frame_y : frame_y + 3, frame_x : frame_x + 3]
        if not np.all((window == 0) | (window == 1) | (window == 2)):
            raise ValueError(f"record {row} has a b-center cell count outside 0/1/2")
        output_window = window[::-1, :]
        out_board[row, 0] = output_window

        column_empty_state = (
            (1 if np.all(window[:, 1] == 0) else 0)
            | (2 if np.all(window[:, 2] == 0) else 0)
        )
        row_empty_state = (
            (1 if np.all(output_window[1, :] == 0) else 0)
            | (2 if np.all(output_window[2, :] == 0) else 0)
        )
        column_states[column_empty_state] += 1
        row_states[row_empty_state] += 1
        right_margins[right_margin] += 1
        top_margins[top_margin] += 1
        anchors.append(anchor_rank)

        values: list[float] = []
        values.extend(_one_hot(anchor_rank - ANCHOR_RANK_MIN, 4))
        values.extend(_one_hot(right_margin, MARGIN_CLASSES))
        values.extend(_one_hot(top_margin, MARGIN_CLASSES))
        values.extend(_one_hot(column_empty_state, EMPTY_STATE_CLASSES))
        values.extend(_one_hot(row_empty_state, EMPTY_STATE_CLASSES))

        source = context[row]
        offset = 0
        for _slot in range(HAND_SIZE):
            values.extend(float(value) for value in source[offset : offset + 5])
            offset += 5
            present = source[offset - 5] > 0.5
            rank = _rank_scalar_to_index(float(source[offset]), row, "hand")
            offset += 1
            if present:
                delta = rank - anchor_index
                deltas.append(delta)
                values.extend(_rank_delta_one_hot(delta, row))
            else:
                values.extend([0.0] * RANK_DELTA_CLASSES)
        values.extend(float(value) for value in source[offset : offset + 21])
        offset += 21
        for _slot in range(HISTORY_SIZE):
            values.extend(float(value) for value in source[offset : offset + 9])
            present = source[offset] > 0.5
            offset += 9
            rank = _rank_scalar_to_index(float(source[offset]), row, "history")
            offset += 1
            if present:
                delta = rank - anchor_index
                deltas.append(delta)
                values.extend(_rank_delta_one_hot(delta, row))
            else:
                values.extend([0.0] * RANK_DELTA_CLASSES)
            values.extend(float(value) for value in source[offset : offset + 2])
            offset += 2
        if offset != VALUE_CONTEXT_SIZE:
            raise AssertionError(f"unexpected consumed context size: {offset}")
        if len(values) != BOARD_CENTERED_V1_CONTEXT_SIZE:
            raise AssertionError(f"unexpected b-center context size: {len(values)}")
        out_context[row] = np.asarray(values, dtype=np.float32)

        # Board card ranks are also audited against the same anchor contract.
        card_rank_counts = color_rank[row].sum(axis=(0, 2, 3))
        for rank, count in enumerate(card_rank_counts):
            if count:
                delta = rank - anchor_index
                _rank_delta_one_hot(delta, row)

    if not deltas:
        min_delta = max_delta = 0
    else:
        min_delta = min(deltas)
        max_delta = max(deltas)
    return out_board, out_context, BoardCenteredStats(
        records=board.shape[0],
        min_anchor_rank=min(anchors),
        max_anchor_rank=max(anchors),
        min_rank_delta=min_delta,
        max_rank_delta=max_delta,
        left_margin_counts=tuple(right_margins),
        top_margin_counts=tuple(top_margins),
        column_empty_state_counts=tuple(column_states),
        row_empty_state_counts=tuple(row_states),
    )


def _rank_scalar_to_index(value: float, row: int, field: str) -> int:
    rank = int(round(value * 6))
    if abs(value - rank / 6) > 1e-4:
        raise ValueError(f"record {row} {field} rank is not a V1 scalar: {value}")
    if not 0 <= rank <= 6:
        raise ValueError(f"record {row} {field} rank is outside 0..6: {rank}")
    return rank


def _rank_delta_one_hot(delta: int, row: int) -> list[float]:
    _validate_class(delta, RANK_DELTA_MIN, RANK_DELTA_MAX, "rank_delta", row)
    return _one_hot(delta - RANK_DELTA_MIN, RANK_DELTA_CLASSES)


def _frame_delta_one_hot(delta: int, row: int) -> list[float]:
    _validate_class(delta, FRAME_DELTA_MIN, FRAME_DELTA_MAX, "frame_delta", row)
    return _one_hot(delta - FRAME_DELTA_MIN, FRAME_DELTA_CLASSES)


def _validate_class(value: int, minimum: int, maximum: int, name: str, row: int) -> None:
    if not minimum <= value <= maximum:
        raise ValueError(
            f"record {row} {name} outside {minimum}..{maximum}: {value}"
        )


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if index == value else 0.0 for value in range(size)]
