"""Original V1 tensors with an explicit pending refill-card count."""

from __future__ import annotations

from yellowstone.types import RefillAction, RefillSource
from yellowstone.value_canonicalization import (
    CANONICALIZATION_NAME,
    canonicalize_value_tensors_with_stats,
)
from yellowstone.value_learning import VALUE_CONTEXT_SIZE, ValueRecord


CANONICALIZATION_REFILL_COUNT = f"{CANONICALIZATION_NAME}_refill_count_v1"
CANONICALIZATION_REFILL_COUNT_SCALAR = (
    f"{CANONICALIZATION_NAME}_refill_count_scalar_v1"
)
REFILL_COUNT_CLASSES = 7
VALUE_CONTEXT_SIZE_REFILL_COUNT = VALUE_CONTEXT_SIZE + REFILL_COUNT_CLASSES
VALUE_CONTEXT_SIZE_REFILL_COUNT_SCALAR = VALUE_CONTEXT_SIZE + 1


def refill_count_metadata() -> dict[str, object]:
    return {
        "input_canonicalization": CANONICALIZATION_REFILL_COUNT,
        "base_input_canonicalization": CANONICALIZATION_NAME,
        "refill_count_range": [0, 6],
        "refill_count_classes": REFILL_COUNT_CLASSES,
        "refill_count_encoding": "one_hot",
        "refill_count_semantics": (
            "number of cards that will be received by the pending refill action"
        ),
    }


def refill_count_scalar_metadata() -> dict[str, object]:
    return {
        "input_canonicalization": CANONICALIZATION_REFILL_COUNT_SCALAR,
        "base_input_canonicalization": CANONICALIZATION_NAME,
        "refill_count_range": [0, 6],
        "refill_count_features": 1,
        "refill_count_encoding": "scalar_count_divided_by_6",
        "refill_count_semantics": (
            "number of cards that will be received by the pending refill action"
        ),
    }


def refill_count_for_action(record: ValueRecord, action: RefillAction | None) -> int:
    if action is None or action.source == RefillSource.NONE:
        return 0
    player = record.state.players[record.perspective_player_index]
    needed = max(0, 6 - len(player.hand))
    if action.source == RefillSource.DECK:
        return min(needed, len(record.state.deck))
    if action.source == RefillSource.NEGATIVE_CARDS:
        return min(needed, len(player.negative_cards))
    raise ValueError(f"unsupported refill source: {action.source}")


def append_refill_count_context(context, counts):
    import numpy as np

    count_array = np.asarray(tuple(counts), dtype=np.int64)
    if context.ndim != 2 or context.shape[1] != VALUE_CONTEXT_SIZE:
        raise ValueError(f"expected context shape [N,{VALUE_CONTEXT_SIZE}], got {context.shape}")
    if len(count_array) != context.shape[0]:
        raise ValueError("refill count length differs from context rows")
    if np.any(count_array < 0) or np.any(count_array >= REFILL_COUNT_CLASSES):
        raise ValueError("refill count outside 0..6")
    encoded = np.zeros((context.shape[0], REFILL_COUNT_CLASSES), dtype=context.dtype)
    encoded[np.arange(context.shape[0]), count_array] = 1.0
    return np.concatenate((context, encoded), axis=1)


def append_refill_count_scalar_context(context, counts):
    import numpy as np

    count_array = np.asarray(tuple(counts), dtype=np.float32)
    if context.ndim != 2 or context.shape[1] != VALUE_CONTEXT_SIZE:
        raise ValueError(f"expected context shape [N,{VALUE_CONTEXT_SIZE}], got {context.shape}")
    if len(count_array) != context.shape[0]:
        raise ValueError("refill count length differs from context rows")
    if np.any(count_array < 0) or np.any(count_array > 6):
        raise ValueError("refill count outside 0..6")
    return np.concatenate((context, (count_array / 6.0)[:, None].astype(context.dtype)), axis=1)


def canonicalize_refill_count_tensors(board, context, counts):
    board, context, stats = canonicalize_value_tensors_with_stats(board, context)
    return board, append_refill_count_context(context, counts), stats


def canonicalize_refill_count_scalar_tensors(board, context, counts):
    board, context, stats = canonicalize_value_tensors_with_stats(board, context)
    return board, append_refill_count_scalar_context(context, counts), stats
