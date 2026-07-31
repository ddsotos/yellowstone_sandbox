import numpy as np

from yellowstone.cnn import build_win_value_net_v2_lite
from yellowstone.symmetry import transform_state
from yellowstone.types import (
    Card,
    Color,
    GameState,
    Phase,
    PlayerState,
    Position,
)
from yellowstone.value_v2 import PendingRefillSource
from yellowstone.value_v2_lite import (
    BOARD_CHANNELS_V2_LITE,
    VALUE_CONTEXT_SIZE_V2_LITE,
    ValueRecordV2Lite,
    canonical_tensors_v2_lite,
)
from yellowstone.value_v2_lite_action import (
    VALUE_CONTEXT_SIZE_V2_LITE_ACTION,
    action_cards_from_transition,
    build_win_value_net_v2_lite_action,
    canonical_tensors_v2_lite_action,
)


def _record(*, horizontal: bool = False) -> ValueRecordV2Lite:
    before = GameState(
        players=(
            PlayerState(
                hand=(Card(Color.BLUE, 2), Card(Color.RED, 4)),
                negative_cards=(Card(Color.YELLOW, 1),),
                loss_score=5,
            ),
            PlayerState(loss_score=6),
            PlayerState(loss_score=7),
            PlayerState(loss_score=8),
        ),
        board={Position(1, 2): (Card(Color.GREEN, 2),)},
        deck=(Card(Color.YELLOW, 0),) * 8,
        current_player_index=0,
        phase=Phase.PLAY,
    )
    after = GameState(
        players=(
            PlayerState(
                hand=(Card(Color.RED, 4),),
                negative_cards=(Card(Color.YELLOW, 1),),
                loss_score=5,
            ),
            PlayerState(loss_score=6),
            PlayerState(loss_score=7),
            PlayerState(loss_score=8),
        ),
        board={
            Position(1, 2): (Card(Color.GREEN, 2),),
            Position(3, 2): (Card(Color.BLUE, 2),),
        },
        deck=before.deck,
        current_player_index=1,
        phase=Phase.PLAY,
    )
    if horizontal:
        color_map = {color: color for color in Color}
        before = transform_state(
            before,
            color_map=color_map,
            horizontal_reflection=True,
            vertical_reflection=False,
        )
        after = transform_state(
            after,
            color_map=color_map,
            horizontal_reflection=True,
            vertical_reflection=False,
        )
    return ValueRecordV2Lite(
        game_id=1,
        perspective_player_index=0,
        state_before_turn=before,
        state=after,
        history_before_turn=(),
        pending_refill_source=PendingRefillSource.NO_PENDING,
        target=1.0,
    )


def test_v2_lite_context_and_signed_board_delta() -> None:
    board, context, _ = canonical_tensors_v2_lite(_record())
    assert board.shape == (BOARD_CHANNELS_V2_LITE, 7, 7)
    assert context.shape == (VALUE_CONTEXT_SIZE_V2_LITE,)
    assert VALUE_CONTEXT_SIZE_V2_LITE == 138
    after = board[:29]
    delta = board[29:]
    before = after - delta
    assert int(after[-1].sum()) == 2
    assert int(before[-1].sum()) == 1
    assert int(delta[-1].sum()) == 1


def test_v2_lite_horizontal_symmetry_collapses() -> None:
    first_board, first_context, _ = canonical_tensors_v2_lite(_record())
    mirrored_board, mirrored_context, _ = canonical_tensors_v2_lite(
        _record(horizontal=True)
    )
    assert np.array_equal(first_board, mirrored_board)
    assert np.array_equal(first_context, mirrored_context)


def test_v2_lite_network_accepts_compact_inputs() -> None:
    import torch

    model = build_win_value_net_v2_lite()
    result = model(
        torch.zeros((2, BOARD_CHANNELS_V2_LITE, 7, 7)),
        torch.zeros((2, VALUE_CONTEXT_SIZE_V2_LITE)),
    )
    assert result.shape == (2,)


def test_v2_lite_action_adds_unordered_played_cards() -> None:
    import torch

    record = _record()
    cards = action_cards_from_transition(record)
    assert cards == (Card(Color.BLUE, 2),)
    board, context, _ = canonical_tensors_v2_lite_action(record)
    assert board.shape == (BOARD_CHANNELS_V2_LITE, 7, 7)
    assert context.shape == (VALUE_CONTEXT_SIZE_V2_LITE_ACTION,)
    assert VALUE_CONTEXT_SIZE_V2_LITE_ACTION == 150
    assert context[-12] == 1.0
    assert np.count_nonzero(context[-6:]) == 0
    result = build_win_value_net_v2_lite_action()(
        torch.from_numpy(board[None]), torch.from_numpy(context[None])
    )
    assert result.shape == (1,)
