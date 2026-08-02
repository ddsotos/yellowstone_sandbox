from itertools import permutations
from random import Random

import numpy as np
import pytest

from yellowstone.bots import HeuristicBot
from yellowstone.cnn import build_win_value_net, win_value_architecture_from_checkpoint
from yellowstone.convert_replay_v2_to_v1_board_centered import convert_replay_shards
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.replay_v2 import RULES_VERSION_V2, ReplayGameV2, write_replay_shard
from yellowstone.symmetry import transform_state
from yellowstone.train_value import train_from_archive
from yellowstone.types import Card, Color, GameState, Phase, PlayerState, Position
from yellowstone.value_board_centered import (
    BOARD_CENTERED_V1,
    BOARD_CENTERED_V1_CANONICALIZATIONS,
    BOARD_CENTERED_V1_CHAIN_CONTEXT_SIZE,
    BOARD_CENTERED_V1_CHAIN_HISTORY,
    BOARD_CENTERED_V1_CONTEXT_SIZE,
    BOARD_CENTERED_V1_HISTORY_NONE,
    BOARD_CENTERED_V1_HISTORY_OWN_FRAME_DELTA_2CYCLE,
    BOARD_CENTERED_V1_HISTORY_TURN_LOCAL,
    board_center_record_for_player,
    board_center_records_with_stats,
    board_center_value_tensors,
    board_centered_metadata,
)
from yellowstone.value_canonicalization import canonicalize_value_tensors
from yellowstone.value_learning import (
    COLOR_ORDER,
    RecentPlacement,
    ValueRecord,
    board_tensor_for_player,
    context_tensor_for_player,
)


def _context_with_ranks(*, hand_rank: int = 6, history_rank: int = 4) -> np.ndarray:
    values: list[float] = []
    values.extend([1.0, 1.0, 0.0, 0.0, 0.0, hand_rank / 6])
    values.extend([0.0] * (5 * 6))
    values.extend([0.0] * 21)
    values.extend([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, history_rank / 6, 0.0, 0.0])
    values.extend([0.0] * 12)
    assert len(values) == 81
    return np.asarray([values], dtype=np.float32)


def _board_from_cells(cells: dict[tuple[int, int], int]) -> np.ndarray:
    board = np.zeros((1, 29, 7, 7), dtype=np.float32)
    for (x, y), count in cells.items():
        board[0, 0 * 7 + y, y, x] = count
        board[0, -1, y, x] = count
    return board


def _bcenter_context(board: np.ndarray, context: np.ndarray) -> np.ndarray:
    _, centered_context = board_center_value_tensors(board, context)
    return centered_context[0]


def test_board_centered_v1_shape_anchor_delta_and_margins() -> None:
    board = _board_from_cells({(1, 3): 1, (2, 5): 2})
    centered_board, centered_context = board_center_value_tensors(
        board, _context_with_ranks(hand_rank=6, history_rank=4)
    )

    assert centered_board.shape == (1, 1, 3, 3)
    assert centered_context.shape == (1, BOARD_CENTERED_V1_CONTEXT_SIZE)
    np.testing.assert_array_equal(
        centered_board[0, 0],
        np.asarray([[0, 2, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(centered_context[0, 0:4], [0, 0, 1, 0])
    np.testing.assert_array_equal(centered_context[0, 4:8], [0, 1, 0, 0])
    np.testing.assert_array_equal(centered_context[0, 8:12], [0, 1, 0, 0])
    first_hand_delta = centered_context[0, 25:35]
    np.testing.assert_array_equal(first_hand_delta, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])


@pytest.mark.parametrize(
    ("cells", "expected_column_state", "expected_row_state"),
    [
        ({(0, 3): 1, (1, 4): 1, (2, 5): 1}, 0, 0),
        ({(0, 3): 1, (2, 5): 1}, 1, 1),
        ({(0, 3): 1, (1, 5): 1}, 2, 1),
        ({(0, 3): 1}, 3, 3),
        ({(0, 4): 1, (1, 5): 1}, 2, 2),
        ({(0, 5): 1}, 3, 3),
    ],
)
def test_board_centered_empty_state_classes(
    cells: dict[tuple[int, int], int],
    expected_column_state: int,
    expected_row_state: int,
) -> None:
    context = _bcenter_context(_board_from_cells(cells), _context_with_ranks())
    column_state = int(np.argmax(context[12:16]))
    row_state = int(np.argmax(context[16:20]))

    assert column_state == expected_column_state
    assert row_state == expected_row_state


def test_board_centered_contract_errors_are_not_clipped() -> None:
    with pytest.raises(ValueError, match="anchor_rank outside 4..7"):
        board_center_value_tensors(
            _board_from_cells({(0, 2): 1}),
            _context_with_ranks(),
        )
    malformed_context = _context_with_ranks()
    malformed_context[0, 5] = 8 / 6
    with pytest.raises(ValueError, match="hand rank is outside 0..6"):
        board_center_value_tensors(_board_from_cells({(0, 3): 1}), malformed_context)
    with pytest.raises(ValueError, match="right_margin outside 0..3"):
        board_center_value_tensors(
            _board_from_cells({(4, 5): 1}),
            _context_with_ranks(),
        )
    with pytest.raises(ValueError, match="cell count outside 0/1/2"):
        board_center_value_tensors(
            _board_from_cells({(0, 5): 3}),
            _context_with_ranks(),
        )


def _record() -> ValueRecord:
    state = GameState(
        players=(
            PlayerState(),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(0, 1): (Card(Color.RED, 1),),
            Position(0, 2): (Card(Color.GREEN, 2),),
            Position(1, 3): (Card(Color.BLUE, 3),),
        },
        current_player_index=1,
        phase=Phase.REFILL,
    )
    return ValueRecord(
        game_id=0,
        perspective_player_index=0,
        state=state,
        history=(),
        target=1.0,
    )


def _transform_record(record: ValueRecord, color_map, horizontal, vertical) -> ValueRecord:
    def transform_card(card: Card) -> Card:
        return Card(
            color_map[card.color],
            6 - card.rank_index if vertical else card.rank_index,
        )

    return ValueRecord(
        game_id=record.game_id,
        perspective_player_index=record.perspective_player_index,
        state=transform_state(
            record.state,
            color_map=color_map,
            horizontal_reflection=horizontal,
            vertical_reflection=vertical,
        ),
        history=tuple(
            RecentPlacement(
                placement.player_index,
                transform_card(placement.card),
                placement.score_delta,
                placement.negative_card_delta,
            )
            for placement in record.history
        ),
        target=record.target,
    )


def test_board_centered_record_encoding_uses_v2_aligned_orientation() -> None:
    board, context = board_center_record_for_player(_record())

    assert board.shape == (1, 3, 3)
    assert context.shape == (BOARD_CENTERED_V1_CONTEXT_SIZE,)
    np.testing.assert_array_equal(
        board[0],
        np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(context[:4], [0, 0, 1, 0])
    np.testing.assert_array_equal(context[4:8], [1, 0, 0, 0])


def test_board_centered_record_encoding_collapses_v2_symmetries() -> None:
    record = _record()
    expected_board, expected_context = board_center_record_for_player(record)
    for permuted in permutations(COLOR_ORDER):
        color_map = dict(zip(COLOR_ORDER, permuted, strict=True))
        for horizontal in (False, True):
            for vertical in (False, True):
                transformed = _transform_record(
                    record, color_map, horizontal, vertical
                )
                board, context = board_center_record_for_player(transformed)
                assert np.array_equal(board, expected_board)
                assert np.array_equal(context, expected_context)


def _encoded(record: ValueRecord) -> tuple[np.ndarray, np.ndarray]:
    return (
        board_tensor_for_player(record)[None, ...],
        context_tensor_for_player(record)[None, ...],
    )


def test_board_centered_preserves_fast_canonicalization_invariance() -> None:
    record = _record()
    color_map = {
        Color.RED: Color.YELLOW,
        Color.BLUE: Color.GREEN,
        Color.GREEN: Color.RED,
        Color.YELLOW: Color.BLUE,
    }
    transformed = ValueRecord(
        game_id=record.game_id,
        perspective_player_index=record.perspective_player_index,
        state=transform_state(
            record.state,
            color_map=color_map,
            horizontal_reflection=True,
            vertical_reflection=True,
        ),
        history=tuple(
            RecentPlacement(
                placement.player_index,
                Card(color_map[placement.card.color], 6 - placement.card.rank_index),
                placement.score_delta,
                placement.negative_card_delta,
            )
            for placement in record.history
        ),
        target=record.target,
    )

    board, context = board_center_value_tensors(
        *canonicalize_value_tensors(*_encoded(record))
    )
    transformed_board, transformed_context = board_center_value_tensors(
        *canonicalize_value_tensors(*_encoded(transformed))
    )

    np.testing.assert_array_equal(transformed_board, board)
    np.testing.assert_array_equal(transformed_context, context)


def test_board_centered_metadata_and_cnn_shape() -> None:
    torch = pytest.importorskip("torch")
    metadata = board_centered_metadata(BOARD_CENTERED_V1)
    model = build_win_value_net(
        context_size=BOARD_CENTERED_V1_CONTEXT_SIZE,
        board_channels=1,
        board_size=3,
    )

    output = model(
        torch.zeros((2, 1, 3, 3)),
        torch.zeros((2, BOARD_CENTERED_V1_CONTEXT_SIZE)),
    )

    assert metadata["base_input_canonicalization"] == "strict_residual_v2_aligned_v1"
    assert metadata["spatial_canonicalization"] == "v2_residual_lexicographic_board"
    assert metadata["rank_delta_range"] == [-6, 3]
    assert output.shape == (2,)
    assert win_value_architecture_from_checkpoint(
        {"board_channels": 1, "board_size": 3}
    )["model_architecture"].endswith("board1x3x3")


def test_board_centered_history_variants_keep_shape() -> None:
    for canonicalization in BOARD_CENTERED_V1_CANONICALIZATIONS:
        metadata = board_centered_metadata(canonicalization)
        assert metadata["input_canonicalization"] == canonicalization
        assert metadata["board_center_history_slots"] == (
            4 if canonicalization == BOARD_CENTERED_V1_CHAIN_HISTORY else 2
        )


def test_board_centered_history_none_zeros_history_slots() -> None:
    board, context, _ = board_center_records_with_stats(
        (_record(),),
        canonicalization=BOARD_CENTERED_V1_HISTORY_NONE,
    )

    assert board.shape == (1, 1, 3, 3)
    assert context.shape == (1, BOARD_CENTERED_V1_CONTEXT_SIZE)
    np.testing.assert_array_equal(context[0, -42:], np.zeros(42, dtype=np.float32))


def test_board_centered_own_frame_delta_history_uses_prior_frame() -> None:
    record = _record()
    record = ValueRecord(
        game_id=record.game_id,
        perspective_player_index=record.perspective_player_index,
        state=record.state,
        history=record.history,
        target=record.target,
        board_center_frame_history=((1, 1),),
    )

    _, context, _ = board_center_records_with_stats(
        (record,),
        canonicalization=BOARD_CENTERED_V1_HISTORY_OWN_FRAME_DELTA_2CYCLE,
    )

    history = context[0, -42:]
    assert history[:21].sum() == 0
    assert history[21] == 1.0
    assert history[22:31].sum() == 1.0
    assert history[31:40].sum() == 1.0


def test_board_centered_turn_local_accepts_one_history_slot() -> None:
    record = _record()
    record = ValueRecord(
        game_id=record.game_id,
        perspective_player_index=record.perspective_player_index,
        state=record.state,
        history=(
            RecentPlacement(
                player_index=0,
                card=Card(Color.RED, 3),
                score_delta=0,
                negative_card_delta=0,
            ),
        ),
        target=record.target,
    )

    _, context, _ = board_center_records_with_stats(
        (record,),
        canonicalization=BOARD_CENTERED_V1_HISTORY_TURN_LOCAL,
    )

    assert context.shape == (1, BOARD_CENTERED_V1_CONTEXT_SIZE)
    assert context[0, -42] == 1.0
    assert context[0, -21:].sum() == 0


def test_board_centered_chain_history_padding_and_deltas() -> None:
    record = _record()
    before = GameState(
        players=record.state.players,
        board={
            Position(0, 1): (Card(Color.RED, 1),),
            Position(1, 3): (Card(Color.BLUE, 3),),
        },
        current_player_index=record.state.current_player_index,
        phase=record.state.phase,
    )
    record = ValueRecord(
        game_id=record.game_id,
        perspective_player_index=record.perspective_player_index,
        state=record.state,
        history=(),
        target=record.target,
        board_center_chain_states=(before,),
    )

    _, context, _ = board_center_records_with_stats(
        (record,),
        canonicalization=BOARD_CENTERED_V1_CHAIN_HISTORY,
    )

    assert context.shape == (1, BOARD_CENTERED_V1_CHAIN_CONTEXT_SIZE)
    history = context[0, -8:]
    assert history[0] == 1.0
    assert history[1] == 0.0
    np.testing.assert_array_equal(history[2:], np.zeros(6, dtype=np.float32))
    assert board_centered_metadata(BOARD_CENTERED_V1_CHAIN_HISTORY)[
        "board_center_chain_history_rank_delta_range"
    ] == [-6, 6]


def test_board_centered_training_checkpoint_contract(tmp_path) -> None:
    pytest.importorskip("torch")
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(31)
    np.savez_compressed(
        data / "part_000000.npz",
        board=rng.normal(size=(10, 1, 3, 3)).astype(np.float32),
        context=rng.normal(size=(10, BOARD_CENTERED_V1_CONTEXT_SIZE)).astype(np.float32),
        target=np.asarray([0, 1] * 5, dtype=np.float32),
        game_id=np.arange(10, dtype=np.int64),
    )
    checkpoint_path = tmp_path / "bcenter.pt"

    train_from_archive(
        data,
        checkpoint_path,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        seed=7,
        input_canonicalization=BOARD_CENTERED_V1,
        checkpoint_metadata=board_centered_metadata(BOARD_CENTERED_V1),
    )

    import torch

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["input_canonicalization"] == BOARD_CENTERED_V1
    assert checkpoint["context_size"] == BOARD_CENTERED_V1_CONTEXT_SIZE
    assert checkpoint["board_channels"] == 1
    assert checkpoint["board_size"] == 3
    assert checkpoint["rank_delta_range"] == [-6, 3]


def test_board_centered_replay_conversion_smoke(tmp_path) -> None:
    state = create_initial_state(4, seed=11)
    initial_state = state
    rng = Random(1011)
    bot = HeuristicBot()
    actions = []
    while state.phase != Phase.GAME_OVER:
        action = bot.choose_action(state)
        assert action is not None
        actions.append(action)
        state = apply_known_legal_action(state, action, rng=rng)
    replay = ReplayGameV2(
        game_id=0,
        initial_seed=11,
        gameplay_seed=1011,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=(),
        winners=state.winners,
        teacher_checkpoint="test",
        teacher_sha256="0" * 64,
        teacher_generation=0,
        privileged_teacher_deck=False,
        rules_version=RULES_VERSION_V2,
    )
    source = tmp_path / "replays"
    output = tmp_path / "bcenter"
    source.mkdir()
    write_replay_shard((replay,), source / "part_000000.jsonl.gz")

    manifest = convert_replay_shards(source, output, expected_games=1)

    assert manifest["canonicalization"] == BOARD_CENTERED_V1
    with np.load(output / "part_000000.npz") as archive:
        assert archive["board"].shape[1:] == (1, 3, 3)
        assert archive["context"].shape[1] == BOARD_CENTERED_V1_CONTEXT_SIZE
        assert set(archive["game_id"]) == {0}
