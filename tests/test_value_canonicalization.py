from itertools import permutations

import numpy as np

from yellowstone.canonicalize_value_data import canonicalize_archives
from yellowstone.symmetry import transform_state
from yellowstone.types import Card, Color, GameState, Phase, PlayerState, Position
from yellowstone.value_canonicalization import canonicalize_value_tensors
from yellowstone.value_learning import (
    COLOR_ORDER,
    RecentPlacement,
    ValueRecord,
    board_tensor_for_player,
    context_tensor_for_player,
)


def _record() -> ValueRecord:
    state = GameState(
        players=(
            PlayerState(
                hand=(
                    Card(Color.BLUE, 0),
                    Card(Color.BLUE, 1),
                    Card(Color.RED, 2),
                    Card(Color.GREEN, 3),
                )
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(1, 0): (Card(Color.RED, 0),),
            Position(1, 1): (Card(Color.RED, 1),),
            Position(4, 2): (Card(Color.GREEN, 2),),
        },
        current_player_index=1,
        phase=Phase.REFILL,
    )
    return ValueRecord(
        game_id=3,
        perspective_player_index=0,
        state=state,
        history=(
            RecentPlacement(3, Card(Color.RED, 1), 0, 0),
            RecentPlacement(0, Card(Color.GREEN, 2), 1, 0),
        ),
        target=1.0,
    )


def _encoded(record: ValueRecord):
    return (
        board_tensor_for_player(record)[None, ...],
        context_tensor_for_player(record)[None, ...],
    )


def _transform_record(
    record: ValueRecord,
    *,
    color_map,
    horizontal_reflection: bool,
    vertical_reflection: bool,
) -> ValueRecord:
    def card(card_value: Card) -> Card:
        return Card(
            color_map[card_value.color],
            6 - card_value.rank_index
            if vertical_reflection
            else card_value.rank_index,
        )

    return ValueRecord(
        game_id=record.game_id,
        perspective_player_index=record.perspective_player_index,
        state=transform_state(
            record.state,
            color_map=color_map,
            horizontal_reflection=horizontal_reflection,
            vertical_reflection=vertical_reflection,
        ),
        history=tuple(
            RecentPlacement(
                placement.player_index,
                card(placement.card),
                placement.score_delta,
                placement.negative_card_delta,
            )
            for placement in record.history
        ),
        target=record.target,
    )


def test_canonicalization_is_idempotent() -> None:
    board, context = canonicalize_value_tensors(*_encoded(_record()))
    second_board, second_context = canonicalize_value_tensors(board, context)
    np.testing.assert_array_equal(second_board, board)
    np.testing.assert_array_equal(second_context, context)


def test_non_tied_symmetry_orbit_collapses_to_one_input() -> None:
    record = _record()
    results = set()
    for permuted in permutations(COLOR_ORDER):
        color_map = dict(zip(COLOR_ORDER, permuted, strict=True))
        for horizontal in (False, True):
            for vertical in (False, True):
                transformed = _transform_record(
                    record,
                    color_map=color_map,
                    horizontal_reflection=horizontal,
                    vertical_reflection=vertical,
                )
                board, context = canonicalize_value_tensors(*_encoded(transformed))
                results.add((board.tobytes(), context.tobytes()))
    assert len(results) == 1


def test_canonical_direction_and_board_color_order() -> None:
    board, _ = canonicalize_value_tensors(*_encoded(_record()))
    occupancy = board[0, -1]
    assert occupancy[:, :3].sum() >= occupancy[:, 4:].sum()
    assert occupancy[:3, :].sum() >= occupancy[4:, :].sum()

    color_rank = board[0, :28].reshape(4, 7, 7, 7)
    occupied_x_by_color = [
        np.flatnonzero(color_rank[color].sum(axis=(0, 1)))
        for color in range(4)
    ]
    assert occupied_x_by_color[1].size
    assert occupied_x_by_color[0].size
    assert occupied_x_by_color[1][0] > occupied_x_by_color[0][0]


def test_archive_conversion_preserves_labels_and_is_resumable(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    board, context = _encoded(_record())
    np.savez_compressed(
        source / "part_000100.npz",
        board=board,
        context=context,
        target=np.asarray([1.0], dtype=np.float32),
        game_id=np.asarray([100], dtype=np.int64),
    )

    first = canonicalize_archives(source, output, start_part=100, end_part=100)
    second = canonicalize_archives(source, output, start_part=100, end_part=100)

    assert first["converted_files"] == 1
    assert second["skipped_files"] == 1
    with np.load(output / "part_000100.npz") as converted:
        np.testing.assert_array_equal(converted["target"], [1.0])
        np.testing.assert_array_equal(converted["game_id"], [100])
