import pytest

from yellowstone.value_learning import (
    RANK_BOARD_CHANNELS,
    VALUE_CONTEXT_SIZE,
    board_tensor_for_player,
    collect_heuristic_games,
    context_tensor_for_player,
    split_game_ids,
)
from yellowstone.value_policy import enumerate_turn_end_candidates
from yellowstone.game import create_initial_state


def test_heuristic_collector_labels_every_turn_end_from_each_player() -> None:
    records = collect_heuristic_games(game_count=10, seed=7)

    assert records
    assert {record.game_id for record in records} == set(range(10))
    assert all(record.target in (0.0, 0.25, 1 / 3, 0.5, 1.0) for record in records)
    assert {record.perspective_player_index for record in records} == {0, 1, 2, 3}


def test_player_perspective_tensors_have_expected_shapes() -> None:
    np = pytest.importorskip("numpy")
    record = collect_heuristic_games(game_count=10, seed=3)[0]

    board = board_tensor_for_player(record)
    context = context_tensor_for_player(record)

    assert board.shape == (RANK_BOARD_CHANNELS, 7, 7)
    assert board.dtype == np.float32
    assert context.shape == (VALUE_CONTEXT_SIZE,)
    assert context.dtype == np.float32


def test_game_split_is_disjoint_and_complete() -> None:
    train, validation, test = split_game_ids(100, seed=4)

    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(range(100))
    assert (len(train), len(validation), len(test)) == (80, 10, 10)


def test_turn_candidates_cover_one_and_two_card_turn_end_states() -> None:
    state = create_initial_state(4, seed=11)

    candidates = enumerate_turn_end_candidates(state)

    assert candidates
    assert any(len(candidate.actions) == 2 for candidate in candidates)
    assert any(candidate.record.state.phase.value == "refill" for candidate in candidates)
    assert all(candidate.record.perspective_player_index == 0 for candidate in candidates)
