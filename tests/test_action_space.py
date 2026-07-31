import pytest

from yellowstone.action_space import (
    ACTION_SPACE_SIZE,
    END_TURN_ACTION_INDEX,
    REFILL_ACTION_START_INDEX,
    action_from_index,
    action_space_metadata,
    action_to_index,
    legal_action_indices,
    legal_action_mask,
)
from yellowstone.game import create_initial_state, legal_actions
from yellowstone.types import (
    Card,
    Color,
    EndTurnAction,
    Frame,
    GameState,
    Phase,
    PlaceCardAction,
    PlayerState,
    Position,
    RefillAction,
    RefillSource,
)


def test_action_space_metadata_has_fixed_size() -> None:
    # 行動空間の長さと主要indexが固定されることを確認する。
    metadata = action_space_metadata()

    assert ACTION_SPACE_SIZE == 1054
    assert metadata["action_space_size"] == ACTION_SPACE_SIZE
    assert END_TURN_ACTION_INDEX == 1050
    assert REFILL_ACTION_START_INDEX == 1051


def test_place_action_round_trips_through_index() -> None:
    # 配置Actionをindex化し、現在手札のrankからyを復元できることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.GREEN, 4),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    action = PlaceCardAction(
        hand_index=0,
        position=Position(3, 4),
        frame=Frame(2, 1),
    )

    index = action_to_index(action)

    assert action_from_index(index, state) == action


def test_non_place_actions_round_trip_through_index() -> None:
    # ターン終了と補充Actionをindexから復元できることを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    assert action_to_index(EndTurnAction()) == END_TURN_ACTION_INDEX
    assert action_from_index(END_TURN_ACTION_INDEX, state) == EndTurnAction()
    assert (
        action_from_index(REFILL_ACTION_START_INDEX, state)
        == RefillAction(RefillSource.DECK)
    )
    assert (
        action_from_index(REFILL_ACTION_START_INDEX + 1, state)
        == RefillAction(RefillSource.NEGATIVE_CARDS)
    )
    assert (
        action_from_index(REFILL_ACTION_START_INDEX + 2, state)
        == RefillAction(RefillSource.NONE)
    )


def test_legal_action_mask_matches_legal_actions() -> None:
    # legal_actionsとlegal_action_maskのTrue位置が一致することを確認する。
    state = create_initial_state(4, seed=1)
    expected_indexes = {action_to_index(action) for action in legal_actions(state)}

    mask = legal_action_mask(state)

    assert len(mask) == ACTION_SPACE_SIZE
    assert {index for index, value in enumerate(mask) if value} == expected_indexes
    assert set(legal_action_indices(state)) == expected_indexes


def test_refill_phase_mask_contains_only_refill_actions() -> None:
    # 補充フェーズでは補充Actionだけが合法indexになることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    assert legal_action_indices(state) == (
        REFILL_ACTION_START_INDEX,
        REFILL_ACTION_START_INDEX + 2,
    )


def test_action_from_index_rejects_unavailable_hand_slot() -> None:
    # 存在しない手札スロットの配置indexは復元時に拒否することを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    with pytest.raises(ValueError, match="hand index is not available"):
        action_from_index(0, state)


def test_action_from_index_rejects_out_of_range_index() -> None:
    # 行動空間外のindexを拒否することを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    with pytest.raises(ValueError, match="action index out of range"):
        action_from_index(ACTION_SPACE_SIZE, state)
