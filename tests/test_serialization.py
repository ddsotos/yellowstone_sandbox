import json

from yellowstone.game import create_initial_state, legal_actions
from yellowstone.serialization import (
    action_from_dict,
    action_to_dict,
    actions_from_dicts,
    actions_to_dicts,
    game_state_from_dict,
    game_state_to_dict,
)
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


def test_game_state_round_trips_through_json_dict() -> None:
    # GameStateをJSON互換dictに変換して元に戻せることを確認する。
    state = GameState(
        players=(
            PlayerState(
                hand=(Card(Color.RED, 0), Card(Color.BLUE, 6)),
                negative_cards=(Card(Color.GREEN, 2),),
                loss_score=7,
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(1, 0): (Card(Color.RED, 0), Card(Color.YELLOW, 0)),
            Position(3, 6): (Card(Color.BLUE, 6),),
        },
        deck=(Card(Color.GREEN, 1),),
        current_player_index=2,
        phase=Phase.REFILL,
        cards_played_this_turn=2,
        winners=(1,),
        settlement_count=3,
        last_turn_play_counts=(1, 2, 1, 0),
    )

    encoded = game_state_to_dict(state)
    decoded = game_state_from_dict(json.loads(json.dumps(encoded)))

    assert decoded == state


def test_actions_round_trip_through_json_dict() -> None:
    # 各ActionをJSON互換dictに変換して元に戻せることを確認する。
    actions = (
        PlaceCardAction(
            hand_index=1,
            position=Position(2, 3),
            frame=Frame(1, 2),
        ),
        EndTurnAction(),
        RefillAction(RefillSource.NEGATIVE_CARDS),
    )

    encoded = actions_to_dicts(actions)
    decoded = actions_from_dicts(json.loads(json.dumps(encoded)))

    assert decoded == actions


def test_legal_actions_are_json_serializable() -> None:
    # 合法手一覧をJSON化して復元できることを確認する。
    state = create_initial_state(4, seed=1)
    actions = legal_actions(state)

    encoded = actions_to_dicts(actions)
    decoded = actions_from_dicts(json.loads(json.dumps(encoded)))

    assert decoded == actions


def test_action_from_dict_rejects_unknown_type() -> None:
    # 未知のAction種別は明示的に拒否することを確認する。
    try:
        action_from_dict({"type": "unknown"})
    except ValueError as error:
        assert "unsupported action type" in str(error)
    else:
        raise AssertionError("ValueError was not raised")


def test_action_to_dict_rejects_unknown_action_object() -> None:
    # 未対応のActionオブジェクトは明示的に拒否することを確認する。
    try:
        action_to_dict(object())  # type: ignore[arg-type]
    except TypeError as error:
        assert "unsupported action" in str(error)
    else:
        raise AssertionError("TypeError was not raised")
