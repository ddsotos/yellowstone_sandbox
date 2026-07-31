from random import Random

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_action, create_initial_state, legal_actions
from yellowstone.symmetry import (
    rotate_90,
    transform_action,
    transform_state,
)
from yellowstone.types import Card, Color, Frame, GameState, PlaceCardAction, PlayerState, Position


COLOR_SWAP = {
    Color.RED: Color.BLUE,
    Color.BLUE: Color.RED,
    Color.GREEN: Color.YELLOW,
    Color.YELLOW: Color.GREEN,
}

ORDER_SHIFT = {
    Color.RED: Color.GREEN,
    Color.BLUE: Color.YELLOW,
    Color.GREEN: Color.BLUE,
    Color.YELLOW: Color.RED,
}


def test_color_and_horizontal_symmetry_preserve_legal_actions_and_transitions() -> None:
    # 色置換と水平反射後も合法手と状態遷移が対応することを確認する。
    for seed in range(12):
        state = create_initial_state(4, seed=seed)
        transformed_state = transform_state(
            state,
            color_map=COLOR_SWAP,
            horizontal_reflection=True,
            vertical_reflection=True,
        )
        transformed_legal_actions = set(legal_actions(transformed_state))
        for action in legal_actions(state):
            transformed_action = transform_action(
                state,
                action,
                color_map=COLOR_SWAP,
                horizontal_reflection=True,
                vertical_reflection=True,
            )
            assert transformed_action in transformed_legal_actions
            assert apply_action(transformed_state, transformed_action) == transform_state(
                apply_action(state, action),
                color_map=COLOR_SWAP,
                horizontal_reflection=True,
                vertical_reflection=True,
            )


def test_action_transform_resolves_sorted_hand_index_by_card() -> None:
    # 色置換で手札順が変わるため元のhand indexをコピーしないことを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.BLUE, 2), Card(Color.GREEN, 2))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )
    action = PlaceCardAction(0, Position(0, 2), Frame(0, 0))
    transformed_state = transform_state(state, color_map=ORDER_SHIFT)
    transformed_action = transform_action(state, action, color_map=ORDER_SHIFT)

    assert isinstance(transformed_action, PlaceCardAction)
    original_card = state.players[0].hand[action.hand_index]
    assert transformed_state.players[0].hand[transformed_action.hand_index] == Card(
        color=ORDER_SHIFT[original_card.color],
        rank_index=original_card.rank_index,
    )
    assert transformed_action.hand_index == 1
    assert transformed_action in legal_actions(transformed_state)


def test_vertical_reflection_is_an_involution() -> None:
    state = create_initial_state(4, seed=9)
    assert transform_state(
        transform_state(state, vertical_reflection=True),
        vertical_reflection=True,
    ) == state


def test_symmetry_commutes_along_complete_heuristic_games() -> None:
    bot = HeuristicBot()
    for seed in range(3):
        state = create_initial_state(4, seed=seed)
        transformed = transform_state(
            state,
            color_map=COLOR_SWAP,
            horizontal_reflection=True,
            vertical_reflection=True,
        )
        step = 0
        while not state.winners:
            action = bot.choose_action(state)
            assert action is not None
            transformed_action = transform_action(
                state,
                action,
                color_map=COLOR_SWAP,
                horizontal_reflection=True,
                vertical_reflection=True,
            )
            assert transformed_action in legal_actions(transformed)
            state = apply_action(state, action, rng=Random(seed * 10_000 + step))
            transformed = apply_action(
                transformed,
                transformed_action,
                rng=Random(seed * 10_000 + step),
            )
            assert transformed == transform_state(
                state,
                color_map=COLOR_SWAP,
                horizontal_reflection=True,
                vertical_reflection=True,
            )
            step += 1


def test_quarter_turn_is_rejected() -> None:
    try:
        rotate_90()
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported transform was accepted")
