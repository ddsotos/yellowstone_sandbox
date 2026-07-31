from dataclasses import replace

from yellowstone.game import apply_known_legal_action, create_initial_state, legal_actions
from yellowstone.types import Phase, PlaceCardAction
from yellowstone.value_evaluation import EvaluationResult, ValueTurnPlayer


def test_value_turn_player_commits_to_a_legal_complete_turn_plan() -> None:
    state = create_initial_state(4, seed=9)
    player = ValueTurnPlayer(0, estimate=lambda record: float(len(record.state.board)))

    first = player.choose_action(state)
    assert isinstance(first, PlaceCardAction)
    after_first = apply_known_legal_action(state, first)
    player.observe(state, first, after_first)

    second = player.choose_action(after_first)
    assert second in legal_actions(after_first)


def test_value_turn_player_handles_empty_hand_play_start_via_heuristic_refill() -> None:
    state = create_initial_state(4, seed=9)
    empty_player = replace(state.players[0], hand=())
    state = replace(state, players=(empty_player, *state.players[1:]), phase=Phase.PLAY)
    player = ValueTurnPlayer(0, estimate=lambda record: float(len(record.state.board)))

    action = player.choose_action(state)

    assert action in legal_actions(state)
    assert not isinstance(action, PlaceCardAction)


def test_value_turn_player_supports_approximate_new_color_neighbor_limit() -> None:
    state = create_initial_state(4, seed=9)
    player = ValueTurnPlayer(
        0,
        estimate=lambda record: float(len(record.state.board)),
        approximate_new_color_neighbor_limit=True,
    )

    action = player.choose_action(state)

    assert action in legal_actions(state)


def test_evaluation_result_reports_one_card_turn_rate() -> None:
    result = EvaluationResult(
        games=10,
        wins=2.0,
        evaluated_player_one_card_turns=3,
        evaluated_player_two_card_turns=7,
    )

    assert result.evaluated_player_one_card_turn_rate == 0.3
