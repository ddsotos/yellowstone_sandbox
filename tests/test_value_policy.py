import pytest

from yellowstone.game import apply_known_legal_action, create_initial_state, legal_actions
from yellowstone.types import Card, Color, GameState, Phase, PlaceCardAction, PlayerState, Position
from yellowstone.value_policy import (
    enumerate_best_turn_card_group,
    enumerate_grouped_turn_action_pools,
    enumerate_grouped_turn_pools,
    TorchWinValueEstimator,
    _candidate_actions,
    _representative_frame_actions,
    _turn_public_result_key,
    enumerate_turn_end_candidates,
    turn_card_group_keys,
)


def test_targeted_card_groups_match_full_grouped_enumeration() -> None:
    state = create_initial_state(4, seed=13)
    full = enumerate_grouped_turn_pools(
        state, approximate_new_color_neighbor_limit=True
    )
    player_index = state.current_player_index

    for play_count, expected_groups in (
        (1, full.one_card_groups),
        (2, full.two_card_groups),
    ):
        expected = {group.cards: group for group in expected_groups}
        assert set(turn_card_group_keys(state, play_count=play_count)) == set(
            expected
        )
        for cards, expected_group in expected.items():
            actual, _ = enumerate_best_turn_card_group(
                state,
                cards,
                approximate_new_color_neighbor_limit=True,
            )
            assert actual is not None
            assert actual.negative_card_increase == (
                expected_group.negative_card_increase
            )
            assert actual.score_bonus == expected_group.score_bonus
            assert {
                _turn_public_result_key(
                    candidate.record.state, player_index
                )
                for candidate in actual.candidates
            } == {
                _turn_public_result_key(
                    candidate.record.state, player_index
                )
                for candidate in expected_group.candidates
            }


def test_action_only_groups_match_materialized_groups() -> None:
    state = create_initial_state(4, seed=17)
    lazy = enumerate_grouped_turn_action_pools(
        state, approximate_new_color_neighbor_limit=True
    )
    full = enumerate_grouped_turn_pools(
        state, approximate_new_color_neighbor_limit=True
    )

    assert lazy.enumerated_candidate_count == full.enumerated_candidate_count
    for action_groups, candidate_groups in (
        (lazy.one_card_groups, full.one_card_groups),
        (lazy.two_card_groups, full.two_card_groups),
    ):
        assert [
            (
                group.cards,
                group.negative_card_increase,
                group.score_bonus,
                group.outcomes,
            )
            for group in action_groups
        ] == [
            (
                group.cards,
                group.negative_card_increase,
                group.score_bonus,
                tuple(
                    candidate.actions for candidate in group.candidates
                ),
            )
            for group in candidate_groups
        ]


def _action_signature(action) -> tuple:
    return (
        type(action).__name__,
        getattr(action, "hand_index", None),
        getattr(getattr(action, "position", None), "x", None),
        getattr(getattr(action, "position", None), "y", None),
        getattr(getattr(action, "frame", None), "x", None),
        getattr(getattr(action, "frame", None), "y", None),
    )


def test_candidate_actions_are_subset_of_legal_actions_at_turn_start() -> None:
    state = create_initial_state(4, seed=9)

    exact = {_action_signature(action) for action in _candidate_actions(state, approximate_new_color_neighbor_limit=False)}
    legal = {_action_signature(action) for action in legal_actions(state)}

    assert exact <= legal
    assert len(exact) < len(legal)


def test_candidate_actions_are_subset_of_legal_actions_after_first_play() -> None:
    state = create_initial_state(4, seed=9)
    first = next(action for action in legal_actions(state) if isinstance(action, PlaceCardAction))
    after_first = apply_known_legal_action(state, first)

    exact = {
        _action_signature(action)
        for action in _candidate_actions(after_first, approximate_new_color_neighbor_limit=False)
    }
    legal = {_action_signature(action) for action in legal_actions(after_first)}

    assert exact <= legal
    assert len(exact) < len(legal)


def test_representative_frame_actions_keeps_single_zero_loss_frame() -> None:
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.GREEN, 4),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={Position(3, 3): (Card(Color.GREEN, 3),)},
        phase=Phase.PLAY,
    )

    actions = _representative_frame_actions(state, 0, Position(3, 4))

    assert len(actions) == 1


def test_representative_frame_actions_dedupes_same_received_negative_cards() -> None:
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 6),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={Position(3, 3): (Card(Color.RED, 3),)},
        phase=Phase.PLAY,
    )

    actions = _representative_frame_actions(state, 0, Position(3, 6))

    assert len(actions) == 1


def test_representative_frame_actions_keeps_distinct_received_negative_cards() -> None:
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.GREEN, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(0, 0): (Card(Color.RED, 0),),
            Position(2, 0): (Card(Color.GREEN, 0),),
            Position(4, 0): (Card(Color.YELLOW, 0),),
        },
        phase=Phase.PLAY,
    )

    actions = _representative_frame_actions(state, 0, Position(2, 0))

    assert len(actions) == 2


def test_torch_estimator_restores_conv3_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from yellowstone.cnn import (
        build_win_value_net,
        win_value_architecture_metadata,
    )

    checkpoint = tmp_path / "conv3.pt"
    model = build_win_value_net(convolution_layers=3)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "context_size": 81,
            **win_value_architecture_metadata(convolution_layers=3),
        },
        checkpoint,
    )

    estimator = TorchWinValueEstimator(str(checkpoint))
    candidates = enumerate_turn_end_candidates(create_initial_state(4, seed=4))
    values = estimator.estimate_many(
        tuple(candidate.record for candidate in candidates[:3])
    )

    assert estimator.architecture["convolution_layers"] == 3
    assert len(values) == 3
    assert all(0.0 <= value <= 1.0 for value in values)
