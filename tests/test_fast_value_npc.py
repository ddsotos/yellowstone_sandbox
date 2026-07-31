from random import Random

from yellowstone.bots import HeuristicBot
from yellowstone.fast_value_npc import (
    MODE_HEURISTIC_ONE_VS_TWO,
    MODE_TWO,
    FastValueNpc,
    _best_safe_pool,
    _dedupe_public_results,
    _heuristic_representative,
    choose_refill,
)
from yellowstone.game import (
    apply_known_legal_action,
    create_initial_state,
    legal_actions,
)
from yellowstone.types import Phase, PlaceCardAction, RefillSource
from yellowstone.value_policy import (
    _turn_public_result_key,
    enumerate_loss_safe_turn_pools,
    enumerate_turn_end_candidates,
)


class _IncreasingEstimator:
    def estimate_many(self, records):
        return tuple(index / 100 for index in range(len(records)))


class _ZeroRandom:
    def random(self):
        return 0.0


class _FailOnUseRandom:
    def __getattr__(self, name):
        raise AssertionError(f"heuristic representative used RNG: {name}")


def test_safe_pool_keeps_minimum_negative_then_maximum_bonus() -> None:
    state = create_initial_state(4, seed=11)
    candidates = _dedupe_public_results(
        enumerate_turn_end_candidates(
            state,
            approximate_new_color_neighbor_limit=True,
            collapse_equivalent_frames=True,
        )
    )

    for play_count in (1, 2):
        pool = _best_safe_pool(
            state, candidates, play_count=play_count
        )
        matching = tuple(
            candidate
            for candidate in candidates
            if sum(
                action.__class__.__name__ == "PlaceCardAction"
                for action in candidate.actions
            )
            == play_count
        )
        assert pool
        negative_counts = [
            len(
                candidate.record.state.players[
                    state.current_player_index
                ].negative_cards
            )
            for candidate in matching
        ]
        assert {
            len(
                candidate.record.state.players[
                    state.current_player_index
                ].negative_cards
            )
            for candidate in pool
        } == {min(negative_counts)}
        assert len(
            {
                candidate.record.state.players[
                    state.current_player_index
                ].loss_score
                for candidate in pool
            }
        ) == 1


def test_two_mode_scores_one_candidate_per_play_count() -> None:
    npc = FastValueNpc.__new__(FastValueNpc)
    npc.mode = MODE_TWO
    npc.estimator = _IncreasingEstimator()
    state = create_initial_state(4, seed=13)

    choice = npc.choose_turn(state, (), rng=Random(7))

    assert len(choice.scores) == 2
    assert choice.one_pool_size >= 1
    assert choice.two_pool_size >= 1


def test_heuristic_mode_scores_deterministic_one_and_two_representatives() -> None:
    npc = FastValueNpc.__new__(FastValueNpc)
    npc.mode = MODE_HEURISTIC_ONE_VS_TWO
    npc.estimator = _IncreasingEstimator()
    state = create_initial_state(4, seed=13)
    pools = enumerate_loss_safe_turn_pools(
        state,
        approximate_new_color_neighbor_limit=True,
    )

    choice = npc.choose_turn(state, (), rng=_FailOnUseRandom())

    expected = (
        _heuristic_representative(
            state, pools.one_card_candidates
        ),
        _heuristic_representative(
            state, pools.two_card_candidates
        ),
    )
    assert all(candidate is not None for candidate in expected)
    assert len(choice.scores) == 2
    assert choice.selection_mode == "max_value"
    assert choice.actions == expected[1].actions


def test_heuristic_representative_uses_placement_sort_keys() -> None:
    state = create_initial_state(4, seed=13)
    pools = enumerate_loss_safe_turn_pools(
        state,
        approximate_new_color_neighbor_limit=True,
    )

    for candidates in (
        pools.one_card_candidates,
        pools.two_card_candidates,
    ):
        selected = _heuristic_representative(state, candidates)
        assert selected is not None

        def key(candidate):
            working = state
            result = []
            for action in candidate.actions:
                if isinstance(action, PlaceCardAction):
                    from yellowstone.bots import placement_sort_key

                    result.append(placement_sort_key(working, action))
                working = apply_known_legal_action(working, action)
            return tuple(result)

        assert selected == min(candidates, key=key)


def test_two_card_refill_boundary_can_choose_none() -> None:
    state = create_initial_state(4, seed=17)
    candidate = next(
        candidate
        for candidate in enumerate_turn_end_candidates(state)
        if candidate.record.state.phase == Phase.REFILL
    )

    action, audit = choose_refill(
        candidate.record.state, rng=_ZeroRandom()
    )

    assert action.source == RefillSource.NONE
    assert audit["eligible_no_refill"] is True


def test_streaming_safe_pools_match_full_enumeration() -> None:
    bot = HeuristicBot()
    rng = Random(29)
    state = create_initial_state(4, seed=29)
    checked = 0
    while state.phase != Phase.GAME_OVER and checked < 12:
        if (
            state.phase == Phase.PLAY
            and state.cards_played_this_turn == 0
            and any(
                isinstance(action, PlaceCardAction)
                for action in legal_actions(state)
            )
        ):
            full = _dedupe_public_results(
                enumerate_turn_end_candidates(
                    state,
                    approximate_new_color_neighbor_limit=True,
                    collapse_equivalent_frames=True,
                )
            )
            expected_one = _best_safe_pool(
                state, full, play_count=1
            )
            expected_two = _best_safe_pool(
                state, full, play_count=2
            )
            streamed = enumerate_loss_safe_turn_pools(
                state,
                approximate_new_color_neighbor_limit=True,
            )
            player_index = state.current_player_index

            def keys(candidates):
                return {
                    _turn_public_result_key(
                        candidate.record.state, player_index
                    )
                    for candidate in candidates
                }

            assert keys(streamed.one_card_candidates) == keys(
                expected_one
            )
            assert keys(streamed.two_card_candidates) == keys(
                expected_two
            )
            checked += 1
        action = bot.choose_action(state)
        assert action is not None
        state = apply_known_legal_action(state, action, rng=rng)
    assert checked == 12
