from math import inf, nan

import pytest

from yellowstone.evaluate_value import validate_checkpoint_contract
from yellowstone.game import create_initial_state
from yellowstone.search_one_card_boost import (
    run_maximization_search,
    run_search,
)
from yellowstone.value_policy import (
    adjusted_win_probability,
    select_highest_value_turn,
)


def test_boost_changes_only_one_card_probability_and_caps_at_one() -> None:
    assert adjusted_win_probability(
        0.4, is_one_card=True, boost_percent=50
    ) == pytest.approx(0.6)
    assert adjusted_win_probability(
        0.4, is_one_card=False, boost_percent=50
    ) == 0.4
    assert adjusted_win_probability(
        0.8, is_one_card=True, boost_percent=50
    ) == 1.0
    assert adjusted_win_probability(
        0.4, is_one_card=True, boost_percent=0
    ) == 0.4


@pytest.mark.parametrize("boost", [-1.0, inf, nan])
def test_boost_rejects_negative_and_non_finite_values(boost: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        adjusted_win_probability(
            0.4, is_one_card=True, boost_percent=boost
        )


def test_zero_boost_and_ties_preserve_existing_deterministic_selection() -> None:
    state = create_initial_state(4, seed=9)
    estimate = lambda record: 0.25

    original = select_highest_value_turn(state, estimate)
    zero_boost = select_highest_value_turn(
        state,
        estimate,
        one_card_win_probability_boost_percent=0,
    )

    assert original.candidate.actions == zero_boost.candidate.actions
    assert original.predicted_win_probability == 0.25
    assert original.selection_score == 0.25


def test_checkpoint_contract_hard_fails_for_history_mismatch(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "value_schema": "yellowstone.value.v1_historyfix",
            "history_semantics": "rolling_last_two_placements",
            "input_canonicalization": "fast_lr_ud_color_v1",
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="input contract mismatch"):
        validate_checkpoint_contract(
            checkpoint, current_turn_history_only=True
        )


def test_search_refines_first_passing_coarse_interval() -> None:
    calls: list[int] = []

    def evaluate(boost: int) -> dict[str, object]:
        calls.append(boost)
        return {
            "one_card_win_probability_boost_percent": boost,
            "win_rate": 0.26 if boost >= 14 else 0.25,
            "all_one_card_candidates_saturated": False,
        }

    state, _, minimum = run_search(
        evaluate, coarse_max=20, coarse_step=10, fine_step=2
    )

    assert state == "threshold_found"
    assert minimum == 14
    assert calls[:3] == [0, 10, 20]
    assert calls[3:] == [12, 14]


def test_search_stops_when_all_one_card_scores_saturate() -> None:
    def evaluate(boost: int) -> dict[str, object]:
        return {
            "one_card_win_probability_boost_percent": boost,
            "win_rate": 0.2,
            "all_one_card_candidates_saturated": boost >= 20,
        }

    state, results, minimum = run_search(
        evaluate, coarse_max=20, coarse_step=10, fine_step=2
    )

    assert state == "unreachable_saturated"
    assert minimum is None
    assert [
        result["one_card_win_probability_boost_percent"]
        for result in results
    ] == [0, 10, 20]


def test_maximization_search_halves_only_around_tied_coarse_maxima() -> None:
    calls: list[int] = []
    rates = {
        10: 0.18,
        20: 0.13,
        30: 0.12,
        40: 0.15,
        50: 0.175,
        60: 0.18,
    }

    def evaluate(boost: int) -> dict[str, object]:
        calls.append(boost)
        rate = rates.get(boost, 0.19 if boost == 12 else 0.16)
        return {
            "one_card_win_probability_boost_percent": boost,
            "win_rate": rate,
            "all_one_card_candidates_saturated": False,
        }

    results, best = run_maximization_search(evaluate)

    assert best == 12
    assert {10, 20, 30, 40, 50, 60} <= set(calls)
    assert all(10 <= call <= 60 for call in calls)
    assert len(calls) == len(set(calls))
    assert {int(result["one_card_win_probability_boost_percent"]) for result in results} == set(calls)
