from random import Random

from yellowstone.collect_v2 import (
    CATEGORY_DECK,
    CATEGORY_NEGATIVE,
    CATEGORY_RETAIN,
    _category_probabilities,
    _weighted_choice,
)


def test_three_category_probabilities_are_60_20_20() -> None:
    probabilities = _category_probabilities(
        [CATEGORY_RETAIN, CATEGORY_DECK, CATEGORY_NEGATIVE],
        CATEGORY_NEGATIVE,
    )
    assert probabilities == {
        CATEGORY_RETAIN: 0.2,
        CATEGORY_DECK: 0.2,
        CATEGORY_NEGATIVE: 0.6,
    }


def test_unavailable_category_mass_moves_to_best() -> None:
    probabilities = _category_probabilities(
        [CATEGORY_DECK, CATEGORY_NEGATIVE],
        CATEGORY_DECK,
    )
    assert probabilities == {
        CATEGORY_RETAIN: 0.0,
        CATEGORY_DECK: 0.8,
        CATEGORY_NEGATIVE: 0.2,
    }
    assert _category_probabilities([CATEGORY_DECK], CATEGORY_DECK) == {
        CATEGORY_RETAIN: 0.0,
        CATEGORY_DECK: 1.0,
        CATEGORY_NEGATIVE: 0.0,
    }


def test_weighted_choice_never_returns_unavailable_category() -> None:
    probabilities = _category_probabilities(
        [CATEGORY_DECK, CATEGORY_NEGATIVE],
        CATEGORY_DECK,
    )
    selected = {
        _weighted_choice(probabilities, Random(seed))
        for seed in range(100)
    }
    assert CATEGORY_RETAIN not in selected
    assert selected == {CATEGORY_DECK, CATEGORY_NEGATIVE}
