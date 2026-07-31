from yellowstone.collect_simple_refill_v2 import negative_refill_probability
from yellowstone.types import Card, Color


def _cards(*ranks: int):
    return tuple(Card(Color.BLUE, rank - 1) for rank in ranks)


def test_simple_refill_requires_six_negative_cards() -> None:
    assert negative_refill_probability(_cards(3, 3, 4, 4, 5)) == 0.0


def test_simple_refill_prefers_negative_when_middle_is_at_least_half() -> None:
    assert negative_refill_probability(_cards(1, 2, 3, 4, 5, 7)) == 0.8
    assert negative_refill_probability(_cards(1, 3, 3, 4, 6, 7)) == 0.8


def test_simple_refill_mostly_prefers_deck_below_half() -> None:
    assert negative_refill_probability(_cards(1, 2, 3, 4, 6, 7)) == 0.2
