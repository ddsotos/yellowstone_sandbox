from dataclasses import replace
from random import Random

from yellowstone.exploratory_collection import (
    EMPTY_HAND_DECK_REFILL_PROBABILITY,
    LOW_HAND_NO_REFILL_PROBABILITY,
    ExploratoryValueNpc,
    choose_exploratory_refill,
)
from yellowstone.game import create_initial_state
from yellowstone.types import Phase, RefillSource


class _IncreasingEstimator:
    def estimate_many(self, records):
        return tuple(index / 100 for index in range(len(records)))


class _FixedRandom:
    def __init__(self, value: float):
        self.value = value

    def random(self):
        return self.value

    def choice(self, values):
        return values[0]


def test_hand_six_selects_safe_one_below_thirty_percent() -> None:
    npc = ExploratoryValueNpc.__new__(ExploratoryValueNpc)
    npc.estimator = _IncreasingEstimator()
    state = create_initial_state(4, seed=13)

    choice = npc.choose_turn(state, (), rng=_FixedRandom(0.0))

    assert choice.selection_mode == "random_safe_one"
    assert len(choice.selected_cards) == 1
    assert choice.negative_card_increase == 0
    assert choice.branch_probability == 0.30


def test_hand_six_selects_safe_two_at_thirty_percent_boundary() -> None:
    npc = ExploratoryValueNpc.__new__(ExploratoryValueNpc)
    npc.estimator = _IncreasingEstimator()
    state = create_initial_state(4, seed=13)

    choice = npc.choose_turn(state, (), rng=_FixedRandom(0.30))

    assert choice.selection_mode == "random_safe_two"
    assert len(choice.selected_cards) == 2
    assert choice.negative_card_increase == 0


def test_card_first_hand_six_still_selects_only_zero_loss() -> None:
    npc = ExploratoryValueNpc.__new__(ExploratoryValueNpc)
    npc.card_first = True
    npc.estimator = _IncreasingEstimator()
    state = create_initial_state(4, seed=13)

    choice = npc.choose_turn(state, (), rng=Random(7))

    assert choice.selection_mode in {
        "random_safe_one",
        "random_safe_two",
    }
    assert choice.negative_card_increase == 0
    assert choice.safe_group_counts_exact is False
    assert choice.groups_examined > 0


def test_card_first_fallback_matches_existing_baseline() -> None:
    state = create_initial_state(4, seed=13)
    player = state.players[0]
    state = replace(
        state,
        players=(replace(player, hand=player.hand[:4]), *state.players[1:]),
    )
    old = ExploratoryValueNpc.__new__(ExploratoryValueNpc)
    old.card_first = False
    old.estimator = _IncreasingEstimator()
    fast = ExploratoryValueNpc.__new__(ExploratoryValueNpc)
    fast.card_first = True
    fast.estimator = _IncreasingEstimator()

    old_choice = old.choose_turn(state, (), rng=Random(11))
    fast_choice = fast.choose_turn(state, (), rng=Random(11))

    assert old_choice.selection_mode == fast_choice.selection_mode == "baseline_v1"
    assert old_choice.actions == fast_choice.actions


def test_lazy_single_pass_matches_existing_random_choice() -> None:
    state = create_initial_state(4, seed=13)
    old = ExploratoryValueNpc.__new__(ExploratoryValueNpc)
    old.card_first = False
    old.lazy_single_pass = False
    old.estimator = _IncreasingEstimator()
    lazy = ExploratoryValueNpc.__new__(ExploratoryValueNpc)
    lazy.card_first = False
    lazy.lazy_single_pass = True
    lazy.estimator = _IncreasingEstimator()

    old_choice = old.choose_turn(state, (), rng=Random(7))
    lazy_choice = lazy.choose_turn(state, (), rng=Random(7))

    assert lazy_choice.selection_mode == old_choice.selection_mode
    assert lazy_choice.actions == old_choice.actions
    assert lazy_choice.negative_card_increase == 0
    assert lazy_choice.one_group_count == old_choice.one_group_count
    assert lazy_choice.two_group_count == old_choice.two_group_count


def test_lazy_single_pass_matches_existing_baseline() -> None:
    state = create_initial_state(4, seed=13)
    player = state.players[0]
    state = replace(
        state,
        players=(replace(player, hand=player.hand[:4]), *state.players[1:]),
    )
    old = ExploratoryValueNpc.__new__(ExploratoryValueNpc)
    old.card_first = False
    old.lazy_single_pass = False
    old.estimator = _IncreasingEstimator()
    lazy = ExploratoryValueNpc.__new__(ExploratoryValueNpc)
    lazy.card_first = False
    lazy.lazy_single_pass = True
    lazy.estimator = _IncreasingEstimator()

    old_choice = old.choose_turn(state, (), rng=Random(11))
    lazy_choice = lazy.choose_turn(state, (), rng=Random(11))

    assert lazy_choice.selection_mode == old_choice.selection_mode
    assert lazy_choice.actions == old_choice.actions
    assert lazy_choice.baseline_scores == old_choice.baseline_scores


def test_low_starting_hand_uses_ten_percent_no_refill() -> None:
    state = create_initial_state(4, seed=17)
    player = state.players[0]
    players = (replace(player, hand=player.hand[:2]), *state.players[1:])
    refill = replace(
        state,
        players=players,
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    action, audit = choose_exploratory_refill(
        refill, rng=_FixedRandom(0.0)
    )

    assert action.source == RefillSource.NONE
    assert audit["starting_hand_size"] == 4
    assert audit["no_refill_probability"] == (
        LOW_HAND_NO_REFILL_PROBABILITY
    )


def test_empty_hand_uses_ten_percent_deck_and_ninety_negative() -> None:
    state = create_initial_state(4, seed=19)
    player = state.players[0]
    players = (
        replace(
            player,
            hand=(),
            negative_cards=state.deck[:6],
        ),
        *state.players[1:],
    )
    refill = replace(
        state,
        players=players,
        deck=state.deck[6:],
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    deck, deck_audit = choose_exploratory_refill(
        refill, rng=_FixedRandom(0.0)
    )
    negative, _ = choose_exploratory_refill(
        refill, rng=_FixedRandom(EMPTY_HAND_DECK_REFILL_PROBABILITY)
    )

    assert deck.source == RefillSource.DECK
    assert negative.source == RefillSource.NEGATIVE_CARDS
    assert deck_audit["deck_probability"] == (
        EMPTY_HAND_DECK_REFILL_PROBABILITY
    )


def test_empty_hand_without_six_negative_cards_forces_deck() -> None:
    state = create_initial_state(4, seed=23)
    player = state.players[0]
    players = (
        replace(player, hand=(), negative_cards=state.deck[:5]),
        *state.players[1:],
    )
    refill = replace(
        state,
        players=players,
        deck=state.deck[5:],
        phase=Phase.REFILL,
        cards_played_this_turn=1,
    )

    action, audit = choose_exploratory_refill(
        refill, rng=_FixedRandom(0.99)
    )

    assert action.source == RefillSource.DECK
    assert audit["deck_probability"] == 1.0
