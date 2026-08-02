from dataclasses import replace
from random import Random
from types import SimpleNamespace

from yellowstone.exploratory_collection import (
    EMPTY_HAND_DECK_REFILL_PROBABILITY,
    ExploratoryValueNpc,
    _choose_min_loss_one_or_two_actions,
    _one_off_card_counts,
    _safe_one_card_counts,
    choose_exploratory_refill,
)
from yellowstone.game import create_initial_state
from yellowstone.safe_count_features import rank_color_offset_count_for_player
from yellowstone.types import (
    Card,
    Color,
    GameState,
    Phase,
    PlayerState,
    Position,
    RefillSource,
)


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


def _fake_group(loss: int, label: str):
    return SimpleNamespace(
        negative_card_increase=loss,
        outcomes=((label,),),
    )


def _fake_pools(one_losses, two_losses):
    return SimpleNamespace(
        one_card_groups=tuple(
            _fake_group(loss, f"one-{index}")
            for index, loss in enumerate(one_losses)
        ),
        two_card_groups=tuple(
            _fake_group(loss, f"two-{index}")
            for index, loss in enumerate(two_losses)
        ),
    )


def _count_state(
    board_cards: tuple[Card, ...],
    hand: tuple[Card, ...],
) -> GameState:
    return GameState(
        players=(
            PlayerState(hand=hand),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(index, card.rank_index): (card,)
            for index, card in enumerate(board_cards)
        },
    )


def test_safe_counts_use_rank_window_union() -> None:
    state = _count_state(
        (
            Card(Color.RED, 2),
            Card(Color.BLUE, 4),
        ),
        tuple(Card(Color.RED, rank) for rank in range(7)),
    )

    assert _safe_one_card_counts(state)[0] == 3
    assert _one_off_card_counts(state)[0] == 2


def test_safe_counts_treat_consecutive_two_neighbors_as_safe() -> None:
    state = _count_state(
        (
            Card(Color.RED, 2),
            Card(Color.BLUE, 3),
        ),
        tuple(Card(Color.RED, rank) for rank in range(7)),
    )

    assert _safe_one_card_counts(state)[0] == 4
    assert _one_off_card_counts(state)[0] == 2


def test_safe_counts_treat_single_rank_neighbors_and_next_neighbors_as_safe() -> None:
    state = _count_state(
        (Card(Color.RED, 3),),
        tuple(Card(Color.RED, rank) for rank in range(7)),
    )

    assert _safe_one_card_counts(state)[0] == 5
    assert _one_off_card_counts(state)[0] == 2


def test_safe_counts_add_color_and_rank_offsets() -> None:
    state = _count_state(
        (
            Card(Color.RED, 2),
            Card(Color.BLUE, 3),
            Card(Color.GREEN, 4),
        ),
        (
            Card(Color.RED, 3),
            Card(Color.YELLOW, 3),
            Card(Color.RED, 1),
            Card(Color.YELLOW, 1),
        ),
    )

    assert _safe_one_card_counts(state)[0] == 1
    assert _one_off_card_counts(state)[0] == 2


def test_safe_counts_do_not_penalize_missing_color_with_two_board_colors() -> None:
    state = _count_state(
        (
            Card(Color.RED, 2),
            Card(Color.BLUE, 3),
        ),
        (
            Card(Color.YELLOW, 3),
            Card(Color.YELLOW, 0),
        ),
    )

    assert _safe_one_card_counts(state)[0] == 1
    assert _one_off_card_counts(state)[0] == 1


def test_single_player_offset_count_matches_all_player_count() -> None:
    state = _count_state(
        (
            Card(Color.RED, 2),
            Card(Color.BLUE, 3),
            Card(Color.GREEN, 4),
        ),
        (
            Card(Color.RED, 3),
            Card(Color.YELLOW, 3),
            Card(Color.RED, 1),
            Card(Color.YELLOW, 1),
        ),
    )

    assert rank_color_offset_count_for_player(state, 0) == (
        _safe_one_card_counts(state)[0],
        _one_off_card_counts(state)[0],
    )


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


def test_card_first_mode_remains_distinct_for_low_hand_randomization() -> None:
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

    assert old_choice.selection_mode == "random_safe_two"
    assert fast_choice.selection_mode == "random_safe_two"
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


def test_min_loss_gap_of_three_forces_one_card() -> None:
    pools = _fake_pools([1], [4])

    group, actions, mode, probability, _ = (
        _choose_min_loss_one_or_two_actions(
            pools, starting_hand_size=4, rng=_FixedRandom(0.99)
        )
    )

    assert group.negative_card_increase == 1
    assert actions == ("one-0",)
    assert mode == "random_min_loss_one"
    assert probability == 1.0


def test_hand_six_splits_min_loss_one_and_two_evenly() -> None:
    pools = _fake_pools([2], [3])

    one = _choose_min_loss_one_or_two_actions(
        pools, starting_hand_size=6, rng=_FixedRandom(0.49)
    )
    two = _choose_min_loss_one_or_two_actions(
        pools, starting_hand_size=6, rng=_FixedRandom(0.50)
    )

    assert one[2] == "random_min_loss_one"
    assert one[3] == 0.50
    assert two[2] == "random_min_loss_two"
    assert two[0].negative_card_increase == 3


def test_hand_five_uses_twenty_eighty_min_loss_split() -> None:
    pools = _fake_pools([2], [3])

    one = _choose_min_loss_one_or_two_actions(
        pools, starting_hand_size=5, rng=_FixedRandom(0.19)
    )
    two = _choose_min_loss_one_or_two_actions(
        pools, starting_hand_size=5, rng=_FixedRandom(0.20)
    )

    assert one[2] == "random_min_loss_one"
    assert one[3] == 0.20
    assert two[2] == "random_min_loss_two"


def test_hand_four_or_less_forces_min_loss_two_when_gap_is_small() -> None:
    pools = _fake_pools([2], [3])

    group, _, mode, probability, _ = _choose_min_loss_one_or_two_actions(
        pools, starting_hand_size=4, rng=_FixedRandom(0.0)
    )

    assert mode == "random_min_loss_two"
    assert probability == 0.0
    assert group.negative_card_increase == 3


def test_non_empty_refill_always_uses_deck() -> None:
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

    assert action.source == RefillSource.DECK
    assert audit["starting_hand_size"] == 4
    assert audit["no_refill_probability"] == 0.0
    assert audit["selected_source"] == RefillSource.DECK.value


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
