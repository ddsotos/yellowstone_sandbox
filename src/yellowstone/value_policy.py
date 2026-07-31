"""Turn-level candidate generation for a learned win-value estimator."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Callable

from yellowstone.game import (
    all_frames,
    apply_known_legal_action,
    can_place_card_at,
    columns_containing_color,
    colors_in_column,
    frame_positions,
    frames_containing,
    legal_actions,
    occupied_count_in_frame,
)
from yellowstone.types import (
    Action,
    BOARD_SIZE,
    Card,
    EndTurnAction,
    Frame,
    GameState,
    Phase,
    PlaceCardAction,
    Position,
)
from yellowstone.value_learning import (
    HISTORY_SIZE,
    VALUE_CONTEXT_SIZE,
    RecentPlacement,
    ValueRecord,
)


@dataclass(frozen=True, slots=True)
class TurnCandidate:
    """A complete one- or two-card turn and its turn-end state."""

    actions: tuple[Action, ...]
    record: ValueRecord


@dataclass(frozen=True, slots=True)
class TurnSelection:
    """Selected candidate plus pruning facts for evaluation/audit."""

    candidate: TurnCandidate
    predicted_win_probability: float
    selection_score: float
    pruning_active: bool
    candidate_count_before_pruning: int
    candidate_count_after_pruning: int
    negative_card_increase: int
    one_card_candidate_count: int
    all_one_card_candidates_saturated: bool


@dataclass(frozen=True, slots=True)
class LossSafeTurnPools:
    """Best loss/bonus candidates without materializing inferior records."""

    one_card_candidates: tuple[TurnCandidate, ...]
    two_card_candidates: tuple[TurnCandidate, ...]
    enumerated_candidate_count: int


@dataclass(frozen=True, slots=True)
class GroupedTurnCandidates:
    """Best outcomes grouped by the unordered cards played."""

    cards: tuple[Card, ...]
    negative_card_increase: int
    score_bonus: int
    candidates: tuple[TurnCandidate, ...]


@dataclass(frozen=True, slots=True)
class GroupedTurnPools:
    """One/two-card groups retained at per-card-group best loss/bonus."""

    one_card_groups: tuple[GroupedTurnCandidates, ...]
    two_card_groups: tuple[GroupedTurnCandidates, ...]
    enumerated_candidate_count: int


@dataclass(frozen=True, slots=True)
class GroupedTurnActions:
    """Per-card-group best outcomes without ValueRecord materialization."""

    cards: tuple[Card, ...]
    negative_card_increase: int
    score_bonus: int
    outcomes: tuple[tuple[Action, ...], ...]


@dataclass(frozen=True, slots=True)
class GroupedTurnActionPools:
    one_card_groups: tuple[GroupedTurnActions, ...]
    two_card_groups: tuple[GroupedTurnActions, ...]
    enumerated_candidate_count: int


def turn_card_group_keys(
    state: GameState, *, play_count: int
) -> tuple[tuple[Card, ...], ...]:
    """Return distinct physical-card multisets available for a play count."""
    if play_count not in (1, 2):
        raise ValueError("play_count must be one or two")
    hand = state.players[state.current_player_index].hand
    groups = {
        tuple(sorted(cards, key=_card_sort_key))
        for cards in (
            ((card,) for card in hand)
            if play_count == 1
            else combinations(hand, 2)
        )
    }
    return tuple(sorted(groups, key=lambda cards: tuple(map(_card_sort_key, cards))))


def enumerate_best_turn_card_group(
    state: GameState,
    cards: tuple[Card, ...],
    *,
    history: tuple[RecentPlacement, ...] = (),
    game_id: int = -1,
    approximate_new_color_neighbor_limit: bool = False,
) -> tuple[GroupedTurnCandidates | None, int]:
    """Evaluate only one card multiset and retain its min-loss/max-bonus outcomes."""
    if state.phase != Phase.PLAY or state.cards_played_this_turn != 0:
        raise ValueError("turn candidates require a play-phase turn start")
    if len(cards) not in (1, 2):
        raise ValueError("card group must contain one or two cards")
    target_cards = tuple(sorted(cards, key=_card_sort_key))
    player_index = state.current_player_index
    starting_player = state.players[player_index]
    best_metric: tuple[int, int] | None = None
    retained: dict[tuple[object, ...], tuple[Action, ...]] = {}
    enumerated = 0

    def consider(
        actions: tuple[Action, ...],
        metric: tuple[int, int],
        public_key: tuple[object, ...],
    ) -> None:
        nonlocal best_metric, enumerated
        enumerated += 1
        if best_metric is None or metric < best_metric:
            best_metric = metric
            retained.clear()
        elif metric != best_metric:
            return
        retained.setdefault(public_key, actions)

    for first in _candidate_actions_for_cards(
        state,
        allowed_cards=target_cards,
        approximate_new_color_neighbor_limit=(
            approximate_new_color_neighbor_limit
        ),
    ):
        if not isinstance(first, PlaceCardAction):
            continue
        first_card = starting_player.hand[first.hand_index]
        after_first = apply_known_legal_action(state, first)
        if len(target_cards) == 1:
            if first_card != target_cards[0]:
                continue
            if EndTurnAction() not in legal_actions(after_first):
                continue
            after_end = apply_known_legal_action(
                after_first, EndTurnAction()
            )
            after_player = after_end.players[player_index]
            consider(
                (first, EndTurnAction()),
                (
                    len(after_player.negative_cards)
                    - len(starting_player.negative_cards),
                    after_player.loss_score - starting_player.loss_score,
                ),
                _turn_public_result_key(after_end, player_index),
            )
            continue

        for second in _candidate_actions_for_cards(
            after_first,
            allowed_cards=target_cards,
            approximate_new_color_neighbor_limit=(
                approximate_new_color_neighbor_limit
            ),
        ):
            if not isinstance(second, PlaceCardAction):
                continue
            second_card = after_first.players[player_index].hand[
                second.hand_index
            ]
            if (
                tuple(
                    sorted(
                        (first_card, second_card), key=_card_sort_key
                    )
                )
                != target_cards
            ):
                continue
            metric, public_key = _second_placement_metric_and_key(
                after_first,
                second,
                starting_negative_count=len(
                    starting_player.negative_cards
                ),
                starting_loss_score=starting_player.loss_score,
            )
            consider((first, second), metric, public_key)

    if best_metric is None:
        return None, enumerated
    candidates: list[TurnCandidate] = []
    for actions in retained.values():
        working = state
        resulting_history = history
        for action in actions:
            if isinstance(action, PlaceCardAction):
                working, resulting_history = _apply_with_history(
                    working, action, resulting_history
                )
            else:
                working = apply_known_legal_action(working, action)
        candidates.append(
            TurnCandidate(
                actions=actions,
                record=ValueRecord(
                    game_id,
                    player_index,
                    working,
                    resulting_history,
                    0.0,
                ),
            )
        )
    return (
        GroupedTurnCandidates(
            cards=target_cards,
            negative_card_increase=best_metric[0],
            score_bonus=-best_metric[1],
            candidates=tuple(candidates),
        ),
        enumerated,
    )


def enumerate_grouped_turn_action_pools(
    state: GameState,
    *,
    approximate_new_color_neighbor_limit: bool = False,
) -> GroupedTurnActionPools:
    """Single-pass grouped outcomes without constructing ValueRecords."""
    if state.phase != Phase.PLAY or state.cards_played_this_turn != 0:
        raise ValueError("turn candidates require a play-phase turn start")
    player_index = state.current_player_index
    starting_player = state.players[player_index]
    retained: dict[
        int,
        dict[
            tuple[tuple[str, int], ...],
            tuple[
                tuple[int, int],
                tuple[Card, ...],
                dict[tuple[object, ...], tuple[Action, ...]],
            ],
        ],
    ] = {1: {}, 2: {}}
    enumerated = 0

    def consider(
        play_count: int,
        cards: tuple[Card, ...],
        actions: tuple[Action, ...],
        metric: tuple[int, int],
        public_key: tuple[object, ...],
    ) -> None:
        nonlocal enumerated
        enumerated += 1
        ordered_cards = tuple(
            sorted(cards, key=lambda card: (card.color.value, card.rank_index))
        )
        group_key = tuple(
            (card.color.value, card.rank_index) for card in ordered_cards
        )
        existing = retained[play_count].get(group_key)
        if existing is None or metric < existing[0]:
            retained[play_count][group_key] = (
                metric,
                ordered_cards,
                {public_key: actions},
            )
        elif metric == existing[0]:
            existing[2].setdefault(public_key, actions)

    first_actions = _candidate_actions(
        state,
        approximate_new_color_neighbor_limit=(
            approximate_new_color_neighbor_limit
        ),
        collapse_equivalent_frames=True,
        collapse_identical_hand_cards=True,
    )
    for first in first_actions:
        if not isinstance(first, PlaceCardAction):
            continue
        first_card = starting_player.hand[first.hand_index]
        after_first = apply_known_legal_action(state, first)
        if EndTurnAction() in legal_actions(after_first):
            after_end = apply_known_legal_action(
                after_first, EndTurnAction()
            )
            after_end_player = after_end.players[player_index]
            consider(
                1,
                (first_card,),
                (first, EndTurnAction()),
                (
                    len(after_end_player.negative_cards)
                    - len(starting_player.negative_cards),
                    after_end_player.loss_score - starting_player.loss_score,
                ),
                _turn_public_result_key(after_end, player_index),
            )
        for second in _candidate_actions(
            after_first,
            approximate_new_color_neighbor_limit=(
                approximate_new_color_neighbor_limit
            ),
            collapse_equivalent_frames=True,
            collapse_identical_hand_cards=True,
        ):
            if not isinstance(second, PlaceCardAction):
                continue
            second_card = after_first.players[player_index].hand[
                second.hand_index
            ]
            metric, public_key = _second_placement_metric_and_key(
                after_first,
                second,
                starting_negative_count=len(
                    starting_player.negative_cards
                ),
                starting_loss_score=starting_player.loss_score,
            )
            consider(
                2,
                (first_card, second_card),
                (first, second),
                metric,
                public_key,
            )

    def action_groups(play_count: int) -> tuple[GroupedTurnActions, ...]:
        groups: list[GroupedTurnActions] = []
        for group_key in sorted(retained[play_count]):
            metric, cards, outcomes = retained[play_count][group_key]
            groups.append(
                GroupedTurnActions(
                    cards=cards,
                    negative_card_increase=metric[0],
                    score_bonus=-metric[1],
                    outcomes=tuple(outcomes.values()),
                )
            )
        return tuple(groups)

    return GroupedTurnActionPools(
        one_card_groups=action_groups(1),
        two_card_groups=action_groups(2),
        enumerated_candidate_count=enumerated,
    )


def materialize_turn_candidate(
    state: GameState,
    actions: tuple[Action, ...],
    *,
    history: tuple[RecentPlacement, ...] = (),
    game_id: int = -1,
) -> TurnCandidate:
    """Build one ValueRecord only after an action sequence is selected."""
    working = state
    resulting_history = history
    for action in actions:
        if isinstance(action, PlaceCardAction):
            working, resulting_history = _apply_with_history(
                working, action, resulting_history
            )
        else:
            working = apply_known_legal_action(working, action)
    return TurnCandidate(
        actions=actions,
        record=ValueRecord(
            game_id,
            state.current_player_index,
            working,
            resulting_history,
            0.0,
        ),
    )


def enumerate_grouped_turn_pools(
    state: GameState,
    *,
    history: tuple[RecentPlacement, ...] = (),
    game_id: int = -1,
    approximate_new_color_neighbor_limit: bool = False,
) -> GroupedTurnPools:
    """Materialized compatibility wrapper around action-only enumeration."""
    pools = enumerate_grouped_turn_action_pools(
        state,
        approximate_new_color_neighbor_limit=(
            approximate_new_color_neighbor_limit
        ),
    )

    def materialize(
        groups: tuple[GroupedTurnActions, ...],
    ) -> tuple[GroupedTurnCandidates, ...]:
        return tuple(
            GroupedTurnCandidates(
                cards=group.cards,
                negative_card_increase=group.negative_card_increase,
                score_bonus=group.score_bonus,
                candidates=tuple(
                    materialize_turn_candidate(
                        state,
                        actions,
                        history=history,
                        game_id=game_id,
                    )
                    for actions in group.outcomes
                ),
            )
            for group in groups
        )

    return GroupedTurnPools(
        one_card_groups=materialize(pools.one_card_groups),
        two_card_groups=materialize(pools.two_card_groups),
        enumerated_candidate_count=pools.enumerated_candidate_count,
    )


def enumerate_loss_safe_turn_pools(
    state: GameState,
    *,
    history: tuple[RecentPlacement, ...] = (),
    game_id: int = -1,
    approximate_new_color_neighbor_limit: bool = False,
) -> LossSafeTurnPools:
    """Stream one/two-card outcomes and retain min-loss, max-bonus states.

    This is equivalent to enumerating all turn-end candidates, deduplicating
    identical public results, then applying the loss/bonus filter. It avoids
    constructing ``ValueRecord`` and history objects for inferior outcomes.
    """
    if state.phase != Phase.PLAY or state.cards_played_this_turn != 0:
        raise ValueError("turn candidates require a play-phase turn start")
    player_index = state.current_player_index
    starting_player = state.players[player_index]
    best_metrics: dict[int, tuple[int, int] | None] = {1: None, 2: None}
    retained: dict[
        int, dict[tuple[object, ...], tuple[Action, ...]]
    ] = {1: {}, 2: {}}
    enumerated = 0

    def consider(
        play_count: int,
        actions: tuple[Action, ...],
        metric: tuple[int, int],
        public_key: tuple[object, ...],
    ) -> None:
        nonlocal enumerated
        enumerated += 1
        best = best_metrics[play_count]
        if best is None or metric < best:
            best_metrics[play_count] = metric
            retained[play_count].clear()
        elif metric != best:
            return
        retained[play_count].setdefault(public_key, actions)

    first_actions = _candidate_actions(
        state,
        approximate_new_color_neighbor_limit=(
            approximate_new_color_neighbor_limit
        ),
        collapse_equivalent_frames=True,
        collapse_identical_hand_cards=True,
    )
    for first in first_actions:
        if not isinstance(first, PlaceCardAction):
            continue
        after_first = apply_known_legal_action(state, first)
        if EndTurnAction() in legal_actions(after_first):
            after_end = apply_known_legal_action(
                after_first, EndTurnAction()
            )
            after_end_player = after_end.players[player_index]
            consider(
                1,
                (first, EndTurnAction()),
                (
                    len(after_end_player.negative_cards)
                    - len(starting_player.negative_cards),
                    after_end_player.loss_score - starting_player.loss_score,
                ),
                _turn_public_result_key(after_end, player_index),
            )
        for second in _candidate_actions(
            after_first,
            approximate_new_color_neighbor_limit=(
                approximate_new_color_neighbor_limit
            ),
            collapse_equivalent_frames=True,
            collapse_identical_hand_cards=True,
        ):
            if not isinstance(second, PlaceCardAction):
                continue
            metric, public_key = _second_placement_metric_and_key(
                after_first,
                second,
                starting_negative_count=len(
                    starting_player.negative_cards
                ),
                starting_loss_score=starting_player.loss_score,
            )
            consider(2, (first, second), metric, public_key)

    def records_for(play_count: int) -> tuple[TurnCandidate, ...]:
        candidates: list[TurnCandidate] = []
        for actions in retained[play_count].values():
            working = state
            resulting_history = history
            for action in actions:
                if isinstance(action, PlaceCardAction):
                    working, resulting_history = _apply_with_history(
                        working, action, resulting_history
                    )
                else:
                    working = apply_known_legal_action(working, action)
            candidates.append(
                TurnCandidate(
                    actions=actions,
                    record=ValueRecord(
                        game_id,
                        player_index,
                        working,
                        resulting_history,
                        0.0,
                    ),
                )
            )
        return tuple(candidates)

    return LossSafeTurnPools(
        one_card_candidates=records_for(1),
        two_card_candidates=records_for(2),
        enumerated_candidate_count=enumerated,
    )


def enumerate_turn_end_candidates(
    state: GameState,
    *,
    history: tuple[RecentPlacement, ...] = (),
    game_id: int = -1,
    max_negative_card_increase: int | None = None,
    approximate_new_color_neighbor_limit: bool = False,
    collapse_equivalent_frames: bool = True,
) -> tuple[TurnCandidate, ...]:
    """Enumerate legal one-card-end and two-card-refill-boundary candidates.

    ``state`` must be the evaluating player's turn start. Candidate records
    preserve that player's perspective even when a one-card EndTurn has moved
    ``current_player_index`` to the next player. When
    ``max_negative_card_increase`` is supplied, candidates over that exact
    limit are omitted; a first placement already over the limit skips all of
    its second-placement branches.
    """
    if state.phase != Phase.PLAY or state.cards_played_this_turn != 0:
        raise ValueError("turn candidates require a play-phase turn start")
    if max_negative_card_increase is not None and max_negative_card_increase < 0:
        raise ValueError("max_negative_card_increase must not be negative")
    player_index = state.current_player_index
    initial_negative_count = len(state.players[player_index].negative_cards)
    result: list[TurnCandidate] = []
    for first in _candidate_actions(
        state,
        approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
        collapse_equivalent_frames=collapse_equivalent_frames,
    ):
        if not isinstance(first, PlaceCardAction):
            continue
        after_first, first_history = _apply_with_history(state, first, history)
        first_negative_increase = (
            len(after_first.players[player_index].negative_cards) - initial_negative_count
        )
        if (
            max_negative_card_increase is not None
            and first_negative_increase > max_negative_card_increase
        ):
            continue
        if EndTurnAction() in legal_actions(after_first):
            after_end = apply_known_legal_action(after_first, EndTurnAction())
            result.append(
                TurnCandidate(
                    actions=(first, EndTurnAction()),
                    record=ValueRecord(game_id, player_index, after_end, first_history, 0.0),
                )
            )
        for second in _candidate_actions(
            after_first,
            approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
            collapse_equivalent_frames=collapse_equivalent_frames,
        ):
            if not isinstance(second, PlaceCardAction):
                continue
            after_second, second_history = _apply_with_history(
                after_first, second, first_history
            )
            if (
                max_negative_card_increase is not None
                and len(after_second.players[player_index].negative_cards)
                - initial_negative_count
                > max_negative_card_increase
            ):
                continue
            if after_second.phase != Phase.REFILL:
                raise AssertionError("second placement must reach refill boundary")
            result.append(
                TurnCandidate(
                    actions=(first, second),
                    record=ValueRecord(game_id, player_index, after_second, second_history, 0.0),
                )
            )
    return tuple(result)


def choose_highest_value_turn(
    state: GameState,
    estimate: Callable[[ValueRecord], float],
    *,
    history: tuple[RecentPlacement, ...] = (),
    prune_negative_card_increase_above: int | None = None,
    approximate_new_color_neighbor_limit: bool = False,
) -> TurnCandidate:
    """Return the highest-value completed turn.

    With ``prune_negative_card_increase_above``, pruning activates only when a
    cheap frame-local search proves that a two-placement, zero-negative-card
    candidate exists.
    """
    if prune_negative_card_increase_above is not None:
        if prune_negative_card_increase_above < 0:
            raise ValueError("prune_negative_card_increase_above must not be negative")
    return select_highest_value_turn(
        state,
        estimate,
        history=history,
        prune_negative_card_increase_above=prune_negative_card_increase_above,
        approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
    ).candidate


def select_highest_value_turn(
    state: GameState,
    estimate: Callable[[ValueRecord], float],
    *,
    history: tuple[RecentPlacement, ...] = (),
    prune_negative_card_increase_above: int | None = None,
    approximate_new_color_neighbor_limit: bool = False,
    one_card_win_probability_boost_percent: float = 0.0,
) -> TurnSelection:
    """Select a turn and expose whether loss pruning was used."""
    _validate_one_card_boost_percent(one_card_win_probability_boost_percent)
    if prune_negative_card_increase_above is not None:
        if prune_negative_card_increase_above < 0:
            raise ValueError("prune_negative_card_increase_above must not be negative")
    pruning_limit = (
        prune_negative_card_increase_above
        if prune_negative_card_increase_above is not None
        and _has_zero_negative_two_card_witness(state)
        else None
    )
    candidates = enumerate_turn_end_candidates(
        state,
        history=history,
        max_negative_card_increase=pruning_limit,
        approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
    )
    if not candidates:
        raise ValueError("no completed-turn candidates are legal")
    player_index = state.current_player_index
    negative_before = len(state.players[player_index].negative_cards)
    estimate_many = getattr(estimate, "estimate_many", None)
    if estimate_many is not None:
        raw_scores = tuple(
            float(score)
            for score in estimate_many(
                tuple(candidate.record for candidate in candidates)
            )
        )
    else:
        raw_scores = tuple(
            float(estimate(candidate.record)) for candidate in candidates
        )
    if any(not isfinite(score) for score in raw_scores):
        raise ValueError("estimated win probabilities must be finite")
    scores = tuple(
        adjusted_win_probability(
            raw_score,
            is_one_card=_is_one_card_candidate(candidate),
            boost_percent=one_card_win_probability_boost_percent,
        )
        for candidate, raw_score in zip(candidates, raw_scores, strict=True)
    )
    best_index = max(range(len(candidates)), key=lambda index: scores[index])
    selected = candidates[best_index]
    one_card_scores = tuple(
        score
        for candidate, score in zip(candidates, scores, strict=True)
        if _is_one_card_candidate(candidate)
    )
    return TurnSelection(
        candidate=selected,
        predicted_win_probability=raw_scores[best_index],
        selection_score=scores[best_index],
        pruning_active=pruning_limit is not None,
        candidate_count_before_pruning=-1,
        candidate_count_after_pruning=len(candidates),
        negative_card_increase=(
            len(selected.record.state.players[player_index].negative_cards) - negative_before
        ),
        one_card_candidate_count=len(one_card_scores),
        all_one_card_candidates_saturated=(
            bool(one_card_scores)
            and all(score == 1.0 for score in one_card_scores)
        ),
    )


def adjusted_win_probability(
    probability: float,
    *,
    is_one_card: bool,
    boost_percent: float,
) -> float:
    """Return the candidate selection score after a one-card-only boost."""
    _validate_one_card_boost_percent(boost_percent)
    probability = float(probability)
    if not isfinite(probability):
        raise ValueError("estimated win probability must be finite")
    if not is_one_card:
        return probability
    return min(1.0, probability * (1.0 + boost_percent / 100.0))


def _validate_one_card_boost_percent(boost_percent: float) -> None:
    if not isfinite(boost_percent) or boost_percent < 0:
        raise ValueError(
            "one-card win-probability boost percent must be finite and non-negative"
        )


def _is_one_card_candidate(candidate: TurnCandidate) -> bool:
    return any(isinstance(action, EndTurnAction) for action in candidate.actions)


def _candidate_actions(
    state: GameState,
    *,
    approximate_new_color_neighbor_limit: bool,
    collapse_equivalent_frames: bool = True,
    collapse_identical_hand_cards: bool = False,
) -> tuple[Action, ...]:
    return _candidate_actions_for_cards(
        state,
        allowed_cards=None,
        approximate_new_color_neighbor_limit=(
            approximate_new_color_neighbor_limit
        ),
        collapse_equivalent_frames=collapse_equivalent_frames,
        collapse_identical_hand_cards=collapse_identical_hand_cards,
    )


def _candidate_actions_for_cards(
    state: GameState,
    *,
    allowed_cards: tuple[Card, ...] | None,
    approximate_new_color_neighbor_limit: bool,
    collapse_equivalent_frames: bool = True,
    collapse_identical_hand_cards: bool = True,
) -> tuple[Action, ...]:
    if state.phase != Phase.PLAY:
        return legal_actions(state)
    actions: list[Action] = []
    if state.cards_played_this_turn == 1:
        actions.append(EndTurnAction())
    if state.cards_played_this_turn >= 2:
        return tuple(actions)
    player = state.players[state.current_player_index]
    if not player.hand and state.cards_played_this_turn == 0:
        return legal_actions(state)
    exact_positions = (
        None
        if approximate_new_color_neighbor_limit
        else _exact_legal_positions_by_card(state.board, player.hand)
    )
    seen_cards: set[Card] = set()
    for hand_index, card in enumerate(player.hand):
        if allowed_cards is not None and card not in allowed_cards:
            continue
        if collapse_identical_hand_cards and card in seen_cards:
            continue
        seen_cards.add(card)
        positions = (
            exact_positions[hand_index]
            if exact_positions is not None
            else _candidate_positions_for_card(
                state.board,
                card,
                approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
            )
        )
        for position in positions:
            if collapse_equivalent_frames:
                actions.extend(
                    _representative_frame_actions(state, hand_index, position)
                )
            else:
                actions.extend(
                    PlaceCardAction(
                        hand_index=hand_index,
                        position=position,
                        frame=frame,
                    )
                    for frame in frames_containing(position)
                )
    return tuple(actions)


def _card_sort_key(card: Card) -> tuple[str, int]:
    return card.color.value, card.rank_index


def _representative_frame_actions(
    state: GameState, hand_index: int, position: Position
) -> tuple[PlaceCardAction, ...]:
    """Collapse equivalent frame choices for one placement.

    If any frame yields zero added negative cards, keep the first such frame
    only. Otherwise, keep one representative per distinct received-negative
    multiset among frames with the minimum added-negative-card count.
    """
    frames = frames_containing(position)
    if len(frames) <= 1:
        return (PlaceCardAction(hand_index=hand_index, position=position, frame=frames[0]),) if frames else ()
    best_loss: int | None = None
    representatives: dict[tuple[tuple[str, int], ...], PlaceCardAction] = {}
    player = state.players[state.current_player_index]
    card = player.hand[hand_index]
    board = dict(state.board)
    board[position] = board.get(position, ()) + (card,)
    for frame in frames:
        action = PlaceCardAction(hand_index=hand_index, position=position, frame=frame)
        frame_cells = frame_positions(frame)
        received = tuple(
            card
            for board_position, stack in board.items()
            if board_position not in frame_cells
            for card in stack
        )
        loss = len(received)
        if loss == 0:
            return (action,)
        if best_loss is None or loss < best_loss:
            best_loss = loss
            representatives.clear()
        if loss != best_loss:
            continue
        signature = tuple(sorted((card.color.value, card.rank_index) for card in received))
        representatives.setdefault(signature, action)
    return tuple(representatives.values())


def _turn_public_result_key(
    state: GameState, perspective_player_index: int
) -> tuple[object, ...]:
    player = state.players[perspective_player_index]
    board = tuple(
        sorted(
            (
                position.x,
                position.y,
                tuple(
                    (card.color.value, card.rank_index) for card in stack
                ),
            )
            for position, stack in state.board.items()
        )
    )
    return (
        board,
        tuple((card.color.value, card.rank_index) for card in player.hand),
        tuple(
            (card.color.value, card.rank_index)
            for card in player.negative_cards
        ),
        player.loss_score,
        state.current_player_index,
        state.phase.value,
        state.cards_played_this_turn,
        state.settlement_count,
    )


def _second_placement_metric_and_key(
    state: GameState,
    action: PlaceCardAction,
    *,
    starting_negative_count: int,
    starting_loss_score: int,
) -> tuple[tuple[int, int], tuple[object, ...]]:
    """Compute a second placement outcome without replacing ``GameState``."""
    player = state.players[state.current_player_index]
    card = player.hand[action.hand_index]
    board = dict(state.board)
    frame_cells = frame_positions(action.frame)
    occupied_before = occupied_count_in_frame(board, action.frame)
    was_occupied = action.position in board
    board[action.position] = board.get(action.position, ()) + (card,)
    occupied_after = occupied_count_in_frame(board, action.frame)
    received = tuple(
        received_card
        for position, stack in board.items()
        if position not in frame_cells
        for received_card in stack
    )
    kept_board = {
        position: stack
        for position, stack in board.items()
        if position in frame_cells
    }
    hand = (
        player.hand[: action.hand_index]
        + player.hand[action.hand_index + 1 :]
    )
    if was_occupied:
        score_delta = 0
    elif occupied_before < 8 <= occupied_after:
        score_delta = 1
    elif occupied_before < 9 <= occupied_after:
        score_delta = 3
    else:
        score_delta = 0
    negative_cards = player.negative_cards + received
    loss_score = max(0, player.loss_score - score_delta)
    board_key = tuple(
        sorted(
            (
                position.x,
                position.y,
                tuple(
                    (placed.color.value, placed.rank_index)
                    for placed in stack
                ),
            )
            for position, stack in kept_board.items()
        )
    )
    public_key = (
        board_key,
        tuple((held.color.value, held.rank_index) for held in hand),
        tuple(
            (negative.color.value, negative.rank_index)
            for negative in negative_cards
        ),
        loss_score,
        state.current_player_index,
        Phase.REFILL.value,
        2,
        state.settlement_count,
    )
    metric = (
        len(negative_cards) - starting_negative_count,
        loss_score - starting_loss_score,
    )
    return metric, public_key


def _exact_legal_positions_by_card(
    board: object, hand: tuple[Card, ...]
) -> tuple[tuple[Position, ...], ...]:
    """Return exact legal positions for each hand card with board facts reused."""
    occupied_columns = {position.x for position in board}
    column_colors = {
        column: colors_in_column(board, column) for column in range(BOARD_SIZE)
    }
    distinct_colors = {card.color for card in hand}
    color_columns = {
        color: columns_containing_color(board, color) for color in distinct_colors
    }
    per_card: list[tuple[Position, ...]] = []
    for card in hand:
        columns = color_columns[card.color]
        candidate_columns = (
            sorted(columns)
            if columns
            else [
                column
                for column in range(BOARD_SIZE)
                if column not in occupied_columns or column_colors[column] == {card.color}
            ]
        )
        per_card.append(
            tuple(
                Position(x=column, y=card.rank_index)
                for column in candidate_columns
                if can_place_card_at(board, card, Position(x=column, y=card.rank_index))
            )
        )
    return tuple(per_card)


def _candidate_positions_for_card(
    board: object,
    card: Card,
    *,
    approximate_new_color_neighbor_limit: bool,
) -> tuple[Position, ...]:
    color_columns = columns_containing_color(board, card.color)
    if color_columns or not approximate_new_color_neighbor_limit:
        candidate_columns = range(BOARD_SIZE)
    else:
        candidate_columns = _approximate_new_color_columns(board)
    return tuple(
        position
        for position in (Position(x=x, y=card.rank_index) for x in candidate_columns)
        if can_place_card_at(board, card, position)
    )


def _approximate_new_color_columns(board: object) -> tuple[int, ...]:
    occupied_columns = sorted({position.x for position in board})
    if not occupied_columns:
        return tuple(range(BOARD_SIZE))
    radius = 2 if len(occupied_columns) == 1 else 1
    return tuple(
        column
        for column in range(BOARD_SIZE)
        if any(abs(column - occupied) <= radius for occupied in occupied_columns)
    )


def _has_zero_negative_two_card_witness(state: GameState) -> bool:
    """Return whether two legal placements can stay inside one 3x3 frame.

    The board is already contained in some 3x3 frame at a turn start. For each
    such frame this checks first and second actions *with that same frame*.
    This is exact: any zero-negative two-card turn has a final frame containing
    the original board and both placements, and that final frame could also
    have been selected for its first placement. It also applies the first card
    before testing the second, preserving color-column rules.
    """
    occupied = set(state.board)
    for frame in all_frames():
        if not occupied <= frame_positions(frame):
            continue
        first_actions = (
            action
            for action in _candidate_actions(
                state,
                approximate_new_color_neighbor_limit=False,
                collapse_equivalent_frames=True,
            )
            if isinstance(action, PlaceCardAction) and action.frame == frame
        )
        for first in first_actions:
            after_first = apply_known_legal_action(state, first)
            if any(
                isinstance(second, PlaceCardAction) and second.frame == frame
                for second in _candidate_actions(
                    after_first,
                    approximate_new_color_neighbor_limit=False,
                    collapse_equivalent_frames=True,
                )
            ):
                return True
    return False


class TorchWinValueEstimator:
    """Batched inference adapter for a saved :func:`build_win_value_net` model."""

    def __init__(self, checkpoint_path: str) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise ImportError("win-value inference requires `pip install -e .[value]`") from error
        from yellowstone.cnn import (
            build_win_value_net,
            win_value_architecture_from_checkpoint,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        architecture = win_value_architecture_from_checkpoint(checkpoint)
        self._torch = torch
        self._model = build_win_value_net(
            convolution_layers=int(architecture["convolution_layers"]),
            hidden_channels=int(architecture["hidden_channels"]),
            hidden_size=int(architecture["hidden_size"]),
            context_size=int(
                checkpoint.get("context_size", VALUE_CONTEXT_SIZE)
            ),
        )
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()
        self._input_canonicalization = checkpoint.get("input_canonicalization")
        self.architecture = architecture

    def __call__(self, record: ValueRecord) -> float:
        return self.estimate_many((record,))[0]

    def estimate_many(self, records: tuple[ValueRecord, ...]) -> tuple[float, ...]:
        """Estimate all candidates together, avoiding per-candidate CNN calls."""
        import numpy as np

        from yellowstone.value_learning import (
            board_tensor_for_player,
            context_tensor_for_player,
        )

        if not records:
            return ()
        board_array = np.stack([board_tensor_for_player(record) for record in records])
        context_array = np.stack(
            [context_tensor_for_player(record) for record in records]
        )
        if self._input_canonicalization is not None:
            from yellowstone.value_canonicalization import (
                CANONICALIZATION_NAME,
                canonicalize_value_tensors,
            )

            if self._input_canonicalization != CANONICALIZATION_NAME:
                raise ValueError(
                    "unsupported value-input canonicalization: "
                    f"{self._input_canonicalization}"
                )
            board_array, context_array = canonicalize_value_tensors(
                board_array, context_array
            )
        with self._torch.no_grad():
            board = self._torch.from_numpy(board_array)
            context = self._torch.from_numpy(context_array)
            values = self._torch.sigmoid(self._model(board, context)).tolist()
        return tuple(float(value) for value in values)


def _apply_with_history(
    state: GameState,
    action: PlaceCardAction,
    history: tuple[RecentPlacement, ...],
) -> tuple[GameState, tuple[RecentPlacement, ...]]:
    player_index = state.current_player_index
    card = state.players[player_index].hand[action.hand_index]
    after = apply_known_legal_action(state, action)
    placement = RecentPlacement(
        player_index=player_index,
        card=card,
        score_delta=(
            state.players[player_index].loss_score - after.players[player_index].loss_score
        ),
        negative_card_delta=(
            len(after.players[player_index].negative_cards)
            - len(state.players[player_index].negative_cards)
        ),
    )
    return after, (*history, placement)[-HISTORY_SIZE:]
