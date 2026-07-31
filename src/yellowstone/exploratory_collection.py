"""Diverse stochastic replay collection for Yellowstone value learning."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from time import monotonic
from typing import Any

from yellowstone.bots import HeuristicBot
from yellowstone.fast_value_npc import (
    NO_REFILL_PROBABILITY,
    _add_totals,
    _append_history,
    _heuristic_representative,
    _mixed_game_seed,
)
from yellowstone.game import (
    apply_known_legal_action,
    create_initial_state,
    legal_actions,
)
from yellowstone.replay_v2 import (
    RULES_VERSION_V2,
    ReplayGameV2,
    file_sha256,
    write_replay_shard,
)
from yellowstone.serialization import action_to_dict
from yellowstone.types import (
    Action,
    GameState,
    Phase,
    PlaceCardAction,
    RefillAction,
    RefillSource,
)
from yellowstone.value_learning import HISTORY_SIZE, RecentPlacement
from yellowstone.value_policy import (
    GroupedTurnActions,
    GroupedTurnCandidates,
    TorchWinValueEstimator,
    TurnCandidate,
    enumerate_best_turn_card_group,
    enumerate_grouped_turn_action_pools,
    enumerate_grouped_turn_pools,
    enumerate_loss_safe_turn_pools,
    materialize_turn_candidate,
    turn_card_group_keys,
)


POLICY_NAME = "diverse_hand_conditioned_exploration_v1"
POLICY_NAME_CARD_FIRST = "diverse_hand_conditioned_exploration_card_first_v2"
ONE_CARD_PROBABILITY_BY_HAND = {6: 0.30, 5: 0.10}
RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY = 0.20
LOW_HAND_NO_REFILL_PROBABILITY = 0.10
EMPTY_HAND_DECK_REFILL_PROBABILITY = 0.10


@dataclass(frozen=True, slots=True)
class ExploratoryTurnChoice:
    actions: tuple[Action, ...]
    selection_mode: str
    branch_probability: float | None
    random_draw: float | None
    selected_cards: tuple[tuple[str, int], ...]
    negative_card_increase: int
    score_bonus: int
    one_group_count: int
    two_group_count: int
    safe_one_group_count: int
    safe_two_group_count: int
    selected_group_outcome_count: int
    baseline_scores: tuple[float, ...]
    baseline_selected_index: int | None
    enumerated_candidate_count: int
    enumeration_seconds: float
    inference_seconds: float
    groups_examined: int = 0
    safe_group_counts_exact: bool = True


class ExploratoryValueNpc:
    """Use deliberate stochastic branches and a V1 baseline fallback."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        card_first: bool = False,
        lazy_single_pass: bool = False,
    ):
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = file_sha256(checkpoint)
        self.estimator = TorchWinValueEstimator(str(checkpoint))
        self.card_first = card_first
        self.lazy_single_pass = lazy_single_pass

    def choose_turn(
        self,
        state: GameState,
        history: tuple[RecentPlacement, ...],
        *,
        rng: Random,
    ) -> ExploratoryTurnChoice:
        if getattr(self, "lazy_single_pass", False):
            return self._choose_turn_lazy_single_pass(
                state, history, rng=rng
            )
        if getattr(self, "card_first", False):
            return self._choose_turn_card_first(state, history, rng=rng)
        started = monotonic()
        pools = enumerate_grouped_turn_pools(
            state,
            history=history,
            approximate_new_color_neighbor_limit=True,
        )
        enumeration_seconds = monotonic() - started
        safe_one = tuple(
            group
            for group in pools.one_card_groups
            if group.negative_card_increase == 0
        )
        safe_two = tuple(
            group
            for group in pools.two_card_groups
            if group.negative_card_increase == 0
        )
        starting_hand_size = len(
            state.players[state.current_player_index].hand
        )
        probability = ONE_CARD_PROBABILITY_BY_HAND.get(
            starting_hand_size
        )
        draw: float | None = None
        if probability is not None and safe_one and safe_two:
            draw = rng.random()
            groups = safe_one if draw < probability else safe_two
            selected_group, selected = _choose_grouped_candidate(
                groups, rng=rng
            )
            mode = (
                "random_safe_one"
                if draw < probability
                else "random_safe_two"
            )
            return _choice(
                selected,
                selected_group,
                mode=mode,
                probability=probability,
                draw=draw,
                pools=pools,
                safe_one=safe_one,
                safe_two=safe_two,
                enumeration_seconds=enumeration_seconds,
            )

        if not safe_one and pools.two_card_groups:
            draw = rng.random()
            if draw < RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY:
                selected_group, selected = _choose_grouped_candidate(
                    pools.two_card_groups, rng=rng
                )
                return _choice(
                    selected,
                    selected_group,
                    mode="random_min_loss_two",
                    probability=(
                        RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY
                    ),
                    draw=draw,
                    pools=pools,
                    safe_one=safe_one,
                    safe_two=safe_two,
                    enumeration_seconds=enumeration_seconds,
                )

        inference_started = monotonic()
        baseline_candidates = _baseline_candidates(state, pools)
        if not baseline_candidates:
            raise RuntimeError("exploratory NPC found no turn candidate")
        scores = self.estimator.estimate_many(
            tuple(candidate.record for candidate in baseline_candidates)
        )
        selected_index = max(
            range(len(baseline_candidates)), key=lambda index: scores[index]
        )
        selected = baseline_candidates[selected_index]
        inference_seconds = monotonic() - inference_started
        selected_group = _find_group_for_candidate(pools, selected)
        return _choice(
            selected,
            selected_group,
            mode="baseline_v1",
            probability=(
                RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY
                if not safe_one and pools.two_card_groups
                else None
            ),
            draw=draw,
            pools=pools,
            safe_one=safe_one,
            safe_two=safe_two,
            enumeration_seconds=enumeration_seconds,
            inference_seconds=inference_seconds,
            baseline_scores=tuple(float(value) for value in scores),
            baseline_selected_index=selected_index,
        )

    def _choose_turn_lazy_single_pass(
        self,
        state: GameState,
        history: tuple[RecentPlacement, ...],
        *,
        rng: Random,
    ) -> ExploratoryTurnChoice:
        started = monotonic()
        pools = enumerate_grouped_turn_action_pools(
            state,
            approximate_new_color_neighbor_limit=True,
        )
        safe_one = tuple(
            group
            for group in pools.one_card_groups
            if group.negative_card_increase == 0
        )
        safe_two = tuple(
            group
            for group in pools.two_card_groups
            if group.negative_card_increase == 0
        )
        starting_hand_size = len(
            state.players[state.current_player_index].hand
        )
        probability = ONE_CARD_PROBABILITY_BY_HAND.get(
            starting_hand_size
        )
        draw: float | None = None
        if probability is not None and safe_one and safe_two:
            draw = rng.random()
            groups = safe_one if draw < probability else safe_two
            action_group = rng.choice(groups)
            actions = rng.choice(action_group.outcomes)
            selected = materialize_turn_candidate(
                state, actions, history=history
            )
            group = _materialized_action_group(action_group, selected)
            return _choice(
                selected,
                group,
                mode=(
                    "random_safe_one"
                    if draw < probability
                    else "random_safe_two"
                ),
                probability=probability,
                draw=draw,
                pools=pools,
                safe_one=safe_one,
                safe_two=safe_two,
                enumeration_seconds=monotonic() - started,
            )

        if not safe_one and pools.two_card_groups:
            draw = rng.random()
            if draw < RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY:
                action_group = rng.choice(pools.two_card_groups)
                actions = rng.choice(action_group.outcomes)
                selected = materialize_turn_candidate(
                    state, actions, history=history
                )
                group = _materialized_action_group(
                    action_group, selected
                )
                return _choice(
                    selected,
                    group,
                    mode="random_min_loss_two",
                    probability=(
                        RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY
                    ),
                    draw=draw,
                    pools=pools,
                    safe_one=safe_one,
                    safe_two=safe_two,
                    enumeration_seconds=monotonic() - started,
                )

        baseline_candidates = []
        for groups in (pools.one_card_groups, pools.two_card_groups):
            if not groups:
                continue
            best_metric = min(
                (group.negative_card_increase, -group.score_bonus)
                for group in groups
            )
            outcomes = tuple(
                actions
                for group in groups
                if (group.negative_card_increase, -group.score_bonus)
                == best_metric
                for actions in group.outcomes
            )
            selected_actions = _heuristic_action_representative(
                state, outcomes
            )
            baseline_candidates.append(
                materialize_turn_candidate(
                    state, selected_actions, history=history
                )
            )
        enumeration_seconds = monotonic() - started
        inference_started = monotonic()
        scores = self.estimator.estimate_many(
            tuple(candidate.record for candidate in baseline_candidates)
        )
        selected_index = max(
            range(len(baseline_candidates)), key=lambda index: scores[index]
        )
        selected = baseline_candidates[selected_index]
        selected_groups = (
            pools.one_card_groups
            if sum(
                isinstance(action, PlaceCardAction)
                for action in selected.actions
            )
            == 1
            else pools.two_card_groups
        )
        action_group = next(
            group
            for group in selected_groups
            if selected.actions in group.outcomes
        )
        group = _materialized_action_group(action_group, selected)
        return _choice(
            selected,
            group,
            mode="baseline_v1",
            probability=(
                RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY
                if not safe_one and pools.two_card_groups
                else None
            ),
            draw=draw,
            pools=pools,
            safe_one=safe_one,
            safe_two=safe_two,
            enumeration_seconds=enumeration_seconds,
            inference_seconds=monotonic() - inference_started,
            baseline_scores=tuple(float(value) for value in scores),
            baseline_selected_index=selected_index,
        )

    def _choose_turn_card_first(
        self,
        state: GameState,
        history: tuple[RecentPlacement, ...],
        *,
        rng: Random,
    ) -> ExploratoryTurnChoice:
        started = monotonic()
        one_keys = turn_card_group_keys(state, play_count=1)
        two_keys = turn_card_group_keys(state, play_count=2)
        safe_one, one_examined, enumerated = _find_random_safe_group(
            state, one_keys, history=history, rng=rng
        )
        starting_hand_size = len(
            state.players[state.current_player_index].hand
        )
        probability = ONE_CARD_PROBABILITY_BY_HAND.get(
            starting_hand_size
        )
        safe_two = None
        two_examined = 0
        if probability is not None and safe_one is not None:
            safe_two, two_examined, two_enumerated = (
                _find_random_safe_group(
                    state, two_keys, history=history, rng=rng
                )
            )
            enumerated += two_enumerated
        draw: float | None = None
        if (
            probability is not None
            and safe_one is not None
            and safe_two is not None
        ):
            draw = rng.random()
            group = safe_one if draw < probability else safe_two
            selected = rng.choice(group.candidates)
            mode = (
                "random_safe_one"
                if draw < probability
                else "random_safe_two"
            )
            return _card_first_choice(
                selected,
                group,
                mode=mode,
                probability=probability,
                draw=draw,
                one_group_count=len(one_keys),
                two_group_count=len(two_keys),
                safe_one_found=True,
                safe_two_found=True,
                enumerated=enumerated,
                groups_examined=one_examined + two_examined,
                enumeration_seconds=monotonic() - started,
            )

        if safe_one is None and two_keys:
            draw = rng.random()
            if draw < RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY:
                group_key = rng.choice(two_keys)
                group, group_enumerated = enumerate_best_turn_card_group(
                    state,
                    group_key,
                    history=history,
                    approximate_new_color_neighbor_limit=True,
                )
                enumerated += group_enumerated
                if group is not None:
                    selected = rng.choice(group.candidates)
                    return _card_first_choice(
                        selected,
                        group,
                        mode="random_min_loss_two",
                        probability=(
                            RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY
                        ),
                        draw=draw,
                        one_group_count=len(one_keys),
                        two_group_count=len(two_keys),
                        safe_one_found=False,
                        safe_two_found=False,
                        enumerated=enumerated,
                        groups_examined=one_examined + 1,
                        enumeration_seconds=monotonic() - started,
                    )

        pools = enumerate_loss_safe_turn_pools(
            state,
            history=history,
            approximate_new_color_neighbor_limit=True,
        )
        enumerated += pools.enumerated_candidate_count
        baseline_candidates = tuple(
            candidate
            for pool in (
                pools.one_card_candidates,
                pools.two_card_candidates,
            )
            if (
                candidate := _heuristic_representative(state, pool)
            )
            is not None
        )
        if not baseline_candidates:
            raise RuntimeError("card-first NPC found no baseline candidate")
        enumeration_seconds = monotonic() - started
        inference_started = monotonic()
        scores = self.estimator.estimate_many(
            tuple(candidate.record for candidate in baseline_candidates)
        )
        selected_index = max(
            range(len(baseline_candidates)), key=lambda index: scores[index]
        )
        selected = baseline_candidates[selected_index]
        group = _group_from_candidate(state, selected)
        return _card_first_choice(
            selected,
            group,
            mode="baseline_v1",
            probability=(
                RANDOM_TWO_WHEN_NO_SAFE_ONE_PROBABILITY
                if safe_one is None and two_keys
                else None
            ),
            draw=draw,
            one_group_count=len(one_keys),
            two_group_count=len(two_keys),
            safe_one_found=safe_one is not None,
            safe_two_found=safe_two is not None,
            enumerated=enumerated,
            groups_examined=one_examined + two_examined,
            enumeration_seconds=enumeration_seconds,
            inference_seconds=monotonic() - inference_started,
            baseline_scores=tuple(float(value) for value in scores),
            baseline_selected_index=selected_index,
        )


def _choice(
    selected: TurnCandidate,
    group: GroupedTurnCandidates,
    *,
    mode: str,
    probability: float | None,
    draw: float | None,
    pools,
    safe_one,
    safe_two,
    enumeration_seconds: float,
    inference_seconds: float = 0.0,
    baseline_scores: tuple[float, ...] = (),
    baseline_selected_index: int | None = None,
) -> ExploratoryTurnChoice:
    return ExploratoryTurnChoice(
        actions=selected.actions,
        selection_mode=mode,
        branch_probability=probability,
        random_draw=draw,
        selected_cards=tuple(
            (card.color.value, card.rank_index) for card in group.cards
        ),
        negative_card_increase=group.negative_card_increase,
        score_bonus=group.score_bonus,
        one_group_count=len(pools.one_card_groups),
        two_group_count=len(pools.two_card_groups),
        safe_one_group_count=len(safe_one),
        safe_two_group_count=len(safe_two),
        selected_group_outcome_count=len(group.candidates),
        baseline_scores=baseline_scores,
        baseline_selected_index=baseline_selected_index,
        enumerated_candidate_count=pools.enumerated_candidate_count,
        enumeration_seconds=enumeration_seconds,
        inference_seconds=inference_seconds,
    )


def _materialized_action_group(
    group: GroupedTurnActions, selected: TurnCandidate
) -> GroupedTurnCandidates:
    return GroupedTurnCandidates(
        cards=group.cards,
        negative_card_increase=group.negative_card_increase,
        score_bonus=group.score_bonus,
        candidates=(selected,) * len(group.outcomes),
    )


def _heuristic_action_representative(
    state: GameState,
    outcomes: tuple[tuple[Action, ...], ...],
) -> tuple[Action, ...]:
    if not outcomes:
        raise ValueError("heuristic action selection requires outcomes")

    def key(actions: tuple[Action, ...]) -> tuple[tuple[int, ...], ...]:
        working = state
        placement_keys = []
        for action in actions:
            if isinstance(action, PlaceCardAction):
                from yellowstone.bots import placement_sort_key

                placement_keys.append(
                    placement_sort_key(working, action)
                )
            working = apply_known_legal_action(working, action)
        return tuple(placement_keys)

    return min(outcomes, key=key)


def _card_first_choice(
    selected: TurnCandidate,
    group: GroupedTurnCandidates,
    *,
    mode: str,
    probability: float | None,
    draw: float | None,
    one_group_count: int,
    two_group_count: int,
    safe_one_found: bool,
    safe_two_found: bool,
    enumerated: int,
    groups_examined: int,
    enumeration_seconds: float,
    inference_seconds: float = 0.0,
    baseline_scores: tuple[float, ...] = (),
    baseline_selected_index: int | None = None,
) -> ExploratoryTurnChoice:
    return ExploratoryTurnChoice(
        actions=selected.actions,
        selection_mode=mode,
        branch_probability=probability,
        random_draw=draw,
        selected_cards=tuple(
            (card.color.value, card.rank_index) for card in group.cards
        ),
        negative_card_increase=group.negative_card_increase,
        score_bonus=group.score_bonus,
        one_group_count=one_group_count,
        two_group_count=two_group_count,
        safe_one_group_count=int(safe_one_found),
        safe_two_group_count=int(safe_two_found),
        selected_group_outcome_count=len(group.candidates),
        baseline_scores=baseline_scores,
        baseline_selected_index=baseline_selected_index,
        enumerated_candidate_count=enumerated,
        enumeration_seconds=enumeration_seconds,
        inference_seconds=inference_seconds,
        groups_examined=groups_examined,
        safe_group_counts_exact=False,
    )


def _find_random_safe_group(
    state: GameState,
    groups: tuple[tuple[Any, ...], ...],
    *,
    history: tuple[RecentPlacement, ...],
    rng: Random,
) -> tuple[GroupedTurnCandidates | None, int, int]:
    shuffled = list(groups)
    rng.shuffle(shuffled)
    enumerated = 0
    for index, cards in enumerate(shuffled, start=1):
        group, group_enumerated = enumerate_best_turn_card_group(
            state,
            cards,
            history=history,
            approximate_new_color_neighbor_limit=True,
        )
        enumerated += group_enumerated
        if group is not None and group.negative_card_increase == 0:
            return group, index, enumerated
    return None, len(shuffled), enumerated


def _group_from_candidate(
    state: GameState, candidate: TurnCandidate
) -> GroupedTurnCandidates:
    working = state
    cards = []
    for action in candidate.actions:
        if isinstance(action, PlaceCardAction):
            cards.append(
                working.players[working.current_player_index].hand[
                    action.hand_index
                ]
            )
        working = apply_known_legal_action(working, action)
    player_index = state.current_player_index
    starting = state.players[player_index]
    ending = candidate.record.state.players[player_index]
    return GroupedTurnCandidates(
        cards=tuple(
            sorted(
                cards, key=lambda card: (card.color.value, card.rank_index)
            )
        ),
        negative_card_increase=(
            len(ending.negative_cards) - len(starting.negative_cards)
        ),
        score_bonus=starting.loss_score - ending.loss_score,
        candidates=(candidate,),
    )


def _choose_grouped_candidate(
    groups: tuple[GroupedTurnCandidates, ...], *, rng: Random
) -> tuple[GroupedTurnCandidates, TurnCandidate]:
    group = rng.choice(groups)
    return group, rng.choice(group.candidates)


def _baseline_candidates(state: GameState, pools) -> tuple[TurnCandidate, ...]:
    result: list[TurnCandidate] = []
    for groups in (pools.one_card_groups, pools.two_card_groups):
        if not groups:
            continue
        best_metric = min(
            (group.negative_card_increase, -group.score_bonus)
            for group in groups
        )
        candidates = tuple(
            candidate
            for group in groups
            if (group.negative_card_increase, -group.score_bonus)
            == best_metric
            for candidate in group.candidates
        )
        representative = _heuristic_representative(state, candidates)
        if representative is not None:
            result.append(representative)
    return tuple(result)


def _find_group_for_candidate(
    pools, candidate: TurnCandidate
) -> GroupedTurnCandidates:
    for groups in (pools.one_card_groups, pools.two_card_groups):
        for group in groups:
            if candidate in group.candidates:
                return group
    raise AssertionError("selected candidate is not in grouped pools")


def choose_exploratory_refill(
    state: GameState, *, rng: Random
) -> tuple[RefillAction, dict[str, Any]]:
    """Choose the hand-size-conditioned refill branch."""
    legal_refills = tuple(
        action
        for action in legal_actions(state)
        if isinstance(action, RefillAction)
    )
    if not legal_refills:
        raise RuntimeError("refill choice requested without legal refill")
    player = state.players[state.current_player_index]
    deck_action = RefillAction(RefillSource.DECK)
    negative_action = RefillAction(RefillSource.NEGATIVE_CARDS)
    none_action = RefillAction(RefillSource.NONE)

    if not player.hand:
        if negative_action not in legal_refills:
            return deck_action, {
                "type": "refill",
                "refill_policy": "empty_hand_forced_deck",
                "eligible_no_refill": False,
                "deck_probability": 1.0,
                "random_draw": None,
                "selected_source": RefillSource.DECK.value,
            }
        draw = rng.random()
        selected = (
            deck_action
            if draw < EMPTY_HAND_DECK_REFILL_PROBABILITY
            else negative_action
        )
        return selected, {
            "type": "refill",
            "refill_policy": "empty_hand_deck_vs_negative",
            "eligible_no_refill": False,
            "deck_probability": EMPTY_HAND_DECK_REFILL_PROBABILITY,
            "random_draw": draw,
            "selected_source": selected.source.value,
        }

    if none_action not in legal_refills:
        raise AssertionError("non-empty refill state must allow NONE")
    starting_hand_size = len(player.hand) + state.cards_played_this_turn
    probability = (
        LOW_HAND_NO_REFILL_PROBABILITY
        if state.cards_played_this_turn == 2 and starting_hand_size <= 4
        else NO_REFILL_PROBABILITY
    )
    draw = rng.random()
    selected = none_action if draw < probability else deck_action
    return selected, {
        "type": "refill",
        "refill_policy": (
            "low_starting_hand_none_vs_deck"
            if probability == LOW_HAND_NO_REFILL_PROBABILITY
            else "standard_none_vs_deck"
        ),
        "eligible_no_refill": True,
        "starting_hand_size": starting_hand_size,
        "no_refill_probability": probability,
        "random_draw": draw,
        "selected_source": selected.source.value,
    }


def play_one_exploratory_game(
    npc: ExploratoryValueNpc,
    *,
    game_id: int,
    seed: int,
) -> tuple[ReplayGameV2, dict[str, float]]:
    seed_rng = Random(_mixed_game_seed(seed, game_id))
    initial_seed = seed_rng.randrange(2**63)
    gameplay_seed = seed_rng.randrange(2**63)
    decision_seed = _mixed_game_seed(
        initial_seed ^ gameplay_seed ^ seed, game_id
    )
    state = create_initial_state(4, seed=initial_seed)
    initial_state = state
    gameplay_rng = Random(gameplay_seed)
    decision_rng = Random(decision_seed)
    heuristic = HeuristicBot()
    history: list[RecentPlacement] = []
    actions: list[Action] = []
    decisions: list[dict[str, Any]] = []
    planned: list[Action] = []
    planned_player: int | None = None
    totals: dict[str, float] = {}

    while state.phase != Phase.GAME_OVER:
        player_index = state.current_player_index
        if planned:
            if player_index != planned_player:
                raise AssertionError("planned player changed")
            action = planned.pop(0)
        elif (
            state.phase == Phase.PLAY
            and state.cards_played_this_turn == 0
            and any(
                isinstance(item, PlaceCardAction)
                for item in legal_actions(state)
            )
        ):
            starting_hand_size = len(state.players[player_index].hand)
            choice = npc.choose_turn(
                state, tuple(history[-HISTORY_SIZE:]), rng=decision_rng
            )
            planned = list(choice.actions)
            planned_player = player_index
            action = planned.pop(0)
            play_count = sum(
                isinstance(item, PlaceCardAction)
                for item in choice.actions
            )
            totals["npc_turns"] = totals.get("npc_turns", 0.0) + 1
            totals["one_card_turns"] = totals.get(
                "one_card_turns", 0.0
            ) + int(play_count == 1)
            totals["two_card_turns"] = totals.get(
                "two_card_turns", 0.0
            ) + int(play_count == 2)
            hand_key = f"{play_count}_card_start_hand_{starting_hand_size}"
            totals[hand_key] = totals.get(hand_key, 0.0) + 1
            mode_key = f"selection_{choice.selection_mode}"
            totals[mode_key] = totals.get(mode_key, 0.0) + 1
            mode_hand_key = (
                f"selection_{choice.selection_mode}"
                f"_start_hand_{starting_hand_size}"
            )
            totals[mode_hand_key] = totals.get(mode_hand_key, 0.0) + 1
            if (
                starting_hand_size in ONE_CARD_PROBABILITY_BY_HAND
                and choice.safe_one_group_count
                and choice.safe_two_group_count
            ):
                eligible_key = (
                    "eligible_safe_one_and_two"
                    f"_start_hand_{starting_hand_size}"
                )
                totals[eligible_key] = totals.get(eligible_key, 0.0) + 1
            if (
                choice.safe_one_group_count == 0
                and choice.two_group_count > 0
            ):
                totals["eligible_no_safe_one_with_two"] = totals.get(
                    "eligible_no_safe_one_with_two", 0.0
                ) + 1
            totals["enumeration_seconds"] = totals.get(
                "enumeration_seconds", 0.0
            ) + choice.enumeration_seconds
            totals["inference_seconds"] = totals.get(
                "inference_seconds", 0.0
            ) + choice.inference_seconds
            totals["enumerated_candidate_count"] = totals.get(
                "enumerated_candidate_count", 0.0
            ) + choice.enumerated_candidate_count
            decisions.append(
                {
                    "type": "turn",
                    "game_id": game_id,
                    "player_index": player_index,
                    "starting_hand_size": starting_hand_size,
                    "selection_mode": choice.selection_mode,
                    "branch_probability": choice.branch_probability,
                    "random_draw": choice.random_draw,
                    "selected_cards": list(choice.selected_cards),
                    "negative_card_increase": (
                        choice.negative_card_increase
                    ),
                    "score_bonus": choice.score_bonus,
                    "one_group_count": choice.one_group_count,
                    "two_group_count": choice.two_group_count,
                    "safe_one_group_count": choice.safe_one_group_count,
                    "safe_two_group_count": choice.safe_two_group_count,
                    "safe_group_counts_exact": (
                        choice.safe_group_counts_exact
                    ),
                    "groups_examined": choice.groups_examined,
                    "selected_group_outcome_count": (
                        choice.selected_group_outcome_count
                    ),
                    "baseline_scores": list(choice.baseline_scores),
                    "baseline_selected_index": (
                        choice.baseline_selected_index
                    ),
                    "enumerated_candidate_count": (
                        choice.enumerated_candidate_count
                    ),
                    "selected_actions": [
                        action_to_dict(item) for item in choice.actions
                    ],
                    "enumeration_seconds": choice.enumeration_seconds,
                    "inference_seconds": choice.inference_seconds,
                }
            )
        elif (
            state.phase == Phase.REFILL
            or not any(
                isinstance(item, PlaceCardAction)
                for item in legal_actions(state)
            )
        ):
            action, audit = choose_exploratory_refill(
                state, rng=decision_rng
            )
            totals["refill_decisions"] = totals.get(
                "refill_decisions", 0.0
            ) + 1
            refill_key = f"refill_{audit['selected_source']}"
            totals[refill_key] = totals.get(refill_key, 0.0) + 1
            policy_key = f"refill_policy_{audit['refill_policy']}"
            totals[policy_key] = totals.get(policy_key, 0.0) + 1
            decisions.append(
                {"game_id": game_id, "player_index": player_index, **audit}
            )
        else:
            action = heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("heuristic returned no action")

        if action not in legal_actions(state):
            raise RuntimeError(
                f"exploratory NPC selected illegal action: {action!r}"
            )
        before = state
        state = apply_known_legal_action(
            state, action, rng=gameplay_rng
        )
        actions.append(action)
        _append_history(history, before, action, state)
        if (
            before.current_player_index != state.current_player_index
            or state.phase == Phase.GAME_OVER
        ):
            planned = []
            planned_player = None

    replay = ReplayGameV2(
        game_id=game_id,
        initial_seed=initial_seed,
        gameplay_seed=gameplay_seed,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=tuple(decisions),
        winners=state.winners,
        teacher_checkpoint=npc.checkpoint,
        teacher_sha256=npc.checkpoint_sha256,
        teacher_generation=0,
        privileged_teacher_deck=False,
        rules_version=RULES_VERSION_V2,
    )
    for player_index in range(4):
        totals[f"win_player_{player_index}"] = (
            1.0 / len(state.winners)
            if player_index in state.winners
            else 0.0
        )
    return replay, totals


def collect_exploratory(
    checkpoint: str | Path,
    *,
    seed: int,
    game_id_offset: int,
    output: str | Path,
    stop_file: str | Path,
    status_file: str | Path | None = None,
    shard_games: int = 100,
    max_games: int | None = None,
    card_first: bool = False,
    lazy_single_pass: bool = False,
) -> dict[str, Any]:
    if shard_games <= 0 or max_games is not None and max_games <= 0:
        raise ValueError("shard_games and max_games must be positive")
    npc = ExploratoryValueNpc(
        checkpoint,
        card_first=card_first,
        lazy_single_pass=lazy_single_pass,
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    stop_path = Path(stop_file)
    status_path = Path(status_file) if status_file is not None else None
    manifest_path = output_path / "collection_manifest.json"
    checkpoint_hash = file_sha256(checkpoint)
    expected = {
        "collector": (
            POLICY_NAME_CARD_FIRST if card_first else POLICY_NAME
        ),
        "seed": seed,
        "game_id_offset": game_id_offset,
        "shard_games": shard_games,
        "max_games": max_games,
        "checkpoint_sha256": checkpoint_hash,
    }
    if card_first:
        expected["card_first"] = True
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"manifest differs at {key}")
        games = int(manifest["games"])
        completed_shards = int(manifest["completed_shards"])
        compressed_bytes = int(manifest["compressed_bytes"])
        prior_wall = float(manifest["wall_seconds"])
        totals = {
            key: float(value)
            for key, value in manifest.get("policy_totals", {}).items()
        }
        started_at = str(manifest["started_at"])
    else:
        games = completed_shards = compressed_bytes = 0
        prior_wall = 0.0
        totals: dict[str, float] = {}
        started_at = datetime.now().astimezone().isoformat()

    run_started = monotonic()
    stopped = stop_path.exists()

    def persist(status: str) -> dict[str, Any]:
        wall = prior_wall + monotonic() - run_started
        payload = {
            "schema": "yellowstone.replay.v2.exploratory_collection",
            **expected,
            "rules_version": RULES_VERSION_V2,
            "checkpoint": str(checkpoint),
            "stop_file": str(stop_path),
            "started_at": started_at,
            "updated_at": datetime.now().astimezone().isoformat(),
            "wall_seconds": wall,
            "games": games,
            "completed_shards": completed_shards,
            "compressed_bytes": compressed_bytes,
            "policy_totals": totals,
            "status": status,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        if status_path is not None:
            _write_collection_status(
                status_path,
                state=status,
                output=output_path,
                checkpoint=checkpoint,
                stop_file=stop_path,
                games=games,
                completed_shards=completed_shards,
                message="",
            )
        print(
            json.dumps(
                {
                    "games": games,
                    "wall_seconds": wall,
                    "status": status,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return payload

    already_complete = max_games is not None and games >= max_games
    manifest = persist(
        "stopped_by_user"
        if stopped
        else "complete"
        if already_complete
        else "running"
    )
    while not stopped and (max_games is None or games < max_games):
        shard_start = game_id_offset + games
        remaining = (
            shard_games
            if max_games is None
            else min(shard_games, max_games - games)
        )
        shard: list[ReplayGameV2] = []
        for index in range(remaining):
            if stop_path.exists():
                stopped = True
                break
            replay, facts = play_one_exploratory_game(
                npc,
                game_id=shard_start + index,
                seed=seed,
            )
            shard.append(replay)
            _add_totals(totals, facts)
        if shard:
            destination = (
                output_path / f"part_{shard_start:07d}.jsonl.gz"
            )
            if destination.exists():
                raise FileExistsError(f"shard exists: {destination}")
            storage = write_replay_shard(shard, destination)
            games += len(shard)
            completed_shards += 1
            compressed_bytes += int(storage["compressed_bytes"])
        complete = max_games is not None and games >= max_games
        status = (
            "stopped_by_user"
            if stopped
            else "complete"
            if complete
            else "running"
        )
        manifest = persist(status)
    return manifest


def _write_collection_status(
    path: Path,
    *,
    state: str,
    output: Path,
    checkpoint: str | Path,
    stop_file: Path,
    games: int,
    completed_shards: int,
    message: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "step": "collect",
        "last_completed_step": (
            "write_shard" if completed_shards else ""
        ),
        "message": message,
        "updated_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "output": str(output),
        "checkpoint": str(checkpoint),
        "stop_file": str(stop_file),
        "games": games,
        "completed_shards": completed_shards,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--game-id-offset", type=int, default=1_100_912)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--shard-games", type=int, default=100)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--card-first", action="store_true")
    parser.add_argument("--lazy-single-pass", action="store_true")
    args = parser.parse_args()
    result = collect_exploratory(
        args.checkpoint,
        seed=args.seed,
        game_id_offset=args.game_id_offset,
        output=args.output,
        stop_file=args.stop_file,
        status_file=args.status_file,
        shard_games=args.shard_games,
        max_games=args.max_games,
        card_first=args.card_first,
        lazy_single_pass=args.lazy_single_pass,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
