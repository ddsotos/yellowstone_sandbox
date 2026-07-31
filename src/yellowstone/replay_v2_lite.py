"""Reconstruct transition-aware V2-lite records from existing V2 replays."""

from __future__ import annotations

from random import Random

from yellowstone.game import apply_known_legal_action
from yellowstone.replay_v2 import (
    LEGACY_RULES_VERSION_V2,
    ReplayGameV2,
)
from yellowstone.types import EndTurnAction, Phase, PlaceCardAction, RefillAction
from yellowstone.value_v2 import CompletedTurnTracker, PendingRefillSource
from yellowstone.value_v2_lite import (
    HISTORY_TURNS_V2_LITE,
    ValueRecordV2Lite,
)


def records_from_replay_v2_lite(
    game: ReplayGameV2,
) -> tuple[ValueRecordV2Lite, ...]:
    """Reconstruct post-candidate records and their exact pre-play states."""
    state = game.initial_state
    rng = Random(game.gameplay_seed)
    history = CompletedTurnTracker()
    records: list[tuple[int, object, object, object, PendingRefillSource]] = []
    turn_history_before = None
    turn_player = None
    turn_state_before = None

    for action in game.actions:
        before = state
        if (
            isinstance(action, PlaceCardAction)
            and before.cards_played_this_turn == 0
        ):
            turn_history_before = history.snapshot()[-HISTORY_TURNS_V2_LITE:]
            turn_player = before.current_player_index
            turn_state_before = before

        state = apply_known_legal_action(
            state,
            action,
            rng=rng,
            settle_on_empty_deck=(
                game.rules_version != LEGACY_RULES_VERSION_V2
            ),
        )
        history.observe(before, action, state)

        if isinstance(action, EndTurnAction) and state.phase != Phase.REFILL:
            if (
                turn_history_before is None
                or turn_player is None
                or turn_state_before is None
            ):
                raise AssertionError("one-card completion lacks turn start")
            records.append(
                (
                    turn_player,
                    turn_state_before,
                    state,
                    turn_history_before,
                    PendingRefillSource.NO_PENDING,
                )
            )
            turn_history_before = None
            turn_player = None
            turn_state_before = None
        elif isinstance(action, RefillAction) and turn_player is not None:
            if turn_history_before is None or turn_state_before is None:
                raise AssertionError("refill completion lacks turn start")
            records.append(
                (
                    turn_player,
                    turn_state_before,
                    before,
                    turn_history_before,
                    PendingRefillSource(action.source.value),
                )
            )
            turn_history_before = None
            turn_player = None
            turn_state_before = None

    if state.phase != Phase.GAME_OVER:
        raise ValueError("cannot label an unfinished replay")
    if not state.winners:
        raise AssertionError("finished replay has no winners")
    winner_count = len(state.winners)
    return tuple(
        ValueRecordV2Lite(
            game_id=game.game_id,
            perspective_player_index=player_index,
            state_before_turn=pre_state,
            state=record_state,
            history_before_turn=record_history,
            pending_refill_source=pending,
            target=(
                1.0 / winner_count if player_index in state.winners else 0.0
            ),
        )
        for (
            player_index,
            pre_state,
            record_state,
            record_history,
            pending,
        ) in records
    )

