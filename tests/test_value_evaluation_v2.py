from dataclasses import replace

from yellowstone.game import create_initial_state
from yellowstone.value_evaluation_v2 import choose_v2_turn
from yellowstone.value_v2 import (
    CompletedTurnTracker,
    PendingRefillSource,
    PublicNegativeKnowledgeTracker,
)


class _PendingSourceEstimator:
    def estimate_many(self, records):
        return tuple(
            {
                PendingRefillSource.NO_PENDING: 0.1,
                PendingRefillSource.NONE: 0.2,
                PendingRefillSource.DECK: 0.9,
                PendingRefillSource.NEGATIVE_CARDS: 0.8,
            }[record.pending_refill_source]
            for record in records
        )


def test_choose_v2_turn_builds_viewer_safe_frame_and_refill_records() -> None:
    state = create_initial_state(4, seed=7)
    history = CompletedTurnTracker()
    knowledge = PublicNegativeKnowledgeTracker(4)

    choice = choose_v2_turn(
        state,
        _PendingSourceEstimator(),
        history=history,
        negative_knowledge=knowledge.snapshot(),
        prune_negative_card_increase_above=None,
        approximate_new_color_neighbor_limit=True,
    )

    assert choice.actions
    assert choice.record.perspective_player_index == 0
    assert choice.record.history_before_turn == ()
    assert choice.record.candidate_frame.start_frame is None
    assert choice.record.candidate_frame.start_board_card_count == 1
    assert choice.record.pending_refill_source in (
        PendingRefillSource.NO_PENDING,
        PendingRefillSource.DECK,
    )


def test_choose_v2_turn_excludes_unsupported_no_refill_choice() -> None:
    state = create_initial_state(4, seed=7)
    state = replace(
        state,
        players=(
            replace(state.players[0], hand=state.players[0].hand[:2]),
            *state.players[1:],
        ),
    )

    choice = choose_v2_turn(
        state,
        _PendingSourceEstimator(),
        history=CompletedTurnTracker(),
        negative_knowledge=PublicNegativeKnowledgeTracker(4).snapshot(),
        prune_negative_card_increase_above=None,
        approximate_new_color_neighbor_limit=True,
    )

    assert choice.record.pending_refill_source != PendingRefillSource.NONE
