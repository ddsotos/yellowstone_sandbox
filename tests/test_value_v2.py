from itertools import permutations

import numpy as np

from yellowstone.symmetry import transform_state
from yellowstone.types import (
    Card,
    Color,
    EndTurnAction,
    Frame,
    GameState,
    Phase,
    PlaceCardAction,
    PlayerState,
    Position,
    RefillAction,
    RefillSource,
)
from yellowstone.value_v2 import (
    BOARD_CHANNELS_V2,
    CandidateFrameContext,
    COLOR_ORDER,
    CompletedTurn,
    CompletedTurnTracker,
    PendingRefillSource,
    PublicNegativeKnowledge,
    PublicNegativeKnowledgeTracker,
    PublicNegativePile,
    RefillResult,
    VALUE_CONTEXT_SIZE_V2,
    ValueRecordV2,
    canonical_tensors_v2,
)


def _knowledge() -> PublicNegativeKnowledge:
    return PublicNegativeKnowledge(
        (
            PublicNegativePile((0.0,) * 7, (0.0,) * 4, True),
            PublicNegativePile(
                (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                True,
            ),
            PublicNegativePile(
                (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                (0.5, 0.0, 0.5, 0.0),
                False,
            ),
            PublicNegativePile((0.0,) * 7, (0.0,) * 4, True),
        )
    )


def _record(
    *,
    history: tuple[CompletedTurn, ...] | None = None,
    candidate_frame: CandidateFrameContext | None = None,
) -> ValueRecordV2:
    state = GameState(
        players=(
            PlayerState(
                hand=(
                    Card(Color.GREEN, 1),
                    Card(Color.BLUE, 5),
                ),
                negative_cards=(Card(Color.YELLOW, 3),),
                loss_score=4,
            ),
            PlayerState(hand=(Card(Color.RED, 0),), loss_score=6),
            PlayerState(hand=(), loss_score=7),
            PlayerState(hand=(Card(Color.BLUE, 6),), loss_score=8),
        ),
        board={
            Position(1, 1): (Card(Color.GREEN, 1),),
            Position(2, 3): (Card(Color.RED, 3), Card(Color.RED, 3)),
            Position(4, 5): (Card(Color.BLUE, 5),),
        },
        deck=(Card(Color.YELLOW, 0),) * 8,
        current_player_index=0,
        phase=Phase.REFILL,
        cards_played_this_turn=2,
        settlement_count=1,
    )
    if history is None:
        history = (
            CompletedTurn(
                player_index=2,
                cards=(Card(Color.RED, 0),),
                start_frame=Frame(0, 1),
                end_frame=Frame(1, 1),
                start_board_card_count=4,
                score_delta=1,
                negative_card_delta=0,
                settlement_occurred=False,
                refill_result=RefillResult.NOT_OFFERED,
            ),
            CompletedTurn(
                player_index=3,
                cards=(Card(Color.GREEN, 1), Card(Color.BLUE, 1)),
                start_frame=Frame(1, 1),
                end_frame=Frame(3, 2),
                start_board_card_count=5,
                score_delta=0,
                negative_card_delta=1,
                settlement_occurred=True,
                refill_result=RefillResult.DECK,
            ),
        )
    if candidate_frame is None:
        candidate_frame = CandidateFrameContext(
            start_frame=Frame(3, 2),
            end_frame=Frame(2, 0),
            start_board_card_count=6,
        )
    return ValueRecordV2(
        game_id=10,
        perspective_player_index=0,
        state=state,
        history_before_turn=history,
        candidate_frame=candidate_frame,
        negative_knowledge=_knowledge(),
        pending_refill_source=PendingRefillSource.DECK,
        target=1.0,
    )


def _transform_record(record, color_map, horizontal, vertical):
    def transform_card(card):
        return Card(
            color_map[card.color],
            6 - card.rank_index if vertical else card.rank_index,
        )

    def transform_frame(frame):
        if frame is None:
            return None
        return Frame(
            4 - frame.x if horizontal else frame.x,
            4 - frame.y if vertical else frame.y,
        )

    history = tuple(
        CompletedTurn(
            player_index=turn.player_index,
            cards=tuple(transform_card(card) for card in turn.cards),
            start_frame=transform_frame(turn.start_frame),
            end_frame=transform_frame(turn.end_frame),
            start_board_card_count=turn.start_board_card_count,
            score_delta=turn.score_delta,
            negative_card_delta=turn.negative_card_delta,
            settlement_occurred=turn.settlement_occurred,
            refill_result=turn.refill_result,
        )
        for turn in record.history_before_turn
    )
    old_to_new = {
        COLOR_ORDER.index(old): COLOR_ORDER.index(new)
        for old, new in color_map.items()
    }
    piles = []
    for pile in record.negative_knowledge.piles:
        ranks = (
            tuple(reversed(pile.rank_expected))
            if vertical
            else pile.rank_expected
        )
        colors = [0.0] * 4
        for old_index, value in enumerate(pile.color_expected):
            colors[old_to_new[old_index]] = value
        piles.append(PublicNegativePile(ranks, tuple(colors), pile.exact))
    return ValueRecordV2(
        game_id=record.game_id,
        perspective_player_index=record.perspective_player_index,
        state=transform_state(
            record.state,
            color_map=color_map,
            horizontal_reflection=horizontal,
            vertical_reflection=vertical,
        ),
        history_before_turn=history,
        candidate_frame=CandidateFrameContext(
            start_frame=transform_frame(record.candidate_frame.start_frame),
            end_frame=transform_frame(record.candidate_frame.end_frame),
            start_board_card_count=record.candidate_frame.start_board_card_count,
        ),
        negative_knowledge=PublicNegativeKnowledge(tuple(piles)),
        pending_refill_source=record.pending_refill_source,
        target=record.target,
    )


def test_all_96_visible_symmetries_collapse_exactly() -> None:
    record = _record()
    expected_board, expected_context, _ = canonical_tensors_v2(record)
    for permuted in permutations(COLOR_ORDER):
        color_map = dict(zip(COLOR_ORDER, permuted, strict=True))
        for horizontal in (False, True):
            for vertical in (False, True):
                transformed = _transform_record(
                    record, color_map, horizontal, vertical
                )
                board, context, _ = canonical_tensors_v2(transformed)
                assert np.array_equal(board, expected_board)
                assert np.array_equal(context, expected_context)


def test_cards_inside_history_turn_are_unordered() -> None:
    first = _record()
    old_turn = first.history_before_turn[-1]
    swapped = _record(
        history=(
            first.history_before_turn[0],
            CompletedTurn(
                player_index=old_turn.player_index,
                cards=tuple(reversed(old_turn.cards)),
                start_frame=old_turn.start_frame,
                end_frame=old_turn.end_frame,
                start_board_card_count=old_turn.start_board_card_count,
                score_delta=old_turn.score_delta,
                negative_card_delta=old_turn.negative_card_delta,
                settlement_occurred=old_turn.settlement_occurred,
                refill_result=old_turn.refill_result,
            ),
        )
    )
    first_board, first_context, _ = canonical_tensors_v2(first)
    second_board, second_context, _ = canonical_tensors_v2(swapped)
    assert np.array_equal(first_board, second_board)
    assert np.array_equal(first_context, second_context)


def test_absent_history_differs_from_real_one_card_zero_delta_turn() -> None:
    absent = _record(history=())
    present = _record(
        history=(
            CompletedTurn(
                player_index=3,
                cards=(Card(Color.BLUE, 1),),
                start_frame=None,
                end_frame=Frame(1, 2),
                start_board_card_count=1,
                score_delta=0,
                negative_card_delta=0,
                settlement_occurred=False,
                refill_result=RefillResult.NOT_OFFERED,
            ),
        )
    )
    _, absent_context, _ = canonical_tensors_v2(absent)
    _, present_context, _ = canonical_tensors_v2(present)
    assert not np.array_equal(absent_context, present_context)


def test_candidate_frame_movement_can_split_the_same_resulting_state() -> None:
    first = _record(
        candidate_frame=CandidateFrameContext(
            start_frame=Frame(2, 2),
            end_frame=Frame(2, 2),
            start_board_card_count=6,
        )
    )
    moved = _record(
        candidate_frame=CandidateFrameContext(
            start_frame=Frame(0, 2),
            end_frame=Frame(2, 2),
            start_board_card_count=6,
        )
    )
    first_board, first_context, _ = canonical_tensors_v2(first)
    moved_board, moved_context, _ = canonical_tensors_v2(moved)
    assert np.array_equal(first_board, moved_board)
    assert not np.array_equal(first_context, moved_context)


def test_missing_candidate_start_frame_differs_from_known_start() -> None:
    missing = _record(
        candidate_frame=CandidateFrameContext(
            start_frame=None,
            end_frame=Frame(2, 2),
            start_board_card_count=1,
        )
    )
    known = _record(
        candidate_frame=CandidateFrameContext(
            start_frame=Frame(2, 2),
            end_frame=Frame(2, 2),
            start_board_card_count=1,
        )
    )
    _, missing_context, _ = canonical_tensors_v2(missing)
    _, known_context, _ = canonical_tensors_v2(known)
    assert not np.array_equal(missing_context, known_context)


def test_completed_turn_tracker_represents_one_card_turn() -> None:
    tracker = CompletedTurnTracker()
    card = Card(Color.GREEN, 2)
    before = GameState(
        players=(
            PlayerState(hand=(card, Card(Color.RED, 4))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        current_player_index=0,
    )
    place = PlaceCardAction(0, Position(1, 2), Frame(0, 0))
    after_place = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 4),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={Position(1, 2): (card,)},
        current_player_index=0,
        cards_played_this_turn=1,
    )
    after_end = GameState(
        players=after_place.players,
        board=after_place.board,
        current_player_index=1,
        cards_played_this_turn=0,
    )
    tracker.observe(before, place, after_place)
    tracker.observe(after_place, EndTurnAction(), after_end)
    assert tracker.snapshot() == (
        CompletedTurn(
            player_index=0,
            cards=(card,),
            start_frame=None,
            end_frame=Frame(0, 0),
            start_board_card_count=0,
            score_delta=0,
            negative_card_delta=0,
            settlement_occurred=False,
            refill_result=RefillResult.NOT_OFFERED,
        ),
    )


def test_completed_turn_tracker_carries_frame_and_counts_stacked_cards() -> None:
    tracker = CompletedTurnTracker()
    first_card = Card(Color.GREEN, 2)
    second_card = Card(Color.BLUE, 3)
    players_before = (
        PlayerState(hand=(first_card,)),
        PlayerState(hand=(second_card,)),
        PlayerState(),
        PlayerState(),
    )
    first_before = GameState(
        players=players_before,
        board={Position(3, 3): (Card(Color.RED, 3), Card(Color.RED, 3))},
        current_player_index=0,
    )
    first_place = PlaceCardAction(0, Position(2, 2), Frame(1, 1))
    first_after_place = GameState(
        players=(PlayerState(), *players_before[1:]),
        board={
            Position(3, 3): (Card(Color.RED, 3), Card(Color.RED, 3)),
            Position(2, 2): (first_card,),
        },
        current_player_index=0,
        cards_played_this_turn=1,
    )
    first_after_end = GameState(
        players=first_after_place.players,
        board=first_after_place.board,
        current_player_index=1,
        cards_played_this_turn=0,
    )
    tracker.observe(first_before, first_place, first_after_place)
    tracker.observe(first_after_place, EndTurnAction(), first_after_end)

    second_place = PlaceCardAction(0, Position(4, 3), Frame(2, 1))
    second_after_place = GameState(
        players=(
            first_after_end.players[0],
            PlayerState(),
            *first_after_end.players[2:],
        ),
        board={
            **first_after_end.board,
            Position(4, 3): (second_card,),
        },
        current_player_index=1,
        cards_played_this_turn=1,
    )
    second_after_end = GameState(
        players=second_after_place.players,
        board=second_after_place.board,
        current_player_index=2,
        cards_played_this_turn=0,
    )
    tracker.observe(first_after_end, second_place, second_after_place)
    tracker.observe(second_after_place, EndTurnAction(), second_after_end)

    second_turn = tracker.snapshot()[-1]
    assert second_turn.start_frame == Frame(1, 1)
    assert second_turn.end_frame == Frame(2, 1)
    assert second_turn.start_board_card_count == 3


def test_completed_turn_tracker_marks_settlement_during_refill() -> None:
    tracker = CompletedTurnTracker()
    card = Card(Color.YELLOW, 4)
    before = GameState(
        players=(
            PlayerState(hand=(card,)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        current_player_index=0,
    )
    place = PlaceCardAction(0, Position(2, 4), Frame(1, 2))
    after_place = GameState(
        players=(PlayerState(), *before.players[1:]),
        board={Position(2, 4): (card,)},
        current_player_index=0,
        cards_played_this_turn=1,
    )
    after_end = GameState(
        players=after_place.players,
        board=after_place.board,
        current_player_index=0,
        phase=Phase.REFILL,
        cards_played_this_turn=1,
    )
    after_refill = GameState(
        players=after_place.players,
        board=after_place.board,
        current_player_index=1,
        settlement_count=1,
    )
    tracker.observe(before, place, after_place)
    tracker.observe(after_place, EndTurnAction(), after_end)
    tracker.observe(
        after_end,
        RefillAction(RefillSource.DECK),
        after_refill,
    )
    assert tracker.snapshot()[-1].settlement_occurred


def test_public_negative_tracker_scales_hidden_refill_marginals() -> None:
    tracker = PublicNegativeKnowledgeTracker(4)
    negative = tuple(Card(Color.RED, 0) for _ in range(6)) + tuple(
        Card(Color.BLUE, 6) for _ in range(6)
    )
    empty_players = (PlayerState(),) * 4
    before_place = GameState(players=empty_players, current_player_index=0)
    after_place = GameState(
        players=(PlayerState(negative_cards=negative), *empty_players[1:]),
        current_player_index=0,
    )
    # Feed the public received cards in one placement-shaped transition.
    tracker.observe(
        before_place,
        PlaceCardAction(0, Position(0, 0), Frame(0, 0)),
        after_place,
    )
    after_refill = GameState(
        players=(
            PlayerState(
                hand=negative[:6],
                negative_cards=negative[6:],
            ),
            *empty_players[1:],
        ),
        current_player_index=1,
    )
    tracker.observe(
        after_place,
        RefillAction(RefillSource.NEGATIVE_CARDS),
        after_refill,
    )
    pile = tracker.snapshot().piles[0]
    assert pile.rank_expected[0] == 3.0
    assert pile.rank_expected[6] == 3.0
    assert pile.color_expected[0] == 3.0
    assert pile.color_expected[1] == 3.0
    assert not pile.exact


def test_v2_network_accepts_declared_shapes() -> None:
    import torch

    from yellowstone.cnn import build_win_value_net_v2

    model = build_win_value_net_v2()
    output = model(
        torch.zeros((3, BOARD_CHANNELS_V2, 7, 7)),
        torch.zeros((3, VALUE_CONTEXT_SIZE_V2)),
    )
    assert output.shape == (3,)
