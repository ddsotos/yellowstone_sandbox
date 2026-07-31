import pytest

from yellowstone.types import (
    Card,
    Color,
    Frame,
    GameState,
    Phase,
    PlayerState,
)
from yellowstone.value_v2 import (
    CandidateFrameContext,
    CanonicalTransformV2,
    PendingRefillSource,
    PublicNegativeKnowledge,
    PublicNegativePile,
    ValueRecordV2,
)
from yellowstone.value_v2_exploratory import (
    CANONICALIZATION_V2_EXPLORATORY,
    HISTORY_SEMANTICS_V2_EXPLORATORY,
    ExploratoryV2Estimator,
    VALUE_CONTEXT_SIZE_V2_EXPLORATORY,
    build_win_value_net_v2_exploratory,
    encode_value_record_v2_exploratory,
    refill_risk_features,
)


def _record(*, deck_count: int = 1) -> ValueRecordV2:
    own_negative = (Card(Color.YELLOW, 3), Card(Color.YELLOW, 3))
    opponent_negative = (Card(Color.RED, 0), Card(Color.BLUE, 6))
    state = GameState(
        players=(
            PlayerState(
                hand=(
                    Card(Color.GREEN, 1),
                    Card(Color.BLUE, 5),
                    Card(Color.RED, 2),
                ),
                negative_cards=own_negative,
                loss_score=4,
            ),
            PlayerState(
                hand=(Card(Color.RED, 0),),
                negative_cards=opponent_negative,
                loss_score=6,
            ),
            PlayerState(hand=(), loss_score=7),
            PlayerState(hand=(Card(Color.BLUE, 6),), loss_score=8),
        ),
        board={},
        deck=(Card(Color.GREEN, 0),) * deck_count,
        current_player_index=0,
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )
    knowledge = PublicNegativeKnowledge(
        (
            PublicNegativePile(
                (0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 2.0, 0.0),
                True,
            ),
            PublicNegativePile(
                (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (1.0, 1.0, 0.0, 0.0),
                False,
            ),
            PublicNegativePile((0.0,) * 7, (0.0,) * 4, True),
            PublicNegativePile((0.0,) * 7, (0.0,) * 4, True),
        )
    )
    return ValueRecordV2(
        game_id=1,
        perspective_player_index=0,
        state=state,
        history_before_turn=(),
        candidate_frame=CandidateFrameContext(
            start_frame=None,
            end_frame=Frame(0, 0),
            start_board_card_count=0,
        ),
        negative_knowledge=knowledge,
        pending_refill_source=PendingRefillSource.DECK,
        target=1.0,
    )


def test_refill_risk_marks_shortage_and_exact_exhaustion() -> None:
    assert refill_risk_features(_record(deck_count=1)) == (
        1.0,
        pytest.approx(2 / 6),
    )
    assert refill_risk_features(_record(deck_count=3)) == (1.0, 0.0)
    assert refill_risk_features(_record(deck_count=4)) == (0.0, 0.0)


def test_all_players_use_twelve_ratio_features() -> None:
    identity = CanonicalTransformV2(
        vertical_reflection=False,
        horizontal_reflection=False,
        old_to_new_color=(0, 1, 2, 3),
    )
    _, context = encode_value_record_v2_exploratory(
        _record(), transform=identity
    )
    negative = context[-50:-2]

    assert len(context) == VALUE_CONTEXT_SIZE_V2_EXPLORATORY
    assert len(negative) == 48
    own = negative[:12]
    opponent = negative[12:24]
    assert own[3] == 1.0
    assert sum(own[:7]) == 1.0
    assert sum(own[7:11]) == 1.0
    assert own[11] == 1.0
    assert opponent[0] == 0.5
    assert opponent[6] == 0.5
    assert sum(opponent[:7]) == 1.0
    assert sum(opponent[7:11]) == 1.0
    assert opponent[11] == 0.0


def test_checkpoint_metadata_mismatch_hard_fails(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "wrong.pt"
    torch.save(
        {
            "state_dict": build_win_value_net_v2_exploratory().state_dict(),
            "value_schema": "yellowstone.value.v2",
            "input_canonicalization": CANONICALIZATION_V2_EXPLORATORY,
            "history_semantics": HISTORY_SEMANTICS_V2_EXPLORATORY,
            "context_size": VALUE_CONTEXT_SIZE_V2_EXPLORATORY,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="value_schema"):
        ExploratoryV2Estimator(str(checkpoint))
