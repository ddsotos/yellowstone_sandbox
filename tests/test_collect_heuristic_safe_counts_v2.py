from yellowstone.collect_heuristic_safe_counts_v2 import VARIANT_POLICY, collect_one_game
from yellowstone.privileged_state import records_from_replay_privileged_state


def test_heuristic_safe_count_replay_records_turn_counts() -> None:
    replay, facts = collect_one_game(game_id=1, seed=20260730)

    turn_decisions = [
        decision
        for decision in replay.decisions
        if decision.get("type") == "turn"
    ]
    records = records_from_replay_privileged_state(replay)

    assert turn_decisions
    assert len(turn_decisions) == len(records)
    assert facts["turns"] == len(records)
    for decision in turn_decisions:
        assert len(decision["safe_one_card_counts_by_player"]) == 4
        assert len(decision["one_off_card_counts_by_player"]) == 4
        assert decision["selection_mode"] == "heuristic"


def test_variant_heuristic4_replay_records_branch_facts() -> None:
    replay, facts = collect_one_game(
        game_id=1,
        seed=20260802,
        policy=VARIANT_POLICY,
    )

    turn_decisions = [
        decision
        for decision in replay.decisions
        if decision.get("type") == "turn"
    ]

    assert turn_decisions
    assert replay.teacher_checkpoint.startswith("variant_board5_hand6_oneoff")
    assert facts["turns"] == len(turn_decisions)
    assert any(decision["variant_branch"] is not None for decision in turn_decisions)
    for decision in turn_decisions:
        assert len(decision["safe_one_card_counts_by_player"]) == 4
        assert len(decision["one_off_card_counts_by_player"]) == 4
        assert decision["policy"].startswith("FixedFrameHandSixOneOffMinLossOneCardBot")
