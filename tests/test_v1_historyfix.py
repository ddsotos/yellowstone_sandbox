from random import Random

from yellowstone.bots import HeuristicBot
from yellowstone.convert_replay_v2_to_v1_historyfix import (
    VALUE_SCHEMA_V1_HISTORYFIX,
    records_from_replay_v1_historyfix,
)
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.replay_v2 import RULES_VERSION_V2, ReplayGameV2
from yellowstone.types import Phase
from yellowstone.transform_v1_historyfix import (
    CARDS_PLAYED_INDEX,
    HISTORY_SLOT_SIZE,
    HISTORY_START,
    repair_history_context,
)


def test_historyfix_records_never_mix_players_within_history() -> None:
    state = create_initial_state(4, seed=19)
    initial_state = state
    rng = Random(23)
    bot = HeuristicBot()
    actions = []
    while state.phase != Phase.GAME_OVER:
        action = bot.choose_action(state)
        assert action is not None
        actions.append(action)
        state = apply_known_legal_action(state, action, rng=rng)
    game = ReplayGameV2(
        game_id=7,
        initial_seed=19,
        gameplay_seed=23,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=(),
        winners=state.winners,
        teacher_checkpoint="test",
        teacher_sha256="0" * 64,
        teacher_generation=0,
        privileged_teacher_deck=False,
        rules_version=RULES_VERSION_V2,
    )

    records = records_from_replay_v1_historyfix(game)

    assert VALUE_SCHEMA_V1_HISTORYFIX == "yellowstone.value.v1_historyfix"
    assert records
    assert {len(record.history) for record in records} == {1, 2}
    assert all(
        placement.player_index == record.perspective_player_index
        for record in records
        for placement in record.history
    )


def test_repair_history_context_moves_current_one_card_slot() -> None:
    import numpy as np

    context = np.zeros((3, 81), dtype=np.float32)
    first = np.arange(1, 13, dtype=np.float32)
    second = np.arange(21, 33, dtype=np.float32)
    context[0, HISTORY_START : HISTORY_START + HISTORY_SLOT_SIZE] = first
    context[
        0,
        HISTORY_START + HISTORY_SLOT_SIZE : HISTORY_START + 24,
    ] = second
    context[1, HISTORY_START : HISTORY_START + HISTORY_SLOT_SIZE] = first
    context[2] = context[0]
    context[2, CARDS_PLAYED_INDEX] = 1.0

    fixed, stats = repair_history_context(context)

    assert np.array_equal(
        fixed[0, HISTORY_START : HISTORY_START + HISTORY_SLOT_SIZE],
        second,
    )
    assert not fixed[0, HISTORY_START + HISTORY_SLOT_SIZE :].any()
    assert np.array_equal(
        fixed[1, HISTORY_START : HISTORY_START + HISTORY_SLOT_SIZE],
        first,
    )
    assert not fixed[1, HISTORY_START + HISTORY_SLOT_SIZE :].any()
    assert np.array_equal(fixed[2], context[2])
    assert stats == {
        "records": 3,
        "one_card_records": 2,
        "two_card_records": 1,
        "moved_second_slot_records": 1,
    }
