from random import Random

import numpy as np

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.types import Phase
from yellowstone.value_history3 import (
    BASE_CONTEXT_SIZE,
    PLACEMENT_FEATURES,
    VALUE_CONTEXT_SIZE_HISTORY3,
    History3Tracker,
    _extract_current_turn,
    _write_encoded_history,
)


def _placement(marker: float) -> np.ndarray:
    value = np.zeros(PLACEMENT_FEATURES, dtype=np.float32)
    value[0] = 1.0
    value[1] = 1.0
    value[5] = 1.0
    value[9] = marker
    return value


def test_encoded_history_keeps_three_turn_boundaries() -> None:
    destination = np.zeros(VALUE_CONTEXT_SIZE_HISTORY3, dtype=np.float32)
    history = [
        (_placement(0.1),),
        (_placement(0.2), _placement(0.3)),
        (_placement(0.4),),
    ]

    _write_encoded_history(destination, history, current_player=3)

    present = [
        destination[BASE_CONTEXT_SIZE + slot * PLACEMENT_FEATURES]
        for slot in range(6)
    ]
    assert present == [1, 0, 1, 1, 1, 0]
    relative_players = [
        int(
            np.argmax(
                destination[
                    BASE_CONTEXT_SIZE
                    + slot * PLACEMENT_FEATURES
                    + 1 :
                    BASE_CONTEXT_SIZE
                    + slot * PLACEMENT_FEATURES
                    + 5
                ]
            )
        )
        for slot in (0, 2, 4)
    ]
    assert relative_players == [1, 2, 3]


def test_extract_current_turn_uses_candidate_placements_only() -> None:
    context = np.zeros(81, dtype=np.float32)
    context[57:69] = _placement(0.2)
    context[69:81] = _placement(0.4)
    context[55] = 0.0
    one = _extract_current_turn(context)
    assert len(one) == 1
    assert one[0][9] == np.float32(0.4)

    context[55] = 1.0
    two = _extract_current_turn(context)
    assert [item[9] for item in two] == [
        np.float32(0.2),
        np.float32(0.4),
    ]


def test_history3_tracker_retains_three_completed_turns() -> None:
    state = create_initial_state(4, seed=41)
    rng = Random(43)
    bot = HeuristicBot()
    tracker = History3Tracker()
    completed = 0
    previous_player = state.current_player_index
    while completed < 5 and state.phase != Phase.GAME_OVER:
        action = bot.choose_action(state)
        assert action is not None
        before = state
        state = apply_known_legal_action(state, action, rng=rng)
        tracker.observe(before, action, state)
        if state.current_player_index != previous_player:
            completed += 1
            previous_player = state.current_player_index

    history = tracker.snapshot()
    assert len(history) == 3
    assert all(1 <= len(turn.placements) <= 2 for turn in history)
