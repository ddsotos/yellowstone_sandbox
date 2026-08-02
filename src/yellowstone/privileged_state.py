"""Privileged four-player turn-start state values used only as a teacher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

from yellowstone.game import apply_known_legal_action, sort_hand
from yellowstone.replay_v2 import LEGACY_RULES_VERSION_V2, ReplayGameV2
from yellowstone.safe_count_features import rank_color_offset_counts
from yellowstone.types import Action, Card, GameState, Phase, PlaceCardAction
from yellowstone.value_learning import (
    COLOR_ORDER,
    HISTORY_SIZE,
    RANK_BOARD_CHANNELS,
    RecentPlacement,
)


VALUE_SCHEMA_PRIVILEGED_STATE = "yellowstone.value.privileged-state.v1"
CANONICALIZATION_PRIVILEGED_STATE = "absolute_board_current_player_relative_v1"
HISTORY_SEMANTICS_PRIVILEGED_STATE = "rolling_last_two_placements_before_turn"
FEATURE_CONTRACT_PRIVILEGED_STATE = (
    "rank_color_offset_sum_counts_v1+board_card_count_v1"
)
HAND_FEATURES = 6
PLAYER_HAND_CONTEXT = 6 * HAND_FEATURES
PLAYER_STATE_CONTEXT = PLAYER_HAND_CONTEXT + 7
GLOBAL_CONTEXT = 3
HISTORY_FEATURES = 12
PRIVILEGED_STATE_CONTEXT_SIZE = (
    4 * PLAYER_STATE_CONTEXT + GLOBAL_CONTEXT + HISTORY_SIZE * HISTORY_FEATURES
)


def safe_one_card_count(state: GameState, player_index: int) -> int:
    """Count hand cards with rank/color offset sum zero."""
    return rank_color_offset_counts(state)[0][player_index]


@dataclass(frozen=True, slots=True)
class PrivilegedStateRecord:
    game_id: int
    state: GameState
    history: tuple[RecentPlacement, ...]
    target: tuple[float, float, float, float]
    safe_one_card_counts: tuple[int, int, int, int] | None = None
    one_off_card_counts: tuple[int, int, int, int] | None = None


def records_from_replay_privileged_state(
    game: ReplayGameV2,
) -> tuple[PrivilegedStateRecord, ...]:
    """Return every placement-decision turn start with all-player targets."""
    state = game.initial_state
    rng = Random(game.gameplay_seed)
    history: list[RecentPlacement] = []
    pending: list[tuple[GameState, tuple[RecentPlacement, ...]]] = []
    turn_decisions = tuple(
        decision
        for decision in game.decisions
        if decision.get("type") == "turn"
    )
    decision_index = 0
    for action in game.actions:
        before = state
        if (
            isinstance(action, PlaceCardAction)
            and before.cards_played_this_turn == 0
        ):
            pending.append((before, tuple(history[-HISTORY_SIZE:])))
        state = apply_known_legal_action(
            state,
            action,
            rng=rng,
            settle_on_empty_deck=(
                game.rules_version != LEGACY_RULES_VERSION_V2
            ),
        )
        _append_history(history, before, action, state)
    if not state.winners:
        raise ValueError(f"replay game {game.game_id} did not finish")
    shares = [
        1.0 / len(state.winners) if player in state.winners else 0.0
        for player in range(4)
    ]
    records: list[PrivilegedStateRecord] = []
    for turn_state, turn_history in pending:
        current = turn_state.current_player_index
        target = tuple(shares[(current + offset) % 4] for offset in range(4))
        counts = None
        one_off_counts = None
        if decision_index < len(turn_decisions):
            raw_counts = turn_decisions[decision_index].get(
                "safe_one_card_counts_by_player"
            )
            if raw_counts is not None:
                if len(raw_counts) != 4:
                    raise ValueError(
                        f"replay game {game.game_id} has invalid safe-one count length"
                    )
                counts = tuple(max(0, int(value)) for value in raw_counts)
            raw_one_off = turn_decisions[decision_index].get(
                "one_off_card_counts_by_player"
            )
            if raw_one_off is not None:
                if len(raw_one_off) != 4:
                    raise ValueError(
                        f"replay game {game.game_id} has invalid one-off count length"
                    )
                one_off_counts = tuple(
                    max(0, int(value)) for value in raw_one_off
                )
        decision_index += 1
        records.append(
            PrivilegedStateRecord(
                game_id=game.game_id,
                state=turn_state,
                history=turn_history,
                target=target,
                safe_one_card_counts=counts,
                one_off_card_counts=one_off_counts,
            )
        )
    return tuple(records)


def encode_privileged_state(record: PrivilegedStateRecord):
    """Encode public board plus all exact hands, relative to current player."""
    import numpy as np

    board = np.zeros((RANK_BOARD_CHANNELS, 7, 7), dtype=np.float32)
    for position, stack in record.state.board.items():
        for card in stack:
            channel = COLOR_ORDER.index(card.color) * 7 + card.rank_index
            board[channel, position.y, position.x] += 1.0
            board[-1, position.y, position.x] += 1.0

    values: list[float] = []
    current = record.state.current_player_index
    fallback_safe_counts, fallback_one_off_counts = rank_color_offset_counts(
        record.state
    )
    for offset in range(4):
        player = record.state.players[(current + offset) % 4]
        _append_hand(values, player.hand)
        low, middle, high = _negative_rank_fractions(player.negative_cards)
        if record.safe_one_card_counts is not None:
            safe_count = record.safe_one_card_counts[
                (current + offset) % 4
            ]
        else:
            safe_count = fallback_safe_counts[(current + offset) % 4]
        if record.one_off_card_counts is not None:
            one_off_count = record.one_off_card_counts[
                (current + offset) % 4
            ]
        else:
            one_off_count = fallback_one_off_counts[
                (current + offset) % 4
            ]
        values.extend(
            [
                player.loss_score / 35,
                len(player.negative_cards) / 56,
                low,
                middle,
                high,
                safe_count / 6,
                one_off_count / 6,
            ]
        )
    values.extend(
        [
            len(record.state.deck) / 112,
            record.state.settlement_count / 10,
            sum(len(stack) for stack in record.state.board.values()) / 49,
        ]
    )
    history = record.history[-HISTORY_SIZE:]
    values.extend([0.0] * ((HISTORY_SIZE - len(history)) * HISTORY_FEATURES))
    for placement in history:
        values.extend(
            [
                1.0,
                *_one_hot((placement.player_index - current) % 4, 4),
                *_one_hot(COLOR_ORDER.index(placement.card.color), 4),
                placement.card.rank_index / 6,
                placement.score_delta / 3,
                placement.negative_card_delta / 9,
            ]
        )
    if len(values) != PRIVILEGED_STATE_CONTEXT_SIZE:
        raise AssertionError(
            f"unexpected privileged context: {len(values)} "
            f"!= {PRIVILEGED_STATE_CONTEXT_SIZE}"
        )
    return board, np.asarray(values, dtype=np.float32)


def build_privileged_state_net(
    *, hidden_channels: int = 64, hidden_size: int = 128
):
    import torch
    import torch.nn as nn

    class PrivilegedStateNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(RANK_BOARD_CHANNELS, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(
                    hidden_channels * 7 * 7
                    + PRIVILEGED_STATE_CONTEXT_SIZE,
                    hidden_size,
                ),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(hidden_size, 4)

        def forward(self, board, context):
            encoded = self.board_encoder(board)
            return self.value_head(
                self.trunk(torch.cat((encoded, context), dim=1))
            )

    return PrivilegedStateNet()


class TorchPrivilegedStateEstimator:
    """Strict loader for the non-deployable privileged teacher."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        import torch

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        expected = {
            "value_schema": VALUE_SCHEMA_PRIVILEGED_STATE,
            "input_canonicalization": CANONICALIZATION_PRIVILEGED_STATE,
            "history_semantics": HISTORY_SEMANTICS_PRIVILEGED_STATE,
            "privileged_inputs": True,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"privileged checkpoint differs at {key}")
        self.torch = torch
        self.model = build_privileged_state_net()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def estimate_many(
        self, records: tuple[PrivilegedStateRecord, ...]
    ) -> tuple[tuple[float, float, float, float], ...]:
        import numpy as np

        if not records:
            return ()
        encoded = [encode_privileged_state(record) for record in records]
        boards = np.stack([item[0] for item in encoded])
        contexts = np.stack([item[1] for item in encoded])
        with self.torch.no_grad():
            probabilities = self.torch.softmax(
                self.model(
                    self.torch.from_numpy(boards),
                    self.torch.from_numpy(contexts),
                ),
                dim=1,
            ).tolist()
        return tuple(tuple(float(value) for value in row) for row in probabilities)

    def __call__(self, record: PrivilegedStateRecord):
        return self.estimate_many((record,))[0]


def _append_hand(values: list[float], hand: tuple[Card, ...]) -> None:
    hand = sort_hand(hand)
    for slot in range(6):
        if slot < len(hand):
            card = hand[slot]
            values.extend(
                [
                    1.0,
                    *_one_hot(COLOR_ORDER.index(card.color), 4),
                    card.rank_index / 6,
                ]
            )
        else:
            values.extend([0.0] * HAND_FEATURES)


def _negative_rank_fractions(cards: tuple[Card, ...]) -> tuple[float, float, float]:
    if not cards:
        return 0.0, 0.0, 0.0
    counts = (
        sum(card.rank_index <= 1 for card in cards),
        sum(2 <= card.rank_index <= 4 for card in cards),
        sum(card.rank_index >= 5 for card in cards),
    )
    return tuple(value / len(cards) for value in counts)


def _append_history(
    history: list[RecentPlacement],
    before: GameState,
    action: Action,
    after: GameState,
) -> None:
    if not isinstance(action, PlaceCardAction):
        return
    player_index = before.current_player_index
    history.append(
        RecentPlacement(
            player_index=player_index,
            card=before.players[player_index].hand[action.hand_index],
            score_delta=(
                before.players[player_index].loss_score
                - after.players[player_index].loss_score
            ),
            negative_card_delta=(
                len(after.players[player_index].negative_cards)
                - len(before.players[player_index].negative_cards)
            ),
        )
    )
    del history[:-HISTORY_SIZE]


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if value == index else 0.0 for value in range(size)]
