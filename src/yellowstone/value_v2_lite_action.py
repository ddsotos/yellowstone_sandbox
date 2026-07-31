"""V2-lite terminal value model with explicit unordered candidate cards."""

from __future__ import annotations

from pathlib import Path

from yellowstone.types import Card
from yellowstone.value_learning import COLOR_ORDER
from yellowstone.value_v2_lite import (
    BOARD_CHANNELS_V2_LITE,
    VALUE_CONTEXT_SIZE_V2_LITE,
    ValueRecordV2Lite,
    canonical_tensors_v2_lite,
)


VALUE_SCHEMA_V2_LITE_ACTION = "yellowstone.value.v2-lite-action.v1"
CANONICALIZATION_V2_LITE_ACTION = (
    "strict_residual_v2_lite_transition_plus_unordered_action_cards"
)
HISTORY_SEMANTICS_V2_LITE_ACTION = "last_two_completed_turns_before_turn"
ACTION_CARD_CONTEXT_SIZE = 2 * 6
VALUE_CONTEXT_SIZE_V2_LITE_ACTION = (
    VALUE_CONTEXT_SIZE_V2_LITE + ACTION_CARD_CONTEXT_SIZE
)


def action_cards_from_transition(
    record: ValueRecordV2Lite,
) -> tuple[Card, ...]:
    """Recover the one or two played cards from the exact own-hand change."""
    viewer = record.perspective_player_index
    remaining = list(record.state.players[viewer].hand)
    played = []
    for card in record.state_before_turn.players[viewer].hand:
        try:
            remaining.remove(card)
        except ValueError:
            played.append(card)
    if remaining:
        raise ValueError("resulting hand is not a subset of pre-play hand")
    if not 1 <= len(played) <= 2:
        raise ValueError(
            f"V2-lite action must contain one or two cards, got {len(played)}"
        )
    return tuple(played)


def canonical_tensors_v2_lite_action(record: ValueRecordV2Lite):
    """Return the base V2-lite tensors plus two canonical card slots."""
    import numpy as np

    board, context, transform = canonical_tensors_v2_lite(record)
    cards = []
    for card in action_cards_from_transition(record):
        old_color = COLOR_ORDER.index(card.color)
        color = transform.old_to_new_color[old_color]
        rank = (
            6 - card.rank_index
            if transform.vertical_reflection
            else card.rank_index
        )
        cards.append((color, rank))
    cards.sort()
    values = list(float(value) for value in context)
    for slot in range(2):
        if slot < len(cards):
            color, rank = cards[slot]
            values.extend([1.0, *_one_hot(color, 4), rank / 6])
        else:
            values.extend([0.0] * 6)
    if len(values) != VALUE_CONTEXT_SIZE_V2_LITE_ACTION:
        raise AssertionError("unexpected V2-lite-action context size")
    return board, np.asarray(values, dtype=np.float32), transform


def build_win_value_net_v2_lite_action(
    *, hidden_channels: int = 64, hidden_size: int = 128
):
    import torch
    import torch.nn as nn

    class YellowstoneWinValueNetV2LiteAction(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(
                    BOARD_CHANNELS_V2_LITE,
                    hidden_channels,
                    3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(
                    hidden_channels * 7 * 7
                    + VALUE_CONTEXT_SIZE_V2_LITE_ACTION,
                    hidden_size,
                ),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, board, context):
            encoded = self.board_encoder(board)
            hidden = self.trunk(torch.cat((encoded, context), dim=1))
            return self.value_head(hidden).squeeze(-1)

    return YellowstoneWinValueNetV2LiteAction()


class TorchWinValueEstimatorV2LiteAction:
    def __init__(self, checkpoint_path: str | Path) -> None:
        import torch

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        expected = {
            "value_schema": VALUE_SCHEMA_V2_LITE_ACTION,
            "input_canonicalization": CANONICALIZATION_V2_LITE_ACTION,
            "history_semantics": HISTORY_SEMANTICS_V2_LITE_ACTION,
            "context_size": VALUE_CONTEXT_SIZE_V2_LITE_ACTION,
            "opponent_private_inputs": False,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"V2-lite-action checkpoint differs at {key}")
        self.torch = torch
        self.model = build_win_value_net_v2_lite_action()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def __call__(self, record: ValueRecordV2Lite) -> float:
        return self.estimate_many((record,))[0]

    def estimate_many(
        self, records: tuple[ValueRecordV2Lite, ...]
    ) -> tuple[float, ...]:
        import numpy as np

        if not records:
            return ()
        encoded = [
            canonical_tensors_v2_lite_action(record) for record in records
        ]
        board = np.stack([item[0] for item in encoded])
        context = np.stack([item[1] for item in encoded])
        with self.torch.no_grad():
            probabilities = self.torch.sigmoid(
                self.model(
                    self.torch.from_numpy(board),
                    self.torch.from_numpy(context),
                )
            ).tolist()
        return tuple(float(value) for value in probabilities)


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if value == index else 0.0 for value in range(size)]
