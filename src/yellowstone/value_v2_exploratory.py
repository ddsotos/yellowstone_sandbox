"""V2-compatible value input with uniform pile ratios and refill risk."""

from __future__ import annotations

from itertools import permutations, product

from yellowstone.types import HAND_SIZE
from yellowstone.value_v2 import (
    BOARD_CHANNELS_V2,
    COLOR_ORDER,
    COLOR_COUNT,
    OPPONENT_NEGATIVE_CONTEXT_V2,
    OWN_NEGATIVE_CONTEXT_V2,
    RANK_COUNT,
    VALUE_CONTEXT_SIZE_V2,
    CanonicalTransformV2,
    ValueRecordV2,
    _hand_rank_key,
    _occupancy_key,
    encode_value_record_v2,
)


VALUE_SCHEMA_V2_EXPLORATORY = "yellowstone.value.v2-exploratory-refill.v1"
CANONICALIZATION_V2_EXPLORATORY = (
    "strict_residual_v2_uniform_negative_ratios_refill_risk_v1"
)
HISTORY_SEMANTICS_V2_EXPLORATORY = "rolling_last_three_completed_turns_v2"
BASE_NEGATIVE_CONTEXT_SIZE = (
    OWN_NEGATIVE_CONTEXT_V2 + OPPONENT_NEGATIVE_CONTEXT_V2
)
UNIFORM_NEGATIVE_CONTEXT_SIZE = 4 * (RANK_COUNT + COLOR_COUNT + 1)
REFILL_RISK_CONTEXT_SIZE = 2
VALUE_CONTEXT_SIZE_V2_EXPLORATORY = (
    VALUE_CONTEXT_SIZE_V2
    - BASE_NEGATIVE_CONTEXT_SIZE
    + UNIFORM_NEGATIVE_CONTEXT_SIZE
    + REFILL_RISK_CONTEXT_SIZE
)


def refill_risk_features(record: ValueRecordV2) -> tuple[float, float]:
    """Return hypothetical deck exhaustion and normalized draw shortage."""
    hand_count = len(
        record.state.players[record.perspective_player_index].hand
    )
    needed = max(0, HAND_SIZE - hand_count)
    deck_count = len(record.state.deck)
    exhausts = needed > 0 and deck_count <= needed
    shortage = max(0, needed - deck_count)
    return float(exhausts), shortage / HAND_SIZE


def canonical_tensors_v2_exploratory(record: ValueRecordV2):
    """Canonicalize using only information represented by the new schema."""
    best = None
    for transform in _residual_transforms_exploratory(record):
        board, context = encode_value_record_v2_exploratory(
            record, transform=transform
        )
        key = tuple(float(value) for value in board.reshape(-1)) + tuple(
            float(value) for value in context
        )
        candidate = (key, board, context, transform)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("canonicalization produced no transforms")
    return best[1], best[2], best[3]


def encode_value_record_v2_exploratory(
    record: ValueRecordV2, *, transform: CanonicalTransformV2
):
    """Encode one record with common per-player marginal pile features."""
    import numpy as np

    board, base_context = encode_value_record_v2(
        record, transform=transform
    )
    values = list(
        float(value)
        for value in base_context[:-BASE_NEGATIVE_CONTEXT_SIZE]
    )
    values.extend(_uniform_negative_features(record, transform))
    values.extend(refill_risk_features(record))
    if len(values) != VALUE_CONTEXT_SIZE_V2_EXPLORATORY:
        raise AssertionError(
            f"unexpected exploratory V2 context: {len(values)} "
            f"!= {VALUE_CONTEXT_SIZE_V2_EXPLORATORY}"
        )
    return board, np.asarray(values, dtype=np.float32)


def _uniform_negative_features(
    record: ValueRecordV2, transform: CanonicalTransformV2
) -> list[float]:
    viewer = record.perspective_player_index
    mapping = transform.old_to_new_color
    values: list[float] = []
    for offset in range(4):
        absolute_index = (viewer + offset) % len(record.state.players)
        pile_count = len(record.state.players[absolute_index].negative_cards)
        if offset == 0:
            ranks = [0.0] * RANK_COUNT
            colors = [0.0] * COLOR_COUNT
            for card in record.state.players[absolute_index].negative_cards:
                ranks[card.rank_index] += 1.0
                colors[COLOR_ORDER.index(card.color)] += 1.0
            exact = True
        else:
            pile = record.negative_knowledge.piles[absolute_index]
            ranks = list(pile.rank_expected)
            colors = list(pile.color_expected)
            exact = pile.exact
        if transform.vertical_reflection:
            ranks.reverse()
        transformed_colors = [0.0] * COLOR_COUNT
        for old_color, count in enumerate(colors):
            transformed_colors[mapping[old_color]] = count
        denominator = float(pile_count)
        if pile_count:
            values.extend(value / denominator for value in ranks)
            values.extend(
                value / denominator for value in transformed_colors
            )
        else:
            values.extend([0.0] * (RANK_COUNT + COLOR_COUNT))
        values.append(1.0 if exact else 0.0)
    return values


def _residual_transforms_exploratory(
    record: ValueRecordV2,
) -> tuple[CanonicalTransformV2, ...]:
    vertical_candidates = [False, True]
    vertical_board_keys = {
        vertical: min(
            _occupancy_key(record.state, vertical, horizontal)
            for horizontal in (False, True)
        )
        for vertical in vertical_candidates
    }
    minimum = min(vertical_board_keys.values())
    vertical_candidates = [
        value
        for value in vertical_candidates
        if vertical_board_keys[value] == minimum
    ]
    if len(vertical_candidates) > 1:
        hand_keys = {
            vertical: _hand_rank_key(record, vertical)
            for vertical in vertical_candidates
        }
        minimum_hand = min(hand_keys.values())
        vertical_candidates = [
            value
            for value in vertical_candidates
            if hand_keys[value] == minimum_hand
        ]

    spatial = [
        (vertical, horizontal)
        for vertical in vertical_candidates
        for horizontal in (False, True)
    ]
    spatial_keys = {
        pair: _occupancy_key(record.state, pair[0], pair[1])
        for pair in spatial
    }
    minimum_spatial = min(spatial_keys.values())
    spatial = [
        pair for pair in spatial if spatial_keys[pair] == minimum_spatial
    ]

    result: list[CanonicalTransformV2] = []
    for vertical, horizontal in spatial:
        signatures = [
            _color_signature_exploratory(
                record, old_color, vertical, horizontal
            )
            for old_color in range(COLOR_COUNT)
        ]
        ordered_groups: list[list[int]] = []
        for old_color in sorted(
            range(COLOR_COUNT), key=lambda index: signatures[index]
        ):
            if (
                ordered_groups
                and signatures[ordered_groups[-1][0]]
                == signatures[old_color]
            ):
                ordered_groups[-1].append(old_color)
            else:
                ordered_groups.append([old_color])
        group_orders = [
            tuple(permutations(group))
            if len(group) > 1
            else (tuple(group),)
            for group in ordered_groups
        ]
        for chosen_groups in product(*group_orders):
            old_in_new_order = tuple(
                old_color
                for group in chosen_groups
                for old_color in group
            )
            mapping = [0] * COLOR_COUNT
            for new_color, old_color in enumerate(old_in_new_order):
                mapping[old_color] = new_color
            result.append(
                CanonicalTransformV2(
                    vertical_reflection=vertical,
                    horizontal_reflection=horizontal,
                    old_to_new_color=tuple(mapping),
                )
            )
    return tuple(result)


def _color_signature_exploratory(
    record: ValueRecordV2,
    old_color: int,
    vertical: bool,
    horizontal: bool,
) -> tuple[float, ...]:
    color = COLOR_ORDER[old_color]
    board = [0.0] * 49
    for position, stack in record.state.board.items():
        x = 6 - position.x if horizontal else position.x
        y = 6 - position.y if vertical else position.y
        board[y * 7 + x] += sum(card.color == color for card in stack)

    hand = [0.0] * RANK_COUNT
    viewer = record.perspective_player_index
    for card in record.state.players[viewer].hand:
        if card.color == color:
            rank = 6 - card.rank_index if vertical else card.rank_index
            hand[rank] += 1.0

    history: list[float] = []
    missing = 3 - len(record.history_before_turn[-3:])
    history.extend([0.0] * (missing * RANK_COUNT))
    for turn in record.history_before_turn[-3:]:
        counts = [0.0] * RANK_COUNT
        for card in turn.cards:
            if card.color == color:
                rank = 6 - card.rank_index if vertical else card.rank_index
                counts[rank] += 1.0
        history.extend(counts)

    pile_colors: list[float] = []
    for offset in range(4):
        absolute_index = (viewer + offset) % len(record.state.players)
        if offset == 0:
            count = sum(
                card.color == color
                for card in record.state.players[
                    absolute_index
                ].negative_cards
            )
            pile_colors.append(float(count))
        else:
            pile_colors.append(
                record.negative_knowledge.piles[
                    absolute_index
                ].color_expected[old_color]
            )
    return tuple(board + hand + history + pile_colors)


def build_win_value_net_v2_exploratory(
    *, hidden_channels: int = 64, hidden_size: int = 128
):
    """Build the Conv2/FC128 network for the exploratory V2 context."""
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as error:
        raise ImportError(
            "exploratory V2 support requires `pip install -e .[value]`"
        ) from error

    class YellowstoneWinValueNetV2Exploratory(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(BOARD_CHANNELS_V2, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(
                    hidden_channels * 7 * 7
                    + VALUE_CONTEXT_SIZE_V2_EXPLORATORY,
                    hidden_size,
                ),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, board, context):
            encoded = self.board_encoder(board)
            hidden = self.trunk(
                torch.cat((encoded, context), dim=1)
            )
            return self.value_head(hidden).squeeze(-1)

    return YellowstoneWinValueNetV2Exploratory()


class ExploratoryV2Estimator:
    """Validated inference adapter for exploratory V2 checkpoints."""

    def __init__(self, checkpoint_path: str):
        try:
            import torch
        except ModuleNotFoundError as error:
            raise ImportError(
                "exploratory V2 inference requires torch"
            ) from error
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        expected = {
            "value_schema": VALUE_SCHEMA_V2_EXPLORATORY,
            "input_canonicalization": CANONICALIZATION_V2_EXPLORATORY,
            "history_semantics": HISTORY_SEMANTICS_V2_EXPLORATORY,
            "context_size": VALUE_CONTEXT_SIZE_V2_EXPLORATORY,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"checkpoint differs at {key}")
        self._torch = torch
        self._model = build_win_value_net_v2_exploratory()
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()

    def estimate_many(
        self, records: tuple[ValueRecordV2, ...]
    ) -> tuple[float, ...]:
        import numpy as np

        if not records:
            return ()
        tensors = [
            canonical_tensors_v2_exploratory(record)
            for record in records
        ]
        boards = np.stack([item[0] for item in tensors])
        contexts = np.stack([item[1] for item in tensors])
        with self._torch.no_grad():
            probabilities = self._torch.sigmoid(
                self._model(
                    self._torch.from_numpy(boards),
                    self._torch.from_numpy(contexts),
                )
            ).tolist()
        return tuple(float(value) for value in probabilities)
