"""Reduced V2 value records with an explicit before/after board transition."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

from yellowstone.types import GameState, Phase
from yellowstone.value_v2 import (
    BOARD_CHANNELS_V2,
    COLOR_COUNT,
    COLOR_ORDER,
    HAND_SIZE,
    RANK_COUNT,
    CanonicalTransformV2,
    CompletedTurn,
    PendingRefillSource,
    RefillResult,
)


HISTORY_TURNS_V2_LITE = 2
HISTORY_TURN_CONTEXT_V2_LITE = 1 + 4 + 1 + 2 * 6 + 2 + 4 + 1
HAND_CONTEXT_V2_LITE = HAND_SIZE * 6
PLAYERS_CONTEXT_V2_LITE = 4 * 3
TURN_CONTEXT_V2_LITE = 4 + 3 + 1 + 1 + 4
PENDING_REFILL_CONTEXT_V2_LITE = 4
HISTORY_CONTEXT_V2_LITE = (
    HISTORY_TURNS_V2_LITE * HISTORY_TURN_CONTEXT_V2_LITE
)
OWN_NEGATIVE_CONTEXT_V2_LITE = RANK_COUNT + COLOR_COUNT
PRE_PLAYERS_CONTEXT_V2_LITE = 4 * 3
VALUE_CONTEXT_SIZE_V2_LITE = (
    HAND_CONTEXT_V2_LITE
    + PLAYERS_CONTEXT_V2_LITE
    + TURN_CONTEXT_V2_LITE
    + PENDING_REFILL_CONTEXT_V2_LITE
    + HISTORY_CONTEXT_V2_LITE
    + OWN_NEGATIVE_CONTEXT_V2_LITE
    + PRE_PLAYERS_CONTEXT_V2_LITE
)
BOARD_CHANNELS_V2_LITE = BOARD_CHANNELS_V2 * 2
VALUE_SCHEMA_V2_LITE = "yellowstone.value.v2-lite-transition"
CANONICALIZATION_V2_LITE = "strict_residual_v2_lite_transition"


@dataclass(frozen=True, slots=True)
class ValueRecordV2Lite:
    """One post-candidate state paired with its public pre-play state."""

    game_id: int
    perspective_player_index: int
    state_before_turn: GameState
    state: GameState
    history_before_turn: tuple[CompletedTurn, ...]
    pending_refill_source: PendingRefillSource
    target: float

    def __post_init__(self) -> None:
        if len(self.history_before_turn) > HISTORY_TURNS_V2_LITE:
            raise ValueError("V2-lite stores at most two completed turns")
        if len(self.state.players) != len(self.state_before_turn.players):
            raise ValueError("pre-play and resulting player counts differ")


def canonical_tensors_v2_lite(record: ValueRecordV2Lite):
    """Return canonical after/delta board, compact context, and transform."""
    best = None
    for transform in _residual_transforms_lite(record):
        board, context = encode_value_record_v2_lite(record, transform=transform)
        key = tuple(float(value) for value in board.reshape(-1)) + tuple(
            float(value) for value in context
        )
        candidate = (key, board, context, transform)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("V2-lite canonicalization produced no transforms")
    return best[1], best[2], best[3]


def encode_value_record_v2_lite(
    record: ValueRecordV2Lite, *, transform: CanonicalTransformV2
):
    """Encode after-board plus signed after-minus-before board transition."""
    import numpy as np

    after = _encode_board(record.state, transform)
    before = _encode_board(record.state_before_turn, transform)
    board = np.concatenate((after, after - before), axis=0)
    mapping = transform.old_to_new_color
    viewer = record.perspective_player_index
    values: list[float] = []

    hand = sorted(
        (
            mapping[COLOR_ORDER.index(card.color)],
            6 - card.rank_index
            if transform.vertical_reflection
            else card.rank_index,
        )
        for card in record.state.players[viewer].hand
    )
    for slot in range(HAND_SIZE):
        if slot < len(hand):
            color, rank = hand[slot]
            values.extend([1.0, *_one_hot(color, COLOR_COUNT), rank / 6])
        else:
            values.extend([0.0] * 6)

    _append_player_summaries(values, record.state, viewer)
    values.extend(
        _one_hot(
            (record.state.current_player_index - viewer)
            % len(record.state.players),
            4,
        )
    )
    values.extend(
        _one_hot(
            (Phase.PLAY, Phase.REFILL, Phase.GAME_OVER).index(
                record.state.phase
            ),
            3,
        )
    )
    values.extend(
        [
            record.state.cards_played_this_turn / 2,
            record.state.settlement_count / 10,
        ]
    )
    values.extend(_one_hot(_deck_bucket(len(record.state.deck)), 4))
    values.extend(
        _one_hot(
            (
                PendingRefillSource.NO_PENDING,
                PendingRefillSource.NONE,
                PendingRefillSource.DECK,
                PendingRefillSource.NEGATIVE_CARDS,
            ).index(record.pending_refill_source),
            4,
        )
    )

    history = record.history_before_turn[-HISTORY_TURNS_V2_LITE:]
    values.extend(
        [0.0]
        * (
            (HISTORY_TURNS_V2_LITE - len(history))
            * HISTORY_TURN_CONTEXT_V2_LITE
        )
    )
    for turn in history:
        values.append(1.0)
        values.extend(
            _one_hot(
                (turn.player_index - viewer) % len(record.state.players), 4
            )
        )
        values.append(len(turn.cards) / 2)
        cards = sorted(
            (
                mapping[COLOR_ORDER.index(card.color)],
                6 - card.rank_index
                if transform.vertical_reflection
                else card.rank_index,
            )
            for card in turn.cards
        )
        for slot in range(2):
            if slot < len(cards):
                color, rank = cards[slot]
                values.extend([1.0, *_one_hot(color, COLOR_COUNT), rank / 6])
            else:
                values.extend([0.0] * 6)
        values.extend([turn.score_delta / 3, turn.negative_card_delta / 9])
        values.extend(
            _one_hot(
                (
                    RefillResult.NOT_OFFERED,
                    RefillResult.NONE,
                    RefillResult.DECK,
                    RefillResult.NEGATIVE_CARDS,
                ).index(turn.refill_result),
                4,
            )
        )
        values.append(1.0 if turn.settlement_occurred else 0.0)

    rank_counts = [0.0] * RANK_COUNT
    color_counts = [0.0] * COLOR_COUNT
    for card in record.state.players[viewer].negative_cards:
        rank = (
            6 - card.rank_index
            if transform.vertical_reflection
            else card.rank_index
        )
        rank_counts[rank] += 1.0
        color_counts[mapping[COLOR_ORDER.index(card.color)]] += 1.0
    values.extend(value / 56 for value in rank_counts)
    values.extend(value / 56 for value in color_counts)
    _append_player_summaries(values, record.state_before_turn, viewer)

    if len(values) != VALUE_CONTEXT_SIZE_V2_LITE:
        raise AssertionError(
            f"unexpected V2-lite context size: {len(values)} "
            f"!= {VALUE_CONTEXT_SIZE_V2_LITE}"
        )
    return board, np.asarray(values, dtype=np.float32)


def _encode_board(state: GameState, transform: CanonicalTransformV2):
    import numpy as np

    board = np.zeros((BOARD_CHANNELS_V2, 7, 7), dtype=np.float32)
    for position, stack in state.board.items():
        x = (
            6 - position.x
            if transform.horizontal_reflection
            else position.x
        )
        y = (
            6 - position.y
            if transform.vertical_reflection
            else position.y
        )
        for card in stack:
            old_color = COLOR_ORDER.index(card.color)
            rank = (
                6 - card.rank_index
                if transform.vertical_reflection
                else card.rank_index
            )
            channel = transform.old_to_new_color[old_color] * RANK_COUNT + rank
            board[channel, y, x] += 1.0
            board[-1, y, x] += 1.0
    return board


def _append_player_summaries(
    values: list[float], state: GameState, viewer: int
) -> None:
    for offset in range(4):
        player = state.players[(viewer + offset) % len(state.players)]
        values.extend(
            [
                player.loss_score / 35,
                len(player.hand) / HAND_SIZE,
                len(player.negative_cards) / 56,
            ]
        )


def _residual_transforms_lite(
    record: ValueRecordV2Lite,
) -> tuple[CanonicalTransformV2, ...]:
    vertical_candidates = [False, True]
    vertical_keys = {
        vertical: min(
            _occupancy_key(record.state, vertical, horizontal)
            for horizontal in (False, True)
        )
        for vertical in vertical_candidates
    }
    minimum = min(vertical_keys.values())
    vertical_candidates = [
        value for value in vertical_candidates if vertical_keys[value] == minimum
    ]
    if len(vertical_candidates) > 1:
        rank_keys = {
            vertical: _hand_rank_key(record, vertical)
            for vertical in vertical_candidates
        }
        minimum = min(rank_keys.values())
        vertical_candidates = [
            value for value in vertical_candidates if rank_keys[value] == minimum
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
    spatial = [pair for pair in spatial if spatial_keys[pair] == minimum_spatial]

    result: list[CanonicalTransformV2] = []
    for vertical, horizontal in spatial:
        signatures = [
            _color_signature_lite(record, old_color, vertical, horizontal)
            for old_color in range(COLOR_COUNT)
        ]
        groups: list[list[int]] = []
        for old_color in sorted(
            range(COLOR_COUNT), key=lambda index: signatures[index]
        ):
            if groups and signatures[groups[-1][0]] == signatures[old_color]:
                groups[-1].append(old_color)
            else:
                groups.append([old_color])
        orders = [
            tuple(permutations(group)) if len(group) > 1 else (tuple(group),)
            for group in groups
        ]
        for chosen_groups in product(*orders):
            old_in_new_order = tuple(
                old for group in chosen_groups for old in group
            )
            mapping = [0] * COLOR_COUNT
            for new, old in enumerate(old_in_new_order):
                mapping[old] = new
            result.append(
                CanonicalTransformV2(
                    vertical_reflection=vertical,
                    horizontal_reflection=horizontal,
                    old_to_new_color=tuple(mapping),
                )
            )
    return tuple(result)


def _occupancy_key(
    state: GameState, vertical: bool, horizontal: bool
) -> tuple[int, ...]:
    cells = [[0] * 7 for _ in range(7)]
    for position, stack in state.board.items():
        x = 6 - position.x if horizontal else position.x
        y = 6 - position.y if vertical else position.y
        cells[y][x] += len(stack)
    return tuple(value for row in cells for value in row)


def _hand_rank_key(
    record: ValueRecordV2Lite, vertical: bool
) -> tuple[int, ...]:
    counts = [0] * RANK_COUNT
    for card in record.state.players[record.perspective_player_index].hand:
        rank = 6 - card.rank_index if vertical else card.rank_index
        counts[rank] += 1
    return tuple(counts)


def _color_signature_lite(
    record: ValueRecordV2Lite,
    old_color: int,
    vertical: bool,
    horizontal: bool,
) -> tuple[float, ...]:
    color = COLOR_ORDER[old_color]

    def board_color(state: GameState) -> list[float]:
        result = [0.0] * 49
        for position, stack in state.board.items():
            x = 6 - position.x if horizontal else position.x
            y = 6 - position.y if vertical else position.y
            result[y * 7 + x] += sum(card.color == color for card in stack)
        return result

    after_board = board_color(record.state)
    before_board = board_color(record.state_before_turn)
    delta = [
        after - before
        for after, before in zip(after_board, before_board, strict=True)
    ]
    viewer = record.perspective_player_index
    hand = [0.0] * RANK_COUNT
    for card in record.state.players[viewer].hand:
        if card.color == color:
            rank = 6 - card.rank_index if vertical else card.rank_index
            hand[rank] += 1.0
    history: list[float] = []
    turns = record.history_before_turn[-HISTORY_TURNS_V2_LITE:]
    history.extend(
        [0.0] * ((HISTORY_TURNS_V2_LITE - len(turns)) * RANK_COUNT)
    )
    for turn in turns:
        counts = [0.0] * RANK_COUNT
        for card in turn.cards:
            if card.color == color:
                rank = 6 - card.rank_index if vertical else card.rank_index
                counts[rank] += 1.0
        history.extend(counts)
    negative = [0.0] * RANK_COUNT
    for card in record.state.players[viewer].negative_cards:
        if card.color == color:
            rank = 6 - card.rank_index if vertical else card.rank_index
            negative[rank] += 1.0
    return tuple(after_board + delta + hand + history + negative)


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if value == index else 0.0 for value in range(size)]


def _deck_bucket(deck_count: int) -> int:
    if deck_count == 0:
        return 0
    if deck_count <= HAND_SIZE:
        return 1
    if deck_count <= HAND_SIZE * 3:
        return 2
    return 3

