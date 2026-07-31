"""Replay-derived, public-information value records for V2 learning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import permutations, product
from typing import Iterable

from yellowstone.types import (
    Action,
    BOARD_SIZE,
    Card,
    Color,
    EndTurnAction,
    FRAME_SIZE,
    Frame,
    GameState,
    Phase,
    PlaceCardAction,
    RefillAction,
    RefillSource,
)


COLOR_ORDER = (Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW)
COLOR_COUNT = len(COLOR_ORDER)
RANK_COUNT = 7
HAND_SIZE = 6
HISTORY_TURNS = 3
FRAME_AXIS_SIZE = BOARD_SIZE - FRAME_SIZE + 1
BOARD_CHANNELS_V2 = COLOR_COUNT * RANK_COUNT + 1


class RefillResult(str, Enum):
    """Public refill result attached to a completed historical turn."""

    NOT_OFFERED = "not_offered"
    NONE = "none"
    DECK = "deck"
    NEGATIVE_CARDS = "negative_cards"


class PendingRefillSource(str, Enum):
    """A refill decision whose random result is not part of the value state."""

    NO_PENDING = "no_pending"
    NONE = "none"
    DECK = "deck"
    NEGATIVE_CARDS = "negative_cards"


@dataclass(frozen=True, slots=True)
class CompletedTurn:
    """One public completed turn; card order is intentionally not semantic."""

    player_index: int
    cards: tuple[Card, ...]
    start_frame: Frame | None
    end_frame: Frame
    start_board_card_count: int
    score_delta: int
    negative_card_delta: int
    settlement_occurred: bool
    refill_result: RefillResult

    def __post_init__(self) -> None:
        if not 1 <= len(self.cards) <= 2:
            raise ValueError("a completed turn must contain one or two cards")
        if self.start_board_card_count < 0:
            raise ValueError("start_board_card_count must not be negative")


@dataclass(frozen=True, slots=True)
class CandidateFrameContext:
    """Frame movement for the candidate represented by a value record."""

    start_frame: Frame | None
    end_frame: Frame
    start_board_card_count: int

    def __post_init__(self) -> None:
        if self.start_board_card_count < 0:
            raise ValueError("start_board_card_count must not be negative")


@dataclass(frozen=True, slots=True)
class PublicNegativePile:
    """Public marginal knowledge about one player's negative-card pile."""

    rank_expected: tuple[float, ...]
    color_expected: tuple[float, ...]
    exact: bool

    def __post_init__(self) -> None:
        if len(self.rank_expected) != RANK_COUNT:
            raise ValueError("rank_expected must contain seven values")
        if len(self.color_expected) != COLOR_COUNT:
            raise ValueError("color_expected must contain four values")


@dataclass(frozen=True, slots=True)
class PublicNegativeKnowledge:
    """Public negative-pile knowledge indexed by absolute player index."""

    piles: tuple[PublicNegativePile, ...]


@dataclass(frozen=True, slots=True)
class ValueRecordV2:
    """A viewer-safe semantic record before strict canonicalization."""

    game_id: int
    perspective_player_index: int
    state: GameState
    history_before_turn: tuple[CompletedTurn, ...]
    candidate_frame: CandidateFrameContext
    negative_knowledge: PublicNegativeKnowledge
    pending_refill_source: PendingRefillSource
    target: float

    def __post_init__(self) -> None:
        if len(self.history_before_turn) > HISTORY_TURNS:
            raise ValueError("V2 stores at most three completed turns")
        if len(self.negative_knowledge.piles) != len(self.state.players):
            raise ValueError("negative knowledge player count differs from state")


@dataclass(frozen=True, slots=True)
class CanonicalTransformV2:
    vertical_reflection: bool
    horizontal_reflection: bool
    old_to_new_color: tuple[int, ...]


# hand 6*(present + color4 + rank), players 4*3,
# current-player4 + phase3 + cards-played + settlement + deck-bucket4,
# pending-refill4, history 3*(present + player4 + play-count +
# frame(start-present + start-x5 + start-y5 + end-x5 + end-y5 +
# start-board-count + abs-dx + abs-dy) +
# cards2*(present + color4 + rank) + deltas2 + settlement + refill4),
# candidate frame(start-present + start-x5 + start-y5 + end-x5 + end-y5 +
# start-board-count + abs-dx + abs-dy),
# own negative color*rank28, opponents 3*(rank7 + color4 + exact).
HAND_CONTEXT_V2 = HAND_SIZE * 6
PLAYERS_CONTEXT_V2 = 4 * 3
TURN_CONTEXT_V2 = 4 + 3 + 1 + 1 + 4
PENDING_REFILL_CONTEXT_V2 = 4
FRAME_MOVEMENT_CONTEXT_V2 = 1 + 4 * FRAME_AXIS_SIZE + 1 + 2
HISTORY_TURN_CONTEXT_V2 = (
    1 + 4 + 1 + FRAME_MOVEMENT_CONTEXT_V2 + 2 * 6 + 2 + 1 + 4
)
HISTORY_CONTEXT_V2 = HISTORY_TURNS * HISTORY_TURN_CONTEXT_V2
CANDIDATE_FRAME_CONTEXT_V2 = FRAME_MOVEMENT_CONTEXT_V2
OWN_NEGATIVE_CONTEXT_V2 = COLOR_COUNT * RANK_COUNT
OPPONENT_NEGATIVE_CONTEXT_V2 = 3 * (RANK_COUNT + COLOR_COUNT + 1)
VALUE_CONTEXT_SIZE_V2 = (
    HAND_CONTEXT_V2
    + PLAYERS_CONTEXT_V2
    + TURN_CONTEXT_V2
    + PENDING_REFILL_CONTEXT_V2
    + HISTORY_CONTEXT_V2
    + CANDIDATE_FRAME_CONTEXT_V2
    + OWN_NEGATIVE_CONTEXT_V2
    + OPPONENT_NEGATIVE_CONTEXT_V2
)


class PublicNegativeKnowledgeTracker:
    """Track legal public marginal knowledge without exposing hidden remainders."""

    def __init__(self, player_count: int) -> None:
        self._rank = [[0.0] * RANK_COUNT for _ in range(player_count)]
        self._color = [[0.0] * COLOR_COUNT for _ in range(player_count)]
        self._exact = [True] * player_count

    def snapshot(self) -> PublicNegativeKnowledge:
        return PublicNegativeKnowledge(
            tuple(
                PublicNegativePile(
                    rank_expected=tuple(self._rank[index]),
                    color_expected=tuple(self._color[index]),
                    exact=self._exact[index],
                )
                for index in range(len(self._rank))
            )
        )

    def observe(self, before: GameState, action: Action, after: GameState) -> None:
        """Apply one public event to the marginal knowledge state."""
        if after.settlement_count > before.settlement_count:
            for index in range(len(self._rank)):
                self._rank[index] = [0.0] * RANK_COUNT
                self._color[index] = [0.0] * COLOR_COUNT
                self._exact[index] = True
            return

        player_index = before.current_player_index
        if isinstance(action, PlaceCardAction):
            old_count = len(before.players[player_index].negative_cards)
            received = after.players[player_index].negative_cards[old_count:]
            for card in received:
                self._rank[player_index][card.rank_index] += 1.0
                self._color[player_index][COLOR_ORDER.index(card.color)] += 1.0
            return

        if (
            isinstance(action, RefillAction)
            and action.source == RefillSource.NEGATIVE_CARDS
        ):
            old_count = len(before.players[player_index].negative_cards)
            new_count = len(after.players[player_index].negative_cards)
            if old_count <= 0:
                raise AssertionError("negative refill requires a non-empty pile")
            factor = new_count / old_count
            self._rank[player_index] = [
                value * factor for value in self._rank[player_index]
            ]
            self._color[player_index] = [
                value * factor for value in self._color[player_index]
            ]
            self._exact[player_index] = new_count == 0


class CompletedTurnTracker:
    """Build complete one/two-card turns and retain the last three."""

    def __init__(self) -> None:
        self._history: list[CompletedTurn] = []
        self._current_frame: Frame | None = None
        self._active_player: int | None = None
        self._cards: list[Card] = []
        self._start_frame: Frame | None = None
        self._end_frame: Frame | None = None
        self._start_board_card_count = 0
        self._score_delta = 0
        self._negative_delta = 0
        self._settlement_occurred = False

    def snapshot(self) -> tuple[CompletedTurn, ...]:
        return tuple(self._history)

    @property
    def current_frame(self) -> Frame | None:
        """Return the latest publicly selected frame, if one exists."""
        return self._current_frame

    def observe(self, before: GameState, action: Action, after: GameState) -> None:
        if isinstance(action, PlaceCardAction):
            player_index = before.current_player_index
            if self._active_player is None:
                self._active_player = player_index
                self._start_frame = self._current_frame
                self._start_board_card_count = _board_card_count(before)
            elif self._active_player != player_index:
                raise AssertionError("turn player changed before turn completion")
            card = before.players[player_index].hand[action.hand_index]
            self._cards.append(card)
            self._end_frame = action.frame
            self._score_delta += (
                before.players[player_index].loss_score
                - after.players[player_index].loss_score
            )
            self._negative_delta += (
                len(after.players[player_index].negative_cards)
                - len(before.players[player_index].negative_cards)
            )
            return

        if self._active_player is not None:
            self._settlement_occurred = (
                self._settlement_occurred
                or after.settlement_count > before.settlement_count
            )
        if isinstance(action, EndTurnAction):
            if after.phase != Phase.REFILL:
                self._finish(RefillResult.NOT_OFFERED)
            return

        if isinstance(action, RefillAction) and self._active_player is not None:
            self._finish(RefillResult(action.source.value))

    def _finish(self, refill_result: RefillResult) -> None:
        if self._active_player is None or not self._cards or self._end_frame is None:
            raise AssertionError("cannot finish an empty tracked turn")
        self._history.append(
            CompletedTurn(
                player_index=self._active_player,
                cards=tuple(self._cards),
                start_frame=self._start_frame,
                end_frame=self._end_frame,
                start_board_card_count=self._start_board_card_count,
                score_delta=self._score_delta,
                negative_card_delta=self._negative_delta,
                settlement_occurred=self._settlement_occurred,
                refill_result=refill_result,
            )
        )
        del self._history[:-HISTORY_TURNS]
        self._current_frame = self._end_frame
        self._active_player = None
        self._cards = []
        self._start_frame = None
        self._end_frame = None
        self._start_board_card_count = 0
        self._score_delta = 0
        self._negative_delta = 0
        self._settlement_occurred = False


def canonical_tensors_v2(record: ValueRecordV2):
    """Return strict canonical board/context tensors and the chosen transform."""
    transforms = _residual_transforms(record)
    best: tuple[tuple[float, ...], object, object, CanonicalTransformV2] | None = None
    for transform in transforms:
        board, context = encode_value_record_v2(record, transform=transform)
        key = tuple(float(value) for value in board.reshape(-1)) + tuple(
            float(value) for value in context
        )
        candidate = (key, board, context, transform)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("canonicalization produced no transforms")
    return best[1], best[2], best[3]


def encode_value_record_v2(
    record: ValueRecordV2, *, transform: CanonicalTransformV2
):
    """Encode one V2 record using an already selected legal transform."""
    import numpy as np

    board = np.zeros((BOARD_CHANNELS_V2, 7, 7), dtype=np.float32)
    mapping = transform.old_to_new_color
    for position, stack in record.state.board.items():
        x = 6 - position.x if transform.horizontal_reflection else position.x
        y = 6 - position.y if transform.vertical_reflection else position.y
        for card in stack:
            old_color = COLOR_ORDER.index(card.color)
            rank = 6 - card.rank_index if transform.vertical_reflection else card.rank_index
            channel = mapping[old_color] * RANK_COUNT + rank
            board[channel, y, x] += 1.0
            board[-1, y, x] += 1.0

    values: list[float] = []
    viewer = record.perspective_player_index
    hand = sorted(
        (
            mapping[COLOR_ORDER.index(card.color)],
            6 - card.rank_index if transform.vertical_reflection else card.rank_index,
        )
        for card in record.state.players[viewer].hand
    )
    for slot in range(HAND_SIZE):
        if slot < len(hand):
            color, rank = hand[slot]
            values.extend([1.0, *_one_hot(color, COLOR_COUNT), rank / 6])
        else:
            values.extend([0.0] * 6)

    for offset in range(4):
        player = record.state.players[(viewer + offset) % len(record.state.players)]
        values.extend(
            [
                player.loss_score / 35,
                len(player.hand) / HAND_SIZE,
                len(player.negative_cards) / 56,
            ]
        )

    values.extend(
        _one_hot(
            (record.state.current_player_index - viewer) % len(record.state.players),
            4,
        )
    )
    values.extend(
        _one_hot(
            (Phase.PLAY, Phase.REFILL, Phase.GAME_OVER).index(record.state.phase),
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

    history = record.history_before_turn[-HISTORY_TURNS:]
    missing = HISTORY_TURNS - len(history)
    values.extend([0.0] * (missing * HISTORY_TURN_CONTEXT_V2))
    for turn in history:
        values.append(1.0)
        values.extend(
            _one_hot((turn.player_index - viewer) % len(record.state.players), 4)
        )
        values.append(len(turn.cards) / 2)
        _append_frame_movement(
            values,
            start_frame=turn.start_frame,
            end_frame=turn.end_frame,
            start_board_card_count=turn.start_board_card_count,
            transform=transform,
        )
        cards = sorted(
            (
                mapping[COLOR_ORDER.index(card.color)],
                6 - card.rank_index
                if transform.vertical_reflection
                else card.rank_index,
            )
            for card in turn.cards
        )
        for card_slot in range(2):
            if card_slot < len(cards):
                color, rank = cards[card_slot]
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

    _append_frame_movement(
        values,
        start_frame=record.candidate_frame.start_frame,
        end_frame=record.candidate_frame.end_frame,
        start_board_card_count=record.candidate_frame.start_board_card_count,
        transform=transform,
    )

    own_counts = [[0.0] * RANK_COUNT for _ in range(COLOR_COUNT)]
    for card in record.state.players[viewer].negative_cards:
        color = mapping[COLOR_ORDER.index(card.color)]
        rank = 6 - card.rank_index if transform.vertical_reflection else card.rank_index
        own_counts[color][rank] += 1.0
    values.extend(value / 56 for row in own_counts for value in row)

    for offset in range(1, 4):
        absolute_index = (viewer + offset) % len(record.state.players)
        pile = record.negative_knowledge.piles[absolute_index]
        ranks = (
            tuple(reversed(pile.rank_expected))
            if transform.vertical_reflection
            else pile.rank_expected
        )
        colors = [0.0] * COLOR_COUNT
        for old_color, count in enumerate(pile.color_expected):
            colors[mapping[old_color]] = count
        values.extend(value / 56 for value in ranks)
        values.extend(value / 56 for value in colors)
        values.append(1.0 if pile.exact else 0.0)

    if len(values) != VALUE_CONTEXT_SIZE_V2:
        raise AssertionError(
            f"unexpected V2 context size: {len(values)} != {VALUE_CONTEXT_SIZE_V2}"
        )
    return board, np.asarray(values, dtype=np.float32)


def _residual_transforms(
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
        value for value in vertical_candidates if vertical_board_keys[value] == minimum
    ]
    if len(vertical_candidates) > 1:
        hand_keys = {
            vertical: _hand_rank_key(record, vertical)
            for vertical in vertical_candidates
        }
        minimum_hand = min(hand_keys.values())
        vertical_candidates = [
            value for value in vertical_candidates if hand_keys[value] == minimum_hand
        ]

    spatial: list[tuple[bool, bool]] = []
    spatial_keys: dict[tuple[bool, bool], tuple[int, ...]] = {}
    for vertical in vertical_candidates:
        for horizontal in (False, True):
            pair = (vertical, horizontal)
            spatial.append(pair)
            spatial_keys[pair] = _occupancy_key(
                record.state, vertical, horizontal
            )
    minimum_spatial = min(spatial_keys.values())
    spatial = [pair for pair in spatial if spatial_keys[pair] == minimum_spatial]

    result: list[CanonicalTransformV2] = []
    for vertical, horizontal in spatial:
        signatures = [
            _color_signature(record, old_color, vertical, horizontal)
            for old_color in range(COLOR_COUNT)
        ]
        ordered_groups: list[list[int]] = []
        for old_color in sorted(range(COLOR_COUNT), key=lambda index: signatures[index]):
            if (
                ordered_groups
                and signatures[ordered_groups[-1][0]] == signatures[old_color]
            ):
                ordered_groups[-1].append(old_color)
            else:
                ordered_groups.append([old_color])
        group_orders = [
            tuple(permutations(group)) if len(group) > 1 else (tuple(group),)
            for group in ordered_groups
        ]
        for chosen_groups in product(*group_orders):
            old_in_new_order = tuple(
                old_color for group in chosen_groups for old_color in group
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


def _occupancy_key(
    state: GameState, vertical: bool, horizontal: bool
) -> tuple[int, ...]:
    cells = [[0] * 7 for _ in range(7)]
    for position, stack in state.board.items():
        x = 6 - position.x if horizontal else position.x
        y = 6 - position.y if vertical else position.y
        cells[y][x] += len(stack)
    return tuple(value for row in cells for value in row)


def _hand_rank_key(record: ValueRecordV2, vertical: bool) -> tuple[int, ...]:
    counts = [0] * RANK_COUNT
    hand = record.state.players[record.perspective_player_index].hand
    for card in hand:
        rank = 6 - card.rank_index if vertical else card.rank_index
        counts[rank] += 1
    return tuple(counts)


def _color_signature(
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
    missing = HISTORY_TURNS - len(record.history_before_turn[-HISTORY_TURNS:])
    history.extend([0.0] * (missing * RANK_COUNT))
    for turn in record.history_before_turn[-HISTORY_TURNS:]:
        counts = [0.0] * RANK_COUNT
        for card in turn.cards:
            if card.color == color:
                rank = 6 - card.rank_index if vertical else card.rank_index
                counts[rank] += 1.0
        history.extend(counts)

    own_negative = [0.0] * RANK_COUNT
    for card in record.state.players[viewer].negative_cards:
        if card.color == color:
            rank = 6 - card.rank_index if vertical else card.rank_index
            own_negative[rank] += 1.0

    opponent_colors = [
        record.negative_knowledge.piles[
            (viewer + offset) % len(record.state.players)
        ].color_expected[old_color]
        for offset in range(1, 4)
    ]
    return tuple(board + hand + history + own_negative + opponent_colors)


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if value == index else 0.0 for value in range(size)]


def _append_frame_movement(
    values: list[float],
    *,
    start_frame: Frame | None,
    end_frame: Frame,
    start_board_card_count: int,
    transform: CanonicalTransformV2,
) -> None:
    transformed_end = _transform_frame_for_value(end_frame, transform)
    values.append(1.0 if start_frame is not None else 0.0)
    if start_frame is None:
        values.extend([0.0] * (2 * FRAME_AXIS_SIZE))
        abs_dx = 0
        abs_dy = 0
    else:
        transformed_start = _transform_frame_for_value(start_frame, transform)
        values.extend(_one_hot(transformed_start.x, FRAME_AXIS_SIZE))
        values.extend(_one_hot(transformed_start.y, FRAME_AXIS_SIZE))
        abs_dx = abs(end_frame.x - start_frame.x)
        abs_dy = abs(end_frame.y - start_frame.y)
    values.extend(_one_hot(transformed_end.x, FRAME_AXIS_SIZE))
    values.extend(_one_hot(transformed_end.y, FRAME_AXIS_SIZE))
    values.extend(
        [
            start_board_card_count / 56,
            abs_dx / (FRAME_AXIS_SIZE - 1),
            abs_dy / (FRAME_AXIS_SIZE - 1),
        ]
    )


def _transform_frame_for_value(
    frame: Frame, transform: CanonicalTransformV2
) -> Frame:
    return Frame(
        x=(
            FRAME_AXIS_SIZE - 1 - frame.x
            if transform.horizontal_reflection
            else frame.x
        ),
        y=(
            FRAME_AXIS_SIZE - 1 - frame.y
            if transform.vertical_reflection
            else frame.y
        ),
    )


def _board_card_count(state: GameState) -> int:
    return sum(len(stack) for stack in state.board.values())


def _deck_bucket(deck_count: int) -> int:
    if deck_count == 0:
        return 0
    if deck_count <= HAND_SIZE:
        return 1
    if deck_count <= HAND_SIZE * 3:
        return 2
    return 3


def card_multiset_key(cards: Iterable[Card]) -> tuple[tuple[str, int], ...]:
    """Return a stable semantic multiset key used by collectors and audits."""
    return tuple(sorted((card.color.value, card.rank_index) for card in cards))


class TorchWinValueEstimatorV2:
    """Batched strict-canonical inference for a V2 checkpoint."""

    def __init__(self, checkpoint_path: str) -> None:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise ImportError("V2 inference requires `pip install -e .[value]`") from error
        from yellowstone.cnn import build_win_value_net_v2

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("value_schema") != "yellowstone.value.v2":
            raise ValueError("checkpoint is not a Yellowstone value V2 model")
        self._torch = torch
        self._model = build_win_value_net_v2()
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()

    def __call__(self, record: ValueRecordV2) -> float:
        return self.estimate_many((record,))[0]

    def estimate_many(
        self, records: tuple[ValueRecordV2, ...]
    ) -> tuple[float, ...]:
        import numpy as np

        if not records:
            return ()
        tensors = [canonical_tensors_v2(record) for record in records]
        board_array = np.stack([item[0] for item in tensors])
        context_array = np.stack([item[1] for item in tensors])
        with self._torch.no_grad():
            board = self._torch.from_numpy(board_array)
            context = self._torch.from_numpy(context_array)
            values = self._torch.sigmoid(self._model(board, context)).tolist()
        return tuple(float(value) for value in values)
