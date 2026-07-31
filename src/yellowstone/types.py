"""Core data types for the Yellowstone Park game model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


BOARD_SIZE = 7
FRAME_SIZE = 3
MIN_PLAYERS = 4
MAX_PLAYERS = 5
HAND_SIZE = 6
DEFAULT_LOSS_SCORE = 5
GAME_END_LOSS_SCORE = 35


class Color(str, Enum):
    """Card colors.

    The exact physical color names are not important for rule logic. Stable
    values make JSON conversion and reinforcement-learning encodings easier.
    """

    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"


class Phase(str, Enum):
    """Current game phase."""

    PLAY = "play"
    REFILL = "refill"
    GAME_OVER = "game_over"


class RefillSource(str, Enum):
    """Source used during a refill action."""

    DECK = "deck"
    NEGATIVE_CARDS = "negative_cards"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class Card:
    """A card with a zero-based rank index."""

    color: Color
    rank_index: int

    @property
    def rank(self) -> int:
        """Return the display/rule rank, from 1 to 7."""
        return self.rank_index + 1


@dataclass(frozen=True, slots=True, order=True)
class Position:
    """A 7x7 board coordinate.

    (0, 0) is the upper-right cell. rank_index maps directly to y.
    """

    x: int
    y: int


@dataclass(frozen=True, slots=True, order=True)
class Frame:
    """A 3x3 frame identified by its upper-right coordinate."""

    x: int
    y: int


Board: TypeAlias = dict[Position, tuple[Card, ...]]


@dataclass(frozen=True, slots=True)
class PlayerState:
    """Per-player public and private state."""

    hand: tuple[Card, ...] = ()
    negative_cards: tuple[Card, ...] = ()
    loss_score: int = DEFAULT_LOSS_SCORE


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete immutable game state."""

    players: tuple[PlayerState, ...]
    board: Board = field(default_factory=dict)
    deck: tuple[Card, ...] = ()
    current_player_index: int = 0
    phase: Phase = Phase.PLAY
    cards_played_this_turn: int = 0
    winners: tuple[int, ...] = ()
    settlement_count: int = 0
    last_turn_play_counts: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PlaceCardAction:
    """Place one card from the current player's hand and choose a frame."""

    hand_index: int
    position: Position
    frame: Frame


@dataclass(frozen=True, slots=True)
class EndTurnAction:
    """End the turn after placing exactly one card."""


@dataclass(frozen=True, slots=True)
class RefillAction:
    """Resolve the refill phase after placing two cards."""

    source: RefillSource


Action: TypeAlias = PlaceCardAction | EndTurnAction | RefillAction
