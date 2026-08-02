"""Fast board-summary counts for zero/one-offset hand cards."""

from __future__ import annotations

from typing import Iterable

from yellowstone.types import BOARD_SIZE, FRAME_SIZE, Card, Color, GameState


def rank_color_offset_counts(state: GameState) -> tuple[list[int], list[int]]:
    """Return per-player cards with rank/color offset sums of 0 and 1.

    A hand card is counted as safe when ``rank_offset + color_offset == 0``
    and as one-off when the sum is exactly 1. Larger sums are not counted.
    """
    board_cards = tuple(
        card for stack in state.board.values() for card in stack
    )
    safe_ranks, one_off_ranks = _rank_offset_sets(
        card.rank_index for card in board_cards
    )
    board_colors = {card.color for card in board_cards}

    safe_counts: list[int] = []
    one_off_counts: list[int] = []
    for player in state.players:
        safe = one_off = 0
        for card in player.hand:
            offset = _rank_offset(
                card,
                safe_ranks=safe_ranks,
                one_off_ranks=one_off_ranks,
            ) + _color_offset(card.color, board_colors)
            if offset == 0:
                safe += 1
            elif offset == 1:
                one_off += 1
        safe_counts.append(safe)
        one_off_counts.append(one_off)
    return safe_counts, one_off_counts


def rank_color_offset_count_for_player(
    state: GameState, player_index: int
) -> tuple[int, int]:
    """Return zero/one-offset hand-card counts for one player."""
    board_cards = tuple(
        card for stack in state.board.values() for card in stack
    )
    safe_ranks, one_off_ranks = _rank_offset_sets(
        card.rank_index for card in board_cards
    )
    board_colors = {card.color for card in board_cards}
    safe = one_off = 0
    for card in state.players[player_index].hand:
        offset = _rank_offset(
            card,
            safe_ranks=safe_ranks,
            one_off_ranks=one_off_ranks,
        ) + _color_offset(card.color, board_colors)
        if offset == 0:
            safe += 1
        elif offset == 1:
            one_off += 1
    return safe, one_off


def _rank_offset_sets(
    ranks: Iterable[int],
) -> tuple[frozenset[int], frozenset[int]]:
    existing = frozenset(int(rank) for rank in ranks)
    if not existing:
        safe = frozenset(range(BOARD_SIZE))
    else:
        safe_windows = tuple(
            range(start, start + FRAME_SIZE)
            for start in range(BOARD_SIZE - FRAME_SIZE + 1)
            if existing <= set(range(start, start + FRAME_SIZE))
        )
        safe = frozenset(rank for window in safe_windows for rank in window)
    one_off = frozenset(
        rank
        for safe_rank in safe
        for rank in (safe_rank - 1, safe_rank + 1)
        if 0 <= rank < BOARD_SIZE and rank not in safe
    )
    return safe, one_off


def _rank_offset(
    card: Card,
    *,
    safe_ranks: frozenset[int],
    one_off_ranks: frozenset[int],
) -> int:
    if card.rank_index in safe_ranks:
        return 0
    if card.rank_index in one_off_ranks:
        return 1
    return 2


def _color_offset(color: Color, board_colors: set[Color]) -> int:
    if color in board_colors or len(board_colors) <= 2:
        return 0
    return 1
