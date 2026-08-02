"""Pure game-rule helpers for Yellowstone Park."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from random import Random
from typing import Iterable, Mapping

from yellowstone.types import (
    BOARD_SIZE,
    DEFAULT_LOSS_SCORE,
    FRAME_SIZE,
    GAME_END_LOSS_SCORE,
    HAND_SIZE,
    MAX_PLAYERS,
    MIN_PLAYERS,
    Action,
    Board,
    Card,
    Color,
    EndTurnAction,
    Frame,
    GameState,
    Phase,
    PlaceCardAction,
    PlayerState,
    Position,
    RefillAction,
    RefillSource,
)


class InvalidActionError(ValueError):
    """Raised when an action is not legal for the current state."""


def create_deck() -> tuple[Card, ...]:
    """Create the full 56-card deck in deterministic order."""
    return tuple(
        Card(color=color, rank_index=rank_index)
        for color in Color
        for rank_index in range(BOARD_SIZE)
        for _ in range(2)
    )


def sort_hand(hand: Iterable[Card]) -> tuple[Card, ...]:
    """Return a deterministic rank-first hand order."""
    return tuple(sorted(hand, key=lambda card: (card.rank_index, card.color.value)))


def create_initial_state(
    player_count: int,
    *,
    seed: int | None = None,
    rng: Random | None = None,
    starting_loss_score: int = DEFAULT_LOSS_SCORE,
) -> GameState:
    """Create a shuffled initial state for a 4-5 player game."""
    if not MIN_PLAYERS <= player_count <= MAX_PLAYERS:
        raise ValueError("player_count must be 4 or 5")
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")

    local_rng = rng if rng is not None else Random(seed)
    deck = list(create_deck())
    local_rng.shuffle(deck)

    players: list[PlayerState] = []
    for _ in range(player_count):
        hand = sort_hand(_draw_from_list(deck, HAND_SIZE))
        players.append(PlayerState(hand=hand, loss_score=starting_loss_score))

    initial_card = deck.pop(0)
    initial_position = Position(x=BOARD_SIZE // 2, y=initial_card.rank_index)
    board: Board = {initial_position: (initial_card,)}

    return GameState(
        players=tuple(players),
        board=board,
        deck=tuple(deck),
        current_player_index=0,
        last_turn_play_counts=tuple(0 for _ in players),
    )


@cache
def all_frames() -> tuple[Frame, ...]:
    """Return all valid 3x3 frames."""
    return tuple(
        Frame(x=x, y=y)
        for y in range(BOARD_SIZE - FRAME_SIZE + 1)
        for x in range(BOARD_SIZE - FRAME_SIZE + 1)
    )


@cache
def frame_positions(frame: Frame) -> frozenset[Position]:
    """Return positions inside a valid 3x3 frame."""
    if not is_valid_frame(frame):
        raise ValueError(f"invalid frame: {frame}")
    return frozenset(
        Position(x=x, y=y)
        for y in range(frame.y, frame.y + FRAME_SIZE)
        for x in range(frame.x, frame.x + FRAME_SIZE)
    )


def is_valid_position(position: Position) -> bool:
    """Return whether a position is inside the 7x7 board."""
    return 0 <= position.x < BOARD_SIZE and 0 <= position.y < BOARD_SIZE


def is_valid_frame(frame: Frame) -> bool:
    """Return whether a frame fits inside the board."""
    return (
        0 <= frame.x <= BOARD_SIZE - FRAME_SIZE
        and 0 <= frame.y <= BOARD_SIZE - FRAME_SIZE
    )


@cache
def frames_containing(position: Position) -> tuple[Frame, ...]:
    """Return valid frames that contain a position."""
    if not is_valid_position(position):
        return ()
    return tuple(frame for frame in all_frames() if position in frame_positions(frame))


def board_fits_in_some_frame(board: Board) -> bool:
    """Return whether all occupied cells fit together in at least one frame."""
    positions = board.keys()
    if not positions:
        return True

    min_x = min(position.x for position in positions)
    max_x = max(position.x for position in positions)
    min_y = min(position.y for position in positions)
    max_y = max(position.y for position in positions)
    return (
        0 <= min_x <= max_x < BOARD_SIZE
        and 0 <= min_y <= max_y < BOARD_SIZE
        and max_x - min_x < FRAME_SIZE
        and max_y - min_y < FRAME_SIZE
    )


def legal_actions(state: GameState) -> tuple[Action, ...]:
    """Return legal actions for the current state."""
    if state.phase == Phase.GAME_OVER:
        return ()
    if state.phase == Phase.REFILL:
        return _legal_refill_actions(state)
    if state.phase != Phase.PLAY:
        return ()

    actions: list[Action] = []
    if state.cards_played_this_turn == 1:
        actions.append(EndTurnAction())
    if state.cards_played_this_turn >= 2:
        return tuple(actions)

    player = state.players[state.current_player_index]
    # A settlement can leave the next player with no hand at turn start.
    # They must resolve a refill rather than leaving the game with no legal
    # action.  This is the same refill choice used after an empty one-card
    # turn, but it occurs before any card has been played this turn.
    if not player.hand and state.cards_played_this_turn == 0:
        return _legal_refill_actions(state)
    color_columns, column_colors = _placement_index(state.board)
    for hand_index, card in enumerate(player.hand):
        for position in _legal_positions_from_index(card, color_columns, column_colors):
            for frame in frames_containing(position):
                actions.append(
                    PlaceCardAction(
                        hand_index=hand_index,
                        position=position,
                        frame=frame,
                    )
                )
    return tuple(actions)


def legal_positions_for_card(board: Board, card: Card) -> tuple[Position, ...]:
    """Return legal board positions for a card before frame selection."""
    return _legal_positions_from_index(card, *_placement_index(board))


def _legal_positions_from_index(
    card: Card,
    color_columns: Mapping[Color, frozenset[int]],
    column_colors: tuple[frozenset[Color], ...],
) -> tuple[Position, ...]:
    """Return legal positions using a precomputed summary of the board."""
    existing_columns = color_columns.get(card.color)
    if existing_columns:
        return tuple(
            Position(x=x, y=card.rank_index)
            for x in sorted(existing_columns)
            if 0 <= x < BOARD_SIZE
        )
    return tuple(
        Position(x=x, y=card.rank_index)
        for x, colors in enumerate(column_colors)
        if not colors or colors == {card.color}
    )


def _placement_index(
    board: Board,
) -> tuple[dict[Color, frozenset[int]], tuple[frozenset[Color], ...]]:
    """Summarize board colors once for repeated placement checks."""
    columns_by_color: dict[Color, set[int]] = {}
    colors_by_column = [set() for _ in range(BOARD_SIZE)]
    for position, stack in board.items():
        for card in stack:
            columns_by_color.setdefault(card.color, set()).add(position.x)
            if 0 <= position.x < BOARD_SIZE:
                colors_by_column[position.x].add(card.color)
    return (
        {
            color: frozenset(columns)
            for color, columns in columns_by_color.items()
        },
        tuple(frozenset(colors) for colors in colors_by_column),
    )


def can_place_card_at(board: Board, card: Card, position: Position) -> bool:
    """Return whether a card may be placed at a board position."""
    if not is_valid_position(position):
        return False
    if position.y != card.rank_index:
        return False

    color_columns = columns_containing_color(board, card.color)
    if color_columns:
        return position.x in color_columns

    column_colors = colors_in_column(board, position.x)
    return not column_colors or column_colors == {card.color}


def apply_action(state: GameState, action: Action, *, rng: Random | None = None) -> GameState:
    """Apply a legal action and return a new state."""
    if action not in legal_actions(state):
        raise InvalidActionError(f"illegal action: {action}")
    return apply_known_legal_action(state, action, rng=rng)


def apply_known_legal_action(
    state: GameState,
    action: Action,
    *,
    rng: Random | None = None,
    settle_on_empty_deck: bool = True,
) -> GameState:
    """Apply an already validated legal action and return a new state."""
    if isinstance(action, PlaceCardAction):
        return _apply_place_card(state, action)
    if isinstance(action, EndTurnAction):
        return _apply_end_turn(state)
    if isinstance(action, RefillAction):
        return _apply_refill(
            state,
            action,
            rng=rng,
            settle_on_empty_deck=settle_on_empty_deck,
        )
    raise InvalidActionError(f"unknown action: {action}")


def columns_containing_color(board: Board, color: Color) -> frozenset[int]:
    """Return columns that contain at least one card of a color."""
    return frozenset(
        position.x
        for position, stack in board.items()
        if any(card.color == color for card in stack)
    )


def colors_in_column(board: Board, x: int) -> set[Color]:
    """Return all colors currently present in one column."""
    return {
        card.color
        for position, stack in board.items()
        if position.x == x
        for card in stack
    }


def occupied_count_in_frame(board: Board, frame: Frame) -> int:
    """Count occupied cells in a frame; stacked cards count as one cell."""
    positions = frame_positions(frame)
    return sum(1 for position in board if position in positions)


def winners(state: GameState) -> tuple[int, ...]:
    """Return winner indexes for a finished state or an empty tuple otherwise."""
    if all(player.loss_score < GAME_END_LOSS_SCORE for player in state.players):
        return ()
    lowest_score = min(player.loss_score for player in state.players)
    return tuple(
        index
        for index, player in enumerate(state.players)
        if player.loss_score == lowest_score
    )


def _apply_place_card(state: GameState, action: PlaceCardAction) -> GameState:
    player = state.players[state.current_player_index]
    card = player.hand[action.hand_index]
    board = {position: stack for position, stack in state.board.items()}

    frame_cells = frame_positions(action.frame)
    occupied_before = occupied_count_in_frame(board, action.frame)
    was_occupied = action.position in board
    board[action.position] = board.get(action.position, ()) + (card,)
    occupied_after = occupied_count_in_frame(board, action.frame)

    negative_cards: list[Card] = []
    kept_board: Board = {}
    for position, stack in board.items():
        if position in frame_cells:
            kept_board[position] = stack
        else:
            negative_cards.extend(stack)

    new_hand = _remove_at(player.hand, action.hand_index)
    score_delta = _positive_score_delta(
        occupied_before=occupied_before,
        occupied_after=occupied_after,
        was_occupied=was_occupied,
    )
    new_player = replace(
        player,
        hand=new_hand,
        negative_cards=player.negative_cards + tuple(negative_cards),
        loss_score=max(0, player.loss_score - score_delta),
    )
    players = _replace_player(state.players, state.current_player_index, new_player)
    cards_played = state.cards_played_this_turn + 1
    phase = Phase.REFILL if cards_played == 2 else Phase.PLAY

    return replace(
        state,
        players=players,
        board=kept_board,
        phase=phase,
        cards_played_this_turn=cards_played,
    )


def _apply_refill(
    state: GameState,
    action: RefillAction,
    *,
    rng: Random | None,
    settle_on_empty_deck: bool,
) -> GameState:
    player = state.players[state.current_player_index]
    deck = list(state.deck)
    hand = list(player.hand)
    negative_cards = list(player.negative_cards)

    if action.source == RefillSource.DECK:
        draw_count = max(0, HAND_SIZE - len(hand))
        hand.extend(_draw_from_list(deck, draw_count))
    elif action.source == RefillSource.NEGATIVE_CARDS:
        local_rng = rng if rng is not None else Random()
        local_rng.shuffle(negative_cards)
        hand.extend(_draw_from_list(negative_cards, HAND_SIZE))
    elif action.source == RefillSource.NONE:
        pass
    else:
        raise InvalidActionError(f"unknown refill source: {action.source}")

    new_player = replace(
        player,
        hand=sort_hand(hand),
        negative_cards=tuple(negative_cards),
    )
    players = _replace_player(state.players, state.current_player_index, new_player)
    next_state = replace(state, players=players, deck=tuple(deck))

    deck_was_exhausted = not deck and (
        settle_on_empty_deck or len(hand) < HAND_SIZE
    )
    if action.source == RefillSource.DECK and deck_was_exhausted:
        return _settle_after_deck_exhaustion(next_state, rng=rng)

    return _advance_to_next_player(next_state)


def _settle_after_deck_exhaustion(
    state: GameState,
    *,
    rng: Random | None,
) -> GameState:
    settled_players: list[PlayerState] = []
    collected_negative_cards: list[Card] = []
    for player in state.players:
        collected_negative_cards.extend(player.negative_cards)
        settled_players.append(
            replace(
                player,
                negative_cards=(),
                loss_score=player.loss_score + len(player.negative_cards),
            )
        )

    settled_state = replace(
        state,
        players=tuple(settled_players),
        settlement_count=state.settlement_count + 1,
    )
    game_winners = winners(settled_state)
    if game_winners:
        return replace(
            settled_state,
            phase=Phase.GAME_OVER,
            cards_played_this_turn=0,
            winners=game_winners,
            last_turn_play_counts=_updated_last_turn_play_counts(state),
        )

    local_rng = rng if rng is not None else Random()
    local_rng.shuffle(collected_negative_cards)
    return replace(
        settled_state,
        deck=tuple(collected_negative_cards),
        current_player_index=(state.current_player_index + 1) % len(state.players),
        phase=Phase.PLAY,
        cards_played_this_turn=0,
        last_turn_play_counts=_updated_last_turn_play_counts(state),
    )


def _legal_refill_actions(state: GameState) -> tuple[Action, ...]:
    player = state.players[state.current_player_index]
    if player.hand:
        return (RefillAction(RefillSource.DECK), RefillAction(RefillSource.NONE))
    actions: list[Action] = [RefillAction(RefillSource.DECK)]
    if len(player.negative_cards) >= HAND_SIZE:
        actions.append(RefillAction(RefillSource.NEGATIVE_CARDS))
    return tuple(actions)


def _apply_end_turn(state: GameState) -> GameState:
    player = state.players[state.current_player_index]
    if player.hand:
        return _advance_to_next_player(state)
    return replace(state, phase=Phase.REFILL)


def _advance_to_next_player(state: GameState) -> GameState:
    return replace(
        state,
        current_player_index=(state.current_player_index + 1) % len(state.players),
        phase=Phase.PLAY,
        cards_played_this_turn=0,
        last_turn_play_counts=_updated_last_turn_play_counts(state),
    )


def _updated_last_turn_play_counts(state: GameState) -> tuple[int, ...]:
    counts = list(_last_turn_play_counts(state))
    counts[state.current_player_index] = state.cards_played_this_turn
    return tuple(counts)


def _last_turn_play_counts(state: GameState) -> tuple[int, ...]:
    if len(state.last_turn_play_counts) == len(state.players):
        return state.last_turn_play_counts
    return tuple(0 for _ in state.players)


def _positive_score_delta(
    *,
    occupied_before: int,
    occupied_after: int,
    was_occupied: bool,
) -> int:
    if was_occupied:
        return 0
    if occupied_before < 8 <= occupied_after:
        return 1
    if occupied_before < 9 <= occupied_after:
        return 3
    return 0


def _replace_player(
    players: tuple[PlayerState, ...],
    index: int,
    player: PlayerState,
) -> tuple[PlayerState, ...]:
    return players[:index] + (player,) + players[index + 1 :]


def _remove_at(cards: tuple[Card, ...], index: int) -> tuple[Card, ...]:
    return cards[:index] + cards[index + 1 :]


def _draw_from_list(cards: list[Card], count: int) -> tuple[Card, ...]:
    draw_count = min(count, len(cards))
    drawn = tuple(cards[:draw_count])
    del cards[:draw_count]
    return drawn


def _flatten_board_cards(board: Board) -> Iterable[Card]:
    for stack in board.values():
        yield from stack
