"""Text rendering helpers for developer-facing game inspection."""

from __future__ import annotations

from yellowstone.types import BOARD_SIZE, Board, Card, GameState, Position


COLOR_LABELS = {
    "red": "R",
    "blue": "B",
    "green": "G",
    "yellow": "Y",
}


def render_card(card: Card) -> str:
    """Render a card as a compact color/rank label."""
    return f"{COLOR_LABELS[card.color.value]}{card.rank}"


def render_stack(stack: tuple[Card, ...]) -> str:
    """Render one board cell stack."""
    if not stack:
        return "."
    top = render_card(stack[-1])
    if len(stack) == 1:
        return top
    return f"{top}+{len(stack) - 1}"


def render_board(board: Board) -> str:
    """Render the 7x7 board.

    Rows are printed from y=0 to y=6. A stacked cell shows the top card plus
    the number of hidden cards below it, for example R3+1.
    """
    cell_width = _board_cell_width(board)
    header_cells = [str(x).rjust(cell_width) for x in range(BOARD_SIZE)]
    lines = [f"{'y/r':>4} | {' '.join(header_cells)}"]
    separator_width = 7 + BOARD_SIZE * (cell_width + 1)
    lines.append("-" * separator_width)

    for y in range(BOARD_SIZE):
        rendered_cells = [
            render_stack(board.get(Position(x=x, y=y), ())).rjust(cell_width)
            for x in range(BOARD_SIZE)
        ]
        lines.append(f"{y}/{y + 1:>1} | {' '.join(rendered_cells)}")
    return "\n".join(lines)


def render_player_summary(state: GameState) -> str:
    """Render per-player hand, negative-card, and loss-score counts."""
    lines = []
    for index, player in enumerate(state.players):
        marker = "*" if index == state.current_player_index else " "
        lines.append(
            f"{marker}P{index}: hand={len(player.hand)} "
            f"negative={len(player.negative_cards)} loss={player.loss_score}"
        )
    return "\n".join(lines)


def render_state(state: GameState) -> str:
    """Render a complete developer-facing state summary."""
    winners = ",".join(str(winner) for winner in state.winners) or "-"
    sections = [
        (
            f"phase={state.phase.value} current=P{state.current_player_index} "
            f"played={state.cards_played_this_turn} deck={len(state.deck)} "
            f"settlements={state.settlement_count} winners={winners}"
        ),
        render_board(state.board),
        render_player_summary(state),
    ]
    return "\n\n".join(sections)


def _board_cell_width(board: Board) -> int:
    rendered_cells = [render_stack(stack) for stack in board.values()]
    return max([4, *(len(cell) for cell in rendered_cells)])
