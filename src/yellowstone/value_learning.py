"""Player-perspective Monte-Carlo win-value data and training utilities.

The collector intentionally plays only :class:`HeuristicBot` games.  Each
record is a completed turn, labelled after the game finishes, from the player
who completed that turn's perspective.  No deck order or opponent card is
encoded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.types import Card, Color, GameState, Phase, PlaceCardAction


COLOR_ORDER = (Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW)
RANK_BOARD_CHANNELS = len(COLOR_ORDER) * 7 + 1
HAND_SIZE = 6
HISTORY_SIZE = 2
# own hand 6*(present + color one-hot + rank), players 4*3, turn 4+3+2,
# and two placements*(present + relative-player one-hot + color one-hot +
# rank + score delta + negative-card delta).
VALUE_CONTEXT_SIZE = HAND_SIZE * 6 + 12 + 9 + HISTORY_SIZE * 12


@dataclass(frozen=True, slots=True)
class RecentPlacement:
    """One public card placement, expressed later relative to a viewer."""

    player_index: int
    card: Card
    score_delta: int
    negative_card_delta: int


@dataclass(frozen=True, slots=True)
class ValueRecord:
    """One terminally-labelled, player-perspective turn-end observation."""

    game_id: int
    perspective_player_index: int
    state: GameState
    history: tuple[RecentPlacement, ...]
    target: float
    board_center_frame_history: tuple[tuple[int, int], ...] = ()
    board_center_chain_states: tuple[GameState, ...] = ()
    refill_count: int = 0


def collect_heuristic_games(
    *, game_count: int, seed: int = 0, game_id_offset: int = 0
) -> list[ValueRecord]:
    """Collect completed-turn records from deterministic heuristic games."""
    if game_count <= 0:
        raise ValueError("game_count must be positive")
    seed_rng = Random(seed)
    records: list[ValueRecord] = []
    bot = HeuristicBot()
    for local_game_id in range(game_count):
        game_id = game_id_offset + local_game_id
        state = create_initial_state(4, seed=seed_rng.randrange(2**63))
        history: list[RecentPlacement] = []
        pending: list[tuple[int, GameState, tuple[RecentPlacement, ...]]] = []
        while state.phase != Phase.GAME_OVER:
            player_index = state.current_player_index
            action = bot.choose_action(state)
            if action is None:
                raise RuntimeError("heuristic bot returned no action before game end")
            before = state
            state = apply_known_legal_action(state, action, rng=seed_rng)
            if isinstance(action, PlaceCardAction):
                card = before.players[player_index].hand[action.hand_index]
                history.append(
                    RecentPlacement(
                        player_index=player_index,
                        card=card,
                        score_delta=(
                            before.players[player_index].loss_score
                            - state.players[player_index].loss_score
                        ),
                        negative_card_delta=(
                            len(state.players[player_index].negative_cards)
                            - len(before.players[player_index].negative_cards)
                        ),
                    )
                )
                del history[:-HISTORY_SIZE]
                # The second card ends a normal turn at the public refill
                # boundary.  A one-card turn is recorded when EndTurn occurs.
                if state.phase == Phase.REFILL:
                    pending.append((player_index, state, tuple(history)))
            elif before.cards_played_this_turn == 1:
                pending.append((player_index, state, tuple(history)))
        winner_count = len(state.winners)
        if winner_count == 0:
            raise AssertionError("finished game has no winners")
        for player_index, snapshot, snapshot_history in pending:
            records.append(
                ValueRecord(
                    game_id=game_id,
                    perspective_player_index=player_index,
                    state=snapshot,
                    history=snapshot_history,
                    target=(1.0 / winner_count if player_index in state.winners else 0.0),
                )
            )
    return records


def board_tensor_for_player(record: ValueRecord):
    """Return [29, 7, 7] public board features; player does not affect board."""
    np = _numpy()
    tensor = np.zeros((RANK_BOARD_CHANNELS, 7, 7), dtype=np.float32)
    color_index = {color: index for index, color in enumerate(COLOR_ORDER)}
    for position, stack in record.state.board.items():
        for card in stack:
            channel = color_index[card.color] * 7 + card.rank_index
            tensor[channel, position.y, position.x] += 1.0
            tensor[-1, position.y, position.x] += 1.0
    return tensor


def context_tensor_for_player(record: ValueRecord):
    """Return public-plus-own-hand features in the record player's frame."""
    np = _numpy()
    state = record.state
    viewer = record.perspective_player_index
    values: list[float] = []
    own_hand = state.players[viewer].hand
    for slot in range(HAND_SIZE):
        if slot < len(own_hand):
            card = own_hand[slot]
            values.extend([1.0, *_one_hot(COLOR_ORDER.index(card.color), 4), card.rank_index / 6])
        else:
            values.extend([0.0] * 6)
    for offset in range(4):
        player = state.players[(viewer + offset) % 4]
        values.extend([player.loss_score / 35, len(player.hand) / 6, len(player.negative_cards) / 56])
    values.extend(_one_hot((state.current_player_index - viewer) % 4, 4))
    values.extend(_one_hot((Phase.PLAY, Phase.REFILL, Phase.GAME_OVER).index(state.phase), 3))
    values.extend([state.cards_played_this_turn / 2, state.settlement_count / 10])
    for placement in record.history[-HISTORY_SIZE:]:
        values.extend(
            [
                1.0,
                *_one_hot((placement.player_index - viewer) % 4, 4),
                *_one_hot(COLOR_ORDER.index(placement.card.color), 4),
                placement.card.rank_index / 6,
                placement.score_delta / 3,
                placement.negative_card_delta / 9,
            ]
        )
    missing = HISTORY_SIZE - len(record.history[-HISTORY_SIZE:])
    values.extend([0.0] * (missing * 12))
    if len(values) != VALUE_CONTEXT_SIZE:
        raise AssertionError(f"unexpected context size: {len(values)}")
    return np.asarray(values, dtype=np.float32)


def split_game_ids(game_count: int, *, seed: int = 0) -> tuple[set[int], set[int], set[int]]:
    """Split game IDs 80/10/10, never splitting states from one game."""
    if game_count < 10:
        raise ValueError("at least 10 games are required for an 80/10/10 split")
    ids = list(range(game_count))
    Random(seed).shuffle(ids)
    train_end = game_count * 8 // 10
    validation_end = game_count * 9 // 10
    return set(ids[:train_end]), set(ids[train_end:validation_end]), set(ids[validation_end:])


def save_records(records: Iterable[ValueRecord], path: str | Path) -> None:
    """Save encoded data in a portable NumPy archive."""
    np = _numpy()
    rows = list(records)
    np.savez_compressed(
        path,
        board=np.stack([board_tensor_for_player(row) for row in rows]),
        context=np.stack([context_tensor_for_player(row) for row in rows]),
        target=np.asarray([row.target for row in rows], dtype=np.float32),
        game_id=np.asarray([row.game_id for row in rows], dtype=np.int64),
    )


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if index == value else 0.0 for value in range(size)]


def _numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise ImportError("value data requires `pip install -e .[value]`") from error
    return np


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Yellowstone heuristic value data")
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("data/heuristic_value_data")
    )
    parser.add_argument("--chunk-games", type=int, default=100)
    parser.add_argument("--game-id-offset", type=int, default=0)
    args = parser.parse_args()
    if args.chunk_games <= 0:
        raise ValueError("chunk-games must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    total_records = 0
    for offset in range(0, args.games, args.chunk_games):
        count = min(args.chunk_games, args.games - offset)
        game_offset = args.game_id_offset + offset
        records = collect_heuristic_games(
            game_count=count,
            seed=args.seed + game_offset,
            game_id_offset=game_offset,
        )
        path = args.output / f"part_{game_offset:06d}.npz"
        save_records(records, path)
        total_records += len(records)
        print(f"saved {len(records)} records to {path}")
    print(f"saved {total_records} turn-end records from {args.games} games to {args.output}")


if __name__ == "__main__":
    main()
