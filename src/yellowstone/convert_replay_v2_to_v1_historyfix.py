"""Convert V2 raw replays to canonical V1 tensors with turn-local history."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from random import Random

from yellowstone.game import apply_known_legal_action
from yellowstone.replay_v2 import (
    LEGACY_RULES_VERSION_V2,
    ReplayGameV2,
    read_replay_shard,
)
from yellowstone.types import EndTurnAction, Phase, PlaceCardAction, RefillAction
from yellowstone.value_canonicalization import (
    CANONICALIZATION_NAME,
    canonicalize_value_tensors_with_stats,
)
from yellowstone.value_learning import (
    HISTORY_SIZE,
    RecentPlacement,
    ValueRecord,
    board_tensor_for_player,
    context_tensor_for_player,
)


VALUE_SCHEMA_V1_HISTORYFIX = "yellowstone.value.v1_historyfix"


def records_from_replay_v1_historyfix(
    game: ReplayGameV2,
) -> tuple[ValueRecord, ...]:
    """Rebuild V1 records whose history contains only the evaluated turn."""
    state = game.initial_state
    rng = Random(game.gameplay_seed)
    pending: list[
        tuple[int, object, tuple[RecentPlacement, ...]]
    ] = []
    turn_placements: list[RecentPlacement] = []
    turn_player: int | None = None

    for action in game.actions:
        before = state
        if isinstance(action, PlaceCardAction):
            if before.cards_played_this_turn == 0:
                turn_placements = []
                turn_player = before.current_player_index
            if turn_player != before.current_player_index:
                raise AssertionError("turn player changed during placement")
            card = before.players[turn_player].hand[action.hand_index]

        state = apply_known_legal_action(
            state,
            action,
            rng=rng,
            settle_on_empty_deck=(
                game.rules_version != LEGACY_RULES_VERSION_V2
            ),
        )

        if isinstance(action, PlaceCardAction):
            turn_placements.append(
                RecentPlacement(
                    player_index=turn_player,
                    card=card,
                    score_delta=(
                        before.players[turn_player].loss_score
                        - state.players[turn_player].loss_score
                    ),
                    negative_card_delta=(
                        len(state.players[turn_player].negative_cards)
                        - len(before.players[turn_player].negative_cards)
                    ),
                )
            )
            if len(turn_placements) > HISTORY_SIZE:
                raise AssertionError("a turn contains more than two placements")
            if state.phase == Phase.REFILL:
                pending.append(
                    (turn_player, state, tuple(turn_placements))
                )
        elif isinstance(action, EndTurnAction):
            if turn_player is None or not turn_placements:
                raise AssertionError("one-card completion lacks a placement")
            pending.append((turn_player, state, tuple(turn_placements)))
            turn_placements = []
            turn_player = None
        elif isinstance(action, RefillAction):
            turn_placements = []
            turn_player = None

    if state.phase != Phase.GAME_OVER:
        raise ValueError(f"replay game {game.game_id} did not finish")
    if state.winners != game.winners:
        raise ValueError(f"replay winners differ for game {game.game_id}")
    if not state.winners:
        raise AssertionError("finished replay has no winners")
    winner_count = len(state.winners)
    return tuple(
        ValueRecord(
            game_id=game.game_id,
            perspective_player_index=player_index,
            state=snapshot,
            history=history,
            target=(
                1.0 / winner_count
                if player_index in state.winners
                else 0.0
            ),
        )
        for player_index, snapshot, history in pending
    )


def convert_replay_shards(
    source: str | Path,
    output: str | Path,
    *,
    expected_games: int | None = None,
) -> dict[str, object]:
    """Convert restartable raw replay shards to canonical V1 archives."""
    import numpy as np

    source_path = Path(source)
    output_path = Path(output)
    paths = tuple(sorted(source_path.glob("part_*.jsonl.gz")))
    if not paths:
        raise FileNotFoundError(f"no replay shards found at {source_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    converted_files = skipped_files = 0
    horizontal = vertical = 0
    for index, path in enumerate(paths, start=1):
        destination = output_path / path.name.replace(".jsonl.gz", ".npz")
        if destination.is_file():
            skipped_files += 1
            continue
        rows = [
            record
            for game in read_replay_shard(path)
            for record in records_from_replay_v1_historyfix(game)
        ]
        if not rows:
            raise ValueError(f"replay shard produced no records: {path}")
        board = np.stack(
            [board_tensor_for_player(record) for record in rows]
        )
        context = np.stack(
            [context_tensor_for_player(record) for record in rows]
        )
        board, context, stats = canonicalize_value_tensors_with_stats(
            board, context
        )
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            board=board,
            context=context,
            target=np.asarray(
                [record.target for record in rows], dtype=np.float32
            ),
            game_id=np.asarray(
                [record.game_id for record in rows], dtype=np.int64
            ),
        )
        os.replace(temporary, destination)
        converted_files += 1
        horizontal += stats.horizontal_reflections
        vertical += stats.vertical_reflections
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(
                f"progress={index}/{len(paths)} "
                f"converted={converted_files} skipped={skipped_files}",
                flush=True,
            )

    archive_paths = tuple(sorted(output_path.glob("part_*.npz")))
    game_ids: set[int] = set()
    records = one_card_records = two_card_records = 0
    compressed_bytes = 0
    for path in archive_paths:
        with np.load(path) as archive:
            count = len(archive["target"])
            records += count
            game_ids.update(int(value) for value in archive["game_id"])
            history = archive["context"][:, -HISTORY_SIZE * 12 :]
            second_present = history[:, 12] > 0.5
            two_card_records += int(second_present.sum())
            one_card_records += int(count - second_present.sum())
        compressed_bytes += path.stat().st_size
    if expected_games is not None and len(game_ids) != expected_games:
        raise ValueError(
            f"converted game count differs: {len(game_ids)} != {expected_games}"
        )
    result: dict[str, object] = {
        "value_schema": VALUE_SCHEMA_V1_HISTORYFIX,
        "canonicalization": CANONICALIZATION_NAME,
        "source": str(source_path),
        "output": str(output_path),
        "source_shards": len(paths),
        "converted_files": converted_files,
        "skipped_files": skipped_files,
        "games": len(game_ids),
        "records": records,
        "one_card_records": one_card_records,
        "two_card_records": two_card_records,
        "compressed_bytes": compressed_bytes,
        "history_semantics": "evaluated_turn_only_one_card_zero_padded",
    }
    temporary_manifest = output_path / "conversion_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(
        temporary_manifest, output_path / "conversion_manifest.json"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int)
    args = parser.parse_args()
    convert_replay_shards(
        args.source,
        args.output,
        expected_games=args.expected_games,
    )


if __name__ == "__main__":
    main()
