"""Convert replay shards to V2-lite tensors with explicit action cards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from yellowstone.replay_v2 import read_replay_shard
from yellowstone.replay_v2_lite import records_from_replay_v2_lite
from yellowstone.value_v2_lite_action import (
    CANONICALIZATION_V2_LITE_ACTION,
    HISTORY_SEMANTICS_V2_LITE_ACTION,
    VALUE_SCHEMA_V2_LITE_ACTION,
    action_cards_from_transition,
    canonical_tensors_v2_lite_action,
)


def convert_replay_shards_v2_lite_action(
    source: str | Path,
    output: str | Path,
    *,
    game_id_rebase: int,
    expected_games: int | None = None,
    expected_source_game_id_min: int | None = None,
    expected_source_game_id_max: int | None = None,
) -> dict:
    import numpy as np

    source_path = Path(source)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        sorted(
            source_path.glob("part_*.jsonl.gz"),
            key=lambda path: int(path.name[5 : -len(".jsonl.gz")]),
        )
    )
    if not paths:
        raise FileNotFoundError(f"no replay shards found at {source_path}")
    progress_path = output_path / "conversion_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8-sig"))
        if progress_path.exists()
        else {"shards": {}}
    )
    expected = {
        "value_schema": VALUE_SCHEMA_V2_LITE_ACTION,
        "canonicalization": CANONICALIZATION_V2_LITE_ACTION,
        "history_semantics": HISTORY_SEMANTICS_V2_LITE_ACTION,
        "opponent_private_inputs": False,
        "source": str(source_path),
        "game_id_rebase": game_id_rebase,
    }
    for key, value in expected.items():
        if key in progress and progress[key] != value:
            raise ValueError(f"V2-lite-action progress differs at {key}")
        progress[key] = value

    for path in paths:
        destination = output_path / path.name.replace(".jsonl.gz", ".npz")
        if destination.is_file() and progress["shards"].get(path.name):
            continue
        rows = [
            record
            for game in read_replay_shard(path)
            for record in records_from_replay_v2_lite(game)
        ]
        if not rows:
            raise ValueError(f"replay shard produced no records: {path}")
        encoded = [
            canonical_tensors_v2_lite_action(record) for record in rows
        ]
        cards = [action_cards_from_transition(record) for record in rows]
        source_game_ids = np.asarray(
            [record.game_id for record in rows], dtype=np.int64
        )
        temporary = destination.with_suffix(f".npz.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                board=np.stack([item[0] for item in encoded]),
                context=np.stack([item[1] for item in encoded]),
                target=np.asarray(
                    [record.target for record in rows], dtype=np.float32
                ),
                game_id=source_game_ids - game_id_rebase,
                source_game_id=source_game_ids,
                perspective=np.asarray(
                    [record.perspective_player_index for record in rows],
                    dtype=np.int8,
                ),
                play_count=np.asarray(
                    [len(value) for value in cards], dtype=np.int8
                ),
                canonical_transform=np.asarray(
                    [
                        (
                            int(item[2].vertical_reflection),
                            int(item[2].horizontal_reflection),
                            *item[2].old_to_new_color,
                        )
                        for item in encoded
                    ],
                    dtype=np.int8,
                ),
            )
        os.replace(temporary, destination)
        progress["shards"][path.name] = {
            "games": len({record.game_id for record in rows}),
            "records": len(rows),
            "one_card_records": sum(len(value) == 1 for value in cards),
            "two_card_records": sum(len(value) == 2 for value in cards),
            "compressed_bytes": destination.stat().st_size,
        }
        _write_json_atomic(progress_path, progress)

    facts = tuple(progress["shards"].values())
    source_ids = set()
    rebased_ids = set()
    for path in sorted(output_path.glob("part_*.npz")):
        with np.load(path) as archive:
            source_ids.update(int(value) for value in np.unique(archive["source_game_id"]))
            rebased_ids.update(int(value) for value in np.unique(archive["game_id"]))
    if not source_ids:
        raise ValueError("converted V2-lite-action data has no game IDs")
    source_min, source_max = min(source_ids), max(source_ids)
    if (
        expected_source_game_id_min is not None
        and source_min != expected_source_game_id_min
    ):
        raise ValueError("unexpected minimum source game ID")
    if (
        expected_source_game_id_max is not None
        and source_max != expected_source_game_id_max
    ):
        raise ValueError("unexpected maximum source game ID")
    if source_ids != set(range(source_min, source_max + 1)):
        raise ValueError("source game IDs are not continuous")
    games = len(source_ids)
    if expected_games is not None and games != expected_games:
        raise ValueError(f"converted {games} games, expected {expected_games}")
    if rebased_ids != set(range(games)):
        raise ValueError("rebased game IDs are not continuous from zero")
    manifest = {
        **expected,
        "status": "complete",
        "source_shards": len(paths),
        "converted_files": len(facts),
        "games": games,
        "records": sum(int(item["records"]) for item in facts),
        "one_card_records": sum(
            int(item["one_card_records"]) for item in facts
        ),
        "two_card_records": sum(
            int(item["two_card_records"]) for item in facts
        ),
        "compressed_bytes": sum(
            int(item["compressed_bytes"]) for item in facts
        ),
        "source_game_id_min": source_min,
        "source_game_id_max": source_max,
        "rebased_game_id_min": min(rebased_ids),
        "rebased_game_id_max": max(rebased_ids),
        "labels_derived_from_terminal_winners": True,
        "action_cards_recovered_from_own_hand_transition": True,
        "unordered_action_cards": True,
    }
    _write_json_atomic(output_path / "manifest.json", manifest)
    return manifest


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game-id-rebase", type=int, required=True)
    parser.add_argument("--expected-games", type=int)
    parser.add_argument("--expected-source-game-id-min", type=int)
    parser.add_argument("--expected-source-game-id-max", type=int)
    args = parser.parse_args()
    result = convert_replay_shards_v2_lite_action(
        args.source,
        args.output,
        game_id_rebase=args.game_id_rebase,
        expected_games=args.expected_games,
        expected_source_game_id_min=args.expected_source_game_id_min,
        expected_source_game_id_max=args.expected_source_game_id_max,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
