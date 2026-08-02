"""Convert V2 replays to V2 board-columns value tensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from yellowstone.replay_v2 import read_replay_shard, records_from_replay
from yellowstone.value_board_columns_v2 import (
    BOARD_COLUMNS_LEFT_MARGIN_CLASSES,
    CANONICALIZATION_BOARD_COLUMNS_V2,
    VALUE_SCHEMA_BOARD_COLUMNS_V2,
    board_columns_from_canonical_board,
    board_columns_v2_metadata,
)
from yellowstone.value_v2 import canonical_tensors_v2


def convert_replay_shards(
    source: str | Path,
    output: str | Path,
    *,
    expected_games: int | None = None,
    game_id_rebase: int = 0,
) -> dict[str, object]:
    import numpy as np

    source_path = Path(source)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = tuple(sorted(source_path.glob("part_*.jsonl.gz")))
    if not paths:
        raise FileNotFoundError(f"no replay shards found at {source_path}")
    converted_files = skipped_files = records = 0
    margin_counts = [0] * BOARD_COLUMNS_LEFT_MARGIN_CLASSES
    for path in paths:
        destination = output_path / path.name.replace(".jsonl.gz", ".npz")
        if destination.is_file():
            skipped_files += 1
            continue
        rows = [
            record
            for game in read_replay_shard(path)
            for record in records_from_replay(game)
        ]
        encoded = [canonical_tensors_v2(record) for record in rows]
        board, context, stats = board_columns_from_canonical_board(
            np.stack([item[0] for item in encoded]),
            np.stack([item[1] for item in encoded]),
        )
        source_game_ids = np.asarray([record.game_id for record in rows], dtype=np.int64)
        temporary = destination.with_suffix(f".npz.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                board=board,
                context=context,
                target=np.asarray([record.target for record in rows], dtype=np.float32),
                game_id=source_game_ids - game_id_rebase,
                source_game_id=source_game_ids,
                perspective=np.asarray(
                    [record.perspective_player_index for record in rows],
                    dtype=np.int8,
                ),
            )
        os.replace(temporary, destination)
        converted_files += 1
        records += stats.records
        for index, value in enumerate(stats.left_margin_counts):
            margin_counts[index] += value

    game_ids: set[int] = set()
    source_game_ids: set[int] = set()
    compressed_bytes = 0
    for path in sorted(output_path.glob("part_*.npz")):
        with np.load(path) as archive:
            if archive["board"].shape[1:] != (1, 7, 3):
                raise ValueError(f"unexpected board shape in {path.name}")
            game_ids.update(int(value) for value in np.unique(archive["game_id"]))
            source_game_ids.update(int(value) for value in np.unique(archive["source_game_id"]))
        compressed_bytes += path.stat().st_size
    if not game_ids:
        raise ValueError("conversion produced no game IDs")
    if expected_games is not None and len(game_ids) != expected_games:
        raise ValueError(f"converted {len(game_ids)} games, expected {expected_games}")
    if game_ids != set(range(len(game_ids))):
        raise ValueError("rebased game IDs are not continuous from zero")
    manifest = {
        **board_columns_v2_metadata(preplay=False),
        "status": "complete",
        "canonicalization": CANONICALIZATION_BOARD_COLUMNS_V2,
        "schema": VALUE_SCHEMA_BOARD_COLUMNS_V2,
        "source": str(source_path),
        "output": str(output_path),
        "source_shards": len(paths),
        "converted_files": converted_files,
        "skipped_files": skipped_files,
        "games": len(game_ids),
        "records_from_newly_converted_files": records,
        "left_margin_counts_from_newly_converted_files": margin_counts,
        "game_id_rebase": game_id_rebase,
        "source_game_id_min": min(source_game_ids),
        "source_game_id_max": max(source_game_ids),
        "rebased_game_id_min": min(game_ids),
        "rebased_game_id_max": max(game_ids),
        "compressed_bytes": compressed_bytes,
    }
    _write_json(output_path / "conversion_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int)
    parser.add_argument("--game-id-rebase", type=int, default=0)
    args = parser.parse_args()
    convert_replay_shards(
        args.source,
        args.output,
        expected_games=args.expected_games,
        game_id_rebase=args.game_id_rebase,
    )


if __name__ == "__main__":
    main()
