"""Convert replay V2 shards into exploratory V2 tensor archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yellowstone.replay_v2 import read_replay_shard, records_from_replay
from yellowstone.value_v2_exploratory import (
    CANONICALIZATION_V2_EXPLORATORY,
    VALUE_CONTEXT_SIZE_V2_EXPLORATORY,
    VALUE_SCHEMA_V2_EXPLORATORY,
    canonical_tensors_v2_exploratory,
)


def convert_replay_shards_v2_exploratory(
    source: str | Path,
    output: str | Path,
    *,
    start_part: int | None = None,
    end_part: int | None = None,
) -> dict[str, object]:
    import numpy as np

    source_path = Path(source)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = (
        tuple(sorted(source_path.glob("part_*.jsonl.gz")))
        if source_path.is_dir()
        else (source_path,)
    )
    if (start_part is None) != (end_part is None):
        raise ValueError("start_part and end_part must be supplied together")
    if start_part is not None:
        if start_part < 0 or end_part is None or end_part < start_part:
            raise ValueError("invalid inclusive part range")
        paths = tuple(
            path
            for path in paths
            if start_part <= _part_number(path) <= end_part
        )
    if not paths:
        raise FileNotFoundError(f"no replay shards found at {source_path}")
    converted_games = converted_records = compressed_bytes = 0
    for path in paths:
        destination = output_path / path.name.replace(
            ".jsonl.gz", ".npz"
        )
        if destination.exists():
            with np.load(destination) as archive:
                if archive["context"].shape[1] != (
                    VALUE_CONTEXT_SIZE_V2_EXPLORATORY
                ):
                    raise ValueError(
                        f"existing archive has wrong context: {destination}"
                    )
                converted_games += len(set(archive["game_id"].tolist()))
                converted_records += int(len(archive["target"]))
            compressed_bytes += destination.stat().st_size
            continue
        boards = []
        contexts = []
        targets = []
        game_ids = []
        perspectives = []
        transforms = []
        games = tuple(read_replay_shard(path))
        for game in games:
            for record in records_from_replay(game):
                board, context, transform = (
                    canonical_tensors_v2_exploratory(record)
                )
                boards.append(board)
                contexts.append(context)
                targets.append(record.target)
                game_ids.append(record.game_id)
                perspectives.append(record.perspective_player_index)
                transforms.append(
                    (
                        int(transform.vertical_reflection),
                        int(transform.horizontal_reflection),
                        *transform.old_to_new_color,
                    )
                )
        if not boards:
            raise ValueError(f"replay shard has no records: {path}")
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                board=np.stack(boards),
                context=np.stack(contexts),
                target=np.asarray(targets, dtype=np.float32),
                game_id=np.asarray(game_ids, dtype=np.int64),
                perspective=np.asarray(perspectives, dtype=np.int8),
                canonical_transform=np.asarray(
                    transforms, dtype=np.int8
                ),
            )
        temporary.replace(destination)
        converted_games += len(games)
        converted_records += len(boards)
        compressed_bytes += destination.stat().st_size
    manifest: dict[str, object] = {
        "schema": VALUE_SCHEMA_V2_EXPLORATORY,
        "canonicalization": CANONICALIZATION_V2_EXPLORATORY,
        "context_size": VALUE_CONTEXT_SIZE_V2_EXPLORATORY,
        "source": str(source_path),
        "shards": len(paths),
        "games": converted_games,
        "records": converted_records,
        "compressed_bytes": compressed_bytes,
    }
    temporary_manifest = output_path / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(output_path / "manifest.json")
    return manifest


def _part_number(path: Path) -> int:
    try:
        return int(
            path.stem.removeprefix("part_").removesuffix(".jsonl")
        )
    except ValueError as error:
        raise ValueError(f"invalid replay shard name: {path.name}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-part", type=int)
    parser.add_argument("--end-part", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            convert_replay_shards_v2_exploratory(
                args.source,
                args.output,
                start_part=args.start_part,
                end_part=args.end_part,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
