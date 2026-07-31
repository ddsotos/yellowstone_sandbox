"""Convert replayable V2 game shards into strict-canonical tensor archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yellowstone.replay_v2 import read_replay_shard, records_from_replay
from yellowstone.value_v2 import canonical_tensors_v2


def convert_replay_shards(
    source: str | Path,
    output: str | Path,
) -> dict[str, int]:
    import numpy as np

    source_path = Path(source)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = (
        tuple(sorted(source_path.glob("part_*.jsonl.gz")))
        if source_path.is_dir()
        else (source_path,)
    )
    if not paths:
        raise FileNotFoundError(f"no replay shards found at {source_path}")
    converted_games = 0
    converted_records = 0
    compressed_bytes = 0
    for path in paths:
        destination = output_path / path.name.replace(".jsonl.gz", ".npz")
        boards = []
        contexts = []
        targets = []
        game_ids = []
        perspectives = []
        transforms = []
        games = tuple(read_replay_shard(path))
        for game in games:
            for record in records_from_replay(game):
                board, context, transform = canonical_tensors_v2(record)
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
            raise ValueError(f"replay shard contains no value records: {path}")
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                board=np.stack(boards),
                context=np.stack(contexts),
                target=np.asarray(targets, dtype=np.float32),
                game_id=np.asarray(game_ids, dtype=np.int64),
                perspective=np.asarray(perspectives, dtype=np.int8),
                canonical_transform=np.asarray(transforms, dtype=np.int8),
            )
        temporary.replace(destination)
        converted_games += len(games)
        converted_records += len(boards)
        compressed_bytes += destination.stat().st_size
    manifest = {
        "schema": "yellowstone.value.v2",
        "canonicalization": "strict_residual_v2",
        "source": str(source_path),
        "shards": len(paths),
        "games": converted_games,
        "records": converted_records,
        "compressed_bytes": compressed_bytes,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert V2 replay shards")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            convert_replay_shards(args.source, args.output),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
