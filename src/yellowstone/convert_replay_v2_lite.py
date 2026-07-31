"""Convert V2 replays to restartable V2-lite transition tensor shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from yellowstone.replay_v2 import read_replay_shard
from yellowstone.replay_v2_lite import records_from_replay_v2_lite
from yellowstone.value_v2_lite import (
    CANONICALIZATION_V2_LITE,
    VALUE_SCHEMA_V2_LITE,
    canonical_tensors_v2_lite,
)


def convert_replay_shards_v2_lite(
    source: str | Path, output: str | Path
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
    if not paths:
        raise FileNotFoundError(f"no replay shards found at {source_path}")
    progress_path = output_path / "conversion_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"shards": {}}
    )
    expected = {
        "schema": VALUE_SCHEMA_V2_LITE,
        "canonicalization": CANONICALIZATION_V2_LITE,
        "source": str(source_path),
    }
    for key, value in expected.items():
        if key in progress and progress[key] != value:
            raise ValueError(f"conversion progress differs at {key}")
        progress[key] = value

    for path in paths:
        destination = output_path / path.name.replace(".jsonl.gz", ".npz")
        saved = progress["shards"].get(path.name)
        if destination.is_file() and saved:
            continue
        boards = []
        contexts = []
        targets = []
        game_ids = []
        perspectives = []
        transforms = []
        games = tuple(read_replay_shard(path))
        for game in games:
            for record in records_from_replay_v2_lite(game):
                board, context, transform = canonical_tensors_v2_lite(record)
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
            raise ValueError(f"replay shard contains no records: {path}")
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
        os.replace(temporary, destination)
        progress["shards"][path.name] = {
            "games": len(games),
            "records": len(boards),
            "compressed_bytes": destination.stat().st_size,
        }
        _write_json_atomic(progress_path, progress)

    facts = tuple(progress["shards"].values())
    manifest = {
        **expected,
        "status": "complete",
        "shards": len(facts),
        "games": sum(int(item["games"]) for item in facts),
        "records": sum(int(item["records"]) for item in facts),
        "compressed_bytes": sum(
            int(item["compressed_bytes"]) for item in facts
        ),
    }
    _write_json_atomic(output_path / "manifest.json", manifest)
    return manifest


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert V2 replays to V2-lite")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            convert_replay_shards_v2_lite(args.source, args.output),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

