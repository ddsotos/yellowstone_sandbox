"""Restartable conversion of generation-0 replays to privileged state tensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from yellowstone.privileged_state import (
    CANONICALIZATION_PRIVILEGED_STATE,
    HISTORY_SEMANTICS_PRIVILEGED_STATE,
    VALUE_SCHEMA_PRIVILEGED_STATE,
    encode_privileged_state,
    records_from_replay_privileged_state,
)
from yellowstone.replay_v2 import read_replay_shard


def convert_privileged_state(source: str | Path, output: str | Path) -> dict:
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
        raise FileNotFoundError(f"no replay shards at {source_path}")
    progress_path = output_path / "conversion_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8-sig"))
        if progress_path.exists()
        else {"shards": {}}
    )
    expected = {
        "schema": VALUE_SCHEMA_PRIVILEGED_STATE,
        "canonicalization": CANONICALIZATION_PRIVILEGED_STATE,
        "history_semantics": HISTORY_SEMANTICS_PRIVILEGED_STATE,
        "source": str(source_path),
        "privileged_inputs": True,
    }
    for key, value in expected.items():
        if key in progress and progress[key] != value:
            raise ValueError(f"conversion progress differs at {key}")
        progress[key] = value

    for path in paths:
        destination = output_path / path.name.replace(".jsonl.gz", ".npz")
        if destination.is_file() and path.name in progress["shards"]:
            continue
        boards, contexts, targets, game_ids = [], [], [], []
        games = tuple(read_replay_shard(path))
        for game in games:
            for record in records_from_replay_privileged_state(game):
                board, context = encode_privileged_state(record)
                boards.append(board)
                contexts.append(context)
                targets.append(record.target)
                game_ids.append(record.game_id)
        if not boards:
            raise ValueError(f"no decision turns in {path}")
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                board=np.stack(boards),
                context=np.stack(contexts),
                target=np.asarray(targets, dtype=np.float32),
                game_id=np.asarray(game_ids, dtype=np.int64),
            )
        os.replace(temporary, destination)
        progress["shards"][path.name] = {
            "games": len(games),
            "records": len(boards),
            "compressed_bytes": destination.stat().st_size,
        }
        _write_json(progress_path, progress)

    facts = tuple(progress["shards"].values())
    manifest = {
        **expected,
        "status": "complete",
        "shards": len(facts),
        "games": sum(int(item["games"]) for item in facts),
        "records": sum(int(item["records"]) for item in facts),
        "compressed_bytes": sum(int(item["compressed_bytes"]) for item in facts),
    }
    _write_json(output_path / "manifest.json", manifest)
    return manifest


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            convert_privileged_state(args.source, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
