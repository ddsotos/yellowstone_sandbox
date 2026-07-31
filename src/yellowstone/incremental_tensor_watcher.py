"""Incrementally tensorize completed exploratory replay shards."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from yellowstone.convert_privileged_state import convert_privileged_state
from yellowstone.convert_replay_v2_to_v1_original import (
    HISTORY_SEMANTICS_V1_ORIGINAL,
    VALUE_SCHEMA_V1_ORIGINAL,
    records_from_replay_v1_original,
)
from yellowstone.replay_v2 import file_sha256, read_replay_shard
from yellowstone.value_canonicalization import (
    CANONICALIZATION_NAME,
    canonicalize_value_tensors_with_stats,
)
from yellowstone.value_learning import (
    board_tensor_for_player,
    context_tensor_for_player,
)


WATCHER_SCHEMA = "yellowstone.incremental_tensor_watcher.v1"
CARDS_PLAYED_CONTEXT_INDEX = 55


def convert_new_v1_shards(
    source: str | Path,
    output: str | Path,
    *,
    game_id_rebase: int,
) -> dict[str, Any]:
    """Convert only unseen shards and aggregate facts from progress metadata."""
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
    progress_path = output_path / "conversion_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8-sig"))
        if progress_path.exists()
        else {"shards": {}}
    )
    expected = {
        "schema": VALUE_SCHEMA_V1_ORIGINAL,
        "canonicalization": CANONICALIZATION_NAME,
        "history_semantics": HISTORY_SEMANTICS_V1_ORIGINAL,
        "source": str(source_path),
        "game_id_rebase": game_id_rebase,
    }
    for key, value in expected.items():
        if key in progress and progress[key] != value:
            raise ValueError(f"V1 conversion progress differs at {key}")
        progress[key] = value

    converted = 0
    for path in paths:
        destination = output_path / path.name.replace(
            ".jsonl.gz", ".npz"
        )
        prior = progress["shards"].get(path.name)
        if prior is not None:
            if not destination.is_file():
                raise FileNotFoundError(
                    f"progress output is missing: {destination}"
                )
            if prior.get("source_sha256") != file_sha256(path):
                raise ValueError(f"source shard changed: {path}")
            continue
        if destination.exists():
            raise ValueError(
                f"untracked V1 destination already exists: {destination}"
            )
        games = tuple(read_replay_shard(path))
        rows = [
            record
            for game in games
            for record in records_from_replay_v1_original(game)
        ]
        if not rows:
            raise ValueError(f"replay shard produced no V1 records: {path}")
        board = np.stack(
            [board_tensor_for_player(record) for record in rows]
        )
        context = np.stack(
            [context_tensor_for_player(record) for record in rows]
        )
        board, context, stats = canonicalize_value_tensors_with_stats(
            board, context
        )
        source_game_ids = np.asarray(
            [record.game_id for record in rows], dtype=np.int64
        )
        game_ids = source_game_ids - game_id_rebase
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            board=board,
            context=context,
            target=np.asarray(
                [record.target for record in rows], dtype=np.float32
            ),
            game_id=game_ids,
            source_game_id=source_game_ids,
            perspective_player_index=np.asarray(
                [record.perspective_player_index for record in rows],
                dtype=np.int8,
            ),
        )
        os.replace(temporary, destination)
        unique_source_ids = sorted(set(int(value) for value in source_game_ids))
        if unique_source_ids != list(
            range(unique_source_ids[0], unique_source_ids[-1] + 1)
        ):
            raise ValueError(f"source IDs are not continuous in {path}")
        one_card = context[:, CARDS_PLAYED_CONTEXT_INDEX] < 0.75
        progress["shards"][path.name] = {
            "source_sha256": file_sha256(path),
            "source_game_id_min": unique_source_ids[0],
            "source_game_id_max": unique_source_ids[-1],
            "games": len(unique_source_ids),
            "records": len(rows),
            "one_card_records": int(one_card.sum()),
            "two_card_records": int(len(rows) - one_card.sum()),
            "compressed_bytes": destination.stat().st_size,
            "horizontal_reflections": stats.horizontal_reflections,
            "vertical_reflections": stats.vertical_reflections,
        }
        _write_json(progress_path, progress)
        converted += 1

    entries = [
        progress["shards"][name]
        for name in sorted(
            progress["shards"],
            key=lambda name: int(name[5 : -len(".jsonl.gz")]),
        )
    ]
    if entries:
        expected_next = game_id_rebase
        for entry in entries:
            if int(entry["source_game_id_min"]) != expected_next:
                raise ValueError(
                    "V1 progress source game IDs are not continuous"
                )
            expected_next = int(entry["source_game_id_max"]) + 1
    manifest = {
        **expected,
        "status": "in_progress",
        "source_shards_seen": len(paths),
        "converted_shards": len(entries),
        "converted_this_run": converted,
        "games": sum(int(entry["games"]) for entry in entries),
        "records": sum(int(entry["records"]) for entry in entries),
        "one_card_records": sum(
            int(entry["one_card_records"]) for entry in entries
        ),
        "two_card_records": sum(
            int(entry["two_card_records"]) for entry in entries
        ),
        "compressed_bytes": sum(
            int(entry["compressed_bytes"]) for entry in entries
        ),
        "horizontal_reflections": sum(
            int(entry["horizontal_reflections"]) for entry in entries
        ),
        "vertical_reflections": sum(
            int(entry["vertical_reflections"]) for entry in entries
        ),
        "source_game_id_min": (
            int(entries[0]["source_game_id_min"]) if entries else None
        ),
        "source_game_id_max": (
            int(entries[-1]["source_game_id_max"]) if entries else None
        ),
        "rebased_game_id_min": 0 if entries else None,
        "rebased_game_id_max": (
            sum(int(entry["games"]) for entry in entries) - 1
            if entries
            else None
        ),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    _write_json(output_path / "conversion_manifest.json", manifest)
    return manifest


def watch_tensors(
    source: str | Path,
    v1_output: str | Path,
    preplay_output: str | Path,
    *,
    game_id_rebase: int,
    source_manifest: str | Path,
    status_file: str | Path,
    stop_file: str | Path,
    poll_seconds: float = 15.0,
    max_cycles: int | None = None,
) -> dict[str, Any]:
    if poll_seconds <= 0 or max_cycles is not None and max_cycles <= 0:
        raise ValueError("poll_seconds and max_cycles must be positive")
    source_path = Path(source)
    v1_path = Path(v1_output)
    preplay_path = Path(preplay_output)
    source_manifest_path = Path(source_manifest)
    status_path = Path(status_file)
    stop_path = Path(stop_file)
    cycles = 0
    result: dict[str, Any] = {}

    while True:
        stopped = stop_path.exists()
        source_facts = (
            json.loads(
                source_manifest_path.read_text(encoding="utf-8-sig")
            )
            if source_manifest_path.exists()
            else {}
        )
        source_shards = len(tuple(source_path.glob("part_*.jsonl.gz")))
        v1 = convert_new_v1_shards(
            source_path, v1_path, game_id_rebase=game_id_rebase
        )
        preplay = _convert_preplay_if_needed(
            source_path, preplay_path, source_shards
        )
        cycles += 1
        caught_up = (
            int(v1["converted_shards"]) == source_shards
            and int(preplay["shards"]) == source_shards
        )
        source_terminal = source_facts.get("status") in {
            "complete",
            "stopped_by_user",
        }
        state = (
            "stopped_by_user"
            if stopped
            else "complete"
            if source_terminal and caught_up
            else "running"
        )
        result = {
            "schema": WATCHER_SCHEMA,
            "state": state,
            "step": "tensorize_completed_shards",
            "last_completed_step": (
                "convert_v1_and_preplay" if caught_up else ""
            ),
            "updated_at": datetime.now().astimezone().isoformat(),
            "pid": os.getpid(),
            "source": str(source_path),
            "source_status": source_facts.get("status", "unknown"),
            "source_shards": source_shards,
            "v1_output": str(v1_path),
            "v1_shards": int(v1["converted_shards"]),
            "v1_games": int(v1["games"]),
            "v1_records": int(v1["records"]),
            "preplay_output": str(preplay_path),
            "preplay_shards": int(preplay["shards"]),
            "preplay_games": int(preplay["games"]),
            "preplay_records": int(preplay["records"]),
            "stop_file": str(stop_path),
            "cycles": cycles,
        }
        _write_json(status_path, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if (
            state != "running"
            or max_cycles is not None and cycles >= max_cycles
        ):
            break
        time.sleep(poll_seconds)
    return result


def _convert_preplay_if_needed(
    source: Path, output: Path, source_shards: int
) -> dict[str, Any]:
    progress_path = output / "conversion_progress.json"
    converted = 0
    if progress_path.exists():
        progress = json.loads(
            progress_path.read_text(encoding="utf-8-sig")
        )
        converted = len(progress.get("shards", {}))
    if converted < source_shards:
        return convert_privileged_state(source, output)
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return convert_privileged_state(source, output)
    return json.loads(manifest_path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v1-output", type=Path, required=True)
    parser.add_argument("--preplay-output", type=Path, required=True)
    parser.add_argument("--game-id-rebase", type=int, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-cycles", type=int)
    args = parser.parse_args()
    result = watch_tensors(
        args.source,
        args.v1_output,
        args.preplay_output,
        game_id_rebase=args.game_id_rebase,
        source_manifest=args.source_manifest,
        status_file=args.status_file,
        stop_file=args.stop_file,
        poll_seconds=args.poll_seconds,
        max_cycles=args.max_cycles,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
