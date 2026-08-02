"""Build a fixed replay snapshot from one or more replay directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from yellowstone.replay_v2 import ReplayGameV2, read_replay_shard, write_replay_shard


def _part_number(path: Path) -> int:
    return int(path.name[5 : -len(".jsonl.gz")])


def _source_paths(source: Path, *, completed_shards: int | None = None) -> tuple[Path, ...]:
    paths = tuple(sorted(source.glob("part_*.jsonl.gz"), key=_part_number))
    if completed_shards is not None:
        paths = paths[:completed_shards]
    if not paths:
        raise FileNotFoundError(f"no replay shards at {source}")
    return paths


def _parse_source(value: str) -> tuple[Path, int | None]:
    if "=" not in value:
        return Path(value), None
    path, limit = value.rsplit("=", 1)
    return Path(path), int(limit)


def _iter_games(source: Path, limit: int | None, *, completed_shards: int | None = None):
    yielded = 0
    for path in _source_paths(source, completed_shards=completed_shards):
        for game in read_replay_shard(path):
            if limit is not None and yielded >= limit:
                return
            yielded += 1
            yield game


def make_snapshot(
    sources: list[tuple[Path, int | None]],
    output: Path,
    *,
    games: int,
    shard_games: int = 100,
    completed_shards: int | None = None,
    source_manifest: Path | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    existing_manifest = output / "collection_manifest.json"
    if existing_manifest.is_file():
        manifest = json.loads(existing_manifest.read_text(encoding="utf-8-sig"))
        if int(manifest.get("games", -1)) != games:
            raise ValueError(f"existing snapshot game count differs: {manifest}")
        return manifest

    source_facts: list[dict[str, object]] = []
    source_manifest_facts = _source_manifest_facts(source_manifest)
    next_game_id = 0
    compressed_bytes = 0
    shards = 0
    shard: list[ReplayGameV2] = []

    if any(output.glob("part_*.jsonl.gz")):
        raise ValueError(
            "snapshot output has shard files but no manifest; use a new output path"
        )

    def flush_shard() -> None:
        nonlocal compressed_bytes, shards, shard
        if not shard:
            return
        start_game_id = shards * shard_games
        facts = write_replay_shard(
            tuple(shard),
            output / f"part_{start_game_id:07d}.jsonl.gz",
        )
        compressed_bytes += int(facts["compressed_bytes"])
        shards += 1
        shard = []

    for source, limit in sources:
        before = next_game_id
        for game in _iter_games(source, limit, completed_shards=completed_shards):
            if next_game_id >= games:
                break
            shard.append(replace(game, game_id=next_game_id))
            next_game_id += 1
            if len(shard) >= shard_games:
                flush_shard()
        source_facts.append(
            {
                "source": str(source),
                "requested": limit,
                "used": next_game_id - before,
                "completed_shards_used": completed_shards,
            }
        )
        if next_game_id >= games:
            break
    flush_shard()
    if next_game_id != games:
        raise ValueError(f"insufficient source games: {next_game_id} < {games}")

    manifest = {
        "schema": "yellowstone.replay.v2.mixed_snapshot.v1",
        "status": "complete",
        "games": games,
        "completed_shards": shards,
        "shard_games": shard_games,
        "game_id_offset": 0,
        "source_facts": source_facts,
        "source_manifest": source_manifest_facts,
        "compressed_bytes": compressed_bytes,
    }
    _write_json(existing_manifest, manifest)
    return manifest


def _source_manifest_facts(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    payload = path.read_bytes()
    manifest = json.loads(payload.decode("utf-8-sig"))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "games": manifest.get("games"),
        "completed_shards": manifest.get("completed_shards"),
        "status": manifest.get("status"),
        "updated_at": manifest.get("updated_at"),
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--shard-games", type=int, default=100)
    parser.add_argument("--completed-shards", type=int)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            make_snapshot(
                [_parse_source(value) for value in args.source],
                args.output,
                games=args.games,
                shard_games=args.shard_games,
                completed_shards=args.completed_shards,
                source_manifest=args.source_manifest,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
