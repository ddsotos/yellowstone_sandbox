"""Validate replay artifacts produced by parallel cloud collectors."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable



def _iter_game_ids(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield int(json.loads(line)["game_id"])


def validate_collection(path: Path, expected_games: int | None = None) -> dict:
    manifest_path = path / "collection_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing collection_manifest.json: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = sorted(list(path.glob("*.jsonl")) + list(path.glob("*.jsonl.gz")))
    game_ids: list[int] = []
    for shard in shards:
        game_ids.extend(_iter_game_ids(shard))
    if len(game_ids) != len(set(game_ids)):
        raise ValueError(f"duplicate game ids within {path}")
    if expected_games is not None and len(game_ids) != expected_games:
        raise ValueError(f"expected {expected_games} games, found {len(game_ids)} in {path}")
    return {
        "path": str(path),
        "manifest": manifest,
        "shard_count": len(shards),
        "game_count": len(game_ids),
        "min_game_id": min(game_ids) if game_ids else None,
        "max_game_id": max(game_ids) if game_ids else None,
    }


def validate_workers(paths: Iterable[Path], expected_games: int | None = None) -> dict:
    reports = [validate_collection(p, expected_games) for p in paths]
    seen: set[int] = set()
    for report, path in zip(reports, paths):
        shards = sorted(list(path.glob("*.jsonl")) + list(path.glob("*.jsonl.gz")))
        ids = set(game_id for shard in shards for game_id in _iter_game_ids(shard))
        overlap = seen & ids
        if overlap:
            raise ValueError(f"duplicate game ids across workers: {sorted(overlap)[:5]}")
        seen.update(ids)
    return {"workers": reports, "total_game_count": len(seen)}
