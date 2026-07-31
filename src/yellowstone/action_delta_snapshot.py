"""Freeze completed action-delta shards into a verified training snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from yellowstone.action_delta import (
    ACTION_DELTA_CONTEXT_SIZE,
    CANONICALIZATION_ACTION_DELTA,
    HISTORY_SEMANTICS_ACTION_DELTA,
    VALUE_SCHEMA_ACTION_DELTA,
)


def build_action_delta_snapshot(
    data_path: str | Path, output_path: str | Path
) -> dict:
    import numpy as np

    root = Path(data_path)
    paths = tuple(sorted(root.glob("part_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no action-delta parts found at {root}")
    parts = []
    totals = {"records": 0, "one_card_records": 0, "two_card_records": 0}
    for path in paths:
        try:
            with np.load(path) as archive:
                required = {
                    "board",
                    "context",
                    "target",
                    "game_id",
                    "turn_id",
                    "play_count",
                }
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(f"missing arrays: {sorted(missing)}")
                records = len(archive["target"])
                shapes = {
                    key: archive[key].shape[0]
                    for key in required
                }
                if any(value != records for value in shapes.values()):
                    raise ValueError(f"record count mismatch: {shapes}")
                if archive["board"].shape[1:] != (58, 7, 7):
                    raise ValueError(
                        f"unexpected board shape: {archive['board'].shape}"
                    )
                if archive["context"].shape[1:] != (
                    ACTION_DELTA_CONTEXT_SIZE,
                ):
                    raise ValueError(
                        f"unexpected context shape: {archive['context'].shape}"
                    )
                play_count = archive["play_count"]
                if not np.isin(play_count, (1, 2)).all():
                    raise ValueError("play_count must contain only 1 or 2")
                one = int(np.count_nonzero(play_count == 1))
                game_ids = archive["game_id"]
                game_min = int(game_ids.min()) if records else None
                game_max = int(game_ids.max()) if records else None
        except Exception as error:
            raise ValueError(f"invalid action-delta part {path}: {error}") from error
        size = path.stat().st_size
        digest = _sha256(path)
        row = {
            "path": path.name,
            "bytes": size,
            "sha256": digest,
            "records": records,
            "one_card_records": one,
            "two_card_records": records - one,
            "game_id_min": game_min,
            "game_id_max": game_max,
        }
        parts.append(row)
        totals["records"] += records
        totals["one_card_records"] += one
        totals["two_card_records"] += records - one
    payload = {
        "schema": VALUE_SCHEMA_ACTION_DELTA,
        "canonicalization": CANONICALIZATION_ACTION_DELTA,
        "history_semantics": HISTORY_SEMANTICS_ACTION_DELTA,
        "opponent_private_inputs": False,
        "data_directory": str(root),
        "part_count": len(parts),
        **totals,
        "ignored_temporary_files": sorted(
            path.name for path in root.glob("*.tmp")
        ),
        "parts": parts,
    }
    payload["snapshot_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json_atomic(Path(output_path), payload)
    return payload


def verified_snapshot_paths(
    snapshot_path: str | Path, *, verify_hashes: bool = True
) -> tuple[dict, tuple[Path, ...]]:
    snapshot_file = Path(snapshot_path)
    payload = json.loads(snapshot_file.read_text(encoding="utf-8-sig"))
    expected = {
        "schema": VALUE_SCHEMA_ACTION_DELTA,
        "canonicalization": CANONICALIZATION_ACTION_DELTA,
        "history_semantics": HISTORY_SEMANTICS_ACTION_DELTA,
        "opponent_private_inputs": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"action-delta snapshot differs at {key}")
    stored_hash = payload.get("snapshot_sha256")
    unhashed = dict(payload)
    unhashed.pop("snapshot_sha256", None)
    actual_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stored_hash != actual_hash:
        raise ValueError("action-delta snapshot metadata hash differs")
    root = Path(payload["data_directory"])
    if not root.is_absolute():
        root = snapshot_file.parent.parent.parent / root
    paths = []
    for row in payload["parts"]:
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"action-delta snapshot part differs: {path}")
        if verify_hashes and _sha256(path) != row["sha256"]:
            raise ValueError(f"action-delta snapshot hash differs: {path}")
        paths.append(path)
    return payload, tuple(paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_action_delta_snapshot(args.data, args.output)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "part_count",
                    "records",
                    "one_card_records",
                    "two_card_records",
                    "snapshot_sha256",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
