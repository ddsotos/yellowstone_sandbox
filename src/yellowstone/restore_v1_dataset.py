"""Atomically restore an archived V1 tensor directory from backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def restore_dataset(
    source: Path,
    destination: Path,
    *,
    expected_games: int,
    expected_schema: str,
    expected_history_semantics: str,
    expected_canonicalization: str,
) -> dict[str, object]:
    manifest_path = source / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"backup manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "games": expected_games,
        "value_schema": expected_schema,
        "history_semantics": expected_history_semantics,
        "canonicalization": expected_canonicalization,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "backup dataset contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    parts = tuple(sorted(source.glob("part_*.npz")))
    if len(parts) != int(manifest["source_shards"]):
        raise ValueError("backup tensor part count differs from manifest")
    destination.mkdir(parents=True, exist_ok=True)
    copied = skipped = repaired = 0
    for index, source_part in enumerate(parts, start=1):
        target = destination / source_part.name
        matches = (
            target.is_file()
            and target.stat().st_size == source_part.stat().st_size
            and file_sha256(target) == file_sha256(source_part)
        )
        if matches:
            skipped += 1
        else:
            repaired += int(target.exists())
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(source_part, temporary)
            if file_sha256(temporary) != file_sha256(source_part):
                raise OSError(f"copied tensor hash differs: {source_part.name}")
            os.replace(temporary, target)
            copied += 1
        if index == 1 or index % 100 == 0 or index == len(parts):
            print(
                f"restore_progress={index}/{len(parts)} "
                f"copied={copied} skipped={skipped}",
                flush=True,
            )
    extra_parts = {
        path.name for path in destination.glob("part_*.npz")
    } - {path.name for path in parts}
    if extra_parts:
        raise ValueError(
            f"destination contains unexpected tensor parts: "
            f"{sorted(extra_parts)[:5]}"
        )
    temporary_manifest = (
        destination / "conversion_manifest.json.tmp"
    )
    shutil.copyfile(manifest_path, temporary_manifest)
    os.replace(
        temporary_manifest, destination / "conversion_manifest.json"
    )
    result = {
        "source": str(source),
        "destination": str(destination),
        "parts": len(parts),
        "copied": copied,
        "skipped": skipped,
        "repaired": repaired,
        "compressed_bytes": sum(path.stat().st_size for path in parts),
        **expected,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("--expected-schema", required=True)
    parser.add_argument("--expected-history-semantics", required=True)
    parser.add_argument("--expected-canonicalization", required=True)
    args = parser.parse_args()
    restore_dataset(
        args.source,
        args.destination,
        expected_games=args.expected_games,
        expected_schema=args.expected_schema,
        expected_history_semantics=args.expected_history_semantics,
        expected_canonicalization=args.expected_canonicalization,
    )


if __name__ == "__main__":
    main()
