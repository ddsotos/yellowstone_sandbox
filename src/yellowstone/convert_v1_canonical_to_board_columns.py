"""Convert fast-canonical Original V1 archives to board-columns V1 tensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from yellowstone.value_board_columns import (
    CANONICALIZATION_BOARD_COLUMNS_V1,
    board_columns_from_canonical_v1_tensors,
    board_columns_metadata,
)


def convert_archives(
    source: str | Path,
    output: str | Path,
    *,
    expected_games: int | None = None,
) -> dict[str, object]:
    import numpy as np

    source_path = Path(source)
    output_path = Path(output)
    paths = tuple(sorted(source_path.glob("part_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no canonical V1 archives at {source_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    converted_files = skipped_files = records = 0
    margin_counts = [0] * 5
    game_ids: set[int] = set()
    for index, path in enumerate(paths, start=1):
        destination = output_path / path.name
        if destination.is_file():
            skipped_files += 1
        else:
            with np.load(path) as archive:
                board, context, stats = board_columns_from_canonical_v1_tensors(
                    archive["board"], archive["context"]
                )
                temporary = destination.with_suffix(".tmp.npz")
                payload = {
                    "board": board,
                    "context": context,
                    "target": archive["target"],
                    "game_id": archive["game_id"],
                }
                for key in (
                    "source_game_id",
                    "perspective_player_index",
                ):
                    if key in archive:
                        payload[key] = archive[key]
                np.savez_compressed(temporary, **payload)
            os.replace(temporary, destination)
            converted_files += 1
            records += stats.records
            margin_counts[0] += stats.left_margin_0
            margin_counts[1] += stats.left_margin_1
            margin_counts[2] += stats.left_margin_2
            margin_counts[3] += stats.left_margin_3
            margin_counts[4] += stats.left_margin_4
        with np.load(destination) as archive:
            game_ids.update(int(value) for value in archive["game_id"])
            if destination.is_file() and skipped_files:
                records += int(len(archive["target"])) if converted_files == 0 else 0
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(f"progress={index}/{len(paths)} part={path.stem}", flush=True)

    if expected_games is not None and len(game_ids) != expected_games:
        raise ValueError(f"game count differs: {len(game_ids)} != {expected_games}")
    result: dict[str, object] = {
        "value_schema": "yellowstone.value.v1",
        "canonicalization": CANONICALIZATION_BOARD_COLUMNS_V1,
        "source": str(source_path),
        "output": str(output_path),
        "source_files": len(paths),
        "converted_files": converted_files,
        "skipped_files": skipped_files,
        "games": len(game_ids),
        "records_from_newly_converted_files": records,
        "left_margin_counts_from_newly_converted_files": margin_counts,
        **board_columns_metadata(),
    }
    temporary = output_path / "conversion_manifest.json.tmp"
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path / "conversion_manifest.json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int)
    args = parser.parse_args()
    convert_archives(
        args.source,
        args.output,
        expected_games=args.expected_games,
    )


if __name__ == "__main__":
    main()
