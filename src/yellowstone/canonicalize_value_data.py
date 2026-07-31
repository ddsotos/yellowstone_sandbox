"""Convert saved win-value archives to the configured canonical input form."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from yellowstone.value_canonicalization import (
    CANONICALIZATION_NAME,
    canonicalize_value_tensors_with_stats,
)


def canonicalize_archives(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    start_part: int,
    end_part: int,
) -> dict[str, int | str]:
    import numpy as np

    source = Path(source_dir)
    output = Path(output_dir)
    if start_part < 0 or end_part < start_part:
        raise ValueError("invalid part range")
    selected = [
        path
        for path in source.glob("part_*.npz")
        if start_part <= _part_number(path) <= end_part
    ]
    selected.sort(key=_part_number)
    if not selected:
        raise FileNotFoundError(
            f"no archives in range {start_part}..{end_part} at {source}"
        )
    output.mkdir(parents=True, exist_ok=True)

    converted_files = skipped_files = records = horizontal = vertical = 0
    for index, path in enumerate(selected, start=1):
        destination = output / path.name
        if destination.is_file():
            skipped_files += 1
            continue
        with np.load(path) as archive:
            canonical_board, canonical_context, stats = (
                canonicalize_value_tensors_with_stats(
                    archive["board"], archive["context"]
                )
            )
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                board=canonical_board,
                context=canonical_context,
                target=archive["target"],
                game_id=archive["game_id"],
            )
        os.replace(temporary, destination)
        converted_files += 1
        records += stats.records
        horizontal += stats.horizontal_reflections
        vertical += stats.vertical_reflections
        if index == 1 or index % 100 == 0 or index == len(selected):
            print(
                f"progress={index}/{len(selected)} part={_part_number(path)} "
                f"records={records}",
                flush=True,
            )

    result: dict[str, int | str] = {
        "canonicalization": CANONICALIZATION_NAME,
        "source": str(source),
        "output": str(output),
        "start_part": start_part,
        "end_part": end_part,
        "selected_files": len(selected),
        "converted_files": converted_files,
        "skipped_files": skipped_files,
        "converted_records": records,
        "horizontal_reflections": horizontal,
        "vertical_reflections": vertical,
    }
    manifest = output / "canonicalization_manifest.json"
    manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _part_number(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("part_"))
    except ValueError as error:
        raise ValueError(f"invalid archive name: {path.name}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-part", type=int, required=True)
    parser.add_argument("--end-part", type=int, required=True)
    args = parser.parse_args()
    canonicalize_archives(
        args.source,
        args.output,
        start_part=args.start_part,
        end_part=args.end_part,
    )


if __name__ == "__main__":
    main()
