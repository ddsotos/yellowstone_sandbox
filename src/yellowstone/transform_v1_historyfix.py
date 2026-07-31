"""Repair one-card history in already encoded canonical V1 archives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from yellowstone.convert_replay_v2_to_v1_historyfix import (
    VALUE_SCHEMA_V1_HISTORYFIX,
)
from yellowstone.train_value import _archive_paths
from yellowstone.value_canonicalization import CANONICALIZATION_NAME


CARDS_PLAYED_INDEX = 55
HISTORY_START = 57
HISTORY_SLOT_SIZE = 12
HISTORY_END = HISTORY_START + 2 * HISTORY_SLOT_SIZE


def repair_history_context(context):
    """Return a copy using only the evaluated turn for one-card records."""
    import numpy as np

    if context.ndim != 2 or context.shape[1] != HISTORY_END:
        raise ValueError(
            f"expected [N,{HISTORY_END}] context, got {context.shape}"
        )
    fixed = context.copy()
    one_card = fixed[:, CARDS_PLAYED_INDEX] < 0.75
    second_present = fixed[:, HISTORY_START + HISTORY_SLOT_SIZE] > 0.5
    move_second = one_card & second_present
    fixed[
        move_second,
        HISTORY_START : HISTORY_START + HISTORY_SLOT_SIZE,
    ] = fixed[move_second, HISTORY_START + HISTORY_SLOT_SIZE : HISTORY_END]
    fixed[one_card, HISTORY_START + HISTORY_SLOT_SIZE : HISTORY_END] = 0.0
    return fixed, {
        "records": int(len(fixed)),
        "one_card_records": int(np.sum(one_card)),
        "two_card_records": int(np.sum(~one_card)),
        "moved_second_slot_records": int(np.sum(move_second)),
    }


def transform_archives(
    source: str | Path,
    output: str | Path,
    *,
    start_part: int,
    end_part: int,
    expected_games: int | None = None,
) -> dict[str, object]:
    """Transform an inclusive part range, safely resuming completed files."""
    import numpy as np

    source_path = Path(source)
    output_path = Path(output)
    paths = _archive_paths(
        source_path, start_part=start_part, end_part=end_part
    )
    output_path.mkdir(parents=True, exist_ok=True)
    converted = skipped = 0
    totals = {
        "records": 0,
        "one_card_records": 0,
        "two_card_records": 0,
        "moved_second_slot_records": 0,
    }
    game_ids: set[int] = set()

    for index, path in enumerate(paths, start=1):
        destination = output_path / path.name
        with np.load(path) as archive:
            fixed, stats = repair_history_context(archive["context"])
            game_ids.update(int(value) for value in archive["game_id"])
            if destination.is_file():
                skipped += 1
            else:
                temporary = destination.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temporary,
                    board=archive["board"],
                    context=fixed,
                    target=archive["target"],
                    game_id=archive["game_id"],
                )
                os.replace(temporary, destination)
                converted += 1
        for key, value in stats.items():
            totals[key] += value
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(
                f"progress={index}/{len(paths)} "
                f"converted={converted} skipped={skipped}",
                flush=True,
            )

    if expected_games is not None and len(game_ids) != expected_games:
        raise ValueError(
            f"transformed game count differs: "
            f"{len(game_ids)} != {expected_games}"
        )
    if game_ids and (
        min(game_ids) != start_part
        or max(game_ids) != end_part + 99
    ):
        raise ValueError(
            f"unexpected game range: {min(game_ids)}..{max(game_ids)}"
        )
    result: dict[str, object] = {
        "value_schema": VALUE_SCHEMA_V1_HISTORYFIX,
        "canonicalization": CANONICALIZATION_NAME,
        "history_semantics": (
            "evaluated_turn_only_one_card_zero_padded"
        ),
        "source": str(source_path),
        "output": str(output_path),
        "start_part": start_part,
        "end_part": end_part,
        "source_files": len(paths),
        "converted_files": converted,
        "skipped_files": skipped,
        "games": len(game_ids),
        **totals,
    }
    temporary_manifest = output_path / "conversion_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(
        temporary_manifest, output_path / "conversion_manifest.json"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-part", type=int, required=True)
    parser.add_argument("--end-part", type=int, required=True)
    parser.add_argument("--expected-games", type=int)
    args = parser.parse_args()
    transform_archives(
        args.source,
        args.output,
        start_part=args.start_part,
        end_part=args.end_part,
        expected_games=args.expected_games,
    )


if __name__ == "__main__":
    main()
