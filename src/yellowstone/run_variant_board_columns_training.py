"""Train board-columns V1 models on the completed variant heuristic replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter


NAME = "v2_variant_board5_hand6_oneoff_tiered_board_columns_v1_training"
SOURCES = (
    Path("data/v2_variant_board5_hand6_oneoff_tiered_heuristic4_20260802"),
    Path("data/v2_variant_board5_hand6_oneoff_tiered_heuristic4_20260802_part2"),
)
SNAPSHOT = Path("data/v2_variant_board5_hand6_oneoff_tiered_heuristic4_300000_snapshot")
CANONICAL_DATA = Path(f"data/{SNAPSHOT.name}_canonical")
BOARD_COLUMNS_DATA = Path(f"data/{SNAPSHOT.name}_board_columns_v1")
STATUS_PATH = Path(f"results/evaluations/{NAME}.status.json")
SUMMARY_PATH = Path(f"results/evaluations/{NAME}.json")
TIMINGS_PATH = Path(f"results/evaluations/{NAME}.timings.json")
TRAINING_SEED = 20260727
MILESTONES = "20,50,100"
TOTAL_GAMES = 300_000
INITIAL_CHECKPOINT = Path(
    "models/v2_heuristic_safe_counts_rank_color_6h_snapshot_training_board_columns_v1_epoch001_pct100.pt"
)


def run_pipeline(*, max_workers: int) -> dict[str, object]:
    started = perf_counter()
    timings: dict[str, float] = {}
    _write_status("validate_sources", "running", {})
    source_facts = _source_facts()
    if sum(int(row["games"]) for row in source_facts) != TOTAL_GAMES:
        raise ValueError(f"variant replay must total {TOTAL_GAMES} games: {source_facts}")
    _write_status("validate_sources", "complete", {"source_facts": source_facts})

    _step(
        "snapshot",
        timings,
        [
            "-m",
            "yellowstone.make_replay_snapshot",
            "--source",
            str(SOURCES[0]),
            "--source",
            str(SOURCES[1]),
            "--output",
            str(SNAPSHOT),
            "--games",
            str(TOTAL_GAMES),
        ],
    )
    _step(
        "convert_canonical",
        timings,
        [
            "-m",
            "yellowstone.convert_replay_v2_to_v1_original",
            "--source",
            str(SNAPSHOT),
            "--output",
            str(CANONICAL_DATA),
            "--expected-games",
            str(TOTAL_GAMES),
        ],
    )
    _step(
        "convert_board_columns_v1",
        timings,
        [
            "-m",
            "yellowstone.convert_v1_canonical_to_board_columns",
            "--source",
            str(CANONICAL_DATA),
            "--output",
            str(BOARD_COLUMNS_DATA),
            "--expected-games",
            str(TOTAL_GAMES),
        ],
    )

    start_part, end_part = _part_range(BOARD_COLUMNS_DATA)
    jobs = [
        {
            "name": "scratch",
            "prefix": Path(f"models/{NAME}_scratch_epoch001"),
            "output": Path(f"results/evaluations/{NAME}_scratch_milestones.json"),
            "initial_checkpoint": None,
        },
        {
            "name": "finetune_from_6h_board_columns_v1",
            "prefix": Path(f"models/{NAME}_finetune_from_6h_epoch001"),
            "output": Path(f"results/evaluations/{NAME}_finetune_from_6h_milestones.json"),
            "initial_checkpoint": INITIAL_CHECKPOINT,
        },
    ]
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_train_job, job, start_part, end_part): str(job["name"])
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _write_status("train", "running", {"completed_jobs": results})

    summary = {
        "status": "complete" if all(row["status"] == "complete" for row in results) else "failed",
        "source_facts": source_facts,
        "snapshot": str(SNAPSHOT),
        "canonical_data": str(CANONICAL_DATA),
        "board_columns_data": str(BOARD_COLUMNS_DATA),
        "games": TOTAL_GAMES,
        "training_seed": TRAINING_SEED,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "max_parallel_training_jobs": max_workers,
        "jobs": sorted(results, key=lambda row: row["name"]),
        "timings": timings,
        "elapsed_seconds": perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    _write_status("complete", summary["status"], {"summary": str(SUMMARY_PATH)})
    return summary


def _source_facts() -> list[dict[str, object]]:
    rows = []
    for source in SOURCES:
        manifest_path = source / "collection_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("status") != "complete":
            raise ValueError(f"source is not complete: {source}")
        rows.append(
            {
                "source": str(source),
                "games": int(manifest["games"]),
                "completed_shards": int(manifest["completed_shards"]),
                "collector": manifest.get("collector"),
                "seed": manifest.get("seed"),
                "updated_at": manifest.get("updated_at"),
            }
        )
    return rows


def _train_job(job: dict[str, object], start_part: int, end_part: int) -> dict[str, object]:
    started = perf_counter()
    name = str(job["name"])
    try:
        prefix = Path(str(job["prefix"]))
        output = Path(str(job["output"]))
        final_checkpoint = Path(f"{prefix}_pct100.pt")
        if final_checkpoint.is_file() and output.is_file():
            metrics = json.loads(output.read_text(encoding="utf-8-sig"))
            return {
                "name": name,
                "status": "complete",
                "checkpoint": str(final_checkpoint),
                "metrics": metrics,
                "elapsed_seconds": perf_counter() - started,
                "skipped": True,
            }
        args = [
            "-m",
            "yellowstone.train_value_milestones",
            "--data",
            str(BOARD_COLUMNS_DATA),
            "--checkpoint-prefix",
            str(prefix),
            "--split-game-count",
            str(TOTAL_GAMES),
            "--start-part",
            str(start_part),
            "--end-part",
            str(end_part),
            "--milestones",
            MILESTONES,
            "--batch-size",
            "256",
            "--learning-rate",
            "1e-3",
            "--seed",
            str(TRAINING_SEED),
            "--value-schema",
            "yellowstone.value.v1",
            "--history-semantics",
            "none",
            "--input-canonicalization",
            "board_columns_v1_history_none",
            "--output",
            str(output),
        ]
        if job["initial_checkpoint"] is not None:
            args.extend(["--initial-checkpoint", str(job["initial_checkpoint"])])
        _run(args)
        metrics = json.loads(output.read_text(encoding="utf-8-sig"))
        return {
            "name": name,
            "status": "complete",
            "checkpoint": str(final_checkpoint),
            "metrics": metrics,
            "elapsed_seconds": perf_counter() - started,
            "initial_checkpoint": (
                str(job["initial_checkpoint"]) if job["initial_checkpoint"] else None
            ),
        }
    except Exception as error:
        return {
            "name": name,
            "status": "failed",
            "error": str(error),
            "elapsed_seconds": perf_counter() - started,
        }


def _step(name: str, timings: dict[str, float], args: list[str]) -> None:
    _write_status(name, "running", {})
    started = perf_counter()
    _run(args)
    timings[name] = perf_counter() - started
    _write_json(TIMINGS_PATH, timings)
    _write_status(name, "complete", {})


def _run(args: list[str]) -> None:
    completed = subprocess.run([sys.executable, *args], check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed with exit code {completed.returncode}")


def _part_range(data: Path) -> tuple[int, int]:
    parts = sorted(data.glob("part_*.npz"))
    if not parts:
        raise FileNotFoundError(f"no tensor parts at {data}")
    numbers = [int(path.stem.removeprefix("part_")) for path in parts]
    return min(numbers), max(numbers)


def _write_status(step: str, state: str, extra: dict[str, object]) -> None:
    _write_json(
        STATUS_PATH,
        {
            "state": state,
            "step": step,
            "last_completed_step": step if state == "complete" else "",
            "updated_at": datetime.now().astimezone().isoformat(),
            "pid": os.getpid(),
            "source": [str(source) for source in SOURCES],
            "summary": str(SUMMARY_PATH),
            **extra,
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 2:
        raise ValueError("--max-workers must be in 1..2")
    summary = run_pipeline(max_workers=args.max_workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
