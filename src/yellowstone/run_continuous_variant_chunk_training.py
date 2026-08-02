"""Watch continuous variant replay and train every 200k completed games."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any


NAME = "v2_variant_board5_hand6_oneoff_tiered_continuous_chunk_training"
SOURCE = Path(
    "data/v2_variant_board5_hand6_oneoff_tiered_heuristic4_continuous_20260802"
)
SOURCE_MANIFEST = SOURCE / "collection_manifest.json"
STATUS_PATH = Path(f"results/evaluations/{NAME}.status.json")
SUMMARY_PATH = Path(f"results/evaluations/{NAME}.json")
TIMINGS_PATH = Path(f"results/evaluations/{NAME}.timings.json")
STOP_PATH = Path(f"results/evaluations/{NAME}.stop")
BASE_INITIAL_CHECKPOINT = Path(
    "models/v2_heuristic_safe_counts_rank_color_6h_snapshot_training_"
    "board_columns_v1_epoch001_pct100.pt"
)
TRAINING_SEED = 20260727
MILESTONES = "20,50,100"


def run_watcher(
    *,
    chunk_games: int,
    poll_seconds: int,
    max_training_workers: int,
    once: bool,
) -> dict[str, Any]:
    if chunk_games <= 0 or poll_seconds <= 0:
        raise ValueError("chunk_games and poll_seconds must be positive")
    if not 1 <= max_training_workers <= 2:
        raise ValueError("max_training_workers must be in 1..2")
    started = perf_counter()
    processed = _load_processed_chunks()
    summary: dict[str, Any] = {
        "schema": "yellowstone.continuous_variant_chunk_training.v1",
        "status": "running",
        "source": str(SOURCE),
        "source_manifest": str(SOURCE_MANIFEST),
        "chunk_games": chunk_games,
        "poll_seconds": poll_seconds,
        "max_training_workers": max_training_workers,
        "training_seed": TRAINING_SEED,
        "processed_chunks": processed,
        "updated_at": _now(),
    }
    _write_json(SUMMARY_PATH, summary)

    while True:
        if STOP_PATH.exists():
            summary["status"] = "stopped_by_user"
            summary["updated_at"] = _now()
            summary["elapsed_seconds"] = perf_counter() - started
            _write_json(SUMMARY_PATH, summary)
            _write_status("stopped_by_user", "stopped_by_user", summary)
            return summary
        manifest = _read_source_manifest()
        available_games = int(manifest.get("games", 0))
        target_games = _next_target(processed, chunk_games, available_games)
        if target_games is None:
            summary.update(
                {
                    "status": "running",
                    "available_games": available_games,
                    "next_target_games": (
                        (max(processed) + chunk_games)
                        if processed
                        else chunk_games
                    ),
                    "source_status": manifest.get("status", "unknown"),
                    "updated_at": _now(),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            _write_json(SUMMARY_PATH, summary)
            _write_status("waiting_for_next_chunk", "running", summary)
            if once:
                return summary
            time.sleep(poll_seconds)
            continue

        result = _run_chunk(target_games, max_training_workers)
        processed.append(result)
        processed = sorted(processed, key=lambda row: int(row["games"]))
        summary.update(
            {
                "status": (
                    "complete"
                    if all(row.get("status") == "complete" for row in processed)
                    else "failed"
                ),
                "available_games": available_games,
                "processed_chunks": processed,
                "updated_at": _now(),
                "elapsed_seconds": perf_counter() - started,
            }
        )
        _write_json(SUMMARY_PATH, summary)
        if result["status"] != "complete":
            _write_status("failed", "failed", summary)
            return summary
        if once:
            return summary


def _load_processed_chunks() -> list[dict[str, Any]]:
    if not SUMMARY_PATH.is_file():
        return []
    payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8-sig"))
    return [
        row
        for row in payload.get("processed_chunks", [])
        if row.get("status") == "complete"
    ]


def _read_source_manifest() -> dict[str, Any]:
    if not SOURCE_MANIFEST.is_file():
        return {}
    return json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8-sig"))


def _next_target(
    processed: list[dict[str, Any]], chunk_games: int, available_games: int
) -> int | None:
    done = {int(row["games"]) for row in processed}
    target = chunk_games
    while target in done:
        target += chunk_games
    return target if available_games >= target else None


def _run_chunk(target_games: int, max_training_workers: int) -> dict[str, Any]:
    stem = f"{NAME}_{target_games:07d}"
    snapshot = Path(f"data/{stem}_snapshot")
    canonical = Path(f"data/{stem}_canonical")
    board_columns = Path(f"data/{stem}_board_columns_v1")
    preplay = Path(f"data/{stem}_preplay_board_columns")
    timings: dict[str, float] = {}
    started = perf_counter()
    _write_status(f"{target_games}:snapshot", "running", {"games": target_games})
    _step(
        "snapshot",
        timings,
        [
            "-m",
            "yellowstone.make_replay_snapshot",
            "--source",
            str(SOURCE),
            "--output",
            str(snapshot),
            "--games",
            str(target_games),
            "--source-manifest",
            str(SOURCE_MANIFEST),
        ],
    )
    _write_status(f"{target_games}:convert_canonical", "running", {"games": target_games})
    _step(
        "convert_canonical",
        timings,
        [
            "-m",
            "yellowstone.convert_replay_v2_to_v1_original",
            "--source",
            str(snapshot),
            "--output",
            str(canonical),
            "--expected-games",
            str(target_games),
        ],
    )
    _write_status(f"{target_games}:convert_board_columns_v1", "running", {"games": target_games})
    _step(
        "convert_board_columns_v1",
        timings,
        [
            "-m",
            "yellowstone.convert_v1_canonical_to_board_columns",
            "--source",
            str(canonical),
            "--output",
            str(board_columns),
            "--expected-games",
            str(target_games),
        ],
    )
    _write_status(f"{target_games}:convert_preplay_board_columns", "running", {"games": target_games})
    _step(
        "convert_preplay_board_columns",
        timings,
        [
            "-m",
            "yellowstone.convert_replay_v2_to_preplay_board_columns",
            "--source",
            str(snapshot),
            "--output",
            str(preplay),
            "--expected-games",
            str(target_games),
        ],
    )

    jobs = [
        _job(
            "scratch",
            board_columns,
            Path(f"models/{stem}_board_columns_v1_scratch_epoch001"),
            Path(f"results/evaluations/{stem}_board_columns_v1_scratch_milestones.json"),
            value_schema="yellowstone.value.v1",
            history_semantics="none",
            canonicalization="board_columns_v1_history_none",
        ),
        _job(
            "finetune",
            board_columns,
            Path(f"models/{stem}_board_columns_v1_finetune_epoch001"),
            Path(f"results/evaluations/{stem}_board_columns_v1_finetune_milestones.json"),
            value_schema="yellowstone.value.v1",
            history_semantics="none",
            canonicalization="board_columns_v1_history_none",
            initial_checkpoint=_initial_finetune_checkpoint(target_games),
        ),
        _job(
            "preplay_board_columns",
            preplay,
            Path(f"models/{stem}_preplay_board_columns_epoch001"),
            Path(f"results/evaluations/{stem}_preplay_board_columns_milestones.json"),
            value_schema="yellowstone.value.preplay-board-columns.v1",
            history_semantics="last_two_completed_turns_before_turn",
            canonicalization="preplay_board_columns_v1",
        ),
    ]
    results = []
    _write_status(f"{target_games}:train", "running", {"games": target_games})
    with ThreadPoolExecutor(max_workers=max_training_workers) as executor:
        futures = {
            executor.submit(_train_job, job, target_games): job["name"]
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _write_status(
                f"{target_games}:train",
                "running",
                {"games": target_games, "completed_jobs": results},
            )

    chunk = {
        "games": target_games,
        "status": (
            "complete"
            if all(row.get("status") == "complete" for row in results)
            else "failed"
        ),
        "snapshot": str(snapshot),
        "canonical_data": str(canonical),
        "board_columns_data": str(board_columns),
        "preplay_board_columns_data": str(preplay),
        "jobs": sorted(results, key=lambda row: row["name"]),
        "timings": timings,
        "elapsed_seconds": perf_counter() - started,
        "updated_at": _now(),
    }
    _write_json(Path(f"results/evaluations/{stem}.json"), chunk)
    _write_json(TIMINGS_PATH, {str(target_games): timings})
    return chunk


def _job(
    name: str,
    data: Path,
    prefix: Path,
    output: Path,
    *,
    value_schema: str,
    history_semantics: str,
    canonicalization: str,
    initial_checkpoint: Path | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "data": data,
        "prefix": prefix,
        "output": output,
        "value_schema": value_schema,
        "history_semantics": history_semantics,
        "canonicalization": canonicalization,
        "initial_checkpoint": initial_checkpoint,
    }


def _initial_finetune_checkpoint(target_games: int) -> Path:
    previous = target_games - 200_000
    if previous > 0:
        path = Path(
            f"models/{NAME}_{previous:07d}_board_columns_v1_"
            "finetune_epoch001_pct100.pt"
        )
        if path.is_file():
            return path
    return BASE_INITIAL_CHECKPOINT


def _train_job(job: dict[str, Any], split_game_count: int) -> dict[str, Any]:
    started = perf_counter()
    try:
        data = Path(job["data"])
        start_part, end_part = _part_range(data)
        prefix = Path(job["prefix"])
        output = Path(job["output"])
        final_checkpoint = Path(f"{prefix}_pct100.pt")
        if final_checkpoint.is_file() and output.is_file():
            metrics = json.loads(output.read_text(encoding="utf-8-sig"))
            return {
                "name": job["name"],
                "status": "complete",
                "checkpoint": str(final_checkpoint),
                "metrics": metrics,
                "skipped": True,
                "elapsed_seconds": perf_counter() - started,
            }
        args = [
            "-m",
            "yellowstone.train_value_milestones",
            "--data",
            str(data),
            "--checkpoint-prefix",
            str(prefix),
            "--split-game-count",
            str(split_game_count),
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
            str(job["value_schema"]),
            "--history-semantics",
            str(job["history_semantics"]),
            "--input-canonicalization",
            str(job["canonicalization"]),
            "--output",
            str(output),
        ]
        if job.get("initial_checkpoint") is not None:
            args.extend(["--initial-checkpoint", str(job["initial_checkpoint"])])
        _run(args)
        metrics = json.loads(output.read_text(encoding="utf-8-sig"))
        return {
            "name": job["name"],
            "status": "complete",
            "checkpoint": str(final_checkpoint),
            "metrics": metrics,
            "initial_checkpoint": (
                str(job["initial_checkpoint"])
                if job.get("initial_checkpoint") is not None
                else None
            ),
            "elapsed_seconds": perf_counter() - started,
        }
    except Exception as error:
        return {
            "name": job["name"],
            "status": "failed",
            "error": str(error),
            "elapsed_seconds": perf_counter() - started,
        }


def _part_range(data: Path) -> tuple[int, int]:
    parts = sorted(data.glob("part_*.npz"))
    if not parts:
        raise FileNotFoundError(f"no tensor parts at {data}")
    numbers = [int(path.stem.removeprefix("part_")) for path in parts]
    return min(numbers), max(numbers)


def _step(name: str, timings: dict[str, float], args: list[str]) -> None:
    started = perf_counter()
    _run(args)
    timings[name] = perf_counter() - started


def _run(args: list[str]) -> None:
    completed = subprocess.run([sys.executable, *args], check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed with exit code {completed.returncode}")


def _write_status(step: str, state: str, extra: dict[str, Any]) -> None:
    _write_json(
        STATUS_PATH,
        {
            "state": state,
            "step": step,
            "last_completed_step": step if state == "complete" else "",
            "updated_at": _now(),
            "pid": os.getpid(),
            "source": str(SOURCE),
            "source_manifest": str(SOURCE_MANIFEST),
            "summary": str(SUMMARY_PATH),
            "stop_file": str(STOP_PATH),
            **extra,
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-games", type=int, default=200_000)
    parser.add_argument("--poll-seconds", type=int, default=3600)
    parser.add_argument("--max-training-workers", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    result = run_watcher(
        chunk_games=args.chunk_games,
        poll_seconds=args.poll_seconds,
        max_training_workers=args.max_training_workers,
        once=args.once,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
