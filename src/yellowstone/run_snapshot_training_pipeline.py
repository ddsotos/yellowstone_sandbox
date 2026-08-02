"""Run the delayed snapshot training pipeline for the active safe-count collection."""

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


SOURCE = Path("data/v2_heuristic_safe_counts_rank_color_20260801")
SOURCE_MANIFEST = SOURCE / "collection_manifest.json"
NAME = "v2_heuristic_safe_counts_rank_color_6h_snapshot_training"
STATUS_PATH = Path(f"results/evaluations/{NAME}.status.json")
SUMMARY_PATH = Path(f"results/evaluations/{NAME}.json")
TIMINGS_PATH = Path(f"results/evaluations/{NAME}.timings.json")
TRAINING_SEED = 20260727
EVALUATION_SEED = 20260725
MILESTONES = "20,50,100"


def run_pipeline(*, wait_seconds: int, eval_games: int, max_workers: int) -> dict[str, object]:
    timings: dict[str, float] = {}
    started = perf_counter()
    _write_status("waiting", "running", {"wait_seconds": wait_seconds})
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    source_manifest_bytes = SOURCE_MANIFEST.read_bytes()
    source_manifest = json.loads(source_manifest_bytes.decode("utf-8-sig"))
    games = int(source_manifest["games"])
    completed_shards = int(source_manifest["completed_shards"])
    if games <= 0 or completed_shards <= 0:
        raise ValueError("source collection has no completed games")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = Path(f"data/v2_heuristic_safe_counts_rank_color_snapshot_{stamp}")
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
            str(games),
            "--completed-shards",
            str(completed_shards),
            "--source-manifest",
            str(SOURCE_MANIFEST),
        ],
    )

    canonical_data = Path(f"data/{snapshot.name}_canonical")
    _step(
        "convert_canonical",
        timings,
        [
            "-m",
            "yellowstone.convert_replay_v2_to_v1_original",
            "--source",
            str(snapshot),
            "--output",
            str(canonical_data),
            "--expected-games",
            str(games),
        ],
    )

    jobs = [
        _job(
            "canonical",
            canonical_data,
            "yellowstone.value.v1",
            "fast_lr_ud_color_v1",
            "rolling_last_two_placements",
            eval_module="yellowstone.evaluate_value",
        ),
        _job(
            "board_columns_v1",
            Path(f"data/{snapshot.name}_board_columns_v1"),
            "yellowstone.value.v1",
            "board_columns_v1_history_none",
            "none",
            convert=[
                "-m",
                "yellowstone.convert_v1_canonical_to_board_columns",
                "--source",
                str(canonical_data),
                "--output",
                f"data/{snapshot.name}_board_columns_v1",
                "--expected-games",
                str(games),
            ],
            eval_module="yellowstone.evaluate_value",
        ),
        _job(
            "board_columns_v2",
            Path(f"data/{snapshot.name}_board_columns_v2"),
            "yellowstone.value.v2-board-columns.v1",
            "board_columns_v2",
            "rolling_last_three_completed_turns_v2",
            convert=[
                "-m",
                "yellowstone.convert_replay_v2_to_board_columns_v2",
                "--source",
                str(snapshot),
                "--output",
                f"data/{snapshot.name}_board_columns_v2",
                "--expected-games",
                str(games),
            ],
            eval_module="yellowstone.evaluate_value_v2_board_columns",
        ),
        _job(
            "preplay_board_columns",
            Path(f"data/{snapshot.name}_preplay_board_columns"),
            "yellowstone.value.preplay-board-columns.v1",
            "preplay_board_columns_v1",
            "last_two_completed_turns_before_turn",
            convert=[
                "-m",
                "yellowstone.convert_replay_v2_to_preplay_board_columns",
                "--source",
                str(snapshot),
                "--output",
                f"data/{snapshot.name}_preplay_board_columns",
                "--expected-games",
                str(games),
            ],
            evaluate=False,
            milestones=False,
        ),
        _job(
            "bcenter_v1_chain_history",
            Path(f"data/{snapshot.name}_bcenter_v1_chain_history"),
            "yellowstone.value.v1",
            "bcenter_v1_chain_history",
            "bcenter_chain_play_after_deltas_4_8_12",
            convert=[
                "-m",
                "yellowstone.convert_replay_v2_to_v1_board_centered",
                "--source",
                str(snapshot),
                "--output",
                f"data/{snapshot.name}_bcenter_v1_chain_history",
                "--expected-games",
                str(games),
                "--input-canonicalization",
                "bcenter_v1_chain_history",
            ],
            eval_module="yellowstone.evaluate_value",
        ),
    ]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_job, spec, games): spec["name"] for spec in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _write_status("jobs", "running", {"completed_jobs": results})

    summary = {
        "status": "complete" if all(row["status"] == "complete" for row in results) else "failed",
        "source": str(SOURCE),
        "source_manifest": str(SOURCE_MANIFEST),
        "snapshot": str(snapshot),
        "games": games,
        "completed_shards": completed_shards,
        "training_seed": TRAINING_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "eval_games_seat0": eval_games,
        "max_parallel_jobs": max_workers,
        "jobs": sorted(results, key=lambda row: row["name"]),
        "timings": timings,
        "elapsed_seconds": perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    _write_status("complete", summary["status"], {"summary": str(SUMMARY_PATH)})
    return summary


def _job(
    name: str,
    data: Path,
    value_schema: str,
    canonicalization: str,
    history_semantics: str,
    *,
    convert: list[str] | None = None,
    eval_module: str | None = None,
    evaluate: bool = True,
    milestones: bool = True,
) -> dict[str, object]:
    return {
        "name": name,
        "data": str(data),
        "value_schema": value_schema,
        "canonicalization": canonicalization,
        "history_semantics": history_semantics,
        "convert": convert,
        "eval_module": eval_module,
        "evaluate": evaluate,
        "milestones": milestones,
    }


def _run_job(spec: dict[str, object], games: int) -> dict[str, object]:
    name = str(spec["name"])
    timings: dict[str, float] = {}
    try:
        if spec["convert"]:
            _run_timed(f"{name}:convert", timings, list(spec["convert"]))
        data = Path(str(spec["data"]))
        start_part, end_part = _part_range(data)
        prefix = Path(f"models/{NAME}_{name}_epoch001")
        if spec["milestones"]:
            metrics_path = Path(f"results/evaluations/{NAME}_{name}_milestones.json")
            _run_timed(
                f"{name}:train_milestones",
                timings,
                [
                    "-m",
                    "yellowstone.train_value_milestones",
                    "--data",
                    str(data),
                    "--checkpoint-prefix",
                    str(prefix),
                    "--split-game-count",
                    str(games),
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
                    str(spec["value_schema"]),
                    "--history-semantics",
                    str(spec["history_semantics"]),
                    "--input-canonicalization",
                    str(spec["canonicalization"]),
                    "--output",
                    str(metrics_path),
                ],
            )
            final_checkpoint = Path(f"{prefix}_pct100.pt")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        else:
            final_checkpoint = Path(f"{prefix}.pt")
            _run_timed(
                f"{name}:train",
                timings,
                [
                    "-m",
                    "yellowstone.train_value",
                    "--data",
                    str(data),
                    "--checkpoint",
                    str(final_checkpoint),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "256",
                    "--learning-rate",
                    "1e-3",
                    "--seed",
                    str(TRAINING_SEED),
                    "--split-game-count",
                    str(games),
                    "--input-canonicalization",
                    str(spec["canonicalization"]),
                    "--value-schema",
                    str(spec["value_schema"]),
                    "--history-semantics",
                    str(spec["history_semantics"]),
                    "--training-games",
                    str(games),
                ],
            )
            metrics = {"checkpoint": str(final_checkpoint)}
        evaluation = None
        if spec["evaluate"]:
            evaluation_path = Path(f"results/evaluations/{NAME}_{name}_seat0_1000.json")
            _run_timed(
                f"{name}:evaluate_seat0",
                timings,
                [
                    "-m",
                    str(spec["eval_module"]),
                    "--checkpoint",
                    str(final_checkpoint),
                    "--games",
                    str(EVAL_GAMES),
                    "--seed",
                    str(EVALUATION_SEED),
                    "--player-index",
                    "0",
                    "--output",
                    str(evaluation_path),
                    *(
                        [
                            "--adaptive-pq-pruning",
                            "--approximate-new-color-neighbors",
                        ]
                        if spec["eval_module"] == "yellowstone.evaluate_value"
                        else []
                    ),
                ],
            )
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8-sig"))
        return {
            "name": name,
            "status": "complete",
            "data": str(data),
            "checkpoint": str(final_checkpoint),
            "metrics": metrics,
            "evaluation": evaluation,
            "timings": timings,
        }
    except Exception as error:
        return {
            "name": name,
            "status": "failed",
            "error": str(error),
            "timings": timings,
        }


def _step(name: str, timings: dict[str, float], args: list[str]) -> None:
    _write_status(name, "running", {})
    _run_timed(name, timings, args)
    _write_json(TIMINGS_PATH, timings)
    _write_status(name, "complete", {})


def _run_timed(name: str, timings: dict[str, float], args: list[str]) -> None:
    started = perf_counter()
    command = [sys.executable, *args]
    completed = subprocess.run(command, check=False)
    timings[name] = perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")


def _part_range(data: Path) -> tuple[int, int]:
    parts = sorted(data.glob("part_*.npz"))
    if not parts:
        raise FileNotFoundError(f"no tensor parts at {data}")
    numbers = [int(path.stem.removeprefix("part_")) for path in parts]
    return min(numbers), max(numbers)


def _write_status(step: str, state: str, extra: dict[str, object]) -> None:
    payload = {
        "state": state,
        "step": step,
        "last_completed_step": step if state == "complete" else "",
        "updated_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "source": str(SOURCE),
        "summary": str(SUMMARY_PATH),
        **extra,
    }
    _write_json(STATUS_PATH, payload)


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
    parser.add_argument("--wait-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    if args.wait_seconds < 0:
        raise ValueError("--wait-seconds must be non-negative")
    if args.eval_games <= 0:
        raise ValueError("--eval-games must be positive")
    if not 1 <= args.max_workers <= 2:
        raise ValueError("--max-workers must be in 1..2")
    global EVAL_GAMES
    EVAL_GAMES = args.eval_games
    summary = run_pipeline(
        wait_seconds=args.wait_seconds,
        eval_games=args.eval_games,
        max_workers=args.max_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


EVAL_GAMES = 1000


if __name__ == "__main__":
    main()
