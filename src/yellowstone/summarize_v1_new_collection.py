"""Validate and summarize the Original V1 50k versus 88,966 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def summarize(
    *,
    manifest_path: Path,
    checkpoint_50000_path: Path,
    checkpoint_88966_path: Path,
    evaluation_directory: Path,
    timings_path: Path,
    output_path: Path,
    games_per_seat: int,
    evaluation_seed: int,
    training_seed: int,
) -> dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise ImportError("summary requires `pip install -e .[value]`") from error

    manifest = _read_json(manifest_path)
    if (
        manifest.get("source_game_id_min") != 954346
        or manifest.get("source_game_id_max") != 1043311
        or manifest.get("game_id_rebase") != 954346
        or manifest.get("games") != 88966
        or manifest.get("rebased_game_id_min") != 0
        or manifest.get("rebased_game_id_max") != 88965
    ):
        raise ValueError("conversion manifest does not describe 0..88965")

    checkpoints = {
        50000: torch.load(checkpoint_50000_path, weights_only=False),
        88966: torch.load(checkpoint_88966_path, weights_only=False),
    }
    limited = checkpoints[50000]
    full = checkpoints[88966]
    shared_keys = (
        "seed",
        "split_game_count",
        "full_train_game_ids_sha256",
        "validation_game_ids_sha256",
        "test_game_ids_sha256",
        "validation_split_games",
        "test_split_games",
    )
    for key in shared_keys:
        if limited.get(key) != full.get(key):
            raise ValueError(f"checkpoint split metadata differs for {key}")
    if limited.get("seed") != training_seed:
        raise ValueError("training seed differs")
    if limited.get("train_game_id_limit") != 50000:
        raise ValueError("50k checkpoint train mask differs")
    if full.get("train_game_id_limit") != 88966:
        raise ValueError("88,966 checkpoint train mask differs")
    if not limited.get("fresh_initialization") or not full.get(
        "fresh_initialization"
    ):
        raise ValueError("both checkpoints must use fresh initialization")
    if limited.get("epochs") != 1 or full.get("epochs") != 1:
        raise ValueError("both checkpoints must use exactly one epoch")

    timings = _read_json(timings_path)
    rows: list[dict[str, object]] = []
    paths = {
        50000: checkpoint_50000_path,
        88966: checkpoint_88966_path,
    }
    for size, checkpoint in checkpoints.items():
        seats = []
        total_wins = 0.0
        total_games = 0
        for player_index in range(4):
            result_path = evaluation_directory / (
                f"v1_original_new_{size}_epoch001_"
                f"{games_per_seat}_same_seed_p{player_index}.json"
            )
            result = _read_json(result_path)
            if result.get("games") != games_per_seat:
                raise ValueError(f"evaluation game count differs: {result_path}")
            wins = float(result["wins"])
            games = int(result["games"])
            total_wins += wins
            total_games += games
            seats.append(
                {
                    "player_index": player_index,
                    "games": games,
                    "wins": wins,
                    "win_rate": float(result["win_rate"]),
                    "result": result_path.as_posix(),
                }
            )
        metrics = checkpoint["metrics"]
        rows.append(
            {
                "source_games": size,
                "actual_train_split_games": checkpoint["train_split_games"],
                "checkpoint": paths[size].as_posix(),
                "test_brier": float(metrics["test_brier"]),
                "test_log_loss": float(metrics["test_log_loss"]),
                "validation_brier": float(metrics["validation_brier"]),
                "training_wall_seconds": float(
                    timings[f"train_{size}_epoch001"]
                ),
                "seats": seats,
                "all_seats_games": total_games,
                "all_seats_wins": total_wins,
                "all_seats_win_rate": total_wins / total_games,
            }
        )

    payload: dict[str, object] = {
        "status": "complete",
        "experiment": "original_v1_new_collection_50000_vs_88966",
        "source": manifest["source"],
        "games": 88966,
        "source_game_id_range": [954346, 1043311],
        "rebased_game_id_range": [0, 88965],
        "history_semantics": "rolling_last_two_placements",
        "input_canonicalization": "fast_lr_ud_color_v1",
        "epochs": 1,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "games_per_seat": games_per_seat,
        "adaptive_pq_pruning": True,
        "approximate_new_color_neighbors": True,
        "traditional_refill": True,
        "split_audit": {
            "shared_population_games": limited["split_game_count"],
            "validation_games": limited["validation_split_games"],
            "test_games": limited["test_split_games"],
            "validation_game_ids_sha256": limited[
                "validation_game_ids_sha256"
            ],
            "test_game_ids_sha256": limited["test_game_ids_sha256"],
            "shared_validation_and_test": True,
            "limited_train_mask_is_id_below_50000": True,
        },
        "rows": rows,
        "timings": timings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(payload, output_path.with_suffix(".md"))
    return payload


def _write_markdown(payload: dict[str, object], path: Path) -> None:
    lines = [
        "# Original V1 新収集データ学習量比較",
        "",
        "| Source games | Actual train | Test Brier | Test logloss | "
        "Seat 0 | Seat 1 | Seat 2 | Seat 3 | All seats | Train time |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        seats = row["seats"]
        lines.append(
            f"| {row['source_games']:,} | "
            f"{row['actual_train_split_games']:,} | "
            f"{row['test_brier']:.6f} | {row['test_log_loss']:.6f} | "
            + " | ".join(f"{seat['win_rate']:.4f}" for seat in seats)
            + f" | {row['all_seats_win_rate']:.4f} | "
            f"{row['training_wall_seconds']:.1f}s |"
        )
    lines.extend(
        [
            "",
            "両モデルは共通の88,966戦母集団から同じvalidation/test splitを使用。"
            "50kはtrain split中のgame ID < 50,000だけを学習に使用。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-50000", type=Path, required=True)
    parser.add_argument("--checkpoint-88966", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=1000)
    parser.add_argument("--evaluation-seed", type=int, default=20260725)
    parser.add_argument("--training-seed", type=int, default=20260727)
    args = parser.parse_args()
    payload = summarize(
        manifest_path=args.manifest,
        checkpoint_50000_path=args.checkpoint_50000,
        checkpoint_88966_path=args.checkpoint_88966,
        evaluation_directory=args.evaluation_directory,
        timings_path=args.timings,
        output_path=args.output,
        games_per_seat=args.games_per_seat,
        evaluation_seed=args.evaluation_seed,
        training_seed=args.training_seed,
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
