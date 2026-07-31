"""Summarize the six V2/V1 history-learning conditions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


_CONDITIONS = (
    {
        "key": "v2_epoch001",
        "model": "V2",
        "epochs": 1,
        "checkpoint": "models/win_value_v2_generation0_197800_epoch001.pt",
        "evaluation": (
            "results/evaluations/"
            "v2_generation0_197800_1000_same_seed_p0_corrected.json"
        ),
        "timing": None,
        "timing_start": "data/v2_generation0_197800_tensors/manifest.json",
    },
    {
        "key": "v2_epoch002",
        "model": "V2",
        "epochs": 2,
        "checkpoint": "models/win_value_v2_generation0_197800_epoch002.pt",
        "evaluation": (
            "results/evaluations/"
            "v2_generation0_197800_epoch002_1000_same_seed_p0.json"
        ),
        "timing": ("epoch2_baselines.timings.json", "train_v2_epoch002"),
    },
    {
        "key": "v1_historyfix_epoch001",
        "model": "V1履歴修正版",
        "epochs": 1,
        "checkpoint": (
            "models/"
            "win_value_v1_historyfix_generation0_197800_epoch001.pt"
        ),
        "evaluation": (
            "results/evaluations/"
            "v1_historyfix_generation0_197800_1000_same_seed_p0.json"
        ),
        "timing": None,
        "timing_start": (
            "data/v1_historyfix_generation0_197800_canonical/"
            "conversion_manifest.json"
        ),
    },
    {
        "key": "v1_historyfix_epoch002",
        "model": "V1履歴修正版",
        "epochs": 2,
        "checkpoint": (
            "models/"
            "win_value_v1_historyfix_generation0_197800_epoch002.pt"
        ),
        "evaluation": (
            "results/evaluations/"
            "v1_historyfix_generation0_197800_epoch002_1000_same_seed_p0.json"
        ),
        "timing": (
            "epoch2_baselines.timings.json",
            "train_v1_historyfix_epoch002",
        ),
    },
    {
        "key": "v1_original_epoch001",
        "model": "V1履歴未修正版",
        "epochs": 1,
        "checkpoint": (
            "models/"
            "win_value_v1_original_generation0_197800_epoch001.pt"
        ),
        "evaluation": (
            "results/evaluations/"
            "v1_original_generation0_197800_epoch001_1000_same_seed_p0.json"
        ),
        "timing": (
            "v1_original_generation0.timings.json",
            "train_v1_original_epoch001",
        ),
    },
    {
        "key": "v1_original_epoch002",
        "model": "V1履歴未修正版",
        "epochs": 2,
        "checkpoint": (
            "models/"
            "win_value_v1_original_generation0_197800_epoch002.pt"
        ),
        "evaluation": (
            "results/evaluations/"
            "v1_original_generation0_197800_epoch002_1000_same_seed_p0.json"
        ),
        "timing": (
            "v1_original_generation0.timings.json",
            "train_v1_original_epoch002",
        ),
    },
)


def build_summary(root: Path) -> dict[str, Any]:
    import torch

    result_directory = root / "results" / "evaluations"
    timing_cache: dict[str, dict[str, float]] = {}
    rows = []
    for condition in _CONDITIONS:
        checkpoint_path = root / condition["checkpoint"]
        evaluation_path = root / condition["evaluation"]
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if not evaluation_path.is_file():
            raise FileNotFoundError(evaluation_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8-sig")
        )
        timing = condition["timing"]
        if timing is None:
            start_path = root / condition["timing_start"]
            training_seconds = (
                checkpoint_path.stat().st_mtime
                - start_path.stat().st_mtime
            )
            timing_source = "inferred_from_artifact_mtime"
        else:
            timing_file, timing_key = timing
            if timing_file not in timing_cache:
                timing_cache[timing_file] = json.loads(
                    (result_directory / timing_file).read_text(
                        encoding="utf-8-sig"
                    )
                )
            training_seconds = float(
                timing_cache[timing_file][timing_key]
            )
            timing_source = "measured_wall_clock"
        metrics = checkpoint["metrics"]
        rows.append(
            {
                **{
                    key: condition[key]
                    for key in ("key", "model", "epochs")
                },
                "checkpoint": condition["checkpoint"],
                "evaluation": condition["evaluation"],
                "test_brier": float(metrics["test_brier"]),
                "test_log_loss": float(metrics["test_log_loss"]),
                "index0_win_rate": float(evaluation["win_rate"]),
                "training_seconds": training_seconds,
                "training_time_source": timing_source,
            }
        )

    original_rows = [
        row for row in rows if row["model"] == "V1履歴未修正版"
    ]
    fixed_rows = [
        row for row in rows if row["model"] == "V1履歴修正版"
    ]
    best_original = max(
        original_rows, key=lambda row: row["index0_win_rate"]
    )
    best_fixed_rate = max(
        row["index0_win_rate"] for row in fixed_rows
    )
    gate = {
        "best_original_key": best_original["key"],
        "best_original_checkpoint": best_original["checkpoint"],
        "best_original_index0_win_rate": best_original[
            "index0_win_rate"
        ],
        "best_fixed_index0_win_rate": best_fixed_rate,
        "at_least_25_percent": (
            best_original["index0_win_rate"] >= 0.25
        ),
        "beats_fixed_by_3_points": (
            best_original["index0_win_rate"] - best_fixed_rate >= 0.03
        ),
    }
    gate["run_all_seats"] = (
        gate["at_least_25_percent"]
        or gate["beats_fixed_by_3_points"]
    )
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "training_games": 197800,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "evaluation_games": 1000,
        "evaluation_seed": 20260725,
        "player_index": 0,
        "adaptive_pq_pruning": True,
        "approximate_new_color_neighbors": True,
        "traditional_refill": True,
        "conditions": rows,
        "all_seats_gate": gate,
    }


def write_summary(summary: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# V2 / V1履歴比較",
        "",
        (
            "全条件: 197,800戦、batch 256、learning rate 1e-3、"
            "index 0、seed 20260725、1,000戦。"
        ),
        "",
        "| モデル | epoch | test Brier | test logloss | index 0勝率 | 学習時間 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        lines.append(
            f"| {row['model']} | {row['epochs']} | "
            f"{row['test_brier']:.6f} | "
            f"{row['test_log_loss']:.6f} | "
            f"{row['index0_win_rate']:.3%} | "
            f"{row['training_seconds'] / 60:.1f}分 |"
        )
    gate = summary["all_seats_gate"]
    lines.extend(
        [
            "",
            (
                "全席評価ゲート: "
                + ("通過" if gate["run_all_seats"] else "未通過")
                + f"（最良未修正版 {gate['best_original_index0_win_rate']:.3%}、"
                + f"最良修正版 {gate['best_fixed_index0_win_rate']:.3%}）"
            ),
            "",
            "注: 既存1epochの学習時間のみ成果物mtime差からの推定値。新規条件は実測wall clock。",
        ]
    )
    output.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd()
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/evaluations/"
            "v2_v1_original_history_comparison.json"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output
    summary = build_summary(root)
    write_summary(summary, output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
