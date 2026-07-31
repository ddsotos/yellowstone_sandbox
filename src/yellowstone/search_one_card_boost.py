"""Resume-safe Historyfix V1 one-card probability-boost screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import monotonic
from typing import Callable

from yellowstone.evaluate_value import (
    evaluation_payload,
    validate_checkpoint_contract,
)
from yellowstone.value_evaluation import evaluate_value_player
from yellowstone.value_policy import TorchWinValueEstimator


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def boost_filename(boost_percent: int) -> str:
    return f"historyfix_v1_epoch002_one_card_boost_x{boost_percent:06d}.json"


def run_search(
    evaluate: Callable[[int], dict[str, object]],
    *,
    coarse_max: int = 200,
    coarse_step: int = 10,
    fine_step: int = 2,
    extension_max: int = 1000,
    extension_step: int = 50,
) -> tuple[str, list[dict[str, object]], int | None]:
    """Run the staged grid, stopping at the minimum fine-grid passing X."""
    results: list[dict[str, object]] = []

    def test(boost: int) -> dict[str, object]:
        result = evaluate(boost)
        results.append(result)
        return result

    previous = 0
    passing_coarse: int | None = None
    for boost in range(0, coarse_max + 1, coarse_step):
        result = test(boost)
        if float(result["win_rate"]) > 0.25:
            passing_coarse = boost
            break
        if bool(result["all_one_card_candidates_saturated"]):
            return "unreachable_saturated", results, None
        previous = boost

    if passing_coarse is not None:
        if passing_coarse == 0:
            return "threshold_found", results, 0
        for boost in range(previous + fine_step, passing_coarse + 1, fine_step):
            result = test(boost)
            if float(result["win_rate"]) > 0.25:
                return "threshold_found", results, boost
        raise AssertionError("passing coarse endpoint did not pass when repeated")

    for boost in range(
        coarse_max + extension_step,
        extension_max + 1,
        extension_step,
    ):
        result = test(boost)
        if float(result["win_rate"]) > 0.25:
            return "threshold_found", results, boost
        if bool(result["all_one_card_candidates_saturated"]):
            return "unreachable_saturated", results, None

    boost = extension_max * 2
    while True:
        result = test(boost)
        if float(result["win_rate"]) > 0.25:
            return "threshold_found", results, boost
        if bool(result["all_one_card_candidates_saturated"]):
            return "unreachable_saturated", results, None
        boost *= 2


def run_maximization_search(
    evaluate: Callable[[int], dict[str, object]],
    *,
    minimum: int = 10,
    maximum: int = 60,
    coarse_step: int = 10,
    resolution: int = 2,
) -> tuple[list[dict[str, object]], int]:
    """Maximize a deterministic screen by halving around coarse maxima."""
    if minimum >= maximum:
        raise ValueError("minimum must be less than maximum")
    if coarse_step <= 0 or resolution <= 0:
        raise ValueError("steps must be positive")
    cache: dict[int, dict[str, object]] = {}

    def test(boost: int) -> dict[str, object]:
        if boost not in cache:
            cache[boost] = evaluate(boost)
        return cache[boost]

    coarse = list(range(minimum, maximum + 1, coarse_step))
    if coarse[-1] != maximum:
        coarse.append(maximum)
    for boost in coarse:
        test(boost)
    best_rate = max(float(test(boost)["win_rate"]) for boost in coarse)
    coarse_best = [
        boost
        for boost in coarse
        if float(test(boost)["win_rate"]) == best_rate
    ]

    intervals: set[tuple[int, int]] = set()
    for boost in coarse_best:
        index = coarse.index(boost)
        if index:
            intervals.add((coarse[index - 1], boost))
        if index + 1 < len(coarse):
            intervals.add((boost, coarse[index + 1]))

    for initial_low, initial_high in sorted(intervals):
        low, high = initial_low, initial_high
        while high - low > resolution:
            midpoint = (low + high) // 2
            points = (low, midpoint, high)
            for boost in points:
                test(boost)
            best = max(
                points,
                key=lambda boost: (
                    float(test(boost)["win_rate"]),
                    -boost,
                ),
            )
            if best == low:
                high = midpoint
            elif best == high:
                low = midpoint
            else:
                left_midpoint = (low + midpoint) // 2
                right_midpoint = (midpoint + high) // 2
                for boost in (left_midpoint, right_midpoint):
                    test(boost)
                local = max(
                    (low, left_midpoint, midpoint, right_midpoint, high),
                    key=lambda boost: (
                        float(test(boost)["win_rate"]),
                        -boost,
                    ),
                )
                if local < midpoint:
                    high = midpoint
                elif local > midpoint:
                    low = midpoint
                else:
                    low, high = left_midpoint, right_midpoint

    best_boost = max(
        cache,
        key=lambda boost: (
            float(cache[boost]["win_rate"]),
            -boost,
        ),
    )
    return [cache[boost] for boost in sorted(cache)], best_boost


def write_markdown(path: Path, comparison: dict[str, object]) -> None:
    rows = [
        "# Historyfix V1 one-card probability-boost screen",
        "",
        "席0・同一seedの100戦による方策調整screenであり、"
        "正規の4席モデル成績ではない。",
        "",
        f"- state: `{comparison['state']}`",
        f"- best X: `{comparison.get('best_boost_percent')}`",
        f"- best win rate: `{comparison.get('best_win_rate')}`",
        "",
        "| X (%) | fractional wins | win rate | one-card rate | raw p mean | adjusted mean | seconds | fingerprint |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for result in comparison["results"]:
        rows.append(
            "| {x} | {wins:.3f} | {rate:.3%} | {one:.3%} | {raw:.6f} | "
            "{adjusted:.6f} | {seconds:.1f} | `{fingerprint}` |".format(
                x=result["one_card_win_probability_boost_percent"],
                wins=result["fractional_wins"],
                rate=result["win_rate"],
                one=result["evaluated_player_one_card_turn_rate"],
                raw=result["selected_raw_win_probability"]["mean"],
                adjusted=result["selected_adjusted_score"]["mean"],
                seconds=result["elapsed_seconds"],
                fingerprint=result["policy_fingerprint"][:12],
            )
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, default=0)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--coarse-max", type=int, default=200)
    parser.add_argument("--coarse-step", type=int, default=10)
    parser.add_argument("--fine-step", type=int, default=2)
    parser.add_argument("--extension-max", type=int, default=1000)
    parser.add_argument("--extension-step", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=("threshold", "maximize-range"),
        default="threshold",
    )
    parser.add_argument("--range-min", type=int, default=10)
    parser.add_argument("--range-max", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=2)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.player_index != 0:
        parser.error("this screen is intentionally restricted to player index 0")
    for name in (
        "coarse_max",
        "coarse_step",
        "fine_step",
        "extension_max",
        "extension_step",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    contract = validate_checkpoint_contract(
        args.checkpoint, current_turn_history_only=True
    )
    estimator = TorchWinValueEstimator(str(args.checkpoint))
    args.output_directory.mkdir(parents=True, exist_ok=True)

    def write_status(
        state: str, step: str, last_completed_step: str = "", message: str = ""
    ) -> None:
        atomic_write_json(
            args.status,
            {
                "state": state,
                "step": step,
                "last_completed_step": last_completed_step,
                "message": message,
                "updated_at": __import__("datetime").datetime.now().astimezone().isoformat(),
                "pid": os.getpid(),
                "checkpoint": str(args.checkpoint),
                "output_directory": str(args.output_directory),
                "comparison": str(args.comparison),
            },
        )

    completed_step = ""

    def evaluate(boost: int) -> dict[str, object]:
        nonlocal completed_step
        output = args.output_directory / boost_filename(boost)
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
            completed_step = f"x_{boost}"
            return payload
        write_status("running", f"x_{boost}", completed_step)
        started = monotonic()
        result = evaluate_value_player(
            estimator,
            games=args.games,
            seed=args.seed,
            player_index=0,
            adaptive_pq_pruning=True,
            approximate_new_color_neighbor_limit=True,
            current_turn_history_only=True,
            one_card_win_probability_boost_percent=float(boost),
        )
        payload = evaluation_payload(
            result,
            boost_percent=float(boost),
            checkpoint_contract=contract,
            elapsed_seconds=monotonic() - started,
        )
        payload.update(
            {
                "seed": args.seed,
                "player_index": 0,
                "adaptive_pq_pruning": True,
                "approximate_new_color_neighbors": True,
            }
        )
        atomic_write_json(output, payload)
        completed_step = f"x_{boost}"
        write_status("running", completed_step, completed_step)
        return payload

    write_status("running", "initialize", completed_step)
    try:
        if args.mode == "maximize-range":
            results, best_boost = run_maximization_search(
                evaluate,
                minimum=args.range_min,
                maximum=args.range_max,
                coarse_step=args.coarse_step,
                resolution=args.resolution,
            )
            state = "maximum_found"
            minimum = None
        else:
            state, results, minimum = run_search(
                evaluate,
                coarse_max=args.coarse_max,
                coarse_step=args.coarse_step,
                fine_step=args.fine_step,
                extension_max=args.extension_max,
                extension_step=args.extension_step,
            )
            best_boost = max(
                (
                    int(result["one_card_win_probability_boost_percent"])
                    for result in results
                ),
                key=lambda boost: (
                    float(
                        next(
                            result["win_rate"]
                            for result in results
                            if int(
                                result[
                                    "one_card_win_probability_boost_percent"
                                ]
                            )
                            == boost
                        )
                    ),
                    -boost,
                ),
            )
        unique = {
            int(result["one_card_win_probability_boost_percent"]): result
            for result in results
        }
        comparison = {
            "screen_kind": "seat0_100_game_policy_tuning_screen",
            "formal_four_seat_model_evaluation": False,
            "state": state,
            "games": args.games,
            "seed": args.seed,
            "player_index": 0,
            "pass_threshold": 0.25,
            "pass_operator": ">",
            "minimum_passing_boost_percent": minimum,
            "search_mode": args.mode,
            "range_min_percent": (
                args.range_min if args.mode == "maximize-range" else None
            ),
            "range_max_percent": (
                args.range_max if args.mode == "maximize-range" else None
            ),
            "resolution_percent": (
                args.resolution if args.mode == "maximize-range" else None
            ),
            "best_boost_percent": best_boost,
            "best_win_rate": float(unique[best_boost]["win_rate"]),
            "checkpoint": str(args.checkpoint),
            "checkpoint_contract": contract,
            "results": [unique[key] for key in sorted(unique)],
        }
        atomic_write_json(args.comparison, comparison)
        write_markdown(args.markdown, comparison)
        write_status("complete", "complete", completed_step)
    except Exception as error:
        write_status("failed", "failed", completed_step, str(error))
        raise


if __name__ == "__main__":
    main()
