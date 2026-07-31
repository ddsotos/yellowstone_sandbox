"""Evaluate an exploratory V2 checkpoint against heuristic opponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yellowstone.value_evaluation_v2 import evaluate_v2_value_player
from yellowstone.value_v2_exploratory import ExploratoryV2Estimator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--player-index", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    estimator = ExploratoryV2Estimator(str(args.checkpoint))
    result = evaluate_v2_value_player(
        estimator,
        games=args.games,
        seed=args.seed,
        player_index=args.player_index,
    )
    result.update(
        {
            "checkpoint": str(args.checkpoint),
            "value_schema": "yellowstone.value.v2-exploratory-refill.v1",
            "input_canonicalization": "strict_residual_v2_uniform_negative_ratios_refill_risk_v1",
            "history_semantics": "rolling_last_three_completed_turns_v2",
            "opponent_private_inputs": False,
        }
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
