import json

import pytest

from yellowstone.summarize_v1_conv3 import summarize


def test_summarize_conv3_uses_all_four_seats_and_matched_baseline(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "conv3.pt"
    torch.save(
        {
            "seed": 23,
            "training_games": 100,
            "metrics": {
                "validation_brier": 0.13,
                "test_brier": 0.14,
                "test_log_loss": 0.42,
            },
            "value_schema": "yellowstone.value.v1",
            "history_semantics": "rolling_last_two_placements",
            "input_canonicalization": "fast_lr_ud_color_v1",
            "model_architecture": (
                "yellowstone.win_value.v1.conv3_64_fc128"
            ),
            "convolution_layers": 3,
            "hidden_channels": 64,
            "hidden_size": 128,
        },
        checkpoint,
    )
    baseline = tmp_path / "baseline.json"
    baseline_checkpoint = tmp_path / "baseline.pt"
    torch.save(
        {"metrics": {"test_brier": 0.15}},
        baseline_checkpoint,
    )
    baseline.write_text(
        json.dumps(
            {
                "models": {"original": str(baseline_checkpoint)},
                "rows": [
                    {
                        "checkpoint_model": "original",
                        "history_semantics_match": True,
                        "seats": [
                            {"win_rate": value}
                            for value in (0.2, 0.21, 0.22, 0.23)
                        ],
                        "all_seats_games": 40,
                        "all_seats_win_rate": 0.215,
                        "all_seats_one_card_turn_rate": 0.36,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    for player_index, wins in enumerate((2.0, 3.0, 4.0, 5.0)):
        path = evaluations / (
            "v1_original_conv3_generation0_197800_epoch002_"
            f"10_seed7_p{player_index}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "games": 10,
                    "fractional_wins": wins,
                    "win_rate": wins / 10,
                    "evaluated_player_one_card_turns": 4,
                    "evaluated_player_two_card_turns": 6,
                    "evaluated_player_one_card_turn_rate": 0.4,
                    "elapsed_seconds": 1.0,
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "summary.json"

    result = summarize(
        checkpoint_path=checkpoint,
        baseline_path=baseline,
        baseline_comparison_path=None,
        evaluation_directory=evaluations,
        timings_path=tmp_path / "missing-timings.json",
        output_path=output,
        games_per_seat=10,
        seed=7,
    )

    assert result["conv3"]["all_seats_games"] == 40
    assert result["conv3"]["all_seats_win_rate"] == 0.35
    assert result["conv3"]["all_seats_one_card_turn_rate"] == 0.4
    assert result["conv3_minus_conv2_win_rate"] == pytest.approx(0.135)
    assert output.with_suffix(".md").is_file()
