import json

import pytest

from yellowstone.summarize_v2_lite import summarize


def test_summarize_v2_lite_accepts_bom_timings(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "context_size": 138,
            "seed": 7,
            "metrics": {
                "test_brier": 0.13,
                "test_log_loss": 0.42,
            },
        },
        checkpoint,
    )
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    for player_index in range(4):
        path = evaluations / (
            "v2_lite_transition_generation0_197800_epoch001_"
            f"10_seed9_p{player_index}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "games": 10,
                    "wins": 2.0,
                    "win_rate": 0.2,
                    "player_index": player_index,
                    "turns": 20,
                    "one_card_turns": 8,
                    "two_card_turns": 12,
                    "one_card_turn_rate": 0.4,
                }
            ),
            encoding="utf-8",
        )
    timings = tmp_path / "timings.json"
    timings.write_text('{"train": 1.0}', encoding="utf-8-sig")

    result = summarize(
        checkpoint_path=checkpoint,
        evaluation_directory=evaluations,
        output_path=tmp_path / "summary.json",
        games_per_seat=10,
        seed=9,
        timings_path=timings,
    )

    assert result["all_seats_win_rate"] == 0.2
    assert result["timings_seconds"] == {"train": 1.0}
