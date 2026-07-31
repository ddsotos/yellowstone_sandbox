import json

from yellowstone.summarize_action_delta_selected_all_seats import (
    summarize_selected_all_seats,
)


def test_summarize_selected_action_delta_requires_all_four_seats(
    tmp_path,
) -> None:
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                "training": {"seed": 7},
                "milestones": [
                    {
                        "percent": percent,
                        "checkpoint": f"pct{percent}.pt",
                        "processed_train_records": percent * 10,
                        "actual_fraction": percent / 100,
                        "metrics": {"test_all_mae": 0.1},
                    }
                    for percent in (30, 100)
                ],
            }
        ),
        encoding="utf-8",
    )
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    for percent in (30, 100):
        for player in range(4):
            (evaluations / (
                f"action_delta_milestone_pct{percent:03d}_10_seed9_p"
                f"{player}.json"
            )).write_text(
                json.dumps(
                    {
                        "games": 10,
                        "seed": 9,
                        "player_index": player,
                        "fractional_wins": 2.5,
                        "win_rate": 0.25,
                        "evaluated_player_one_card_turns": 4,
                        "evaluated_player_two_card_turns": 6,
                        "evaluated_player_one_card_turn_rate": 0.4,
                    }
                ),
                encoding="utf-8",
            )
    result = summarize_selected_all_seats(
        training,
        evaluations,
        tmp_path / "summary.json",
        percentages=(30, 100),
        games_per_seat=10,
        seed=9,
    )
    assert result["status"] == "complete"
    assert [row["all_seats_games"] for row in result["milestones"]] == [
        40,
        40,
    ]
    assert all(
        row["all_seats_win_rate"] == 0.25
        for row in result["milestones"]
    )
