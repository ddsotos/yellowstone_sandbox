import json

from yellowstone.one_card_preference_audit import run_one_card_preference_audit


def test_one_card_preference_audit_saves_summary_and_cases(tmp_path) -> None:
    summary = run_one_card_preference_audit(
        checkpoint="models/win_value.pt", games=3, seed=7, output=tmp_path
    )

    assert summary["matched_turns"] > 0
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "top10_cases.json").is_file()
    assert (tmp_path / "top10_cases.md").is_file()
    cases = json.loads((tmp_path / "top10_cases.json").read_text(encoding="utf-8"))
    assert cases
    assert "complete_state" in cases[0]
