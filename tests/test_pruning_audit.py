import json

from yellowstone.pruning_audit import run_pruning_audit


def test_pruning_audit_saves_summary_and_cases(tmp_path) -> None:
    summary = run_pruning_audit(
        checkpoint="models/win_value.pt", games=1, seed=7, output=tmp_path
    )

    assert summary["audited_turns"] > 0
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "top10_cases.json").is_file()
    assert (tmp_path / "top10_cases.md").is_file()
    cases = json.loads((tmp_path / "top10_cases.json").read_text(encoding="utf-8"))
    assert cases
    assert "complete_state" in cases[0]
