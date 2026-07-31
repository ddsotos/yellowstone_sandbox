from pathlib import Path

import pytest

from yellowstone.train_value_milestones import train_value_milestones


def test_continuous_v1_milestones_are_saved_with_contract(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(31)
    np.savez_compressed(
        data / "part_000100.npz",
        board=rng.normal(size=(20, 29, 7, 7)).astype(np.float32),
        context=rng.normal(size=(20, 81)).astype(np.float32),
        target=np.asarray([0, 1] * 10, dtype=np.float32),
        game_id=np.repeat(np.arange(10, dtype=np.int64), 2),
    )
    prefix = tmp_path / "models" / "v1"
    result = train_value_milestones(
        data,
        prefix,
        split_game_count=10,
        start_part=100,
        end_part=100,
        milestones=(10, 50, 100),
        batch_size=4,
        seed=23,
        progress_interval_parts=1,
    )

    assert [row["percent"] for row in result["milestones"]] == [
        10,
        50,
        100,
    ]
    actual_fractions = []
    for percent in (10, 50, 100):
        path = Path(f"{prefix}_pct{percent:03d}.pt")
        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["value_schema"] == "yellowstone.value.v1"
        assert checkpoint["history_semantics"] == (
            "rolling_last_two_placements"
        )
        assert checkpoint["input_canonicalization"] == (
            "fast_lr_ud_color_v1"
        )
        assert checkpoint["split_game_count"] == 10
        assert checkpoint["metrics"]["test_brier"] >= 0
        actual_fractions.append(checkpoint["actual_fraction"])
    assert actual_fractions == sorted(actual_fractions)
    assert actual_fractions[-1] == 1.0
