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


def test_milestones_infer_compact_board_and_context(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    data = tmp_path / "compact"
    data.mkdir()
    rng = np.random.default_rng(41)
    np.savez_compressed(
        data / "part_000000.npz",
        board=rng.normal(size=(20, 1, 7, 3)).astype(np.float32),
        context=rng.normal(size=(20, 155)).astype(np.float32),
        target=np.asarray([0, 1] * 10, dtype=np.float32),
        game_id=np.repeat(np.arange(10, dtype=np.int64), 2),
    )
    prefix = tmp_path / "models" / "compact"

    train_value_milestones(
        data,
        prefix,
        split_game_count=10,
        start_part=0,
        end_part=0,
        milestones=(20, 100),
        batch_size=5,
        seed=23,
        input_canonicalization="board_columns_v2",
        value_schema="yellowstone.value.v2-board-columns.v1",
        history_semantics="rolling_last_three_completed_turns_v2",
    )

    checkpoint = torch.load(Path(f"{prefix}_pct100.pt"), weights_only=False)
    assert checkpoint["context_size"] == 155
    assert checkpoint["board_channels"] == 1
    assert checkpoint["board_height"] == 7
    assert checkpoint["board_width"] == 3
    assert checkpoint["input_canonicalization"] == "board_columns_v2"


def test_milestones_can_start_from_matching_initial_checkpoint(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    data = tmp_path / "board_columns"
    data.mkdir()
    rng = np.random.default_rng(43)
    np.savez_compressed(
        data / "part_000000.npz",
        board=rng.normal(size=(20, 1, 7, 3)).astype(np.float32),
        context=rng.normal(size=(20, 62)).astype(np.float32),
        target=np.asarray([0, 1] * 10, dtype=np.float32),
        game_id=np.repeat(np.arange(10, dtype=np.int64), 2),
    )
    base_prefix = tmp_path / "models" / "base"
    train_value_milestones(
        data,
        base_prefix,
        split_game_count=10,
        start_part=0,
        end_part=0,
        milestones=(100,),
        batch_size=5,
        seed=23,
        input_canonicalization="board_columns_v1_history_none",
        history_semantics="none",
    )

    tuned_prefix = tmp_path / "models" / "tuned"
    train_value_milestones(
        data,
        tuned_prefix,
        split_game_count=10,
        start_part=0,
        end_part=0,
        milestones=(100,),
        batch_size=5,
        seed=23,
        input_canonicalization="board_columns_v1_history_none",
        history_semantics="none",
        initial_checkpoint=Path(f"{base_prefix}_pct100.pt"),
    )

    checkpoint = torch.load(Path(f"{tuned_prefix}_pct100.pt"), weights_only=False)
    assert checkpoint["initial_checkpoint"] == str(Path(f"{base_prefix}_pct100.pt"))
    assert checkpoint["input_canonicalization"] == "board_columns_v1_history_none"
