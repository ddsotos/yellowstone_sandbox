from pathlib import Path

import pytest

from yellowstone.train_value import _archive_paths, train_from_archive
from yellowstone.value_learning import split_game_ids


def test_archive_paths_filters_inclusive_part_range(tmp_path: Path) -> None:
    for number in (0, 100, 200):
        (tmp_path / f"part_{number:06d}.npz").touch()

    assert [path.name for path in _archive_paths(
        tmp_path, start_part=100, end_part=200
    )] == ["part_000100.npz", "part_000200.npz"]


def test_archive_paths_requires_complete_part_range(tmp_path: Path) -> None:
    (tmp_path / "part_000000.npz").touch()
    with pytest.raises(ValueError, match="supplied together"):
        _archive_paths(tmp_path, start_part=0)


def test_shared_split_train_limit_preserves_validation_and_test() -> None:
    train_ids, validation_ids, test_ids = split_game_ids(
        88_966, seed=20260727
    )
    limited_train_ids = {
        game_id for game_id in train_ids if game_id < 50_000
    }

    assert limited_train_ids < train_ids
    assert all(game_id < 50_000 for game_id in limited_train_ids)
    assert len(validation_ids) == 8_897
    assert len(test_ids) == 8_897


def test_epoch_progress_resume_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(17)
    np.savez_compressed(
        data / "part_000000.npz",
        board=rng.normal(size=(10, 29, 7, 7)).astype(np.float32),
        context=rng.normal(size=(10, 81)).astype(np.float32),
        target=np.asarray([0, 1] * 5, dtype=np.float32),
        game_id=np.arange(10, dtype=np.int64),
    )
    uninterrupted = tmp_path / "uninterrupted.pt"
    staged = tmp_path / "staged.pt"
    uninterrupted_progress = tmp_path / "uninterrupted.progress.pt"
    staged_progress = tmp_path / "staged.progress.pt"
    common = {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "seed": 23,
        "convolution_layers": 3,
    }

    uninterrupted_metrics = train_from_archive(
        data,
        uninterrupted,
        progress_checkpoint_path=uninterrupted_progress,
        **common,
    )
    train_from_archive(
        data,
        staged,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        seed=23,
        convolution_layers=3,
        progress_checkpoint_path=staged_progress,
    )
    staged_metrics = train_from_archive(
        data,
        staged,
        progress_checkpoint_path=staged_progress,
        **common,
    )

    uninterrupted_checkpoint = torch.load(
        uninterrupted, weights_only=False
    )
    staged_checkpoint = torch.load(staged, weights_only=False)
    assert uninterrupted_metrics == staged_metrics
    assert staged_checkpoint["epochs"] == 2
    assert staged_checkpoint["model_architecture"] == (
        "yellowstone.win_value.v1.conv3_64_fc128"
    )
    assert all(
        torch.equal(
            uninterrupted_checkpoint["state_dict"][key],
            staged_checkpoint["state_dict"][key],
        )
        for key in uninterrupted_checkpoint["state_dict"]
    )
