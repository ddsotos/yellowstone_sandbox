"""Train and evaluate the CNN win-value model on collected heuristic data."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from yellowstone.cnn import (
    DEFAULT_WIN_VALUE_HIDDEN_CHANNELS,
    DEFAULT_WIN_VALUE_HIDDEN_SIZE,
    build_win_value_net,
    win_value_architecture_from_checkpoint,
    win_value_architecture_metadata,
)
from yellowstone.value_learning import VALUE_CONTEXT_SIZE, split_game_ids


def _game_id_sha256(ids: set[int]) -> str:
    payload = ",".join(str(game_id) for game_id in sorted(ids))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def train_from_archive(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    resume_checkpoint: str | Path | None = None,
    input_canonicalization: str | None = None,
    start_part: int | None = None,
    end_part: int | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
    context_size: int = VALUE_CONTEXT_SIZE,
    split_game_count: int | None = None,
    train_game_id_limit: int | None = None,
    convolution_layers: int = 2,
    hidden_channels: int = DEFAULT_WIN_VALUE_HIDDEN_CHANNELS,
    hidden_size: int = DEFAULT_WIN_VALUE_HIDDEN_SIZE,
    progress_checkpoint_path: str | Path | None = None,
) -> dict[str, float]:
    """Train with game-level splits and return validation/test calibration metrics."""
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as error:
        raise ImportError("training requires `pip install -e .[value]`") from error
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    paths = _archive_paths(
        data_path, start_part=start_part, end_part=end_part
    )
    observed_game_count = (
        max(int(np.load(path)["game_id"].max()) for path in paths) + 1
    )
    game_count = (
        observed_game_count if split_game_count is None else split_game_count
    )
    if game_count != observed_game_count:
        raise ValueError(
            "split game count must match the continuous archive population: "
            f"{game_count} != {observed_game_count}"
        )
    if train_game_id_limit is not None and not (
        0 < train_game_id_limit <= game_count
    ):
        raise ValueError("train game ID limit must be in 1..split game count")
    train_ids, validation_ids, test_ids = split_game_ids(game_count, seed=seed)
    full_train_ids = train_ids
    if train_game_id_limit is not None:
        train_ids = {
            game_id
            for game_id in train_ids
            if game_id < train_game_id_limit
        }
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    architecture = win_value_architecture_metadata(
        convolution_layers=convolution_layers,
        hidden_channels=hidden_channels,
        hidden_size=hidden_size,
    )
    torch.manual_seed(seed)
    model = build_win_value_net(
        context_size=context_size,
        convolution_layers=convolution_layers,
        hidden_channels=hidden_channels,
        hidden_size=hidden_size,
    )
    started_fresh = resume_checkpoint is None
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, weights_only=False)
        resumed_architecture = win_value_architecture_from_checkpoint(
            checkpoint
        )
        if resumed_architecture != architecture:
            raise ValueError("resume checkpoint architecture differs")
        model.load_state_dict(checkpoint["state_dict"])
        resumed_canonicalization = checkpoint.get("input_canonicalization")
        if resumed_canonicalization != input_canonicalization:
            raise ValueError(
                "resume checkpoint canonicalization differs: "
                f"{resumed_canonicalization!r} != {input_canonicalization!r}"
            )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    progress_path = (
        Path(progress_checkpoint_path)
        if progress_checkpoint_path is not None
        else None
    )
    progress_contract = {
        "training_data": str(data_path),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "context_size": context_size,
        "split_game_count": game_count,
        "train_game_id_limit": train_game_id_limit,
        **architecture,
    }
    completed_epochs = 0
    if progress_path is not None and progress_path.exists():
        progress = torch.load(
            progress_path, map_location="cpu", weights_only=False
        )
        if progress.get("progress_contract") != progress_contract:
            raise ValueError("training progress contract differs")
        completed_epochs = int(progress["completed_epochs"])
        if completed_epochs > epochs:
            raise ValueError(
                "training progress is beyond requested epochs"
            )
        model.load_state_dict(progress["state_dict"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        torch.set_rng_state(progress["torch_rng_state"])
        started_fresh = bool(progress["started_fresh"])
    for epoch_index in range(completed_epochs, epochs):
        model.train()
        for path in paths:
            with np.load(path) as archive:
                train_mask = np.isin(
                    archive["game_id"], list(train_ids)
                )
                if not train_mask.any():
                    continue
                train_data = TensorDataset(
                    torch.from_numpy(archive["board"][train_mask]),
                    torch.from_numpy(archive["context"][train_mask]),
                    torch.from_numpy(archive["target"][train_mask]),
                )
            loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            for batch_board, batch_context, batch_target in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(batch_board, batch_context), batch_target)
                loss.backward()
                optimizer.step()
        completed_epochs = epoch_index + 1
        if progress_path is not None:
            _atomic_torch_save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "completed_epochs": completed_epochs,
                    "started_fresh": started_fresh,
                    "progress_contract": progress_contract,
                },
                progress_path,
            )
    metrics = {
        "validation_brier": _archive_brier(model, paths, validation_ids),
        "test_brier": _archive_brier(model, paths, test_ids),
        "test_log_loss": _archive_log_loss(model, paths, test_ids),
    }
    saved_checkpoint = {
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "seed": seed,
        "training_data": str(data_path),
        "epochs": epochs,
        "fresh_initialization": started_fresh,
        "context_size": context_size,
        "split_game_count": game_count,
        "train_game_id_limit": train_game_id_limit,
        "train_split_games": len(train_ids),
        "full_train_split_games": len(full_train_ids),
        "validation_split_games": len(validation_ids),
        "test_split_games": len(test_ids),
        "split_policy": "shared_population_80_10_10_then_train_id_limit",
        "train_game_ids_sha256": _game_id_sha256(train_ids),
        "full_train_game_ids_sha256": _game_id_sha256(full_train_ids),
        "validation_game_ids_sha256": _game_id_sha256(validation_ids),
        "test_game_ids_sha256": _game_id_sha256(test_ids),
        **architecture,
    }
    if input_canonicalization is not None:
        saved_checkpoint["input_canonicalization"] = input_canonicalization
    if checkpoint_metadata:
        saved_checkpoint.update(checkpoint_metadata)
    _atomic_torch_save(saved_checkpoint, Path(checkpoint_path))
    return metrics


def _atomic_torch_save(payload: object, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _probabilities(model, board, context):
    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(board), torch.from_numpy(context))).numpy()


def _brier(model, board, context, target) -> float:
    import numpy as np

    probabilities = _probabilities(model, board, context)
    return float(np.mean((probabilities - target) ** 2))


def _log_loss(model, board, context, target) -> float:
    import numpy as np

    probabilities = np.clip(_probabilities(model, board, context), 1e-7, 1 - 1e-7)
    return float(-np.mean(target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities)))


def _part_number(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("part_"))
    except ValueError as error:
        raise ValueError(f"invalid part archive name: {path.name}") from error


def _archive_paths(
    data_path: str | Path,
    *,
    start_part: int | None = None,
    end_part: int | None = None,
) -> tuple[Path, ...]:
    if (start_part is None) != (end_part is None):
        raise ValueError("start_part and end_part must be supplied together")
    if start_part is not None and (
        start_part < 0 or end_part is None or end_part < start_part
    ):
        raise ValueError("invalid inclusive part range")
    path = Path(data_path)
    paths = tuple(sorted(path.glob("part_*.npz"))) if path.is_dir() else (path,)
    if start_part is not None:
        if not path.is_dir():
            raise ValueError("part range requires a chunk directory")
        paths = tuple(
            item
            for item in paths
            if start_part <= _part_number(item) <= end_part
        )
    if not paths or not all(item.is_file() for item in paths):
        raise FileNotFoundError(f"no value-data archives found at {path}")
    return paths


def _archive_brier(model, paths: tuple[Path, ...], ids: set[int]) -> float:
    import numpy as np

    total, count = 0.0, 0
    for path in paths:
        archive = np.load(path)
        mask = np.isin(archive["game_id"], list(ids))
        if mask.any():
            probabilities = _probabilities(model, archive["board"][mask], archive["context"][mask])
            total += float(np.sum((probabilities - archive["target"][mask]) ** 2))
            count += int(mask.sum())
    if count == 0:
        raise ValueError("split has no records")
    return total / count


def _archive_log_loss(model, paths: tuple[Path, ...], ids: set[int]) -> float:
    import numpy as np

    total, count = 0.0, 0
    for path in paths:
        archive = np.load(path)
        mask = np.isin(archive["game_id"], list(ids))
        if mask.any():
            target = archive["target"][mask]
            probabilities = np.clip(
                _probabilities(model, archive["board"][mask], archive["context"][mask]), 1e-7, 1 - 1e-7
            )
            total += float(-np.sum(target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities)))
            count += int(mask.sum())
    if count == 0:
        raise ValueError("split has no records")
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Yellowstone win-value CNN")
    parser.add_argument("--data", type=Path, required=True, help="chunk directory or one .npz archive")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/win_value.pt")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--input-canonicalization")
    parser.add_argument("--start-part", type=int)
    parser.add_argument("--end-part", type=int)
    parser.add_argument("--value-schema")
    parser.add_argument("--history-semantics")
    parser.add_argument("--training-games", type=int)
    parser.add_argument("--context-size", type=int, default=VALUE_CONTEXT_SIZE)
    parser.add_argument("--split-game-count", type=int)
    parser.add_argument("--train-game-id-limit", type=int)
    parser.add_argument(
        "--convolution-layers",
        type=int,
        choices=(2, 3),
        default=2,
    )
    parser.add_argument("--progress-checkpoint", type=Path)
    args = parser.parse_args()
    metadata = {
        key: value
        for key, value in {
            "value_schema": args.value_schema,
            "history_semantics": args.history_semantics,
            "training_games": args.training_games,
        }.items()
        if value is not None
    }
    metrics = train_from_archive(
        args.data, args.checkpoint, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed, resume_checkpoint=args.resume,
        input_canonicalization=args.input_canonicalization,
        start_part=args.start_part, end_part=args.end_part,
        checkpoint_metadata=metadata,
        context_size=args.context_size,
        split_game_count=args.split_game_count,
        train_game_id_limit=args.train_game_id_limit,
        convolution_layers=args.convolution_layers,
        progress_checkpoint_path=args.progress_checkpoint,
    )
    print(" ".join(f"{key}={value:.6f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
