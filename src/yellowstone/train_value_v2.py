"""Train strict-canonical, refill-conditioned Yellowstone value V2 models."""

from __future__ import annotations

import argparse
from pathlib import Path

from yellowstone.cnn import build_win_value_net_v2
from yellowstone.value_v2 import VALUE_CONTEXT_SIZE_V2


VALUE_SCHEMA_V2 = "yellowstone.value.v2"
CANONICALIZATION_V2 = "strict_residual_v2"


def train_v2(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, float]:
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as error:
        raise ImportError("V2 training requires `pip install -e .[value]`") from error
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    paths = _archive_paths(data_path)
    model = build_win_value_net_v2()
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, weights_only=False)
        if checkpoint.get("value_schema") != VALUE_SCHEMA_V2:
            raise ValueError("resume checkpoint is not V2")
        model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for path in paths:
            archive = np.load(path)
            split = _split_buckets(archive["game_id"], seed)
            mask = split < 8
            if not mask.any():
                continue
            train_data = TensorDataset(
                torch.from_numpy(archive["board"][mask]),
                torch.from_numpy(archive["context"][mask]),
                torch.from_numpy(archive["target"][mask]),
            )
            loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            for batch_board, batch_context, batch_target in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(batch_board, batch_context), batch_target)
                loss.backward()
                optimizer.step()
    metrics = {
        "validation_brier": _metric(model, paths, seed, bucket=8, log_loss=False),
        "test_brier": _metric(model, paths, seed, bucket=9, log_loss=False),
        "test_log_loss": _metric(model, paths, seed, bucket=9, log_loss=True),
    }
    saved = {
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "seed": seed,
        "value_schema": VALUE_SCHEMA_V2,
        "input_canonicalization": CANONICALIZATION_V2,
        "context_size": VALUE_CONTEXT_SIZE_V2,
    }
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, destination)
    return metrics


def _archive_paths(data_path: str | Path) -> tuple[Path, ...]:
    path = Path(data_path)
    paths = tuple(sorted(path.glob("part_*.npz"))) if path.is_dir() else (path,)
    if not paths or not all(item.is_file() for item in paths):
        raise FileNotFoundError(f"no V2 tensor archives found at {path}")
    return paths


def _split_buckets(game_ids, seed: int):
    import numpy as np

    ids = np.asarray(game_ids, dtype=np.uint64)
    mixed = ids * np.uint64(0x9E3779B185EBCA87) + np.uint64(seed & 0xFFFFFFFF)
    mixed ^= mixed >> np.uint64(33)
    return (mixed % np.uint64(10)).astype(np.int8)


def _probabilities(model, board, context):
    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.from_numpy(board), torch.from_numpy(context))
        ).numpy()


def _metric(model, paths, seed: int, *, bucket: int, log_loss: bool) -> float:
    import numpy as np

    total = 0.0
    count = 0
    for path in paths:
        archive = np.load(path)
        mask = _split_buckets(archive["game_id"], seed) == bucket
        if not mask.any():
            continue
        target = archive["target"][mask]
        probabilities = _probabilities(
            model, archive["board"][mask], archive["context"][mask]
        )
        if log_loss:
            probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
            total += float(
                -np.sum(
                    target * np.log(probabilities)
                    + (1 - target) * np.log(1 - probabilities)
                )
            )
        else:
            total += float(np.sum((probabilities - target) ** 2))
        count += int(mask.sum())
    if count == 0:
        raise ValueError(f"split bucket {bucket} contains no records")
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Yellowstone value V2")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    metrics = train_v2(
        args.data,
        args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        resume_checkpoint=args.resume,
    )
    print(" ".join(f"{key}={value:.6f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
