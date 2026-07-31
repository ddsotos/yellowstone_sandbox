"""Train the compact transition-aware Yellowstone V2-lite model."""

from __future__ import annotations

import argparse
from pathlib import Path

from yellowstone.cnn import build_win_value_net_v2_lite
from yellowstone.train_value_v2 import _archive_paths, _split_buckets
from yellowstone.value_v2_lite import (
    CANONICALIZATION_V2_LITE,
    VALUE_CONTEXT_SIZE_V2_LITE,
    VALUE_SCHEMA_V2_LITE,
)


def train_v2_lite(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 20260726,
) -> dict[str, float]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    paths = _archive_paths(data_path)
    model = build_win_value_net_v2_lite()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for path in paths:
            with np.load(path) as archive:
                split = _split_buckets(archive["game_id"], seed)
                mask = split < 8
                if not mask.any():
                    continue
                dataset = TensorDataset(
                    torch.from_numpy(archive["board"][mask]),
                    torch.from_numpy(archive["context"][mask]),
                    torch.from_numpy(archive["target"][mask]),
                )
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            for batch_board, batch_context, batch_target in loader:
                optimizer.zero_grad()
                loss = loss_fn(
                    model(batch_board, batch_context), batch_target
                )
                loss.backward()
                optimizer.step()
    metrics = {
        "validation_brier": _metric(
            model, paths, seed, bucket=8, log_loss=False
        ),
        "test_brier": _metric(model, paths, seed, bucket=9, log_loss=False),
        "test_log_loss": _metric(
            model, paths, seed, bucket=9, log_loss=True
        ),
    }
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "seed": seed,
            "epochs": epochs,
            "value_schema": VALUE_SCHEMA_V2_LITE,
            "input_canonicalization": CANONICALIZATION_V2_LITE,
            "context_size": VALUE_CONTEXT_SIZE_V2_LITE,
        },
        destination,
    )
    return metrics


def _metric(model, paths, seed: int, *, bucket: int, log_loss: bool) -> float:
    import numpy as np
    import torch

    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for path in paths:
            with np.load(path) as archive:
                mask = _split_buckets(archive["game_id"], seed) == bucket
                if not mask.any():
                    continue
                target = archive["target"][mask]
                probabilities = torch.sigmoid(
                    model(
                        torch.from_numpy(archive["board"][mask]),
                        torch.from_numpy(archive["context"][mask]),
                    )
                ).numpy()
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
    parser = argparse.ArgumentParser(description="Train Yellowstone V2-lite")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    metrics = train_v2_lite(
        args.data,
        args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(" ".join(f"{key}={value:.6f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()

