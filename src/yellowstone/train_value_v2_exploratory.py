"""Train exploratory-refill Yellowstone value V2 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yellowstone.train_value_v2 import (
    _archive_paths,
    _metric,
    _split_buckets,
)
from yellowstone.value_v2_exploratory import (
    CANONICALIZATION_V2_EXPLORATORY,
    HISTORY_SEMANTICS_V2_EXPLORATORY,
    VALUE_CONTEXT_SIZE_V2_EXPLORATORY,
    VALUE_SCHEMA_V2_EXPLORATORY,
    build_win_value_net_v2_exploratory,
)


MODEL_ARCHITECTURE_V2_EXPLORATORY = (
    "yellowstone.win_value.v2-exploratory.conv2_64_fc128"
)


def train_v2_exploratory(
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
        raise ImportError(
            "exploratory V2 training requires `pip install -e .[value]`"
        ) from error
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    paths = _archive_paths(data_path)
    for path in paths:
        with np.load(path) as archive:
            if archive["context"].shape[1] != (
                VALUE_CONTEXT_SIZE_V2_EXPLORATORY
            ):
                raise ValueError(
                    f"unexpected context size in {path}: "
                    f"{archive['context'].shape[1]}"
                )
    model = build_win_value_net_v2_exploratory()
    if resume_checkpoint is not None:
        checkpoint = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
        expected = {
            "value_schema": VALUE_SCHEMA_V2_EXPLORATORY,
            "input_canonicalization": CANONICALIZATION_V2_EXPLORATORY,
            "history_semantics": HISTORY_SEMANTICS_V2_EXPLORATORY,
            "context_size": VALUE_CONTEXT_SIZE_V2_EXPLORATORY,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"resume checkpoint differs at {key}")
        model.load_state_dict(checkpoint["state_dict"])
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
                train_data = TensorDataset(
                    torch.from_numpy(archive["board"][mask]),
                    torch.from_numpy(archive["context"][mask]),
                    torch.from_numpy(archive["target"][mask]),
                )
            loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True
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
        "test_brier": _metric(
            model, paths, seed, bucket=9, log_loss=False
        ),
        "test_log_loss": _metric(
            model, paths, seed, bucket=9, log_loss=True
        ),
    }
    saved = {
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "value_schema": VALUE_SCHEMA_V2_EXPLORATORY,
        "input_canonicalization": CANONICALIZATION_V2_EXPLORATORY,
        "history_semantics": HISTORY_SEMANTICS_V2_EXPLORATORY,
        "context_size": VALUE_CONTEXT_SIZE_V2_EXPLORATORY,
        "model_architecture": MODEL_ARCHITECTURE_V2_EXPLORATORY,
        "convolution_layers": 2,
        "hidden_channels": 64,
        "hidden_size": 128,
    }
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, destination)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    result = train_v2_exploratory(
        args.data,
        args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        resume_checkpoint=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
