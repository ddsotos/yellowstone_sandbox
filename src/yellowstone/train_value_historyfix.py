"""Train a fresh V1 history-fix model and attach explicit lineage metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from yellowstone.convert_replay_v2_to_v1_historyfix import (
    VALUE_SCHEMA_V1_HISTORYFIX,
)
from yellowstone.train_value import train_from_archive
from yellowstone.value_canonicalization import CANONICALIZATION_NAME


def train_historyfix(
    data: str | Path,
    checkpoint: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    training_games: int = 197_800,
) -> dict[str, float]:
    """Train from random initialization; resume is intentionally unsupported."""
    metrics = train_from_archive(
        data,
        checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        resume_checkpoint=None,
        input_canonicalization=CANONICALIZATION_NAME,
    )
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.update(
        {
            "value_schema": VALUE_SCHEMA_V1_HISTORYFIX,
            "history_semantics": (
                "evaluated_turn_only_one_card_zero_padded"
            ),
            "training_games": training_games,
            "training_data": str(data),
            "epochs": epochs,
            "fresh_initialization": True,
        }
    )
    torch.save(payload, checkpoint)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-games", type=int, default=197_800)
    args = parser.parse_args()
    metrics = train_historyfix(
        args.data,
        args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        training_games=args.training_games,
    )
    print(
        " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
    )


if __name__ == "__main__":
    main()
