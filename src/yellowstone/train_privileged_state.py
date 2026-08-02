"""Train a four-output privileged state critic."""

from __future__ import annotations

import argparse
from pathlib import Path

from yellowstone.privileged_state import (
    CANONICALIZATION_PRIVILEGED_STATE,
    FEATURE_CONTRACT_PRIVILEGED_STATE,
    HISTORY_SEMANTICS_PRIVILEGED_STATE,
    PRIVILEGED_STATE_CONTEXT_SIZE,
    VALUE_SCHEMA_PRIVILEGED_STATE,
    build_privileged_state_net,
)
from yellowstone.train_value_v2 import (
    _archive_paths as _all_archive_paths,
    _split_buckets,
)


def _archive_paths(
    data_path: str | Path,
    *,
    start_part: int | None = None,
    end_part: int | None = None,
) -> tuple[Path, ...]:
    paths = _all_archive_paths(data_path)
    if (start_part is None) != (end_part is None):
        raise ValueError("start_part and end_part must be supplied together")
    if start_part is None:
        return paths
    if start_part < 0 or end_part is None or end_part < start_part:
        raise ValueError("invalid inclusive part range")
    selected = tuple(
        path
        for path in paths
        if start_part <= int(path.stem.removeprefix("part_")) <= end_part
    )
    if not selected:
        raise FileNotFoundError("no privileged-state shards in requested range")
    return selected


def train_privileged_state(
    data_path: str | Path,
    checkpoint_prefix: str | Path,
    *,
    epochs: int = 2,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 20260727,
    start_part: int | None = None,
    end_part: int | None = None,
) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    paths = _archive_paths(data_path, start_part=start_part, end_part=end_part)
    model = build_privileged_state_net()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    prefix = Path(checkpoint_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    checkpoints: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for path in paths:
            with np.load(path) as archive:
                mask = _split_buckets(archive["game_id"], seed) < 8
                if not mask.any():
                    continue
                dataset = TensorDataset(
                    torch.from_numpy(archive["board"][mask]),
                    torch.from_numpy(archive["context"][mask]),
                    torch.from_numpy(archive["target"][mask]),
                )
            for board, context, target in DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            ):
                optimizer.zero_grad()
                logits = model(board, context)
                loss = -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                loss.backward()
                optimizer.step()
        metrics = _metrics(model, paths, seed)
        destination = prefix.with_name(f"{prefix.name}_epoch{epoch:03d}.pt")
        payload = {
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "seed": seed,
            "epochs": epoch,
            "value_schema": VALUE_SCHEMA_PRIVILEGED_STATE,
            "input_canonicalization": CANONICALIZATION_PRIVILEGED_STATE,
            "history_semantics": HISTORY_SEMANTICS_PRIVILEGED_STATE,
            "feature_contract": FEATURE_CONTRACT_PRIVILEGED_STATE,
            "context_size": PRIVILEGED_STATE_CONTEXT_SIZE,
            "privileged_inputs": True,
            "deployable_policy_model": False,
        }
        torch.save(payload, destination)
        checkpoints.append({"checkpoint": str(destination), **metrics})
    selected = min(checkpoints, key=lambda item: item["validation_logloss"])
    return {"checkpoints": checkpoints, "selected": selected}


def _metrics(model, paths, seed: int) -> dict[str, float]:
    return {
        "validation_logloss": _metric(model, paths, seed, 8, "logloss"),
        "validation_brier": _metric(model, paths, seed, 8, "brier"),
        "test_logloss": _metric(model, paths, seed, 9, "logloss"),
        "test_brier": _metric(model, paths, seed, 9, "brier"),
    }


def _metric(model, paths, seed: int, bucket: int, kind: str) -> float:
    import numpy as np
    import torch

    total = count = 0
    model.eval()
    with torch.no_grad():
        for path in paths:
            with np.load(path) as archive:
                mask = _split_buckets(archive["game_id"], seed) == bucket
                if not mask.any():
                    continue
                target = archive["target"][mask]
                probabilities = torch.softmax(
                    model(
                        torch.from_numpy(archive["board"][mask]),
                        torch.from_numpy(archive["context"][mask]),
                    ),
                    dim=1,
                ).numpy()
            if kind == "logloss":
                total += float(
                    -np.sum(target * np.log(np.clip(probabilities, 1e-7, 1)))
                )
                count += len(target)
            else:
                total += float(np.sum((probabilities - target) ** 2))
                count += target.size
    if not count:
        raise ValueError(f"empty metric bucket {bucket}")
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-prefix", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--start-part", type=int)
    parser.add_argument("--end-part", type=int)
    args = parser.parse_args()
    import json

    result = train_privileged_state(
        args.data,
        args.checkpoint_prefix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        start_part=args.start_part,
        end_part=args.end_part,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.selection_output:
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
