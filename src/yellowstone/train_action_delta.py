"""Train the public action-delta model."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from yellowstone.action_delta import (
    ACTION_DELTA_CONTEXT_SIZE,
    CANONICALIZATION_ACTION_DELTA,
    HISTORY_SEMANTICS_ACTION_DELTA,
    VALUE_SCHEMA_ACTION_DELTA,
    build_action_delta_net,
)
from yellowstone.action_delta_snapshot import verified_snapshot_paths
from yellowstone.train_value_v2 import _archive_paths, _split_buckets


def train_action_delta(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 20260727,
) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    paths = _archive_paths(data_path)
    model = build_action_delta_net()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.SmoothL1Loss()
    for _ in range(epochs):
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
                loss = loss_fn(model(board, context), target)
                loss.backward()
                optimizer.step()
    metrics = {
        "validation_mae": _metric(model, paths, seed, 8, squared=False),
        "validation_rmse": _metric(model, paths, seed, 8, squared=True),
        "test_mae": _metric(model, paths, seed, 9, squared=False),
        "test_rmse": _metric(model, paths, seed, 9, squared=True),
    }
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "seed": seed,
            "epochs": epochs,
            "value_schema": VALUE_SCHEMA_ACTION_DELTA,
            "input_canonicalization": CANONICALIZATION_ACTION_DELTA,
            "history_semantics": HISTORY_SEMANTICS_ACTION_DELTA,
            "context_size": ACTION_DELTA_CONTEXT_SIZE,
            "opponent_private_inputs": False,
        },
        destination,
    )
    return metrics


def train_action_delta_milestones(
    snapshot_path: str | Path,
    checkpoint_prefix: str | Path,
    *,
    milestones: tuple[int, ...] = (10, 30, 50, 100),
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 20260727,
    progress_checkpoint: str | Path | None = None,
    progress_interval_parts: int = 100,
) -> dict:
    """Train one continuous epoch and retain models at stream milestones."""
    import numpy as np
    import torch

    if (
        not milestones
        or tuple(sorted(set(milestones))) != milestones
        or milestones[-1] != 100
        or milestones[0] <= 0
        or milestones[-1] > 100
    ):
        raise ValueError("milestones must be unique, ascending, and end at 100")
    if batch_size <= 0 or progress_interval_parts <= 0:
        raise ValueError("batch size and progress interval must be positive")
    snapshot, paths = verified_snapshot_paths(snapshot_path)
    total_train = _count_train_records(paths, seed)
    if not total_train:
        raise ValueError("action-delta snapshot has no training records")
    prefix = Path(checkpoint_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    progress_path = (
        Path(progress_checkpoint)
        if progress_checkpoint is not None
        else Path(f"{prefix}.progress.pt")
    )
    milestone_targets = {
        percent: math.ceil(total_train * percent / 100)
        for percent in milestones
    }
    torch.manual_seed(seed)
    model = build_action_delta_net()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    processed = 0
    start_part = 0
    start_offset = 0
    if progress_path.is_file():
        progress = torch.load(
            progress_path, map_location="cpu", weights_only=False
        )
        _validate_training_metadata(
            progress,
            snapshot_sha256=snapshot["snapshot_sha256"],
            seed=seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            milestones=milestones,
            total_train=total_train,
        )
        model.load_state_dict(progress["state_dict"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        processed = int(progress["processed_train_records"])
        start_part = int(progress["cursor"]["part_index"])
        start_offset = int(progress["cursor"]["record_offset"])
    for part_index in range(start_part, len(paths)):
        path = paths[part_index]
        with np.load(path) as archive:
            mask = _split_buckets(archive["game_id"], seed) < 8
            board = torch.from_numpy(archive["board"][mask])
            context = torch.from_numpy(archive["context"][mask])
            target = torch.from_numpy(archive["target"][mask])
        generator = torch.Generator().manual_seed(
            (seed + part_index * 1_000_003) & 0x7FFF_FFFF_FFFF_FFFF
        )
        order = torch.randperm(len(target), generator=generator)
        offset = start_offset if part_index == start_part else 0
        if offset < 0 or offset > len(order):
            raise ValueError("resume record offset is outside its part")
        model.train()
        while offset < len(order):
            next_offset = min(offset + batch_size, len(order))
            selected = order[offset:next_offset]
            optimizer.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(
                model(board[selected], context[selected]), target[selected]
            )
            loss.backward()
            optimizer.step()
            processed += next_offset - offset
            offset = next_offset
            for percent, threshold in milestone_targets.items():
                path_at_milestone = _milestone_path(prefix, percent)
                if processed >= threshold and not path_at_milestone.is_file():
                    payload = _training_payload(
                        model=model,
                        optimizer=optimizer,
                        snapshot=snapshot,
                        seed=seed,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        milestones=milestones,
                        total_train=total_train,
                        processed=processed,
                        target_percent=percent,
                        target_records=threshold,
                        cursor_part=part_index,
                        cursor_offset=offset,
                    )
                    _atomic_torch_save(payload, path_at_milestone)
                    _atomic_torch_save(payload, progress_path)
        start_offset = 0
        if (
            (part_index + 1) % progress_interval_parts == 0
            or part_index + 1 == len(paths)
        ):
            payload = _training_payload(
                model=model,
                optimizer=optimizer,
                snapshot=snapshot,
                seed=seed,
                batch_size=batch_size,
                learning_rate=learning_rate,
                milestones=milestones,
                total_train=total_train,
                processed=processed,
                target_percent=None,
                target_records=None,
                cursor_part=part_index + 1,
                cursor_offset=0,
            )
            _atomic_torch_save(payload, progress_path)
    if processed != total_train:
        raise AssertionError(
            f"processed {processed} training records, expected {total_train}"
        )
    results = []
    for percent in milestones:
        path_at_milestone = _milestone_path(prefix, percent)
        checkpoint = torch.load(
            path_at_milestone, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        metrics = _split_metrics(model, paths, seed)
        checkpoint["metrics"] = metrics
        _atomic_torch_save(checkpoint, path_at_milestone)
        results.append(
            {
                "percent": percent,
                "checkpoint": str(path_at_milestone),
                "processed_train_records": checkpoint[
                    "processed_train_records"
                ],
                "actual_fraction": checkpoint["actual_fraction"],
                "metrics": metrics,
            }
        )
    return {
        "snapshot": str(snapshot_path),
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "total_records": int(snapshot["records"]),
        "total_train_records": total_train,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": 1,
        "milestones": results,
    }


def _count_train_records(paths: tuple[Path, ...], seed: int) -> int:
    import numpy as np

    total = 0
    for path in paths:
        with np.load(path) as archive:
            total += int(
                np.count_nonzero(_split_buckets(archive["game_id"], seed) < 8)
            )
    return total


def _training_payload(
    *,
    model,
    optimizer,
    snapshot: dict,
    seed: int,
    batch_size: int,
    learning_rate: float,
    milestones: tuple[int, ...],
    total_train: int,
    processed: int,
    target_percent: int | None,
    target_records: int | None,
    cursor_part: int,
    cursor_offset: int,
) -> dict:
    return {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": None,
        "seed": seed,
        "epochs": 1,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "value_schema": VALUE_SCHEMA_ACTION_DELTA,
        "input_canonicalization": CANONICALIZATION_ACTION_DELTA,
        "history_semantics": HISTORY_SEMANTICS_ACTION_DELTA,
        "context_size": ACTION_DELTA_CONTEXT_SIZE,
        "opponent_private_inputs": False,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "milestone_percentages": milestones,
        "total_train_records": total_train,
        "processed_train_records": processed,
        "target_percent": target_percent,
        "target_records": target_records,
        "actual_fraction": processed / total_train,
        "cursor": {
            "part_index": cursor_part,
            "record_offset": cursor_offset,
        },
    }


def _validate_training_metadata(
    checkpoint: dict,
    *,
    snapshot_sha256: str,
    seed: int,
    batch_size: int,
    learning_rate: float,
    milestones: tuple[int, ...],
    total_train: int,
) -> None:
    expected = {
        "value_schema": VALUE_SCHEMA_ACTION_DELTA,
        "input_canonicalization": CANONICALIZATION_ACTION_DELTA,
        "history_semantics": HISTORY_SEMANTICS_ACTION_DELTA,
        "opponent_private_inputs": False,
        "snapshot_sha256": snapshot_sha256,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "milestone_percentages": milestones,
        "total_train_records": total_train,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"action-delta resume checkpoint differs at {key}")


def _milestone_path(prefix: Path, percent: int) -> Path:
    return Path(f"{prefix}_pct{percent:03d}.pt")


def _atomic_torch_save(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _split_metrics(model, paths: tuple[Path, ...], seed: int) -> dict:
    result = {}
    for name, bucket in (("validation", 8), ("test", 9)):
        result.update(_detailed_metric(model, paths, seed, bucket, name))
    return result


def _detailed_metric(
    model, paths: tuple[Path, ...], seed: int, bucket: int, prefix: str
) -> dict:
    import numpy as np
    import torch

    totals = {
        "all": [0.0, 0.0, 0.0, 0],
        "one_card": [0.0, 0.0, 0.0, 0],
        "two_card": [0.0, 0.0, 0.0, 0],
    }
    model.eval()
    with torch.no_grad():
        for path in paths:
            with np.load(path) as archive:
                mask = _split_buckets(archive["game_id"], seed) == bucket
                if not mask.any():
                    continue
                target = archive["target"][mask]
                play_count = archive["play_count"][mask]
                prediction = model(
                    torch.from_numpy(archive["board"][mask]),
                    torch.from_numpy(archive["context"][mask]),
                ).numpy()
            difference = prediction - target
            for label, selected in (
                ("all", np.ones(len(target), dtype=bool)),
                ("one_card", play_count == 1),
                ("two_card", play_count == 2),
            ):
                if not selected.any():
                    continue
                values = difference[selected]
                totals[label][0] += float(np.sum(np.abs(values)))
                totals[label][1] += float(np.sum(values**2))
                totals[label][2] += float(np.sum(values))
                totals[label][3] += int(np.count_nonzero(selected))
    metrics = {}
    for label, (absolute, squared, signed, count) in totals.items():
        metrics[f"{prefix}_{label}_records"] = count
        metrics[f"{prefix}_{label}_mae"] = absolute / count if count else None
        metrics[f"{prefix}_{label}_rmse"] = (
            (squared / count) ** 0.5 if count else None
        )
        metrics[f"{prefix}_{label}_mean_error"] = (
            signed / count if count else None
        )
    return metrics


def _metric(
    model, paths, seed: int, bucket: int, *, squared: bool
) -> float | None:
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
                prediction = model(
                    torch.from_numpy(archive["board"][mask]),
                    torch.from_numpy(archive["context"][mask]),
                ).numpy()
            difference = prediction - target
            total += float(
                np.sum(difference**2 if squared else np.abs(difference))
            )
            count += len(target)
    if not count:
        return None
    value = total / count
    return value**0.5 if squared else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--checkpoint-prefix", type=Path)
    parser.add_argument("--milestones", default="10,30,50,100")
    parser.add_argument("--progress-checkpoint", type=Path)
    parser.add_argument("--progress-interval-parts", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    if args.snapshot is not None:
        if args.checkpoint_prefix is None:
            parser.error("--checkpoint-prefix is required with --snapshot")
        if args.epochs != 1:
            parser.error("milestone training supports exactly one epoch")
        milestones = tuple(
            int(value.strip())
            for value in args.milestones.split(",")
            if value.strip()
        )
        result = train_action_delta_milestones(
            args.snapshot,
            args.checkpoint_prefix,
            milestones=milestones,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            progress_checkpoint=args.progress_checkpoint,
            progress_interval_parts=args.progress_interval_parts,
        )
        payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return
    if args.data is None or args.checkpoint is None:
        parser.error("--data and --checkpoint are required for standard training")
    print(
        json.dumps(
            train_action_delta(
                args.data,
                args.checkpoint,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
