"""Train V2-lite terminal value with explicit unordered action cards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from yellowstone.train_value import _game_id_sha256
from yellowstone.value_learning import split_game_ids
from yellowstone.value_v2_lite_action import (
    CANONICALIZATION_V2_LITE_ACTION,
    HISTORY_SEMANTICS_V2_LITE_ACTION,
    VALUE_CONTEXT_SIZE_V2_LITE_ACTION,
    VALUE_SCHEMA_V2_LITE_ACTION,
    build_win_value_net_v2_lite_action,
)


def train_v2_lite_action(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    game_count: int,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 20260727,
    progress_checkpoint: str | Path | None = None,
    progress_interval_parts: int = 50,
) -> dict:
    import numpy as np
    import torch

    if epochs != 1:
        raise ValueError("V2-lite-action currently supports exactly one epoch")
    if batch_size <= 0 or progress_interval_parts <= 0:
        raise ValueError("batch size and progress interval must be positive")
    root = Path(data_path)
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8-sig")
    )
    expected_manifest = {
        "status": "complete",
        "value_schema": VALUE_SCHEMA_V2_LITE_ACTION,
        "canonicalization": CANONICALIZATION_V2_LITE_ACTION,
        "history_semantics": HISTORY_SEMANTICS_V2_LITE_ACTION,
        "opponent_private_inputs": False,
        "games": game_count,
        "rebased_game_id_min": 0,
        "rebased_game_id_max": game_count - 1,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"V2-lite-action manifest differs at {key}")
    paths = tuple(sorted(root.glob("part_*.npz")))
    if len(paths) != int(manifest["converted_files"]):
        raise ValueError("V2-lite-action part count differs from manifest")
    train_ids, validation_ids, test_ids = split_game_ids(
        game_count, seed=seed
    )
    train_id_array = np.asarray(sorted(train_ids), dtype=np.int64)
    torch.manual_seed(seed)
    model = build_win_value_net_v2_lite_action()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    progress_path = (
        Path(progress_checkpoint)
        if progress_checkpoint is not None
        else Path(checkpoint_path).with_suffix(".progress.pt")
    )
    contract = {
        "training_data": str(root),
        "value_schema": VALUE_SCHEMA_V2_LITE_ACTION,
        "canonicalization": CANONICALIZATION_V2_LITE_ACTION,
        "history_semantics": HISTORY_SEMANTICS_V2_LITE_ACTION,
        "game_count": game_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "part_count": len(paths),
    }
    start_part = 0
    start_offset = 0
    processed = 0
    if progress_path.is_file():
        progress = torch.load(
            progress_path, map_location="cpu", weights_only=False
        )
        if progress.get("contract") != contract:
            raise ValueError("V2-lite-action progress contract differs")
        model.load_state_dict(progress["state_dict"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        start_part = int(progress["cursor"]["part_index"])
        start_offset = int(progress["cursor"]["record_offset"])
        processed = int(progress["processed_train_records"])
    for part_index in range(start_part, len(paths)):
        with np.load(paths[part_index]) as archive:
            mask = np.isin(archive["game_id"], train_id_array)
            board = torch.from_numpy(archive["board"][mask])
            context = torch.from_numpy(archive["context"][mask])
            target = torch.from_numpy(archive["target"][mask])
        generator = torch.Generator().manual_seed(
            (seed + part_index * 1_000_003) & 0x7FFF_FFFF_FFFF_FFFF
        )
        order = torch.randperm(len(target), generator=generator)
        offset = start_offset if part_index == start_part else 0
        if not 0 <= offset <= len(order):
            raise ValueError("V2-lite-action resume offset is invalid")
        model.train()
        while offset < len(order):
            next_offset = min(offset + batch_size, len(order))
            selected = order[offset:next_offset]
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(board[selected], context[selected]), target[selected]
            )
            loss.backward()
            optimizer.step()
            processed += next_offset - offset
            offset = next_offset
        start_offset = 0
        if (
            (part_index + 1) % progress_interval_parts == 0
            or part_index + 1 == len(paths)
        ):
            _atomic_torch_save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "contract": contract,
                    "processed_train_records": processed,
                    "cursor": {
                        "part_index": part_index + 1,
                        "record_offset": 0,
                    },
                },
                progress_path,
            )
    metrics = {}
    metrics.update(
        _metrics(model, paths, validation_ids, prefix="validation")
    )
    metrics.update(_metrics(model, paths, test_ids, prefix="test"))
    checkpoint = {
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_data": str(root),
        "training_games": game_count,
        "training_records": int(manifest["records"]),
        "processed_train_records": processed,
        "train_split_games": len(train_ids),
        "validation_split_games": len(validation_ids),
        "test_split_games": len(test_ids),
        "train_game_ids_sha256": _game_id_sha256(train_ids),
        "validation_game_ids_sha256": _game_id_sha256(validation_ids),
        "test_game_ids_sha256": _game_id_sha256(test_ids),
        "split_policy": "shared_population_random_game_id_80_10_10",
        "value_schema": VALUE_SCHEMA_V2_LITE_ACTION,
        "input_canonicalization": CANONICALIZATION_V2_LITE_ACTION,
        "history_semantics": HISTORY_SEMANTICS_V2_LITE_ACTION,
        "context_size": VALUE_CONTEXT_SIZE_V2_LITE_ACTION,
        "board_channels": 58,
        "opponent_private_inputs": False,
        "unordered_action_cards": True,
    }
    _atomic_torch_save(checkpoint, Path(checkpoint_path))
    return {
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "processed_train_records": processed,
        "train_split_games": len(train_ids),
        "validation_split_games": len(validation_ids),
        "test_split_games": len(test_ids),
        "validation_game_ids_sha256": checkpoint[
            "validation_game_ids_sha256"
        ],
        "test_game_ids_sha256": checkpoint["test_game_ids_sha256"],
    }


def _metrics(model, paths, ids: set[int], *, prefix: str) -> dict:
    import numpy as np
    import torch

    id_array = np.asarray(sorted(ids), dtype=np.int64)
    totals = {
        "all": [0.0, 0.0, 0],
        "one_card": [0.0, 0.0, 0],
        "two_card": [0.0, 0.0, 0],
    }
    model.eval()
    with torch.no_grad():
        for path in paths:
            with np.load(path) as archive:
                mask = np.isin(archive["game_id"], id_array)
                if not mask.any():
                    continue
                target = archive["target"][mask]
                play_count = archive["play_count"][mask]
                probability = torch.sigmoid(
                    model(
                        torch.from_numpy(archive["board"][mask]),
                        torch.from_numpy(archive["context"][mask]),
                    )
                ).numpy()
            clipped = np.clip(probability, 1e-7, 1 - 1e-7)
            for label, selected in (
                ("all", np.ones(len(target), dtype=bool)),
                ("one_card", play_count == 1),
                ("two_card", play_count == 2),
            ):
                if not selected.any():
                    continue
                y = target[selected]
                p = probability[selected]
                c = clipped[selected]
                totals[label][0] += float(np.sum((p - y) ** 2))
                totals[label][1] += float(
                    -np.sum(y * np.log(c) + (1 - y) * np.log(1 - c))
                )
                totals[label][2] += int(np.count_nonzero(selected))
    result = {}
    for label, (squared, log_loss, count) in totals.items():
        result[f"{prefix}_{label}_records"] = count
        result[f"{prefix}_{label}_brier"] = (
            squared / count if count else None
        )
        result[f"{prefix}_{label}_log_loss"] = (
            log_loss / count if count else None
        )
    return result


def _atomic_torch_save(payload: object, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--game-count", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--progress-checkpoint", type=Path)
    parser.add_argument("--progress-interval-parts", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = train_v2_lite_action(
        args.data,
        args.checkpoint,
        game_count=args.game_count,
        epochs=args.epochs,
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


if __name__ == "__main__":
    main()
