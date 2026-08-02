"""Train one continuous V1 epoch and retain progress milestones."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from yellowstone.cnn import (
    DEFAULT_WIN_VALUE_HIDDEN_CHANNELS,
    DEFAULT_WIN_VALUE_HIDDEN_SIZE,
    build_win_value_net,
    win_value_architecture_from_checkpoint,
    win_value_architecture_metadata,
)
from yellowstone.train_value import (
    _archive_paths,
    _game_id_sha256,
    _probabilities,
)
from yellowstone.value_learning import VALUE_CONTEXT_SIZE, split_game_ids


def train_value_milestones(
    data_path: str | Path,
    checkpoint_prefix: str | Path,
    *,
    split_game_count: int,
    start_part: int,
    end_part: int,
    milestones: tuple[int, ...] = (10, 30, 50, 100),
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 20260727,
    progress_checkpoint: str | Path | None = None,
    progress_interval_parts: int = 25,
    value_schema: str = "yellowstone.value.v1",
    history_semantics: str = "rolling_last_two_placements",
    input_canonicalization: str = "fast_lr_ud_color_v1",
    context_size: int | None = None,
    initial_checkpoint: str | Path | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Train one fixed V1 snapshot and save models along the same epoch."""
    import numpy as np
    import torch

    _validate_arguments(
        milestones=milestones,
        batch_size=batch_size,
        learning_rate=learning_rate,
        split_game_count=split_game_count,
        progress_interval_parts=progress_interval_parts,
    )
    paths = _archive_paths(
        data_path, start_part=start_part, end_part=end_part
    )
    with np.load(paths[0]) as sample:
        board_shape = tuple(int(value) for value in sample["board"].shape[1:])
        observed_context_size = int(sample["context"].shape[1])
    if len(board_shape) != 3:
        raise ValueError(f"unsupported board tensor shape: {board_shape}")
    if context_size is None:
        context_size = observed_context_size
    if context_size != observed_context_size:
        raise ValueError(
            "context size must match archive: "
            f"{context_size} != {observed_context_size}"
        )
    observed_game_count = (
        max(int(np.load(path)["game_id"].max()) for path in paths) + 1
    )
    if observed_game_count != split_game_count:
        raise ValueError(
            "fixed snapshot game count differs: "
            f"{observed_game_count} != {split_game_count}"
        )
    train_ids, validation_ids, test_ids = split_game_ids(
        split_game_count, seed=seed
    )
    total_train = _count_records(paths, train_ids)
    if not total_train:
        raise ValueError("fixed V1 snapshot has no training records")

    prefix = Path(checkpoint_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    progress_path = (
        Path(progress_checkpoint)
        if progress_checkpoint is not None
        else Path(f"{prefix}.progress.pt")
    )
    targets = {
        percent: math.ceil(total_train * percent / 100)
        for percent in milestones
    }
    architecture = win_value_architecture_metadata(
        convolution_layers=2,
        hidden_channels=DEFAULT_WIN_VALUE_HIDDEN_CHANNELS,
        hidden_size=DEFAULT_WIN_VALUE_HIDDEN_SIZE,
        board_channels=board_shape[0],
        board_height=board_shape[1],
        board_width=board_shape[2],
    )
    metadata = _metadata_for_canonicalization(input_canonicalization)
    if checkpoint_metadata:
        metadata.update(checkpoint_metadata)
    contract = {
        "training_data": str(data_path),
        "snapshot_parts": [path.name for path in paths],
        "start_part": start_part,
        "end_part": end_part,
        "split_game_count": split_game_count,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "milestone_percentages": milestones,
        "total_train_records": total_train,
        "value_schema": value_schema,
        "history_semantics": history_semantics,
        "input_canonicalization": input_canonicalization,
        "context_size": context_size,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        **architecture,
        **metadata,
    }

    torch.manual_seed(seed)
    model = build_win_value_net(
        context_size=context_size,
        convolution_layers=2,
        hidden_channels=DEFAULT_WIN_VALUE_HIDDEN_CHANNELS,
        hidden_size=DEFAULT_WIN_VALUE_HIDDEN_SIZE,
        board_channels=board_shape[0],
        board_height=board_shape[1],
        board_width=board_shape[2],
    )
    if initial_checkpoint is not None:
        checkpoint = torch.load(
            initial_checkpoint, map_location="cpu", weights_only=False
        )
        if win_value_architecture_from_checkpoint(checkpoint) != architecture:
            raise ValueError("initial checkpoint architecture differs")
        mismatches = {
            key: {"expected": value, "actual": checkpoint.get(key)}
            for key, value in {
                "value_schema": value_schema,
                "history_semantics": history_semantics,
                "input_canonicalization": input_canonicalization,
            }.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(f"initial checkpoint contract differs: {mismatches}")
        model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    processed = 0
    start_path_index = 0
    start_offset = 0
    if progress_path.is_file():
        progress = torch.load(
            progress_path, map_location="cpu", weights_only=False
        )
        if progress.get("progress_contract") != contract:
            raise ValueError("V1 milestone training progress contract differs")
        model.load_state_dict(progress["state_dict"])
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        processed = int(progress["processed_train_records"])
        start_path_index = int(progress["cursor"]["part_index"])
        start_offset = int(progress["cursor"]["record_offset"])

    train_id_array = np.asarray(sorted(train_ids), dtype=np.int64)
    for path_index in range(start_path_index, len(paths)):
        with np.load(paths[path_index]) as archive:
            mask = np.isin(archive["game_id"], train_id_array)
            board = torch.from_numpy(archive["board"][mask])
            context = torch.from_numpy(archive["context"][mask])
            target = torch.from_numpy(archive["target"][mask])
        generator = torch.Generator().manual_seed(
            (seed + path_index * 1_000_003)
            & 0x7FFF_FFFF_FFFF_FFFF
        )
        order = torch.randperm(len(target), generator=generator)
        offset = start_offset if path_index == start_path_index else 0
        if offset < 0 or offset > len(order):
            raise ValueError("resume record offset is outside its part")
        model.train()
        while offset < len(order):
            next_offset = min(offset + batch_size, len(order))
            selected = order[offset:next_offset]
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(board[selected], context[selected]),
                target[selected],
            )
            loss.backward()
            optimizer.step()
            processed += next_offset - offset
            offset = next_offset
            for percent, threshold in targets.items():
                milestone_path = _milestone_path(prefix, percent)
                if processed >= threshold and not milestone_path.is_file():
                    payload = _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        contract=contract,
                        processed=processed,
                        target_percent=percent,
                        target_records=threshold,
                        cursor_part=path_index,
                        cursor_offset=offset,
                        train_ids=train_ids,
                        validation_ids=validation_ids,
                        test_ids=test_ids,
                    )
                    _atomic_torch_save(payload, milestone_path)
                    _atomic_torch_save(payload, progress_path)
        start_offset = 0
        if (
            (path_index + 1) % progress_interval_parts == 0
            or path_index + 1 == len(paths)
        ):
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                contract=contract,
                processed=processed,
                target_percent=None,
                target_records=None,
                cursor_part=path_index + 1,
                cursor_offset=0,
                train_ids=train_ids,
                validation_ids=validation_ids,
                test_ids=test_ids,
            )
            _atomic_torch_save(payload, progress_path)

    if processed != total_train:
        raise AssertionError(
            f"processed {processed} training records, expected {total_train}"
        )
    results = []
    for percent in milestones:
        milestone_path = _milestone_path(prefix, percent)
        checkpoint = torch.load(
            milestone_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        metrics = _metrics(model, paths, validation_ids, test_ids)
        checkpoint["metrics"] = metrics
        _atomic_torch_save(checkpoint, milestone_path)
        results.append(
            {
                "percent": percent,
                "checkpoint": str(milestone_path),
                "processed_train_records": checkpoint[
                    "processed_train_records"
                ],
                "actual_fraction": checkpoint["actual_fraction"],
                "metrics": metrics,
            }
        )
    return {
        "training_data": str(data_path),
        "start_part": start_part,
        "end_part": end_part,
        "split_game_count": split_game_count,
        "snapshot_shards": len(paths),
        "total_train_records": total_train,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": 1,
        "milestones": results,
    }


def _validate_arguments(
    *,
    milestones: tuple[int, ...],
    batch_size: int,
    learning_rate: float,
    split_game_count: int,
    progress_interval_parts: int,
) -> None:
    if (
        not milestones
        or tuple(sorted(set(milestones))) != milestones
        or milestones[-1] != 100
        or milestones[0] <= 0
        or milestones[-1] > 100
    ):
        raise ValueError(
            "milestones must be unique, ascending, and end at 100"
        )
    if (
        batch_size <= 0
        or learning_rate <= 0
        or split_game_count < 10
        or progress_interval_parts <= 0
    ):
        raise ValueError("training arguments must be positive")


def _metadata_for_canonicalization(canonicalization: str) -> dict[str, object]:
    from yellowstone.value_board_centered import (
        BOARD_CENTERED_V1_CANONICALIZATIONS,
        board_centered_metadata,
    )
    from yellowstone.value_board_columns import (
        CANONICALIZATION_BOARD_COLUMNS_V1,
        board_columns_metadata,
    )
    from yellowstone.value_board_columns_v2 import (
        CANONICALIZATION_BOARD_COLUMNS_V2,
        CANONICALIZATION_PREPLAY_BOARD_COLUMNS,
        board_columns_v2_metadata,
    )

    if canonicalization == CANONICALIZATION_BOARD_COLUMNS_V1:
        return board_columns_metadata()
    if canonicalization == CANONICALIZATION_BOARD_COLUMNS_V2:
        return board_columns_v2_metadata(preplay=False)
    if canonicalization == CANONICALIZATION_PREPLAY_BOARD_COLUMNS:
        return board_columns_v2_metadata(preplay=True)
    if canonicalization in BOARD_CENTERED_V1_CANONICALIZATIONS:
        return board_centered_metadata(canonicalization)
    return {}


def _count_records(paths: tuple[Path, ...], ids: set[int]) -> int:
    import numpy as np

    selected_ids = np.asarray(sorted(ids), dtype=np.int64)
    total = 0
    for path in paths:
        with np.load(path) as archive:
            total += int(
                np.count_nonzero(np.isin(archive["game_id"], selected_ids))
            )
    return total


def _checkpoint_payload(
    *,
    model,
    optimizer,
    contract: dict[str, object],
    processed: int,
    target_percent: int | None,
    target_records: int | None,
    cursor_part: int,
    cursor_offset: int,
    train_ids: set[int],
    validation_ids: set[int],
    test_ids: set[int],
) -> dict[str, object]:
    return {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": None,
        "epochs": 1,
        "fresh_initialization": True,
        "context_size": contract["context_size"],
        "split_policy": "shared_population_80_10_10",
        "train_split_games": len(train_ids),
        "full_train_split_games": len(train_ids),
        "validation_split_games": len(validation_ids),
        "test_split_games": len(test_ids),
        "train_game_ids_sha256": _game_id_sha256(train_ids),
        "full_train_game_ids_sha256": _game_id_sha256(train_ids),
        "validation_game_ids_sha256": _game_id_sha256(validation_ids),
        "test_game_ids_sha256": _game_id_sha256(test_ids),
        "training_games": contract["split_game_count"],
        "processed_train_records": processed,
        "target_percent": target_percent,
        "target_records": target_records,
        "actual_fraction": processed / int(
            contract["total_train_records"]
        ),
        "cursor": {
            "part_index": cursor_part,
            "record_offset": cursor_offset,
        },
        "progress_contract": contract,
        **contract,
    }


def _metrics(
    model,
    paths: tuple[Path, ...],
    validation_ids: set[int],
    test_ids: set[int],
) -> dict[str, float]:
    import numpy as np

    totals = {
        "validation": [0.0, 0.0, 0],
        "test": [0.0, 0.0, 0],
    }
    id_arrays = {
        "validation": np.asarray(sorted(validation_ids), dtype=np.int64),
        "test": np.asarray(sorted(test_ids), dtype=np.int64),
    }
    for path in paths:
        with np.load(path) as archive:
            for name, ids in id_arrays.items():
                mask = np.isin(archive["game_id"], ids)
                if not mask.any():
                    continue
                target = archive["target"][mask]
                probability = np.clip(
                    _probabilities(
                        model,
                        archive["board"][mask],
                        archive["context"][mask],
                    ),
                    1e-7,
                    1 - 1e-7,
                )
                totals[name][0] += float(
                    np.sum((probability - target) ** 2)
                )
                totals[name][1] += float(
                    -np.sum(
                        target * np.log(probability)
                        + (1 - target) * np.log(1 - probability)
                    )
                )
                totals[name][2] += int(mask.sum())
    result = {}
    for name, (brier, log_loss, count) in totals.items():
        if not count:
            raise ValueError(f"{name} split has no records")
        result[f"{name}_brier"] = brier / count
        result[f"{name}_log_loss"] = log_loss / count
    return result


def _milestone_path(prefix: Path, percent: int) -> Path:
    return Path(f"{prefix}_pct{percent:03d}.pt")


def _atomic_torch_save(payload: object, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-prefix", type=Path, required=True)
    parser.add_argument("--split-game-count", type=int, required=True)
    parser.add_argument("--start-part", type=int, required=True)
    parser.add_argument("--end-part", type=int, required=True)
    parser.add_argument("--milestones", default="10,30,50,100")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--progress-checkpoint", type=Path)
    parser.add_argument("--progress-interval-parts", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-schema", default="yellowstone.value.v1")
    parser.add_argument(
        "--history-semantics", default="rolling_last_two_placements"
    )
    parser.add_argument(
        "--input-canonicalization", default="fast_lr_ud_color_v1"
    )
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--initial-checkpoint", type=Path)
    args = parser.parse_args()
    milestones = tuple(
        int(value) for value in args.milestones.split(",")
    )
    result = train_value_milestones(
        args.data,
        args.checkpoint_prefix,
        split_game_count=args.split_game_count,
        start_part=args.start_part,
        end_part=args.end_part,
        milestones=milestones,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        progress_checkpoint=args.progress_checkpoint,
        progress_interval_parts=args.progress_interval_parts,
        value_schema=args.value_schema,
        history_semantics=args.history_semantics,
        input_canonicalization=args.input_canonicalization,
        context_size=args.context_size,
        initial_checkpoint=args.initial_checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json

    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
