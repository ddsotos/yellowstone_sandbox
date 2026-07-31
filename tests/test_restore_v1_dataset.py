import json
from pathlib import Path

import numpy as np

from yellowstone.restore_v1_dataset import restore_dataset


def test_restore_dataset_is_atomic_resumable_and_repairs_corruption(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    np.savez_compressed(
        source / "part_000000.npz",
        value=np.arange(10, dtype=np.int64),
    )
    manifest = {
        "games": 10,
        "value_schema": "yellowstone.value.v1",
        "history_semantics": "rolling_last_two_placements",
        "canonicalization": "fast_lr_ud_color_v1",
        "source_shards": 1,
    }
    (source / "conversion_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    arguments = {
        "expected_games": 10,
        "expected_schema": "yellowstone.value.v1",
        "expected_history_semantics": "rolling_last_two_placements",
        "expected_canonicalization": "fast_lr_ud_color_v1",
    }

    first = restore_dataset(source, destination, **arguments)
    second = restore_dataset(source, destination, **arguments)
    target = destination / "part_000000.npz"
    corrupted = bytearray(target.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    target.write_bytes(corrupted)
    repaired = restore_dataset(source, destination, **arguments)

    assert first["copied"] == 1
    assert second["skipped"] == 1
    assert repaired["copied"] == 1
    assert repaired["repaired"] == 1
    assert target.read_bytes() == (
        source / "part_000000.npz"
    ).read_bytes()
