import json

import numpy as np
import torch

from yellowstone.train_value import _game_id_sha256
from yellowstone.train_value_v2_lite_action import train_v2_lite_action
from yellowstone.value_learning import split_game_ids
from yellowstone.value_v2_lite_action import (
    CANONICALIZATION_V2_LITE_ACTION,
    HISTORY_SEMANTICS_V2_LITE_ACTION,
    VALUE_SCHEMA_V2_LITE_ACTION,
)


def test_train_v2_lite_action_uses_shared_game_split(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    generator = np.random.default_rng(23)
    game_ids = np.repeat(np.arange(10, dtype=np.int64), 2)
    records = len(game_ids)
    np.savez_compressed(
        data / "part_000000.npz",
        board=generator.normal(size=(records, 58, 7, 7)).astype(np.float32),
        context=generator.normal(size=(records, 150)).astype(np.float32),
        target=(game_ids % 4 == 0).astype(np.float32),
        game_id=game_ids,
        source_game_id=game_ids + 100,
        perspective=np.zeros(records, dtype=np.int8),
        play_count=np.where(np.arange(records) % 2, 1, 2).astype(np.int8),
    )
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "value_schema": VALUE_SCHEMA_V2_LITE_ACTION,
                "canonicalization": CANONICALIZATION_V2_LITE_ACTION,
                "history_semantics": HISTORY_SEMANTICS_V2_LITE_ACTION,
                "opponent_private_inputs": False,
                "games": 10,
                "records": records,
                "converted_files": 1,
                "rebased_game_id_min": 0,
                "rebased_game_id_max": 9,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "model.pt"
    result = train_v2_lite_action(
        data,
        checkpoint_path,
        game_count=10,
        batch_size=4,
        seed=7,
        progress_interval_parts=1,
    )
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    _, validation_ids, test_ids = split_game_ids(10, seed=7)
    assert result["validation_game_ids_sha256"] == _game_id_sha256(
        validation_ids
    )
    assert result["test_game_ids_sha256"] == _game_id_sha256(test_ids)
    assert checkpoint["context_size"] == 150
    assert checkpoint["opponent_private_inputs"] is False
    assert checkpoint["unordered_action_cards"] is True
