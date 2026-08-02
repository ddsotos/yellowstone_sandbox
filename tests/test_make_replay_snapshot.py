import json
from pathlib import Path
from random import Random

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action
from yellowstone.make_replay_snapshot import make_snapshot
from yellowstone.replay_v2 import RULES_VERSION_V2, ReplayGameV2, write_replay_shard
from yellowstone.game import create_initial_state
from yellowstone.types import Phase


def _game(game_id: int) -> ReplayGameV2:
    state = create_initial_state(4, seed=game_id)
    initial_state = state
    rng = Random(game_id + 100)
    bot = HeuristicBot()
    actions = []
    while state.phase != Phase.GAME_OVER:
        action = bot.choose_action(state)
        actions.append(action)
        state = apply_known_legal_action(state, action, rng=rng)
    return ReplayGameV2(
        game_id=game_id,
        initial_seed=game_id,
        gameplay_seed=game_id + 100,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=(),
        winners=state.winners,
        teacher_checkpoint="test",
        teacher_sha256="0" * 64,
        teacher_generation=0,
        privileged_teacher_deck=False,
        rules_version=RULES_VERSION_V2,
    )


def test_snapshot_uses_completed_shards_and_manifest_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "snapshot"
    source.mkdir()
    write_replay_shard((_game(10), _game(11)), source / "part_0000000.jsonl.gz")
    write_replay_shard((_game(12), _game(13)), source / "part_0000002.jsonl.gz")
    manifest_path = source / "collection_manifest.json"
    manifest_path.write_text(
        json.dumps({"games": 2, "completed_shards": 1, "status": "running"}),
        encoding="utf-8",
    )

    manifest = make_snapshot(
        [(source, None)],
        output,
        games=2,
        shard_games=2,
        completed_shards=1,
        source_manifest=manifest_path,
    )

    assert manifest["games"] == 2
    assert manifest["completed_shards"] == 1
    assert manifest["source_manifest"]["completed_shards"] == 1
    assert len(manifest["source_manifest"]["sha256"]) == 64
