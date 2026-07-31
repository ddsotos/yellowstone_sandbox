from random import Random

import numpy as np

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.incremental_tensor_watcher import watch_tensors
from yellowstone.replay_v2 import ReplayGameV2, write_replay_shard
from yellowstone.types import Phase


def _replay(game_id: int, seed: int) -> ReplayGameV2:
    state = create_initial_state(4, seed=seed)
    initial_state = state
    gameplay_seed = seed + 1000
    rng = Random(gameplay_seed)
    bot = HeuristicBot()
    actions = []
    while state.phase != Phase.GAME_OVER:
        action = bot.choose_action(state)
        assert action is not None
        actions.append(action)
        state = apply_known_legal_action(state, action, rng=rng)
    return ReplayGameV2(
        game_id=game_id,
        initial_seed=seed,
        gameplay_seed=gameplay_seed,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=(),
        winners=state.winners,
        teacher_checkpoint="test",
        teacher_sha256="0" * 64,
        teacher_generation=0,
        privileged_teacher_deck=False,
    )


def test_watcher_converts_only_new_completed_shards(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    v1 = tmp_path / "v1"
    preplay = tmp_path / "preplay"
    status = tmp_path / "status.json"
    source_manifest = source / "collection_manifest.json"
    stop = tmp_path / "stop"
    write_replay_shard(
        (_replay(100, 31),), source / "part_0000100.jsonl.gz"
    )

    first = watch_tensors(
        source,
        v1,
        preplay,
        game_id_rebase=100,
        source_manifest=source_manifest,
        status_file=status,
        stop_file=stop,
        poll_seconds=0.01,
        max_cycles=1,
    )

    assert first["v1_shards"] == first["preplay_shards"] == 1
    assert first["v1_games"] == first["preplay_games"] == 1
    with np.load(v1 / "part_0000100.npz") as archive:
        assert set(archive["game_id"]) == {0}
        assert set(archive["source_game_id"]) == {100}

    write_replay_shard(
        (_replay(101, 37),), source / "part_0000101.jsonl.gz"
    )
    second = watch_tensors(
        source,
        v1,
        preplay,
        game_id_rebase=100,
        source_manifest=source_manifest,
        status_file=status,
        stop_file=stop,
        poll_seconds=0.01,
        max_cycles=1,
    )
    third = watch_tensors(
        source,
        v1,
        preplay,
        game_id_rebase=100,
        source_manifest=source_manifest,
        status_file=status,
        stop_file=stop,
        poll_seconds=0.01,
        max_cycles=1,
    )

    assert second["v1_shards"] == second["preplay_shards"] == 2
    assert second["v1_games"] == second["preplay_games"] == 2
    assert third["v1_shards"] == third["preplay_shards"] == 2
    with np.load(v1 / "part_0000101.npz") as archive:
        assert set(archive["game_id"]) == {1}
        assert set(archive["source_game_id"]) == {101}
