import gzip
import json
from random import Random

from yellowstone.bots import HeuristicBot
from yellowstone.convert_replay_v2_lite_action import (
    convert_replay_shards_v2_lite_action,
)
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.replay_v2 import (
    RULES_VERSION_V2,
    ReplayGameV2,
    read_replay_shard,
    records_from_replay,
    replay_game,
    replay_game_to_dict,
    verify_replay_dict,
    write_replay_shard,
)
from yellowstone.replay_v2_lite import records_from_replay_v2_lite
from yellowstone.types import Phase
from yellowstone.value_v2 import PendingRefillSource, canonical_tensors_v2


def _heuristic_replay(game_id: int = 7) -> ReplayGameV2:
    initial_seed = 12345
    gameplay_seed = 54321
    state = create_initial_state(4, seed=initial_seed)
    initial_state = state
    rng = Random(gameplay_seed)
    actions = []
    bot = HeuristicBot()
    while state.phase != Phase.GAME_OVER:
        action = bot.choose_action(state)
        assert action is not None
        actions.append(action)
        state = apply_known_legal_action(state, action, rng=rng)
    return ReplayGameV2(
        game_id=game_id,
        initial_seed=initial_seed,
        gameplay_seed=gameplay_seed,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=(),
        winners=state.winners,
        teacher_checkpoint="models/test.pt",
        teacher_sha256="abc",
        teacher_generation=0,
    )


def test_replay_round_trip_and_v2_record_reconstruction(tmp_path) -> None:
    game = _heuristic_replay()
    assert replay_game(game).winners == game.winners
    data = replay_game_to_dict(game)
    assert data["rules_version"] == RULES_VERSION_V2
    verify_replay_dict(data)

    path = tmp_path / "part_000000.jsonl.gz"
    facts = write_replay_shard((game,), path)
    assert facts["games"] == 1
    assert facts["compressed_bytes"] < facts["uncompressed_bytes"]
    loaded = tuple(read_replay_shard(path))
    assert len(loaded) == 1
    assert loaded[0].rules_version == RULES_VERSION_V2
    assert replay_game(loaded[0]).winners == game.winners

    records = records_from_replay(loaded[0])
    assert records
    assert all(len(record.history_before_turn) <= 3 for record in records)
    assert all(record.candidate_frame.end_frame is not None for record in records)
    assert any(
        record.pending_refill_source != PendingRefillSource.NO_PENDING
        for record in records
    )
    for record in records[:10]:
        canonical_tensors_v2(record)


def test_replay_file_is_plain_gzip_jsonl(tmp_path) -> None:
    path = tmp_path / "part.jsonl.gz"
    write_replay_shard((_heuristic_replay(),), path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.loads(stream.readline())
    assert data["schema"] == "yellowstone.replay.v2"


def test_v2_lite_reconstruction_keeps_pre_play_state_and_labels() -> None:
    game = _heuristic_replay()
    original = records_from_replay(game)
    lite = records_from_replay_v2_lite(game)
    assert len(lite) == len(original)
    assert [record.target for record in lite] == [
        record.target for record in original
    ]
    assert all(len(record.history_before_turn) <= 2 for record in lite)
    assert all(record.state_before_turn.phase == Phase.PLAY for record in lite)
    assert all(
        record.state_before_turn.cards_played_this_turn == 0
        for record in lite
    )


def test_v2_lite_action_conversion_rebases_and_stores_cards(tmp_path) -> None:
    import numpy as np

    source = tmp_path / "source"
    source.mkdir()
    write_replay_shard(
        (_heuristic_replay(game_id=954346),),
        source / "part_954346.jsonl.gz",
    )
    output = tmp_path / "output"
    manifest = convert_replay_shards_v2_lite_action(
        source,
        output,
        game_id_rebase=954346,
        expected_games=1,
        expected_source_game_id_min=954346,
        expected_source_game_id_max=954346,
    )
    assert manifest["games"] == 1
    assert manifest["records"] == (
        manifest["one_card_records"] + manifest["two_card_records"]
    )
    with np.load(output / "part_954346.npz") as archive:
        assert archive["context"].shape[1] == 150
        assert set(archive["game_id"]) == {0}
        assert set(archive["source_game_id"]) == {954346}
        assert np.isin(archive["play_count"], (1, 2)).all()
