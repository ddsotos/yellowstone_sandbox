from random import Random

import numpy as np

from yellowstone.bots import HeuristicBot
from yellowstone.convert_replay_v2_to_v1_historyfix import (
    convert_replay_shards as convert_historyfix,
    records_from_replay_v1_historyfix,
)
from yellowstone.convert_replay_v2_to_v1_original import (
    HISTORY_SEMANTICS_V1_ORIGINAL,
    VALUE_SCHEMA_V1_ORIGINAL,
    convert_replay_shards as convert_original,
    records_from_replay_v1_original,
)
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.replay_v2 import (
    LEGACY_RULES_VERSION_V2,
    RULES_VERSION_V2,
    ReplayGameV2,
    write_replay_shard,
)
from yellowstone.types import Phase
from yellowstone.value_learning import split_game_ids


def _heuristic_replay(
    game_id: int,
    *,
    initial_seed: int,
    gameplay_seed: int,
    rules_version: str = RULES_VERSION_V2,
) -> ReplayGameV2:
    state = create_initial_state(4, seed=initial_seed)
    initial_state = state
    rng = Random(gameplay_seed)
    bot = HeuristicBot()
    actions = []
    while state.phase != Phase.GAME_OVER:
        action = bot.choose_action(state)
        assert action is not None
        actions.append(action)
        state = apply_known_legal_action(
            state,
            action,
            rng=rng,
            settle_on_empty_deck=(
                rules_version != LEGACY_RULES_VERSION_V2
            ),
        )
    return ReplayGameV2(
        game_id=game_id,
        initial_seed=initial_seed,
        gameplay_seed=gameplay_seed,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=(),
        winners=state.winners,
        teacher_checkpoint="test",
        teacher_sha256="0" * 64,
        teacher_generation=0,
        privileged_teacher_deck=False,
        rules_version=rules_version,
    )


def test_original_history_matches_v1_rolling_semantics() -> None:
    # This seed starts with a one-card turn, so there is no prior placement.
    game = _heuristic_replay(
        0, initial_seed=23, gameplay_seed=1023
    )
    original = records_from_replay_v1_original(game)
    fixed = records_from_replay_v1_historyfix(game)

    assert VALUE_SCHEMA_V1_ORIGINAL == "yellowstone.value.v1"
    assert HISTORY_SEMANTICS_V1_ORIGINAL == "rolling_last_two_placements"
    assert len(original) == len(fixed)
    assert len(original[0].history) == len(fixed[0].history) == 1
    assert original[0].history == fixed[0].history
    assert any(len(record.history) == 1 for record in fixed)
    assert any(len(record.history) == 2 for record in fixed)

    saw_later_one_card = False
    for original_record, fixed_record in zip(
        original, fixed, strict=True
    ):
        assert original_record.game_id == fixed_record.game_id
        assert (
            original_record.perspective_player_index
            == fixed_record.perspective_player_index
        )
        assert original_record.state == fixed_record.state
        assert original_record.target == fixed_record.target
        if len(fixed_record.history) == 2:
            # A two-card turn replaces both rolling slots.
            assert original_record.history == fixed_record.history
        else:
            # A one-card turn keeps the preceding public placement in slot 1.
            assert original_record.history[-1:] == fixed_record.history
            if original_record is not original[0]:
                saw_later_one_card = True
                assert len(original_record.history) == 2
    assert saw_later_one_card


def test_original_replay_labels_and_legacy_rules() -> None:
    game = _heuristic_replay(
        4,
        initial_seed=31,
        gameplay_seed=41,
        rules_version=LEGACY_RULES_VERSION_V2,
    )
    records = records_from_replay_v1_original(game)

    assert records
    winner_count = len(game.winners)
    assert winner_count
    assert all(
        record.target
        == (
            1.0 / winner_count
            if record.perspective_player_index in game.winners
            else 0.0
        )
        for record in records
    )


def test_original_conversion_matches_historyfix_dataset_identity(
    tmp_path,
) -> None:
    games = (
        _heuristic_replay(0, initial_seed=23, gameplay_seed=1023),
        _heuristic_replay(
            1,
            initial_seed=31,
            gameplay_seed=41,
            rules_version=LEGACY_RULES_VERSION_V2,
        ),
    )
    source = tmp_path / "replays"
    source.mkdir()
    write_replay_shard(games, source / "part_000000.jsonl.gz")
    fixed_path = tmp_path / "fixed"
    original_path = tmp_path / "original"
    fixed_manifest = convert_historyfix(
        source, fixed_path, expected_games=2
    )
    original_manifest = convert_original(
        source,
        original_path,
        expected_games=2,
        reference=fixed_path,
    )

    assert original_manifest["games"] == fixed_manifest["games"] == 2
    assert (
        original_manifest["records"]
        == fixed_manifest["records"]
    )
    assert original_manifest["reference_audit"] == {
        "path": str(fixed_path),
        "records_match": True,
        "game_ids_match": True,
        "labels_match": True,
        "board_and_non_history_context_match": True,
        "split_basis_match": True,
    }
    with np.load(
        original_path / "part_000000.npz"
    ) as original, np.load(
        fixed_path / "part_000000.npz"
    ) as fixed:
        assert np.array_equal(original["game_id"], fixed["game_id"])
        assert np.array_equal(original["target"], fixed["target"])
        assert np.array_equal(original["board"], fixed["board"])
        assert np.array_equal(
            original["context"][:, :57], fixed["context"][:, :57]
        )
        assert np.isfinite(original["context"]).all()
        assert original["context"].dtype == fixed["context"].dtype

    # Both training paths call the same deterministic game-ID split.
    original_split = split_game_ids(10, seed=7)
    fixed_split = split_game_ids(10, seed=7)
    assert original_split == fixed_split


def test_original_conversion_rebases_contiguous_game_ids(tmp_path) -> None:
    games = (
        _heuristic_replay(17, initial_seed=23, gameplay_seed=1023),
        _heuristic_replay(18, initial_seed=31, gameplay_seed=41),
    )
    source = tmp_path / "replays"
    source.mkdir()
    write_replay_shard(games, source / "part_000017.jsonl.gz")

    manifest = convert_original(
        source,
        tmp_path / "original",
        expected_games=2,
        game_id_rebase=17,
        expected_source_game_id_min=17,
        expected_source_game_id_max=18,
    )

    assert manifest["source_game_id_min"] == 17
    assert manifest["source_game_id_max"] == 18
    assert manifest["game_id_rebase"] == 17
    assert manifest["rebased_game_id_min"] == 0
    assert manifest["rebased_game_id_max"] == 1
    with np.load(
        tmp_path / "original" / "part_000017.npz"
    ) as archive:
        assert set(archive["game_id"]) == {0, 1}
        assert set(archive["source_game_id"]) == {17, 18}
        assert np.array_equal(
            archive["game_id"], archive["source_game_id"] - 17
        )
        assert set(archive["perspective_player_index"]) <= {0, 1, 2, 3}
