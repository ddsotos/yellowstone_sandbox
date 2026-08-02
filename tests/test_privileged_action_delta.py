from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from yellowstone.action_delta import (
    ACTION_DELTA_CONTEXT_SIZE,
    ActionDeltaRecord,
    build_action_delta_net,
    encode_action_delta,
    enumerate_action_delta_candidates,
    make_transition_record,
    propose_top_card_groups,
)
from yellowstone.action_delta_snapshot import (
    build_action_delta_snapshot,
    verified_snapshot_paths,
)
from yellowstone.evaluate_action_delta import combined_post_play_probability
from yellowstone.game import create_initial_state
from yellowstone.privileged_state import (
    PRIVILEGED_STATE_CONTEXT_SIZE,
    PrivilegedStateRecord,
    build_privileged_state_net,
    encode_privileged_state,
)
from yellowstone.value_policy import enumerate_turn_end_candidates
from yellowstone.train_action_delta import train_action_delta_milestones
from yellowstone.train_value_v2 import _split_buckets


class _IncreasingEstimator:
    def estimate_many(self, records):
        return tuple(index / 100 for index, _ in enumerate(records))


def _privileged_record(state):
    return PrivilegedStateRecord(
        game_id=1,
        state=state,
        history=(),
        target=(0.25, 0.25, 0.25, 0.25),
    )


def test_recorded_safe_one_counts_are_used_without_reenumeration() -> None:
    state = create_initial_state(4, seed=7)
    record = PrivilegedStateRecord(
        game_id=1,
        state=state,
        history=(),
        target=(0.25, 0.25, 0.25, 0.25),
        safe_one_card_counts=(0, 1, 2, 3),
    )
    _, context = encode_privileged_state(record)
    # Each player block ends with the recorded count, relative to current.
    assert np.allclose(
        context[[41, 84, 127, 170]], np.asarray((0, 1, 2, 3)) / 6
    )


def test_privileged_state_shapes_and_four_output_distribution() -> None:
    record = _privileged_record(create_initial_state(4, seed=7))
    board, context = encode_privileged_state(record)
    assert board.shape == (29, 7, 7)
    assert context.shape == (PRIVILEGED_STATE_CONTEXT_SIZE,)
    logits = build_privileged_state_net()(
        torch.from_numpy(board[None]), torch.from_numpy(context[None])
    )
    assert logits.shape == (1, 4)
    assert torch.allclose(
        torch.softmax(logits, dim=1).sum(dim=1), torch.ones(1)
    )


def test_card_multiset_groups_are_unique_and_limited() -> None:
    state = create_initial_state(4, seed=9)
    proposed = propose_top_card_groups(
        state,
        (),
        _IncreasingEstimator(),
        adaptive_pq_pruning=False,
        approximate_new_color_neighbor_limit=True,
    )
    keys = [item.group_key for item in proposed]
    assert len(keys) == len(set(keys))
    assert sum(len(item.cards) == 1 for item in proposed) <= 3
    assert sum(len(item.cards) == 2 for item in proposed) <= 5
    assert all(tuple(sorted(key)) == key for key in keys)


def test_all_action_delta_candidates_are_not_limited_to_card_groups() -> None:
    state = create_initial_state(4, seed=9)
    expected = enumerate_turn_end_candidates(
        state,
        history=(),
        approximate_new_color_neighbor_limit=True,
        collapse_equivalent_frames=True,
    )
    retained = enumerate_action_delta_candidates(
        state,
        (),
        adaptive_pq_pruning=False,
        approximate_new_color_neighbor_limit=True,
    )
    assert tuple(item.candidate.actions for item in retained) == tuple(
        item.actions for item in expected
    )
    assert len(retained) > 8


def test_opponent_private_hand_changes_critic_not_action_tensor() -> None:
    state = create_initial_state(4, seed=11)
    opponent = state.players[1]
    replacement_hand = state.players[2].hand
    if replacement_hand == opponent.hand:
        replacement_hand = state.players[3].hand
    changed = replace(
        state,
        players=(
            state.players[0],
            replace(opponent, hand=replacement_hand),
            state.players[2],
            state.players[3],
        ),
    )
    _, privileged_a = encode_privileged_state(_privileged_record(state))
    _, privileged_b = encode_privileged_state(_privileged_record(changed))
    assert not np.array_equal(privileged_a, privileged_b)

    candidate_a = enumerate_turn_end_candidates(
        state, approximate_new_color_neighbor_limit=True
    )[0]
    candidate_b = enumerate_turn_end_candidates(
        changed, approximate_new_color_neighbor_limit=True
    )[0]
    record_a = make_transition_record(
        game_id=1,
        state_before=state,
        state_after=candidate_a.record.state,
        history_before=(),
        cards=(
            state.players[0].hand[
                candidate_a.actions[0].hand_index
            ],
        ),
        target=0.0,
    )
    record_b = make_transition_record(
        game_id=1,
        state_before=changed,
        state_after=candidate_b.record.state,
        history_before=(),
        cards=record_a.cards,
        target=0.0,
    )
    board_a, context_a = encode_action_delta(record_a)
    board_b, context_b = encode_action_delta(record_b)
    assert np.array_equal(board_a, board_b)
    assert np.array_equal(context_a, context_b)


def test_action_delta_network_is_bounded_scalar() -> None:
    state = create_initial_state(4, seed=13)
    candidate = enumerate_turn_end_candidates(
        state, approximate_new_color_neighbor_limit=True
    )[0]
    card = state.players[0].hand[candidate.actions[0].hand_index]
    record = make_transition_record(
        game_id=1,
        state_before=state,
        state_after=candidate.record.state,
        history_before=(),
        cards=(card,),
        target=0.2,
    )
    board, context = encode_action_delta(record)
    assert context.shape == (ACTION_DELTA_CONTEXT_SIZE,)
    output = build_action_delta_net()(
        torch.from_numpy(board[None]), torch.from_numpy(context[None])
    )
    assert output.shape == (1,)
    assert -1 <= output.item() <= 1


def test_pre_play_probability_and_delta_are_combined_with_clipping() -> None:
    assert combined_post_play_probability(0.25, 0.1) == 0.35
    assert combined_post_play_probability(0.95, 0.1) == 1.0
    assert combined_post_play_probability(0.05, -0.1) == 0.0


def _write_delta_parts(root: Path, *, seed: int) -> None:
    root.mkdir()
    game_ids = np.arange(200, dtype=np.int64)
    buckets = _split_buckets(game_ids, seed)
    selected = np.concatenate(
        (
            game_ids[buckets < 8][:64],
            game_ids[buckets == 8][:8],
            game_ids[buckets == 9][:8],
        )
    )
    assert len(selected) == 80
    generator = np.random.default_rng(17)
    for part_index, ids in enumerate(np.array_split(selected, 4)):
        records = len(ids)
        np.savez_compressed(
            root / f"part_{part_index:09d}.npz",
            board=generator.normal(size=(records, 58, 7, 7)).astype(
                np.float32
            ),
            context=generator.normal(
                size=(records, ACTION_DELTA_CONTEXT_SIZE)
            ).astype(np.float32),
            target=generator.uniform(-0.2, 0.2, size=records).astype(
                np.float32
            ),
            game_id=ids,
            turn_id=np.arange(records, dtype=np.int32),
            play_count=np.where(np.arange(records) % 2, 1, 2).astype(
                np.int8
            ),
        )


def test_action_delta_snapshot_and_continuous_milestones(tmp_path: Path) -> None:
    seed = 20260727
    data = tmp_path / "data"
    _write_delta_parts(data, seed=seed)
    snapshot = data / "snapshot.json"
    manifest = build_action_delta_snapshot(data, snapshot)
    assert manifest["part_count"] == 4
    assert manifest["records"] == 80
    _, paths = verified_snapshot_paths(snapshot)
    assert len(paths) == 4

    staged_prefix = tmp_path / "staged"
    staged = train_action_delta_milestones(
        snapshot,
        staged_prefix,
        milestones=(10, 30, 50, 100),
        batch_size=8,
        seed=seed,
        progress_interval_parts=1,
    )
    reference_prefix = tmp_path / "reference"
    train_action_delta_milestones(
        snapshot,
        reference_prefix,
        milestones=(100,),
        batch_size=8,
        seed=seed,
        progress_interval_parts=1,
    )
    assert [row["percent"] for row in staged["milestones"]] == [
        10,
        30,
        50,
        100,
    ]
    final_staged = torch.load(
        tmp_path / "staged_pct100.pt", weights_only=False
    )
    final_reference = torch.load(
        tmp_path / "reference_pct100.pt", weights_only=False
    )
    for key, value in final_staged["state_dict"].items():
        assert torch.equal(value, final_reference["state_dict"][key])
    assert final_staged["metrics"]["test_one_card_records"] > 0
    assert final_staged["metrics"]["test_two_card_records"] > 0


def test_action_delta_milestone_resume_matches_continuation(
    tmp_path: Path,
) -> None:
    seed = 20260727
    data = tmp_path / "data"
    _write_delta_parts(data, seed=seed)
    snapshot = data / "snapshot.json"
    build_action_delta_snapshot(data, snapshot)
    prefix = tmp_path / "resume"
    progress = tmp_path / "resume.progress.pt"
    train_action_delta_milestones(
        snapshot,
        prefix,
        batch_size=8,
        seed=seed,
        progress_checkpoint=progress,
        progress_interval_parts=1,
    )
    expected = torch.load(
        tmp_path / "resume_pct100.pt", weights_only=False
    )["state_dict"]
    milestone_30 = torch.load(
        tmp_path / "resume_pct030.pt", weights_only=False
    )
    torch.save(milestone_30, progress)
    (tmp_path / "resume_pct050.pt").unlink()
    (tmp_path / "resume_pct100.pt").unlink()
    train_action_delta_milestones(
        snapshot,
        prefix,
        batch_size=8,
        seed=seed,
        progress_checkpoint=progress,
        progress_interval_parts=1,
    )
    resumed = torch.load(
        tmp_path / "resume_pct100.pt", weights_only=False
    )["state_dict"]
    for key, value in expected.items():
        assert torch.equal(value, resumed[key])
