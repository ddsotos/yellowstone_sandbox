"""Collect restartable next-turn privileged-critic action deltas."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from random import Random
from time import monotonic

from yellowstone.action_delta import (
    CANONICALIZATION_ACTION_DELTA,
    HISTORY_SEMANTICS_ACTION_DELTA,
    VALUE_SCHEMA_ACTION_DELTA,
    _append_recent,
    encode_action_delta,
    make_transition_record,
    propose_top_card_groups,
    validate_proposer_checkpoint,
)
from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action, legal_actions
from yellowstone.privileged_state import (
    PrivilegedStateRecord,
    TorchPrivilegedStateEstimator,
)
from yellowstone.replay_v2 import (
    LEGACY_RULES_VERSION_V2,
    file_sha256,
    read_replay_shard,
)
from yellowstone.types import GameState, Phase, PlaceCardAction
from yellowstone.value_learning import RecentPlacement
from yellowstone.value_policy import TorchWinValueEstimator
from yellowstone.value_v2 import CompletedTurnTracker


def collect_action_delta(
    source: str | Path,
    output: str | Path,
    *,
    proposer_checkpoint: str | Path,
    critic_checkpoint: str | Path,
    max_turns: int | None = None,
    duration_hours: float | None = None,
    shard_turns: int = 100,
) -> dict:
    import numpy as np

    if max_turns is None and duration_hours is None:
        raise ValueError("max_turns or duration_hours is required")
    source_path, output_path = Path(source), Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = tuple(sorted(source_path.glob("part_*.jsonl.gz")))
    if not paths:
        raise FileNotFoundError(f"no replay shards at {source_path}")
    proposer_path, critic_path = Path(proposer_checkpoint), Path(critic_checkpoint)
    validate_proposer_checkpoint(proposer_path)
    proposer = TorchWinValueEstimator(str(proposer_path))
    critic = TorchPrivilegedStateEstimator(critic_path)
    expected = {
        "schema": VALUE_SCHEMA_ACTION_DELTA,
        "canonicalization": CANONICALIZATION_ACTION_DELTA,
        "history_semantics": HISTORY_SEMANTICS_ACTION_DELTA,
        "source": str(source_path),
        "proposer_checkpoint": str(proposer_path),
        "proposer_sha256": file_sha256(proposer_path),
        "critic_checkpoint": str(critic_path),
        "critic_sha256": file_sha256(critic_path),
        "opponent_private_inputs": False,
    }
    progress_path = output_path / "collection_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8-sig"))
        if progress_path.exists()
        else {"turns": 0, "records": 0, "parts": []}
    )
    for key, value in expected.items():
        if key in progress and progress[key] != value:
            raise ValueError(f"delta progress differs at {key}")
        progress[key] = value
    already_complete = int(progress["turns"])
    prior_wall = float(progress.get("wall_seconds_total", 0.0))
    global_turn = 0
    started = monotonic()
    batch: list[tuple] = []

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        start = int(progress["turns"])
        destination = output_path / f"part_{start:09d}.npz"
        boards, contexts, targets = [], [], []
        game_ids, turn_ids, play_counts, proposer_scores = [], [], [], []
        before_values, next_values = [], []
        for row in batch:
            record, game_id, turn_id, score, before_value, next_value = row
            board, context = encode_action_delta(record)
            boards.append(board)
            contexts.append(context)
            targets.append(record.target)
            game_ids.append(game_id)
            turn_ids.append(turn_id)
            play_counts.append(len(record.cards))
            proposer_scores.append(score)
            before_values.append(before_value)
            next_values.append(next_value)
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                board=np.stack(boards),
                context=np.stack(contexts),
                target=np.asarray(targets, dtype=np.float32),
                game_id=np.asarray(game_ids, dtype=np.int64),
                turn_id=np.asarray(turn_ids, dtype=np.int32),
                play_count=np.asarray(play_counts, dtype=np.int8),
                proposer_score=np.asarray(proposer_scores, dtype=np.float32),
                before_value=np.asarray(before_values, dtype=np.float32),
                next_value=np.asarray(next_values, dtype=np.float32),
            )
        os.replace(temporary, destination)
        turns_in_batch = len({(row[1], row[2]) for row in batch})
        progress["turns"] = int(progress["turns"]) + turns_in_batch
        progress["records"] = int(progress["records"]) + len(batch)
        progress["parts"].append(
            {
                "path": destination.name,
                "turns": turns_in_batch,
                "records": len(batch),
                "compressed_bytes": destination.stat().st_size,
            }
        )
        _write_json(progress_path, progress)
        batch = []

    stop = False
    for path in paths:
        if stop:
            break
        for game in read_replay_shard(path):
            rows = _rows_for_game(game, proposer, critic)
            for row in rows:
                if global_turn < already_complete:
                    global_turn += 1
                    continue
                if max_turns is not None and global_turn >= max_turns:
                    stop = True
                    break
                if (
                    duration_hours is not None
                    and prior_wall + monotonic() - started
                    >= duration_hours * 3600
                ):
                    stop = True
                    break
                batch.extend(row)
                global_turn += 1
                if len({(item[1], item[2]) for item in batch}) >= shard_turns:
                    flush()
            if stop:
                break
    flush()
    elapsed = monotonic() - started
    progress["status"] = (
        "complete"
        if duration_hours is not None
        and prior_wall + elapsed >= duration_hours * 3600
        else "stopped"
    )
    progress["wall_seconds_last_run"] = elapsed
    progress["wall_seconds_total"] = prior_wall + elapsed
    _write_json(progress_path, progress)
    _write_json(output_path / "manifest.json", progress)
    return progress


def _rows_for_game(game, proposer, critic):
    state = game.initial_state
    rng = Random(game.gameplay_seed)
    recent: list[RecentPlacement] = []
    completed = CompletedTurnTracker()
    heuristic = HeuristicBot()
    turn_id = 0
    settle = game.rules_version != LEGACY_RULES_VERSION_V2
    for actual_action in game.actions:
        before = state
        if (
            isinstance(actual_action, PlaceCardAction)
            and before.cards_played_this_turn == 0
        ):
            actor = before.current_player_index
            proposed = propose_top_card_groups(
                before, tuple(recent), proposer
            )
            before_record = PrivilegedStateRecord(
                game.game_id,
                before,
                tuple(recent),
                (0.0, 0.0, 0.0, 0.0),
            )
            before_value = critic(before_record)[0]
            branches = []
            for item in proposed:
                branch_rng = Random()
                branch_rng.setstate(rng.getstate())
                branch_state = before
                branch_recent = list(recent)
                branch_completed = copy.deepcopy(completed)
                for action in item.candidate.actions:
                    branch_before = branch_state
                    branch_state = apply_known_legal_action(
                        branch_state,
                        action,
                        rng=branch_rng,
                        settle_on_empty_deck=settle,
                    )
                    _append_recent(
                        branch_recent, branch_before, action, branch_state
                    )
                    branch_completed.observe(
                        branch_before, action, branch_state
                    )
                candidate_state = branch_state
                while (
                    branch_state.phase != Phase.GAME_OVER
                    and not _is_next_decision(branch_state, actor)
                ):
                    action = heuristic.choose_action(branch_state)
                    if action is None:
                        raise RuntimeError("heuristic continuation stopped")
                    branch_before = branch_state
                    branch_state = apply_known_legal_action(
                        branch_state,
                        action,
                        rng=branch_rng,
                        settle_on_empty_deck=settle,
                    )
                    _append_recent(
                        branch_recent, branch_before, action, branch_state
                    )
                    branch_completed.observe(
                        branch_before, action, branch_state
                    )
                if branch_state.phase == Phase.GAME_OVER:
                    next_value = (
                        1.0 / len(branch_state.winners)
                        if actor in branch_state.winners
                        else 0.0
                    )
                    next_record = None
                else:
                    next_record = PrivilegedStateRecord(
                        game.game_id,
                        branch_state,
                        tuple(branch_recent),
                        (0.0, 0.0, 0.0, 0.0),
                    )
                    next_value = None
                branches.append(
                    (item, candidate_state, next_record, next_value)
                )
            pending_records = tuple(
                branch[2] for branch in branches if branch[2] is not None
            )
            pending_values = iter(
                value[0] for value in critic.estimate_many(pending_records)
            )
            branch_facts = []
            for item, candidate_state, next_record, next_value in branches:
                if next_record is not None:
                    next_value = next(pending_values)
                if next_value is None:
                    raise AssertionError("missing next-state value")
                delta = next_value - before_value
                record = make_transition_record(
                    game_id=game.game_id,
                    state_before=before,
                    state_after=candidate_state,
                    history_before=completed.snapshot(),
                    cards=item.cards,
                    target=delta,
                )
                branch_facts.append(
                    (
                        record,
                        game.game_id,
                        turn_id,
                        item.proposer_score,
                        before_value,
                        next_value,
                    )
                )
            yield branch_facts
            turn_id += 1
        state = apply_known_legal_action(
            state,
            actual_action,
            rng=rng,
            settle_on_empty_deck=settle,
        )
        _append_recent(recent, before, actual_action, state)
        completed.observe(before, actual_action, state)


def _is_next_decision(state: GameState, actor: int) -> bool:
    return (
        state.current_player_index == actor
        and state.phase == Phase.PLAY
        and state.cards_played_this_turn == 0
        and any(isinstance(action, PlaceCardAction) for action in legal_actions(state))
    )


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposer-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--duration-hours", type=float)
    parser.add_argument("--shard-turns", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            collect_action_delta(
                args.source,
                args.output,
                proposer_checkpoint=args.proposer_checkpoint,
                critic_checkpoint=args.critic_checkpoint,
                max_turns=args.max_turns,
                duration_hours=args.duration_hours,
                shard_turns=args.shard_turns,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
