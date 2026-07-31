"""Compressed replay logs and V2 record reconstruction."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable, Iterator

from yellowstone.game import apply_known_legal_action
from yellowstone.serialization import (
    action_from_dict,
    action_to_dict,
    game_state_from_dict,
    game_state_to_dict,
)
from yellowstone.types import (
    Action,
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
    RefillAction,
    RefillSource,
)
from yellowstone.value_v2 import (
    CandidateFrameContext,
    CompletedTurnTracker,
    PendingRefillSource,
    PublicNegativeKnowledgeTracker,
    ValueRecordV2,
)


REPLAY_SCHEMA_V2 = "yellowstone.replay.v2"
LEGACY_RULES_VERSION_V2 = "yellowstone-python-2026-07-26"
RULES_VERSION_V2 = "yellowstone-python-2026-07-27-empty-deck-settlement"


@dataclass(frozen=True, slots=True)
class ReplayGameV2:
    game_id: int
    initial_seed: int
    gameplay_seed: int
    initial_state: GameState
    actions: tuple[Action, ...]
    decisions: tuple[dict[str, Any], ...]
    winners: tuple[int, ...]
    teacher_checkpoint: str
    teacher_sha256: str
    teacher_generation: int
    privileged_teacher_deck: bool = True
    rules_version: str = RULES_VERSION_V2


def replay_game_to_dict(game: ReplayGameV2) -> dict[str, Any]:
    terminal_state = replay_game(game)
    return {
        "schema": REPLAY_SCHEMA_V2,
        "rules_version": game.rules_version,
        "game_id": game.game_id,
        "initial_seed": game.initial_seed,
        "gameplay_seed": game.gameplay_seed,
        "initial_state": game_state_to_dict(game.initial_state),
        "actions": [action_to_dict(action) for action in game.actions],
        "decisions": list(game.decisions),
        "winners": list(game.winners),
        "teacher": {
            "checkpoint": game.teacher_checkpoint,
            "sha256": game.teacher_sha256,
            "generation": game.teacher_generation,
            "privileged_actual_deck": game.privileged_teacher_deck,
        },
        "terminal_state_sha256": state_sha256(terminal_state),
    }


def replay_game_from_dict(data: dict[str, Any]) -> ReplayGameV2:
    if data.get("schema") != REPLAY_SCHEMA_V2:
        raise ValueError(f"unsupported replay schema: {data.get('schema')!r}")
    teacher = data["teacher"]
    return ReplayGameV2(
        game_id=int(data["game_id"]),
        initial_seed=int(data["initial_seed"]),
        gameplay_seed=int(data["gameplay_seed"]),
        initial_state=game_state_from_dict(data["initial_state"]),
        actions=tuple(action_from_dict(action) for action in data["actions"]),
        decisions=tuple(data.get("decisions", ())),
        winners=tuple(int(winner) for winner in data["winners"]),
        teacher_checkpoint=str(teacher["checkpoint"]),
        teacher_sha256=str(teacher["sha256"]),
        teacher_generation=int(teacher["generation"]),
        privileged_teacher_deck=bool(teacher["privileged_actual_deck"]),
        rules_version=str(data.get("rules_version", LEGACY_RULES_VERSION_V2)),
    )


def write_replay_shard(
    games: Iterable[ReplayGameV2], path: str | Path
) -> dict[str, int]:
    """Atomically write one gzip JSONL shard and return size facts."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    game_count = 0
    uncompressed_bytes = 0
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        for game in games:
            payload = json.dumps(
                replay_game_to_dict(game),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.write(payload)
            stream.write("\n")
            game_count += 1
            uncompressed_bytes += len(payload.encode("utf-8")) + 1
    temporary.replace(destination)
    return {
        "games": game_count,
        "compressed_bytes": destination.stat().st_size,
        "uncompressed_bytes": uncompressed_bytes,
    }


def read_replay_shard(path: str | Path) -> Iterator[ReplayGameV2]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield replay_game_from_dict(json.loads(line))
            except Exception as error:
                raise ValueError(f"invalid replay line {line_number} in {path}") from error


def replay_game(game: ReplayGameV2) -> GameState:
    state = game.initial_state
    rng = Random(game.gameplay_seed)
    for action in game.actions:
        state = apply_known_legal_action(
            state,
            action,
            rng=rng,
            settle_on_empty_deck=(
                game.rules_version != LEGACY_RULES_VERSION_V2
            ),
        )
    if state.phase != Phase.GAME_OVER:
        raise ValueError(f"replay game {game.game_id} did not finish")
    if state.winners != game.winners:
        raise ValueError(
            f"replay winners differ for game {game.game_id}: "
            f"{state.winners} != {game.winners}"
        )
    return state


def verify_replay_dict(data: dict[str, Any]) -> None:
    game = replay_game_from_dict(data)
    state = replay_game(game)
    expected = data.get("terminal_state_sha256")
    actual = state_sha256(state)
    if expected != actual:
        raise ValueError(
            f"terminal checksum differs for game {game.game_id}: "
            f"{actual} != {expected}"
        )


def records_from_replay(game: ReplayGameV2) -> tuple[ValueRecordV2, ...]:
    """Reconstruct selected-turn V2 records and attach terminal win targets."""
    state = game.initial_state
    rng = Random(game.gameplay_seed)
    history = CompletedTurnTracker()
    knowledge = PublicNegativeKnowledgeTracker(len(state.players))
    records: list[
        tuple[
            int,
            GameState,
            object,
            object,
            PendingRefillSource,
            CandidateFrameContext,
        ]
    ] = []
    turn_history_before = None
    turn_player = None
    turn_start_frame = None
    turn_end_frame = None
    turn_start_board_card_count = None

    for action in game.actions:
        before = state
        knowledge_before = knowledge.snapshot()
        if (
            isinstance(action, PlaceCardAction)
            and before.cards_played_this_turn == 0
        ):
            turn_history_before = history.snapshot()
            turn_player = before.current_player_index
            turn_start_frame = history.current_frame
            turn_start_board_card_count = sum(
                len(stack) for stack in before.board.values()
            )
        if isinstance(action, PlaceCardAction):
            turn_end_frame = action.frame

        state = apply_known_legal_action(
            state,
            action,
            rng=rng,
            settle_on_empty_deck=(
                game.rules_version != LEGACY_RULES_VERSION_V2
            ),
        )
        history.observe(before, action, state)
        knowledge.observe(before, action, state)

        if isinstance(action, EndTurnAction) and state.phase != Phase.REFILL:
            if (
                turn_history_before is None
                or turn_player is None
                or turn_end_frame is None
                or turn_start_board_card_count is None
            ):
                raise AssertionError("one-card completion lacks turn start")
            records.append(
                (
                    turn_player,
                    state,
                    turn_history_before,
                    knowledge.snapshot(),
                    PendingRefillSource.NO_PENDING,
                    CandidateFrameContext(
                        start_frame=turn_start_frame,
                        end_frame=turn_end_frame,
                        start_board_card_count=turn_start_board_card_count,
                    ),
                )
            )
            turn_history_before = None
            turn_player = None
            turn_start_frame = None
            turn_end_frame = None
            turn_start_board_card_count = None
        elif isinstance(action, RefillAction) and turn_player is not None:
            if (
                turn_history_before is None
                or turn_end_frame is None
                or turn_start_board_card_count is None
            ):
                raise AssertionError("refill completion lacks turn start")
            records.append(
                (
                    turn_player,
                    before,
                    turn_history_before,
                    knowledge_before,
                    PendingRefillSource(action.source.value),
                    CandidateFrameContext(
                        start_frame=turn_start_frame,
                        end_frame=turn_end_frame,
                        start_board_card_count=turn_start_board_card_count,
                    ),
                )
            )
            turn_history_before = None
            turn_player = None
            turn_start_frame = None
            turn_end_frame = None
            turn_start_board_card_count = None

    if state.phase != Phase.GAME_OVER:
        raise ValueError("cannot label an unfinished replay")
    winner_count = len(state.winners)
    if winner_count == 0:
        raise AssertionError("finished replay has no winners")
    return tuple(
        ValueRecordV2(
            game_id=game.game_id,
            perspective_player_index=player_index,
            state=record_state,
            history_before_turn=record_history,
            candidate_frame=candidate_frame,
            negative_knowledge=record_knowledge,
            pending_refill_source=pending,
            target=(
                1.0 / winner_count if player_index in state.winners else 0.0
            ),
        )
        for (
            player_index,
            record_state,
            record_history,
            record_knowledge,
            pending,
            candidate_frame,
        ) in records
    )


def state_sha256(state: GameState) -> str:
    payload = json.dumps(
        game_state_to_dict(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
