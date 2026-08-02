"""Convert V2 raw replays to board-centered Original V1 tensors."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from random import Random

from yellowstone.convert_replay_v2_to_v1_original import (
    HISTORY_SEMANTICS_V1_ORIGINAL,
    VALUE_SCHEMA_V1_ORIGINAL,
)
from yellowstone.game import apply_known_legal_action
from yellowstone.replay_v2 import LEGACY_RULES_VERSION_V2, ReplayGameV2
from yellowstone.replay_v2 import read_replay_shard
from yellowstone.types import EndTurnAction, Phase, PlaceCardAction, RefillAction
from yellowstone.value_learning import HISTORY_SIZE, RecentPlacement, ValueRecord
from yellowstone.value_board_centered import (
    BOARD_CENTERED_V1,
    BOARD_CENTERED_V1_CANONICALIZATIONS,
    BOARD_CENTERED_V1_CHAIN_CONTEXT_SIZE,
    BOARD_CENTERED_V1_CHAIN_HISTORY,
    BOARD_CENTERED_V1_CONTEXT_SIZE,
    board_center_frame_origin,
    board_center_records_with_stats,
    board_centered_metadata,
)


def convert_replay_shards(
    source: str | Path,
    output: str | Path,
    *,
    expected_games: int | None = None,
    game_id_rebase: int = 0,
    expected_source_game_id_min: int | None = None,
    expected_source_game_id_max: int | None = None,
    input_canonicalization: str = BOARD_CENTERED_V1,
) -> dict[str, object]:
    """Convert restartable replay shards to board-centered V1 archives."""
    import numpy as np
    if input_canonicalization not in BOARD_CENTERED_V1_CANONICALIZATIONS:
        raise ValueError(f"unsupported b-center canonicalization: {input_canonicalization}")

    source_path = Path(source)
    output_path = Path(output)
    paths = tuple(
        sorted(
            source_path.glob("part_*.jsonl.gz"),
            key=lambda path: int(path.name[5 : -len(".jsonl.gz")]),
        )
    )
    if not paths:
        raise FileNotFoundError(f"no replay shards found at {source_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    converted_files = skipped_files = 0
    combined_stats: dict[str, int | list[int] | None] = {
        "records": 0,
        "min_anchor_rank": None,
        "max_anchor_rank": None,
        "min_rank_delta": None,
        "max_rank_delta": None,
        "left_margin_counts": [0, 0, 0, 0],
        "top_margin_counts": [0, 0, 0, 0],
        "column_empty_state_counts": [0, 0, 0, 0],
        "row_empty_state_counts": [0, 0, 0, 0],
    }
    invalid_records = 0
    invalid_examples: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        destination = output_path / path.name.replace(".jsonl.gz", ".npz")
        if destination.is_file():
            skipped_files += 1
            continue
        rows = [
            record
            for game in read_replay_shard(path)
            for record in records_from_replay_board_centered(
                game,
                input_canonicalization=input_canonicalization,
            )
        ]
        if not rows:
            raise ValueError(f"replay shard produced no records: {path}")
        valid_indices: list[int] = []
        centered_boards: list[np.ndarray] = []
        centered_contexts: list[np.ndarray] = []
        for row_index, record in enumerate(rows):
            try:
                centered_board, centered_context, bcenter_stats = (
                    board_center_records_with_stats(
                        (record,),
                        canonicalization=input_canonicalization,
                    )
                )
            except ValueError as error:
                invalid_records += 1
                if len(invalid_examples) < 20:
                    invalid_examples.append(
                        {
                            "source_shard": path.name,
                            "row_index": row_index,
                            "game_id": record.game_id,
                            "perspective_player_index": (
                                record.perspective_player_index
                            ),
                            "error": str(error),
                        }
                    )
                continue
            valid_indices.append(row_index)
            centered_boards.append(centered_board[0])
            centered_contexts.append(centered_context[0])
            _merge_stats(combined_stats, asdict(bcenter_stats))
        if not valid_indices:
            raise ValueError(f"replay shard produced no valid b-center records: {path}")
        board = np.stack(centered_boards)
        context = np.stack(centered_contexts)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            board=board,
            context=context,
            target=np.asarray(
                [rows[index].target for index in valid_indices], dtype=np.float32
            ),
            game_id=np.asarray(
                [rows[index].game_id - game_id_rebase for index in valid_indices],
                dtype=np.int64,
            ),
            source_game_id=np.asarray(
                [rows[index].game_id for index in valid_indices],
                dtype=np.int64,
            ),
            perspective_player_index=np.asarray(
                [
                    rows[index].perspective_player_index
                    for index in valid_indices
                ],
                dtype=np.int8,
            ),
        )
        os.replace(temporary, destination)
        converted_files += 1
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(
                f"progress={index}/{len(paths)} "
                f"converted={converted_files} skipped={skipped_files}",
                flush=True,
            )

    archive_paths = tuple(sorted(output_path.glob("part_*.npz")))
    game_ids: set[int] = set()
    source_game_ids: set[int] = set()
    records = 0
    compressed_bytes = 0
    for path in archive_paths:
        with np.load(path) as archive:
            count = len(archive["target"])
            records += count
            if archive["board"].shape[1:] != (1, 3, 3):
                raise ValueError(f"unexpected b-center board shape in {path.name}")
            expected_context_size = (
                BOARD_CENTERED_V1_CHAIN_CONTEXT_SIZE
                if input_canonicalization == BOARD_CENTERED_V1_CHAIN_HISTORY
                else BOARD_CENTERED_V1_CONTEXT_SIZE
            )
            if archive["context"].shape[1] != expected_context_size:
                raise ValueError(f"unexpected b-center context shape in {path.name}")
            game_ids.update(int(value) for value in archive["game_id"])
            if "source_game_id" not in archive:
                raise ValueError(f"converted shard lacks source_game_id: {path}")
            shard_source_ids = archive["source_game_id"]
            if not np.array_equal(
                archive["game_id"], shard_source_ids - game_id_rebase
            ):
                raise ValueError(f"rebase differs in {path.name}")
            source_game_ids.update(int(value) for value in shard_source_ids)
        compressed_bytes += path.stat().st_size
    if expected_games is not None and len(game_ids) != expected_games:
        raise ValueError(
            f"converted game count differs: {len(game_ids)} != {expected_games}"
        )
    if not game_ids:
        raise ValueError("conversion produced no game IDs")
    if game_ids != set(range(len(game_ids))):
        raise ValueError(
            "rebased game IDs are not continuous "
            f"0..{len(game_ids) - 1}"
        )
    source_min = min(source_game_ids)
    source_max = max(source_game_ids)
    if expected_source_game_id_min is not None and source_min != expected_source_game_id_min:
        raise ValueError(
            f"source game ID minimum differs: {source_min} != {expected_source_game_id_min}"
        )
    if expected_source_game_id_max is not None and source_max != expected_source_game_id_max:
        raise ValueError(
            f"source game ID maximum differs: {source_max} != {expected_source_game_id_max}"
        )
    if source_game_ids != set(range(source_min, source_max + 1)):
        raise ValueError(
            f"source game IDs are not continuous {source_min}..{source_max}"
        )

    result: dict[str, object] = {
        "value_schema": VALUE_SCHEMA_V1_ORIGINAL,
        "canonicalization": input_canonicalization,
        "source": str(source_path),
        "output": str(output_path),
        "source_shards": len(paths),
        "converted_files": converted_files,
        "skipped_files": skipped_files,
        "games": len(game_ids),
        "game_id_rebase": game_id_rebase,
        "rebased_game_id_min": min(game_ids),
        "rebased_game_id_max": max(game_ids),
        "source_game_id_min": source_min,
        "source_game_id_max": source_max,
        "records": records,
        "compressed_bytes": compressed_bytes,
        "history_semantics": (
            "bcenter_chain_play_after_deltas_4_8_12"
            if input_canonicalization == BOARD_CENTERED_V1_CHAIN_HISTORY
            else HISTORY_SEMANTICS_V1_ORIGINAL
        ),
        "board_centered_stats": combined_stats,
        "board_centered_invalid_records": invalid_records,
        "board_centered_invalid_examples": invalid_examples,
        "tensor_contract": board_centered_metadata(input_canonicalization),
        "source_replay_audit": {
            "game_ids_continuous": True,
            "rebased_game_ids_continuous": True,
            "record_counts_match": True,
            "labels_derived_from_terminal_winners": True,
            "terminal_winners_match_replay": True,
        },
    }
    temporary_manifest = output_path / "conversion_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, output_path / "conversion_manifest.json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def records_from_replay_board_centered(
    game: ReplayGameV2,
    *,
    input_canonicalization: str = BOARD_CENTERED_V1,
) -> tuple[ValueRecord, ...]:
    """Rebuild V1 records with the requested b-center history variant."""
    variant = board_centered_metadata(input_canonicalization)[
        "board_center_history_variant"
    ]
    state = game.initial_state
    rng = Random(game.gameplay_seed)
    rolling_history: list[RecentPlacement] = []
    turn_placements: list[RecentPlacement] = []
    turn_player: int | None = None
    own_frame_history: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    chain_state_history: list[list[object]] = [[] for _ in range(4)]
    pending: list[
        tuple[
            int,
            object,
            tuple[RecentPlacement, ...],
            tuple[tuple[int, int], ...],
            tuple[object, ...],
        ]
    ] = []

    def complete_turn(player_index: int, snapshot) -> None:
        if variant == "none":
            history: tuple[RecentPlacement, ...] = ()
        elif variant == "turn_local":
            history = tuple(turn_placements)
        else:
            history = tuple(rolling_history)
        frame_history = tuple(own_frame_history[player_index][-HISTORY_SIZE:])
        chain_history = tuple(chain_state_history[player_index][-12:])
        pending.append((player_index, snapshot, history, frame_history, chain_history))
        chain_state_history[player_index].append(snapshot)
        del chain_state_history[player_index][:-12]
        try:
            own_frame_history[player_index].append(
                board_center_frame_origin(snapshot)
            )
            del own_frame_history[player_index][:-HISTORY_SIZE]
        except ValueError:
            pass

    for action in game.actions:
        before = state
        player_index = before.current_player_index
        card = None
        if isinstance(action, PlaceCardAction):
            if before.cards_played_this_turn == 0:
                turn_placements = []
                turn_player = player_index
            if turn_player != player_index:
                raise AssertionError("turn player changed during placement")
            card = before.players[player_index].hand[action.hand_index]
        state = apply_known_legal_action(
            state,
            action,
            rng=rng,
            settle_on_empty_deck=(
                game.rules_version != LEGACY_RULES_VERSION_V2
            ),
        )
        if isinstance(action, PlaceCardAction):
            if card is None:
                raise AssertionError("placement card was not captured")
            placement = RecentPlacement(
                player_index=player_index,
                card=card,
                score_delta=(
                    before.players[player_index].loss_score
                    - state.players[player_index].loss_score
                ),
                negative_card_delta=(
                    len(state.players[player_index].negative_cards)
                    - len(before.players[player_index].negative_cards)
                ),
            )
            rolling_history.append(placement)
            del rolling_history[:-HISTORY_SIZE]
            turn_placements.append(placement)
            if len(turn_placements) > HISTORY_SIZE:
                raise AssertionError("a turn contains more than two placements")
            if state.phase == Phase.REFILL:
                complete_turn(player_index, state)
        elif isinstance(action, EndTurnAction):
            if before.cards_played_this_turn != 1 or not turn_placements:
                raise AssertionError("one-card completion lacks a placement")
            complete_turn(player_index, state)
            turn_placements = []
            turn_player = None
        elif isinstance(action, RefillAction):
            turn_placements = []
            turn_player = None

    if state.phase != Phase.GAME_OVER:
        raise ValueError(f"replay game {game.game_id} did not finish")
    if state.winners != game.winners:
        raise ValueError(f"replay winners differ for game {game.game_id}")
    if not state.winners:
        raise AssertionError("finished replay has no winners")
    winner_count = len(state.winners)
    return tuple(
        ValueRecord(
            game_id=game.game_id,
            perspective_player_index=player_index,
            state=snapshot,
            history=history,
            target=(
                1.0 / winner_count
                if player_index in state.winners
                else 0.0
            ),
            board_center_frame_history=frame_history,
            board_center_chain_states=chain_history,
        )
        for player_index, snapshot, history, frame_history, chain_history in pending
    )


def _merge_stats(target: dict[str, object], source: dict[str, object]) -> None:
    target["records"] = int(target["records"]) + int(source["records"])
    for key in ("min_anchor_rank", "min_rank_delta"):
        value = int(source[key])
        target[key] = value if target[key] is None else min(int(target[key]), value)
    for key in ("max_anchor_rank", "max_rank_delta"):
        value = int(source[key])
        target[key] = value if target[key] is None else max(int(target[key]), value)
    for key in (
        "left_margin_counts",
        "top_margin_counts",
        "column_empty_state_counts",
        "row_empty_state_counts",
    ):
        counts = list(target[key])
        for index, value in enumerate(source[key]):
            counts[index] += int(value)
        target[key] = counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int)
    parser.add_argument("--game-id-rebase", type=int, default=0)
    parser.add_argument("--expected-source-game-id-min", type=int)
    parser.add_argument("--expected-source-game-id-max", type=int)
    parser.add_argument(
        "--input-canonicalization",
        choices=BOARD_CENTERED_V1_CANONICALIZATIONS,
        default=BOARD_CENTERED_V1,
    )
    args = parser.parse_args()
    convert_replay_shards(
        args.source,
        args.output,
        expected_games=args.expected_games,
        game_id_rebase=args.game_id_rebase,
        expected_source_game_id_min=args.expected_source_game_id_min,
        expected_source_game_id_max=args.expected_source_game_id_max,
        input_canonicalization=args.input_canonicalization,
    )


if __name__ == "__main__":
    main()
