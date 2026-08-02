"""Convert V2 raw replays to canonical Original V1 rolling-history tensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from random import Random

from yellowstone.game import apply_known_legal_action
from yellowstone.replay_v2 import (
    LEGACY_RULES_VERSION_V2,
    ReplayGameV2,
    read_replay_shard,
)
from yellowstone.types import (
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
    RefillAction,
)
from yellowstone.value_canonicalization import (
    CANONICALIZATION_NAME,
    canonicalize_value_tensors_with_stats,
)
from yellowstone.value_learning import (
    HISTORY_SIZE,
    RecentPlacement,
    ValueRecord,
    board_tensor_for_player,
    context_tensor_for_player,
)
from yellowstone.value_refill_count import (
    CANONICALIZATION_REFILL_COUNT,
    CANONICALIZATION_REFILL_COUNT_SCALAR,
    canonicalize_refill_count_tensors,
    canonicalize_refill_count_scalar_tensors,
    refill_count_for_action,
    refill_count_metadata,
    refill_count_scalar_metadata,
)


VALUE_SCHEMA_V1_ORIGINAL = "yellowstone.value.v1"
HISTORY_SEMANTICS_V1_ORIGINAL = "rolling_last_two_placements"
_CARDS_PLAYED_INDEX = 55


def records_from_replay_v1_original(
    game: ReplayGameV2,
) -> tuple[ValueRecord, ...]:
    """Rebuild Original V1 records with the global last two placements."""
    state = game.initial_state
    rng = Random(game.gameplay_seed)
    history: list[RecentPlacement] = []
    pending: list[
        tuple[int, object, tuple[RecentPlacement, ...]]
    ] = []

    for action in game.actions:
        before = state
        player_index = before.current_player_index
        card = (
            before.players[player_index].hand[action.hand_index]
            if isinstance(action, PlaceCardAction)
            else None
        )
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
            history.append(
                RecentPlacement(
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
            )
            del history[:-HISTORY_SIZE]
            if state.phase == Phase.REFILL:
                pending.append((player_index, state, tuple(history)))
        elif isinstance(action, EndTurnAction):
            if before.cards_played_this_turn != 1 or not history:
                raise AssertionError("one-card completion lacks a placement")
            pending.append((player_index, state, tuple(history)))

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
            history=snapshot_history,
            target=(
                1.0 / winner_count
                if player_index in state.winners
                else 0.0
            ),
        )
        for player_index, snapshot, snapshot_history in pending
    )


def records_from_replay_v1_refill_count(
    game: ReplayGameV2,
) -> tuple[ValueRecord, ...]:
    """Rebuild Original V1 records with explicit pending refill-card counts."""
    state = game.initial_state
    rng = Random(game.gameplay_seed)
    history: list[RecentPlacement] = []
    pending: list[tuple[int, int, GameState, tuple[RecentPlacement, ...]]] = []
    completed: list[tuple[int, ValueRecord]] = []
    next_record_index = 0

    for action in game.actions:
        before = state
        player_index = before.current_player_index
        card = (
            before.players[player_index].hand[action.hand_index]
            if isinstance(action, PlaceCardAction)
            else None
        )
        if isinstance(action, RefillAction):
            if not pending:
                raise AssertionError("refill action lacks a pending V1 record")
            (
                record_index,
                pending_player,
                snapshot,
                snapshot_history,
            ) = pending.pop()
            if pending_player != player_index:
                raise AssertionError("pending refill player differs")
            refill_snapshot = ValueRecord(
                game_id=game.game_id,
                perspective_player_index=pending_player,
                state=snapshot,
                history=snapshot_history,
                target=0.0,
            )
            completed.append(
                (
                    record_index,
                    ValueRecord(
                        game_id=game.game_id,
                        perspective_player_index=pending_player,
                        state=snapshot,
                        history=snapshot_history,
                        target=0.0,
                        refill_count=refill_count_for_action(
                            refill_snapshot,
                            action,
                        ),
                    ),
                )
            )
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
            history.append(
                RecentPlacement(
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
            )
            del history[:-HISTORY_SIZE]
            if state.phase == Phase.REFILL:
                pending.append(
                    (next_record_index, player_index, state, tuple(history))
                )
                next_record_index += 1
        elif isinstance(action, EndTurnAction):
            if before.cards_played_this_turn != 1 or not history:
                raise AssertionError("one-card completion lacks a placement")
            pending.append(
                (next_record_index, player_index, state, tuple(history))
            )
            next_record_index += 1

    if state.phase != Phase.GAME_OVER:
        raise ValueError(f"replay game {game.game_id} did not finish")
    if state.winners != game.winners:
        raise ValueError(f"replay winners differ for game {game.game_id}")
    if not state.winners:
        raise AssertionError("finished replay has no winners")
    winner_count = len(state.winners)
    completed.extend(
        (
            record_index,
            ValueRecord(
                game_id=game.game_id,
                perspective_player_index=player_index,
                state=snapshot,
                history=snapshot_history,
                target=0.0,
                refill_count=0,
            ),
        )
        for record_index, player_index, snapshot, snapshot_history in pending
    )
    return tuple(
        ValueRecord(
            game_id=record.game_id,
            perspective_player_index=record.perspective_player_index,
            state=record.state,
            history=record.history,
            target=(
                1.0 / winner_count
                if record.perspective_player_index in state.winners
                else 0.0
            ),
            refill_count=record.refill_count,
        )
        for _, record in sorted(completed, key=lambda item: item[0])
    )


def convert_replay_shards(
    source: str | Path,
    output: str | Path,
    *,
    expected_games: int | None = None,
    reference: str | Path | None = None,
    game_id_rebase: int = 0,
    expected_source_game_id_min: int | None = None,
    expected_source_game_id_max: int | None = None,
    input_canonicalization: str = CANONICALIZATION_NAME,
) -> dict[str, object]:
    """Convert restartable replay shards and optionally audit a reference set."""
    import numpy as np

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
    horizontal = vertical = 0
    for index, path in enumerate(paths, start=1):
        destination = output_path / path.name.replace(".jsonl.gz", ".npz")
        if destination.is_file():
            skipped_files += 1
            continue
        record_builder = (
            records_from_replay_v1_refill_count
            if input_canonicalization
            in (CANONICALIZATION_REFILL_COUNT, CANONICALIZATION_REFILL_COUNT_SCALAR)
            else records_from_replay_v1_original
        )
        rows = []
        for game in read_replay_shard(path):
            if (
                expected_source_game_id_min is not None
                and game.game_id < expected_source_game_id_min
            ):
                continue
            if (
                expected_source_game_id_max is not None
                and game.game_id > expected_source_game_id_max
            ):
                continue
            rows.extend(record_builder(game))
        if not rows:
            skipped_files += 1
            continue
        board = np.stack(
            [board_tensor_for_player(record) for record in rows]
        )
        context = np.stack(
            [context_tensor_for_player(record) for record in rows]
        )
        if input_canonicalization == CANONICALIZATION_REFILL_COUNT:
            board, context, stats = canonicalize_refill_count_tensors(
                board,
                context,
                [record.refill_count for record in rows],
            )
        elif input_canonicalization == CANONICALIZATION_REFILL_COUNT_SCALAR:
            board, context, stats = canonicalize_refill_count_scalar_tensors(
                board,
                context,
                [record.refill_count for record in rows],
            )
        elif input_canonicalization == CANONICALIZATION_NAME:
            board, context, stats = canonicalize_value_tensors_with_stats(
                board, context
            )
        else:
            raise ValueError(
                f"unsupported V1 original canonicalization: {input_canonicalization}"
            )
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            board=board,
            context=context,
            target=np.asarray(
                [record.target for record in rows], dtype=np.float32
            ),
            game_id=np.asarray(
                [record.game_id - game_id_rebase for record in rows],
                dtype=np.int64,
            ),
            source_game_id=np.asarray(
                [record.game_id for record in rows], dtype=np.int64
            ),
            perspective_player_index=np.asarray(
                [record.perspective_player_index for record in rows],
                dtype=np.int8,
            ),
        )
        os.replace(temporary, destination)
        converted_files += 1
        horizontal += stats.horizontal_reflections
        vertical += stats.vertical_reflections
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(
                f"progress={index}/{len(paths)} "
                f"converted={converted_files} skipped={skipped_files}",
                flush=True,
            )

    archive_paths = tuple(sorted(output_path.glob("part_*.npz")))
    game_ids: set[int] = set()
    source_game_ids: set[int] = set()
    records = one_card_records = two_card_records = 0
    compressed_bytes = 0
    for path in archive_paths:
        with np.load(path) as archive:
            count = len(archive["target"])
            records += count
            game_ids.update(int(value) for value in archive["game_id"])
            if "source_game_id" not in archive:
                raise ValueError(
                    f"converted shard lacks source_game_id: {path}"
                )
            shard_source_ids = archive["source_game_id"]
            if not np.array_equal(
                archive["game_id"], shard_source_ids - game_id_rebase
            ):
                raise ValueError(f"rebase differs in {path.name}")
            source_game_ids.update(int(value) for value in shard_source_ids)
            one_card = archive["context"][:, _CARDS_PLAYED_INDEX] < 0.75
            one_card_records += int(one_card.sum())
            two_card_records += int(count - one_card.sum())
        compressed_bytes += path.stat().st_size
    if expected_games is not None and len(game_ids) != expected_games:
        raise ValueError(
            f"converted game count differs: {len(game_ids)} != {expected_games}"
        )
    if not game_ids:
        raise ValueError("conversion produced no game IDs")
    expected_rebased_ids = set(range(len(game_ids)))
    if game_ids != expected_rebased_ids:
        raise ValueError(
            "rebased game IDs are not continuous "
            f"0..{len(game_ids) - 1}"
        )
    source_min = min(source_game_ids)
    source_max = max(source_game_ids)
    if expected_source_game_id_min is not None and (
        source_min != expected_source_game_id_min
    ):
        raise ValueError(
            f"source game ID minimum differs: "
            f"{source_min} != {expected_source_game_id_min}"
        )
    if expected_source_game_id_max is not None and (
        source_max != expected_source_game_id_max
    ):
        raise ValueError(
            f"source game ID maximum differs: "
            f"{source_max} != {expected_source_game_id_max}"
        )
    if source_game_ids != set(range(source_min, source_max + 1)):
        raise ValueError(
            f"source game IDs are not continuous {source_min}..{source_max}"
        )

    reference_audit = None
    if reference is not None:
        reference_audit = _audit_reference(
            archive_paths, Path(reference), game_ids, records
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
        "one_card_records": one_card_records,
        "two_card_records": two_card_records,
        "compressed_bytes": compressed_bytes,
        "horizontal_reflections": horizontal,
        "vertical_reflections": vertical,
        "history_semantics": HISTORY_SEMANTICS_V1_ORIGINAL,
        "source_replay_audit": {
            "game_ids_continuous": True,
            "rebased_game_ids_continuous": True,
            "record_counts_match": True,
            "labels_derived_from_terminal_winners": True,
            "terminal_winners_match_replay": True,
        },
    }
    if input_canonicalization == CANONICALIZATION_REFILL_COUNT:
        result.update(refill_count_metadata())
    elif input_canonicalization == CANONICALIZATION_REFILL_COUNT_SCALAR:
        result.update(refill_count_scalar_metadata())
    if reference_audit is not None:
        result["reference_audit"] = reference_audit
    temporary_manifest = output_path / "conversion_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(
        temporary_manifest, output_path / "conversion_manifest.json"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def _audit_reference(
    archive_paths: tuple[Path, ...],
    reference: Path,
    game_ids: set[int],
    records: int,
) -> dict[str, object]:
    """Require game IDs, labels, record counts and non-history inputs to match."""
    import numpy as np

    reference_paths = tuple(sorted(reference.glob("part_*.npz")))
    if [path.name for path in archive_paths] != [
        path.name for path in reference_paths
    ]:
        raise ValueError("reference shard names differ")
    reference_game_ids: set[int] = set()
    reference_records = 0
    for actual_path, reference_path in zip(
        archive_paths, reference_paths, strict=True
    ):
        with np.load(actual_path) as actual, np.load(
            reference_path
        ) as expected:
            for key in ("board", "target", "game_id"):
                if not np.array_equal(actual[key], expected[key]):
                    raise ValueError(
                        f"reference {key} differs in {actual_path.name}"
                    )
            if not np.array_equal(
                actual["context"][:, :_CARDS_PLAYED_INDEX + 2],
                expected["context"][:, :_CARDS_PLAYED_INDEX + 2],
            ):
                raise ValueError(
                    f"reference non-history context differs in "
                    f"{actual_path.name}"
                )
            reference_records += len(expected["target"])
            reference_game_ids.update(
                int(value) for value in expected["game_id"]
            )
    if reference_records != records or reference_game_ids != game_ids:
        raise ValueError("reference game set or record count differs")
    return {
        "path": str(reference),
        "records_match": True,
        "game_ids_match": True,
        "labels_match": True,
        "board_and_non_history_context_match": True,
        "split_basis_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--game-id-rebase", type=int, default=0)
    parser.add_argument("--expected-source-game-id-min", type=int)
    parser.add_argument("--expected-source-game-id-max", type=int)
    parser.add_argument(
        "--input-canonicalization",
        choices=(
            CANONICALIZATION_NAME,
            CANONICALIZATION_REFILL_COUNT,
            CANONICALIZATION_REFILL_COUNT_SCALAR,
        ),
        default=CANONICALIZATION_NAME,
    )
    args = parser.parse_args()
    convert_replay_shards(
        args.source,
        args.output,
        expected_games=args.expected_games,
        reference=args.reference,
        game_id_rebase=args.game_id_rebase,
        expected_source_game_id_min=args.expected_source_game_id_min,
        expected_source_game_id_max=args.expected_source_game_id_max,
        input_canonicalization=args.input_canonicalization,
    )


if __name__ == "__main__":
    main()
