"""V1-compatible value features with three prior completed turns."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from yellowstone.game import apply_known_legal_action
from yellowstone.types import (
    Action,
    EndTurnAction,
    GameState,
    Phase,
    PlaceCardAction,
    RefillAction,
)
from yellowstone.value_canonicalization import (
    canonicalize_value_tensors_with_stats,
)
from yellowstone.value_learning import (
    COLOR_ORDER,
    HAND_SIZE,
    RecentPlacement,
    ValueRecord,
    board_tensor_for_player,
)


VALUE_SCHEMA_HISTORY3 = "yellowstone.value.v1_history3"
CANONICALIZATION_HISTORY3 = "fast_lr_ud_color_v1_history3"
HISTORY_TURNS = 3
PLACEMENTS_PER_TURN = 2
PLACEMENT_FEATURES = 12
BASE_CONTEXT_SIZE = 57
VALUE_CONTEXT_SIZE_HISTORY3 = (
    BASE_CONTEXT_SIZE
    + HISTORY_TURNS * PLACEMENTS_PER_TURN * PLACEMENT_FEATURES
)


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    placements: tuple[RecentPlacement, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.placements) <= PLACEMENTS_PER_TURN:
            raise ValueError("history turn must contain one or two placements")


@dataclass(frozen=True, slots=True)
class ValueRecordHistory3:
    game_id: int
    perspective_player_index: int
    state: GameState
    history_before_turn: tuple[HistoryTurn, ...]
    target: float

    def __post_init__(self) -> None:
        if len(self.history_before_turn) > HISTORY_TURNS:
            raise ValueError("history contains more than three turns")


class History3Tracker:
    """Retain placement details for the last three completed turns."""

    def __init__(self) -> None:
        self._history: list[HistoryTurn] = []
        self._current: list[RecentPlacement] = []
        self._active_player: int | None = None

    def snapshot(self) -> tuple[HistoryTurn, ...]:
        return tuple(self._history)

    def observe(
        self, before: GameState, action: Action, after: GameState
    ) -> None:
        if isinstance(action, PlaceCardAction):
            player_index = before.current_player_index
            if self._active_player is None:
                self._active_player = player_index
            elif self._active_player != player_index:
                raise AssertionError("turn player changed during placement")
            card = before.players[player_index].hand[action.hand_index]
            self._current.append(
                RecentPlacement(
                    player_index=player_index,
                    card=card,
                    score_delta=(
                        before.players[player_index].loss_score
                        - after.players[player_index].loss_score
                    ),
                    negative_card_delta=(
                        len(after.players[player_index].negative_cards)
                        - len(before.players[player_index].negative_cards)
                    ),
                )
            )
            return
        if isinstance(action, EndTurnAction):
            if after.phase != Phase.REFILL:
                self._finish()
            return
        if isinstance(action, RefillAction) and self._active_player is not None:
            self._finish()

    def _finish(self) -> None:
        if self._active_player is None or not self._current:
            raise AssertionError("cannot finish an empty tracked turn")
        self._history.append(HistoryTurn(tuple(self._current)))
        del self._history[:-HISTORY_TURNS]
        self._current = []
        self._active_player = None


def context_tensor_history3(record: ValueRecordHistory3):
    """Encode V1 state features plus three fixed two-slot history turns."""
    import numpy as np

    state = record.state
    viewer = record.perspective_player_index
    values: list[float] = []
    own_hand = state.players[viewer].hand
    for slot in range(HAND_SIZE):
        if slot < len(own_hand):
            card = own_hand[slot]
            values.extend(
                [
                    1.0,
                    *_one_hot(COLOR_ORDER.index(card.color), 4),
                    card.rank_index / 6,
                ]
            )
        else:
            values.extend([0.0] * 6)
    for offset in range(4):
        player = state.players[(viewer + offset) % 4]
        values.extend(
            [
                player.loss_score / 35,
                len(player.hand) / 6,
                len(player.negative_cards) / 56,
            ]
        )
    values.extend(
        _one_hot((state.current_player_index - viewer) % 4, 4)
    )
    values.extend(
        _one_hot(
            (Phase.PLAY, Phase.REFILL, Phase.GAME_OVER).index(state.phase),
            3,
        )
    )
    values.extend(
        [state.cards_played_this_turn / 2, state.settlement_count / 10]
    )

    history = record.history_before_turn[-HISTORY_TURNS:]
    values.extend(
        [0.0]
        * (
            (HISTORY_TURNS - len(history))
            * PLACEMENTS_PER_TURN
            * PLACEMENT_FEATURES
        )
    )
    for turn in history:
        for placement in turn.placements:
            values.extend(_placement_values(placement, viewer))
        values.extend(
            [0.0]
            * (
                (PLACEMENTS_PER_TURN - len(turn.placements))
                * PLACEMENT_FEATURES
            )
        )
    if len(values) != VALUE_CONTEXT_SIZE_HISTORY3:
        raise AssertionError(f"unexpected history3 context: {len(values)}")
    return np.asarray(values, dtype=np.float32)


def board_tensor_history3(record: ValueRecordHistory3):
    return board_tensor_for_player(
        ValueRecord(
            game_id=record.game_id,
            perspective_player_index=record.perspective_player_index,
            state=record.state,
            history=(),
            target=record.target,
        )
    )


def transform_v1_to_history3(
    source: str | Path,
    output: str | Path,
    *,
    start_part: int,
    end_part: int,
    expected_games: int | None = None,
) -> dict[str, object]:
    """Rebuild prior-three-turn history from chronological raw V1 records."""
    import numpy as np

    from yellowstone.train_value import _archive_paths

    source_path = Path(source)
    output_path = Path(output)
    paths = _archive_paths(
        source_path, start_part=start_part, end_part=end_part
    )
    output_path.mkdir(parents=True, exist_ok=True)
    game_ids: set[int] = set()
    records = one_card_turns = two_card_turns = 0
    horizontal = vertical = converted = skipped = 0

    for index, path in enumerate(paths, start=1):
        destination = output_path / path.name
        if destination.is_file():
            skipped += 1
            with np.load(destination) as archive:
                game_ids.update(int(value) for value in archive["game_id"])
                records += len(archive["target"])
            continue

        with np.load(path) as archive:
            source_context = archive["context"]
            if source_context.shape[1] != 81:
                raise ValueError(
                    f"expected raw V1 context in {path}: "
                    f"{source_context.shape}"
                )
            context = np.zeros(
                (len(source_context), VALUE_CONTEXT_SIZE_HISTORY3),
                dtype=np.float32,
            )
            context[:, :BASE_CONTEXT_SIZE] = source_context[
                :, :BASE_CONTEXT_SIZE
            ]
            histories: dict[int, list[tuple[object, ...]]] = {}
            turn_counts: dict[int, int] = {}
            for row, game_id_value in enumerate(archive["game_id"]):
                game_id = int(game_id_value)
                history = histories.setdefault(game_id, [])
                current_player = turn_counts.get(game_id, 0) % 4
                _write_encoded_history(
                    context[row], history, current_player=current_player
                )
                current_turn = _extract_current_turn(source_context[row])
                if len(current_turn) == 1:
                    one_card_turns += 1
                else:
                    two_card_turns += 1
                history.append(current_turn)
                del history[:-HISTORY_TURNS]
                turn_counts[game_id] = turn_counts.get(game_id, 0) + 1

            board, context, stats = (
                canonicalize_value_tensors_with_stats(
                    archive["board"], context
                )
            )
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                board=board,
                context=context,
                target=archive["target"],
                game_id=archive["game_id"],
            )
            game_ids.update(int(value) for value in archive["game_id"])
            records += len(archive["target"])
        os.replace(temporary, destination)
        converted += 1
        horizontal += stats.horizontal_reflections
        vertical += stats.vertical_reflections
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(
                f"progress={index}/{len(paths)} "
                f"converted={converted} skipped={skipped}",
                flush=True,
            )

    if expected_games is not None and len(game_ids) != expected_games:
        raise ValueError(
            f"history3 game count differs: "
            f"{len(game_ids)} != {expected_games}"
        )
    result: dict[str, object] = {
        "value_schema": VALUE_SCHEMA_HISTORY3,
        "canonicalization": CANONICALIZATION_HISTORY3,
        "history_semantics": "three_prior_completed_turns_two_slots_each",
        "source": str(source_path),
        "output": str(output_path),
        "start_part": start_part,
        "end_part": end_part,
        "source_files": len(paths),
        "converted_files": converted,
        "skipped_files": skipped,
        "games": len(game_ids),
        "records": records,
        "one_card_turns": one_card_turns,
        "two_card_turns": two_card_turns,
        "horizontal_reflections": horizontal,
        "vertical_reflections": vertical,
        "context_size": VALUE_CONTEXT_SIZE_HISTORY3,
    }
    temporary_manifest = output_path / "conversion_manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(
        temporary_manifest, output_path / "conversion_manifest.json"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _extract_current_turn(context) -> tuple[object, ...]:
    import numpy as np

    slots = [
        np.array(context[offset : offset + PLACEMENT_FEATURES], copy=True)
        for offset in (
            BASE_CONTEXT_SIZE,
            BASE_CONTEXT_SIZE + PLACEMENT_FEATURES,
        )
        if context[offset] > 0.5
    ]
    play_count = 2 if context[55] > 0.75 else 1
    if len(slots) < play_count:
        raise ValueError("V1 record lacks current placement history")
    return tuple(slots[-play_count:])


def _write_encoded_history(
    destination,
    history: list[tuple[object, ...]],
    *,
    current_player: int,
) -> None:
    first_turn = HISTORY_TURNS - len(history)
    for turn_index, placements in enumerate(history):
        age = len(history) - turn_index
        player = (current_player - age) % 4
        relative_player = (player - current_player) % 4
        turn_offset = (
            BASE_CONTEXT_SIZE
            + (first_turn + turn_index)
            * PLACEMENTS_PER_TURN
            * PLACEMENT_FEATURES
        )
        for placement_index, placement in enumerate(placements):
            offset = turn_offset + placement_index * PLACEMENT_FEATURES
            destination[offset : offset + PLACEMENT_FEATURES] = placement
            destination[offset + 1 : offset + 5] = 0.0
            destination[offset + 1 + relative_player] = 1.0


def _placement_values(
    placement: RecentPlacement, viewer: int
) -> list[float]:
    return [
        1.0,
        *_one_hot((placement.player_index - viewer) % 4, 4),
        *_one_hot(COLOR_ORDER.index(placement.card.color), 4),
        placement.card.rank_index / 6,
        placement.score_delta / 3,
        placement.negative_card_delta / 9,
    ]


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if index == value else 0.0 for value in range(size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-part", type=int, required=True)
    parser.add_argument("--end-part", type=int, required=True)
    parser.add_argument("--expected-games", type=int)
    args = parser.parse_args()
    transform_v1_to_history3(
        args.source,
        args.output,
        start_part=args.start_part,
        end_part=args.end_part,
        expected_games=args.expected_games,
    )


if __name__ == "__main__":
    main()
