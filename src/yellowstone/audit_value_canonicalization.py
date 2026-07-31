"""Measure how many canonical forms remain across the 96 supported symmetries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import permutations
from pathlib import Path

from yellowstone.symmetry import transform_state
from yellowstone.types import Card
from yellowstone.value_canonicalization import (
    CANONICALIZATION_NAME,
    canonicalize_value_tensors,
)
from yellowstone.value_learning import (
    COLOR_ORDER,
    RecentPlacement,
    ValueRecord,
    board_tensor_for_player,
    collect_heuristic_games,
    context_tensor_for_player,
)


def audit_canonicalization(
    *, games: int, seed: int, max_records: int
) -> dict[str, object]:
    import numpy as np

    records = collect_heuristic_games(game_count=games, seed=seed)[:max_records]
    distribution: Counter[int] = Counter()
    for record in records:
        boards = []
        contexts = []
        for permuted in permutations(COLOR_ORDER):
            color_map = dict(zip(COLOR_ORDER, permuted, strict=True))
            for horizontal in (False, True):
                for vertical in (False, True):
                    transformed = _transform_record(
                        record,
                        color_map=color_map,
                        horizontal=horizontal,
                        vertical=vertical,
                    )
                    boards.append(board_tensor_for_player(transformed))
                    contexts.append(context_tensor_for_player(transformed))
        canonical_board, canonical_context = canonicalize_value_tensors(
            np.stack(boards), np.stack(contexts)
        )
        forms = {
            (canonical_board[index].tobytes(), canonical_context[index].tobytes())
            for index in range(len(canonical_board))
        }
        distribution[len(forms)] += 1
    collapsed = distribution.get(1, 0)
    return {
        "canonicalization": CANONICALIZATION_NAME,
        "games": games,
        "seed": seed,
        "records": len(records),
        "fully_collapsed_records": collapsed,
        "fully_collapsed_rate": collapsed / len(records) if records else 0.0,
        "canonical_form_count_distribution": {
            str(key): value for key, value in sorted(distribution.items())
        },
    }


def _transform_record(
    record: ValueRecord,
    *,
    color_map,
    horizontal: bool,
    vertical: bool,
) -> ValueRecord:
    def transform_card(card: Card) -> Card:
        return Card(
            color=color_map[card.color],
            rank_index=6 - card.rank_index if vertical else card.rank_index,
        )

    return ValueRecord(
        game_id=record.game_id,
        perspective_player_index=record.perspective_player_index,
        state=transform_state(
            record.state,
            color_map=color_map,
            horizontal_reflection=horizontal,
            vertical_reflection=vertical,
        ),
        history=tuple(
            RecentPlacement(
                player_index=item.player_index,
                card=transform_card(item.card),
                score_delta=item.score_delta,
                negative_card_delta=item.negative_card_delta,
            )
            for item in record.history
        ),
        target=record.target,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_canonicalization(
        games=args.games, seed=args.seed, max_records=args.max_records
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
