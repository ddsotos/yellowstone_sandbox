"""Fast replay collection with heuristic placement and simple refill choice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from random import Random
from time import monotonic
from typing import Any

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.replay_v2 import (
    RULES_VERSION_V2,
    ReplayGameV2,
    write_replay_shard,
)
from yellowstone.types import (
    Phase,
    RefillAction,
    RefillSource,
)


COLLECTOR_NAME = "simple_refill_middle_345_epsilon20_v1"
MIDDLE_RANK_INDICES = frozenset((2, 3, 4))
NEGATIVE_PROBABILITY_WHEN_PREFERRED = 0.8
NEGATIVE_PROBABILITY_WHEN_NOT_PREFERRED = 0.2


def negative_refill_probability(negative_cards) -> float:
    """Return the simple-policy probability of choosing negative cards."""
    if len(negative_cards) < 6:
        return 0.0
    middle_count = sum(
        card.rank_index in MIDDLE_RANK_INDICES for card in negative_cards
    )
    preferred = middle_count * 2 >= len(negative_cards)
    return (
        NEGATIVE_PROBABILITY_WHEN_PREFERRED
        if preferred
        else NEGATIVE_PROBABILITY_WHEN_NOT_PREFERRED
    )


def collect_one_game(
    *,
    game_id: int,
    seed: int,
) -> ReplayGameV2:
    """Collect one complete game without value-model inference."""
    seed_rng = Random(_mixed_game_seed(seed, game_id))
    initial_seed = seed_rng.randrange(2**63)
    gameplay_seed = seed_rng.randrange(2**63)
    decision_seed = seed_rng.randrange(2**63)
    state = create_initial_state(4, seed=initial_seed)
    initial_state = state
    gameplay_rng = Random(gameplay_seed)
    decision_rng = Random(decision_seed)
    heuristic = HeuristicBot()
    actions = []
    decisions: list[dict[str, Any]] = []

    while state.phase != Phase.GAME_OVER:
        player_index = state.current_player_index
        player = state.players[player_index]
        if state.phase == Phase.REFILL and not player.hand:
            negative_probability = negative_refill_probability(
                player.negative_cards
            )
            if negative_probability > 0:
                draw = decision_rng.random()
                source = (
                    RefillSource.NEGATIVE_CARDS
                    if draw < negative_probability
                    else RefillSource.DECK
                )
                action = RefillAction(source)
                middle_count = sum(
                    card.rank_index in MIDDLE_RANK_INDICES
                    for card in player.negative_cards
                )
                decisions.append(
                    {
                        "game_id": game_id,
                        "player_index": player_index,
                        "negative_count": len(player.negative_cards),
                        "middle_345_count": middle_count,
                        "middle_345_fraction": (
                            middle_count / len(player.negative_cards)
                        ),
                        "negative_refill_probability": negative_probability,
                        "random_draw": draw,
                        "selected_source": source.value,
                    }
                )
            else:
                action = RefillAction(RefillSource.DECK)
        else:
            action = heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("heuristic returned no legal action")

        before = state
        state = apply_known_legal_action(state, action, rng=gameplay_rng)
        actions.append(action)
        if before.phase == Phase.GAME_OVER:
            raise AssertionError("action applied after game over")

    policy_hash = hashlib.sha256(COLLECTOR_NAME.encode("utf-8")).hexdigest()
    return ReplayGameV2(
        game_id=game_id,
        initial_seed=initial_seed,
        gameplay_seed=gameplay_seed,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=tuple(decisions),
        winners=state.winners,
        teacher_checkpoint=COLLECTOR_NAME,
        teacher_sha256=policy_hash,
        teacher_generation=0,
        privileged_teacher_deck=False,
        rules_version=RULES_VERSION_V2,
    )


def collect_for_duration(
    *,
    duration_hours: float,
    seed: int,
    game_id_offset: int,
    output: str | Path,
    shard_games: int = 100,
) -> dict[str, Any]:
    """Collect complete games for approximately the requested wall duration."""
    if duration_hours <= 0:
        raise ValueError("duration_hours must be positive")
    if shard_games <= 0:
        raise ValueError("shard_games must be positive")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "collection_manifest.json"
    target_seconds = duration_hours * 3600
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "collector": COLLECTOR_NAME,
            "seed": seed,
            "game_id_offset": game_id_offset,
            "target_seconds": target_seconds,
            "shard_games": shard_games,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"existing manifest differs at {key}: "
                    f"{manifest.get(key)!r} != {value!r}"
                )
        games = int(manifest["games"])
        completed_shards = int(manifest["completed_shards"])
        compressed_bytes = int(manifest["compressed_bytes"])
        prior_wall_seconds = float(manifest["wall_seconds"])
        refill_decisions = int(manifest["refill_decisions"])
        negative_refills = int(manifest["negative_refills"])
        deck_refills = int(manifest["deck_refills"])
        started_at = str(manifest["started_at"])
    else:
        games = completed_shards = compressed_bytes = 0
        prior_wall_seconds = 0.0
        refill_decisions = negative_refills = deck_refills = 0
        started_at = datetime.now().astimezone().isoformat()

    run_started = monotonic()
    while prior_wall_seconds + (monotonic() - run_started) < target_seconds:
        shard_start_id = game_id_offset + games
        shard = []
        for index in range(shard_games):
            if (
                index > 0
                and prior_wall_seconds + (monotonic() - run_started)
                >= target_seconds
            ):
                break
            shard.append(
                collect_one_game(
                    game_id=shard_start_id + index,
                    seed=seed,
                )
            )
        if not shard:
            break
        destination = output_path / f"part_{shard_start_id:06d}.jsonl.gz"
        if destination.exists():
            raise FileExistsError(f"replay shard already exists: {destination}")
        storage = write_replay_shard(shard, destination)
        for game in shard:
            for decision in game.decisions:
                refill_decisions += 1
                if decision["selected_source"] == RefillSource.NEGATIVE_CARDS.value:
                    negative_refills += 1
                else:
                    deck_refills += 1
        games += len(shard)
        completed_shards += 1
        compressed_bytes += int(storage["compressed_bytes"])
        wall_seconds = prior_wall_seconds + (monotonic() - run_started)
        manifest = {
            "schema": "yellowstone.replay.v2.simple_refill_collection",
            "collector": COLLECTOR_NAME,
            "rules_version": RULES_VERSION_V2,
            "seed": seed,
            "game_id_offset": game_id_offset,
            "target_hours": duration_hours,
            "target_seconds": target_seconds,
            "shard_games": shard_games,
            "started_at": started_at,
            "updated_at": datetime.now().astimezone().isoformat(),
            "wall_seconds": wall_seconds,
            "games": games,
            "completed_shards": completed_shards,
            "compressed_bytes": compressed_bytes,
            "refill_decisions": refill_decisions,
            "negative_refills": negative_refills,
            "deck_refills": deck_refills,
            "status": (
                "complete"
                if wall_seconds >= target_seconds
                else "running"
            ),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        print(
            json.dumps(
                {
                    "games": games,
                    "wall_seconds": wall_seconds,
                    "completed_shard": str(destination),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return manifest


def _mixed_game_seed(seed: int, game_id: int) -> int:
    value = (game_id + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value ^= seed & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-hours", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--game-id-offset", type=int, default=231700)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-games", type=int, default=100)
    args = parser.parse_args()
    result = collect_for_duration(
        duration_hours=args.duration_hours,
        seed=args.seed,
        game_id_offset=args.game_id_offset,
        output=args.output,
        shard_games=args.shard_games,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
