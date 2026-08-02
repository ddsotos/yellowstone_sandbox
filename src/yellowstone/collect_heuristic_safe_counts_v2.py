"""Fast heuristic replay collection with safe/one-off turn-start counts."""

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

from yellowstone.bots import (
    FixedFrameHandSixOneOffMinLossOneCardBot,
    HeuristicBot,
)
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.replay_v2 import (
    RULES_VERSION_V2,
    ReplayGameV2,
    write_replay_shard,
)
from yellowstone.safe_count_features import rank_color_offset_counts
from yellowstone.serialization import action_to_dict
from yellowstone.types import Phase, PlaceCardAction


COLLECTOR_NAME = "heuristic_safe_counts_rank_color_v1"
VARIANT_POLICY = "fixed_frame_hand_six_one_off_min_loss_one_card"
VARIANT_COLLECTOR_NAME = (
    "variant_board5_hand6_oneoff_tiered_minloss_onecard_rank_color_v1"
)


def _mixed_game_seed(seed: int, game_id: int) -> int:
    value = (game_id + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value ^= seed & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= value >> 31
    return value


def _add_totals(totals: dict[str, float], facts: dict[str, float]) -> None:
    for key, value in facts.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def collect_one_game(
    *,
    game_id: int,
    seed: int,
    policy: str = "heuristic",
) -> tuple[ReplayGameV2, dict[str, float]]:
    """Collect one heuristic game and record turn-start safe-count facts."""
    seed_rng = Random(_mixed_game_seed(seed, game_id))
    initial_seed = seed_rng.randrange(2**63)
    gameplay_seed = seed_rng.randrange(2**63)
    policy_seed = seed_rng.randrange(2**63)
    state = create_initial_state(4, seed=initial_seed)
    initial_state = state
    gameplay_rng = Random(gameplay_seed)
    if policy == "heuristic":
        bot = HeuristicBot()
        collector_name = COLLECTOR_NAME
        policy_name = "HeuristicBot.choose_action"
    elif policy == VARIANT_POLICY:
        bot = FixedFrameHandSixOneOffMinLossOneCardBot(rng=Random(policy_seed))
        collector_name = VARIANT_COLLECTOR_NAME
        policy_name = (
            "FixedFrameHandSixOneOffMinLossOneCardBot.choose_action_all_players"
        )
    else:
        raise ValueError(f"unknown policy: {policy!r}")
    actions = []
    decisions: list[dict[str, Any]] = []
    totals: dict[str, float] = {}

    while state.phase != Phase.GAME_OVER:
        before = state
        action = bot.choose_action(state)
        if action is None:
            raise RuntimeError("policy returned no legal action")
        if (
            isinstance(action, PlaceCardAction)
            and before.phase == Phase.PLAY
            and before.cards_played_this_turn == 0
        ):
            safe_counts, one_off_counts = rank_color_offset_counts(before)
            hand_size = len(before.players[before.current_player_index].hand)
            branch = getattr(bot, "last_branch", None)
            branch_payload = None
            selection_mode = "heuristic"
            if branch is not None:
                selection_mode = (
                    "variant_one_card_forced"
                    if branch.taken
                    else "variant_heuristic_fallback"
                )
                branch_payload = {
                    "bucket": branch.bucket,
                    "probability": branch.probability,
                    "draw": branch.draw,
                    "taken": branch.taken,
                }
            decisions.append(
                {
                    "type": "turn",
                    "game_id": game_id,
                    "player_index": before.current_player_index,
                    "starting_hand_size": hand_size,
                    "selection_mode": selection_mode,
                    "policy": policy_name,
                    "variant_branch": branch_payload,
                    "safe_one_card_counts_by_player": safe_counts,
                    "one_off_card_counts_by_player": one_off_counts,
                    "selected_actions": [action_to_dict(action)],
                }
            )
            totals["turns"] = totals.get("turns", 0.0) + 1
            totals[f"turn_start_hand_{hand_size}"] = (
                totals.get(f"turn_start_hand_{hand_size}", 0.0) + 1
            )
            if branch is not None:
                totals["variant_eligible"] = totals.get("variant_eligible", 0.0) + 1
                totals[f"variant_bucket_{branch.bucket}"] = (
                    totals.get(f"variant_bucket_{branch.bucket}", 0.0) + 1
                )
                if branch.taken:
                    totals["variant_taken"] = totals.get("variant_taken", 0.0) + 1
                    totals[f"variant_taken_bucket_{branch.bucket}"] = (
                        totals.get(f"variant_taken_bucket_{branch.bucket}", 0.0) + 1
                    )

        state = apply_known_legal_action(
            state, action, rng=gameplay_rng
        )
        actions.append(action)

    policy_hash = hashlib.sha256(collector_name.encode("utf-8")).hexdigest()
    replay = ReplayGameV2(
        game_id=game_id,
        initial_seed=initial_seed,
        gameplay_seed=gameplay_seed,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=tuple(decisions),
        winners=state.winners,
        teacher_checkpoint=collector_name,
        teacher_sha256=policy_hash,
        teacher_generation=0,
        privileged_teacher_deck=False,
        rules_version=RULES_VERSION_V2,
    )
    for player_index in range(4):
        totals[f"win_player_{player_index}"] = (
            1.0 / len(state.winners)
            if player_index in state.winners
            else 0.0
        )
    return replay, totals


def collect_heuristic_safe_counts(
    *,
    seed: int,
    game_id_offset: int,
    output: str | Path,
    stop_file: str | Path,
    status_file: str | Path | None = None,
    shard_games: int = 100,
    max_games: int | None = None,
    policy: str = "heuristic",
) -> dict[str, Any]:
    """Collect replay shards until stopped or ``max_games`` is reached."""
    if shard_games <= 0 or max_games is not None and max_games <= 0:
        raise ValueError("shard_games and max_games must be positive")
    if policy == "heuristic":
        collector_name = COLLECTOR_NAME
        policy_name = "HeuristicBot.choose_action"
    elif policy == VARIANT_POLICY:
        collector_name = VARIANT_COLLECTOR_NAME
        policy_name = (
            "FixedFrameHandSixOneOffMinLossOneCardBot.choose_action_all_players"
        )
    else:
        raise ValueError(f"unknown policy: {policy!r}")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    stop_path = Path(stop_file)
    status_path = Path(status_file) if status_file is not None else None
    manifest_path = output_path / "collection_manifest.json"
    expected = {
        "collector": collector_name,
        "seed": seed,
        "game_id_offset": game_id_offset,
        "shard_games": shard_games,
        "max_games": max_games,
        "policy": policy_name,
        "recorded_turn_features": [
            "safe_one_card_counts_by_player",
            "one_off_card_counts_by_player",
        ],
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"manifest differs at {key}")
        games = int(manifest["games"])
        completed_shards = int(manifest["completed_shards"])
        compressed_bytes = int(manifest["compressed_bytes"])
        prior_wall = float(manifest["wall_seconds"])
        totals = {
            key: float(value)
            for key, value in manifest.get("policy_totals", {}).items()
        }
        started_at = str(manifest["started_at"])
    else:
        games = completed_shards = compressed_bytes = 0
        prior_wall = 0.0
        totals: dict[str, float] = {}
        started_at = datetime.now().astimezone().isoformat()

    run_started = monotonic()
    stopped = stop_path.exists()

    def persist(status: str) -> dict[str, Any]:
        wall = prior_wall + monotonic() - run_started
        payload = {
            "schema": "yellowstone.replay.v2.heuristic_safe_counts_collection",
            **expected,
            "rules_version": RULES_VERSION_V2,
            "stop_file": str(stop_path),
            "started_at": started_at,
            "updated_at": datetime.now().astimezone().isoformat(),
            "wall_seconds": wall,
            "games": games,
            "completed_shards": completed_shards,
            "compressed_bytes": compressed_bytes,
            "policy_totals": totals,
            "status": status,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        if status_path is not None:
            _write_status(
                status_path,
                state=status,
                output=output_path,
                stop_file=stop_path,
                games=games,
                completed_shards=completed_shards,
            )
        print(
            json.dumps(
                {"games": games, "wall_seconds": wall, "status": status},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return payload

    already_complete = max_games is not None and games >= max_games
    manifest = persist(
        "stopped_by_user"
        if stopped
        else "complete"
        if already_complete
        else "running"
    )
    while not stopped and (max_games is None or games < max_games):
        shard_start = game_id_offset + games
        remaining = (
            shard_games
            if max_games is None
            else min(shard_games, max_games - games)
        )
        shard: list[ReplayGameV2] = []
        for index in range(remaining):
            if stop_path.exists():
                stopped = True
                break
            replay, facts = collect_one_game(
                game_id=shard_start + index,
                seed=seed,
                policy=policy,
            )
            shard.append(replay)
            _add_totals(totals, facts)
        if shard:
            destination = output_path / f"part_{shard_start:07d}.jsonl.gz"
            if destination.exists():
                raise FileExistsError(f"shard exists: {destination}")
            storage = write_replay_shard(shard, destination)
            games += len(shard)
            completed_shards += 1
            compressed_bytes += int(storage["compressed_bytes"])
        complete = max_games is not None and games >= max_games
        status = (
            "stopped_by_user"
            if stopped
            else "complete"
            if complete
            else "running"
        )
        manifest = persist(status)
    return manifest


def _write_status(
    path: Path,
    *,
    state: str,
    output: Path,
    stop_file: Path,
    games: int,
    completed_shards: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "step": "collect",
        "last_completed_step": "write_shard" if completed_shards else "",
        "message": "",
        "updated_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "output": str(output),
        "stop_file": str(stop_file),
        "games": games,
        "completed_shards": completed_shards,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--game-id-offset", type=int, default=1_500_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--shard-games", type=int, default=100)
    parser.add_argument("--max-games", type=int)
    parser.add_argument(
        "--policy",
        choices=("heuristic", VARIANT_POLICY),
        default="heuristic",
    )
    args = parser.parse_args()
    result = collect_heuristic_safe_counts(
        seed=args.seed,
        game_id_offset=args.game_id_offset,
        output=args.output,
        stop_file=args.stop_file,
        status_file=args.status_file,
        shard_games=args.shard_games,
        max_games=args.max_games,
        policy=args.policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
