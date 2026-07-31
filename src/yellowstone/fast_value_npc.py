"""Fast two-candidate V1 NPC, evaluation, and replay collection."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from time import monotonic
from typing import Any

from yellowstone.bots import HeuristicBot, placement_sort_key
from yellowstone.collect_simple_refill_v2 import (
    negative_refill_probability,
)
from yellowstone.game import (
    apply_known_legal_action,
    create_initial_state,
    legal_actions,
)
from yellowstone.replay_v2 import (
    RULES_VERSION_V2,
    ReplayGameV2,
    file_sha256,
    write_replay_shard,
)
from yellowstone.serialization import action_to_dict
from yellowstone.types import (
    Action,
    GameState,
    Phase,
    PlaceCardAction,
    RefillAction,
    RefillSource,
)
from yellowstone.value_learning import HISTORY_SIZE, RecentPlacement
from yellowstone.value_policy import (
    TorchWinValueEstimator,
    TurnCandidate,
    enumerate_loss_safe_turn_pools,
    enumerate_turn_end_candidates,
)


POLICY_NAME = "fast_v1_min_negative_max_bonus_v1"
MODE_TWO = "two"
MODE_EIGHT = "eight"
MODE_FULL = "full"
MODE_HEURISTIC_ONE_VS_TWO = "heuristic-one-vs-two"
SUPPORTED_MODES = (
    MODE_TWO,
    MODE_EIGHT,
    MODE_FULL,
    MODE_HEURISTIC_ONE_VS_TWO,
)
EXPLORATION_PROBABILITY = 0.10
NO_REFILL_PROBABILITY = 0.05


@dataclass(frozen=True, slots=True)
class FastTurnChoice:
    actions: tuple[Action, ...]
    scores: tuple[float, ...]
    selected_index: int
    selection_mode: str
    candidate_mode: str
    one_pool_size: int
    two_pool_size: int
    enumerated_candidate_count: int
    enumeration_seconds: float
    inference_seconds: float


class FastValueNpc:
    """Choose from loss-safe one/two-card representatives using V1."""

    def __init__(self, checkpoint: str | Path, *, mode: str = MODE_TWO):
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported fast NPC mode: {mode}")
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = file_sha256(checkpoint)
        self.mode = mode
        self.estimator = TorchWinValueEstimator(str(checkpoint))

    def choose_turn(
        self,
        state: GameState,
        history: tuple[RecentPlacement, ...],
        *,
        rng: Random,
    ) -> FastTurnChoice:
        started = monotonic()
        if self.mode == MODE_FULL:
            all_candidates = _dedupe_public_results(
                enumerate_turn_end_candidates(
                    state,
                    history=history,
                    approximate_new_color_neighbor_limit=True,
                    collapse_equivalent_frames=True,
                )
            )
            one = _best_safe_pool(
                state, all_candidates, play_count=1
            )
            two = _best_safe_pool(
                state, all_candidates, play_count=2
            )
            enumerated_candidate_count = len(all_candidates)
        else:
            pools = enumerate_loss_safe_turn_pools(
                state,
                history=history,
                approximate_new_color_neighbor_limit=True,
            )
            one = pools.one_card_candidates
            two = pools.two_card_candidates
            all_candidates = ()
            enumerated_candidate_count = pools.enumerated_candidate_count
        if not one and not two:
            raise RuntimeError("fast NPC found no turn candidate")
        if self.mode == MODE_HEURISTIC_ONE_VS_TWO:
            safe_candidates = tuple(
                candidate
                for pool in (one, two)
                if (
                    candidate := _heuristic_representative(
                        state, pool
                    )
                )
                is not None
            )
        else:
            safe_limit = 1 if self.mode == MODE_TWO else 4
            safe_candidates = (
                *_sample_without_replacement(one, safe_limit, rng),
                *_sample_without_replacement(two, safe_limit, rng),
            )
        candidates = (
            all_candidates if self.mode == MODE_FULL else safe_candidates
        )
        enumeration_seconds = monotonic() - started

        inference_started = monotonic()
        scores = self.estimator.estimate_many(
            tuple(candidate.record for candidate in candidates)
        )
        inference_seconds = monotonic() - inference_started
        draw = (
            1.0
            if self.mode == MODE_HEURISTIC_ONE_VS_TWO
            else rng.random()
        )
        if draw < EXPLORATION_PROBABILITY:
            exploration_candidates = (
                safe_candidates if self.mode == MODE_FULL else candidates
            )
            selected_candidate = rng.choice(exploration_candidates)
            selected_index = candidates.index(selected_candidate)
            selection_mode = "random_safe"
        else:
            selected_index = max(
                range(len(candidates)), key=lambda index: scores[index]
            )
            selection_mode = "max_value"
        return FastTurnChoice(
            actions=candidates[selected_index].actions,
            scores=tuple(float(value) for value in scores),
            selected_index=selected_index,
            selection_mode=selection_mode,
            candidate_mode=self.mode,
            one_pool_size=len(one),
            two_pool_size=len(two),
            enumerated_candidate_count=enumerated_candidate_count,
            enumeration_seconds=enumeration_seconds,
            inference_seconds=inference_seconds,
        )


def _heuristic_representative(
    state: GameState,
    candidates: tuple[TurnCandidate, ...],
) -> TurnCandidate | None:
    """Return the loss-safe candidate preferred by heuristic tie-breaks."""
    if not candidates:
        return None

    def key(candidate: TurnCandidate) -> tuple[tuple[int, ...], ...]:
        working = state
        placement_keys: list[tuple[int, ...]] = []
        for action in candidate.actions:
            if isinstance(action, PlaceCardAction):
                placement_keys.append(
                    placement_sort_key(working, action)
                )
            working = apply_known_legal_action(working, action)
        return tuple(placement_keys)

    return min(candidates, key=key)


def _best_safe_pool(
    state: GameState,
    candidates: tuple[TurnCandidate, ...],
    *,
    play_count: int,
) -> tuple[TurnCandidate, ...]:
    player_index = state.current_player_index
    starting_player = state.players[player_index]
    matching = tuple(
        candidate
        for candidate in candidates
        if _play_count(candidate) == play_count
    )
    if not matching:
        return ()
    losses = tuple(
        len(candidate.record.state.players[player_index].negative_cards)
        - len(starting_player.negative_cards)
        for candidate in matching
    )
    minimum_loss = min(losses)
    loss_safe = tuple(
        candidate
        for candidate, loss in zip(matching, losses, strict=True)
        if loss == minimum_loss
    )
    bonuses = tuple(
        starting_player.loss_score
        - candidate.record.state.players[player_index].loss_score
        for candidate in loss_safe
    )
    maximum_bonus = max(bonuses)
    return tuple(
        candidate
        for candidate, bonus in zip(loss_safe, bonuses, strict=True)
        if bonus == maximum_bonus
    )


def _dedupe_public_results(
    candidates: tuple[TurnCandidate, ...],
) -> tuple[TurnCandidate, ...]:
    unique: dict[tuple[object, ...], TurnCandidate] = {}
    for candidate in candidates:
        state = candidate.record.state
        player = state.players[candidate.record.perspective_player_index]
        board = tuple(
            sorted(
                (
                    position.x,
                    position.y,
                    tuple(
                        (card.color.value, card.rank_index)
                        for card in stack
                    ),
                )
                for position, stack in state.board.items()
            )
        )
        key = (
            board,
            tuple(
                (card.color.value, card.rank_index) for card in player.hand
            ),
            tuple(
                (card.color.value, card.rank_index)
                for card in player.negative_cards
            ),
            player.loss_score,
            state.current_player_index,
            state.phase.value,
            state.cards_played_this_turn,
            state.settlement_count,
        )
        unique.setdefault(key, candidate)
    return tuple(unique.values())


def _sample_without_replacement(
    candidates: tuple[TurnCandidate, ...], count: int, rng: Random
) -> tuple[TurnCandidate, ...]:
    if len(candidates) <= count:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return tuple(shuffled)
    return tuple(rng.sample(candidates, count))


def _play_count(candidate: TurnCandidate) -> int:
    return sum(
        isinstance(action, PlaceCardAction)
        for action in candidate.actions
    )


def choose_refill(
    state: GameState, *, rng: Random
) -> tuple[RefillAction, dict[str, Any]]:
    """Apply 5% NONE when legal, otherwise the established simple policy."""
    actions = legal_actions(state)
    legal_refills = tuple(
        action for action in actions if isinstance(action, RefillAction)
    )
    if not legal_refills:
        raise RuntimeError("refill choice requested without legal refill")
    player = state.players[state.current_player_index]
    none_action = RefillAction(RefillSource.NONE)
    if none_action in legal_refills:
        draw = rng.random()
        selected = (
            none_action
            if draw < NO_REFILL_PROBABILITY
            else RefillAction(RefillSource.DECK)
        )
        return selected, {
            "type": "refill",
            "eligible_no_refill": True,
            "no_refill_probability": NO_REFILL_PROBABILITY,
            "random_draw": draw,
            "selected_source": selected.source.value,
        }

    probability = negative_refill_probability(player.negative_cards)
    draw = rng.random() if probability > 0 else None
    selected = (
        RefillAction(RefillSource.NEGATIVE_CARDS)
        if draw is not None and draw < probability
        else RefillAction(RefillSource.DECK)
    )
    return selected, {
        "type": "refill",
        "eligible_no_refill": False,
        "negative_refill_probability": probability,
        "random_draw": draw,
        "selected_source": selected.source.value,
    }


def play_one_game(
    npc: FastValueNpc,
    *,
    game_id: int,
    seed: int,
    npc_players: frozenset[int],
    initial_seed: int | None = None,
    gameplay_seed: int | None = None,
    decision_seed: int | None = None,
) -> tuple[ReplayGameV2, dict[str, float]]:
    if initial_seed is None or gameplay_seed is None:
        seed_rng = Random(_mixed_game_seed(seed, game_id))
        initial_seed = seed_rng.randrange(2**63)
        gameplay_seed = seed_rng.randrange(2**63)
    if decision_seed is None:
        decision_seed = _mixed_game_seed(
            initial_seed ^ gameplay_seed ^ seed, game_id
        )
    state = create_initial_state(4, seed=initial_seed)
    initial_state = state
    gameplay_rng = Random(gameplay_seed)
    decision_rng = Random(decision_seed)
    heuristic = HeuristicBot()
    history: list[RecentPlacement] = []
    actions: list[Action] = []
    decisions: list[dict[str, Any]] = []
    planned: list[Action] = []
    planned_player: int | None = None
    totals = {
        "npc_turns": 0.0,
        "enumeration_seconds": 0.0,
        "inference_seconds": 0.0,
        "random_safe_turns": 0.0,
        "one_card_turns": 0.0,
        "two_card_turns": 0.0,
        "eligible_no_refills": 0.0,
        "selected_no_refills": 0.0,
        "candidate_count_sum": 0.0,
    }

    while state.phase != Phase.GAME_OVER:
        player_index = state.current_player_index
        if planned:
            if player_index != planned_player:
                raise AssertionError("planned player changed")
            action = planned.pop(0)
        elif (
            player_index in npc_players
            and state.phase == Phase.PLAY
            and state.cards_played_this_turn == 0
            and any(
                isinstance(action, PlaceCardAction)
                for action in legal_actions(state)
            )
        ):
            choice = npc.choose_turn(
                state, tuple(history), rng=decision_rng
            )
            planned = list(choice.actions)
            planned_player = player_index
            action = planned.pop(0)
            play_count = sum(
                isinstance(item, PlaceCardAction)
                for item in choice.actions
            )
            totals["npc_turns"] += 1
            totals["enumeration_seconds"] += choice.enumeration_seconds
            totals["inference_seconds"] += choice.inference_seconds
            totals["random_safe_turns"] += int(
                choice.selection_mode == "random_safe"
            )
            totals["one_card_turns"] += int(play_count == 1)
            totals["two_card_turns"] += int(play_count == 2)
            totals["candidate_count_sum"] += len(choice.scores)
            decisions.append(
                {
                    "type": "turn",
                    "game_id": game_id,
                    "player_index": player_index,
                    "candidate_mode": choice.candidate_mode,
                    "selection_mode": choice.selection_mode,
                    "one_pool_size": choice.one_pool_size,
                    "two_pool_size": choice.two_pool_size,
                    "enumerated_candidate_count": (
                        choice.enumerated_candidate_count
                    ),
                    "scores": list(choice.scores),
                    "selected_index": choice.selected_index,
                    "selected_actions": [
                        action_to_dict(item) for item in choice.actions
                    ],
                    "enumeration_seconds": choice.enumeration_seconds,
                    "inference_seconds": choice.inference_seconds,
                }
            )
        elif player_index in npc_players and (
            state.phase == Phase.REFILL
            or not any(
                isinstance(item, PlaceCardAction)
                for item in legal_actions(state)
            )
        ):
            action, audit = choose_refill(state, rng=decision_rng)
            totals["eligible_no_refills"] += int(
                audit["eligible_no_refill"]
            )
            totals["selected_no_refills"] += int(
                audit["selected_source"] == RefillSource.NONE.value
            )
            decisions.append(
                {"game_id": game_id, "player_index": player_index, **audit}
            )
        else:
            action = heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("heuristic returned no action")

        if action not in legal_actions(state):
            raise RuntimeError(f"NPC selected illegal action: {action!r}")
        before = state
        state = apply_known_legal_action(
            state, action, rng=gameplay_rng
        )
        actions.append(action)
        _append_history(history, before, action, state)
        if (
            before.current_player_index != state.current_player_index
            or state.phase == Phase.GAME_OVER
        ):
            planned = []
            planned_player = None

    replay = ReplayGameV2(
        game_id=game_id,
        initial_seed=initial_seed,
        gameplay_seed=gameplay_seed,
        initial_state=initial_state,
        actions=tuple(actions),
        decisions=tuple(decisions),
        winners=state.winners,
        teacher_checkpoint=npc.checkpoint,
        teacher_sha256=npc.checkpoint_sha256,
        teacher_generation=0,
        privileged_teacher_deck=False,
        rules_version=RULES_VERSION_V2,
    )
    totals["win_player_0"] = (
        1.0 / len(state.winners) if 0 in state.winners else 0.0
    )
    return replay, totals


def benchmark(
    checkpoint: str | Path,
    *,
    mode: str,
    games: int,
    seed: int,
    all_npc: bool,
) -> dict[str, Any]:
    npc = FastValueNpc(checkpoint, mode=mode)
    started = monotonic()
    seed_rng = Random(seed)
    totals: dict[str, float] = {}
    compressed_bytes = 0
    for game_id in range(games):
        initial_seed = seed_rng.randrange(2**63)
        gameplay_seed = seed_rng.randrange(2**63)
        replay, facts = play_one_game(
            npc,
            game_id=game_id,
            seed=seed,
            npc_players=(
                frozenset(range(4)) if all_npc else frozenset((0,))
            ),
            initial_seed=initial_seed,
            gameplay_seed=gameplay_seed,
        )
        compressed_bytes += len(
            json.dumps(
                {
                    "actions": [
                        action_to_dict(action) for action in replay.actions
                    ],
                    "decisions": replay.decisions,
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )
        _add_totals(totals, facts)
    elapsed = monotonic() - started
    npc_turns = totals.get("npc_turns", 0.0)
    return {
        "games": games,
        "mode": mode,
        "all_npc": all_npc,
        "seconds": elapsed,
        "seconds_per_game": elapsed / games,
        "estimated_games_per_hour": games * 3600 / elapsed,
        "npc_turns": npc_turns,
        "mean_scored_candidates": (
            totals.get("candidate_count_sum", 0.0) / npc_turns
            if npc_turns
            else 0.0
        ),
        "enumeration_seconds": totals.get(
            "enumeration_seconds", 0.0
        ),
        "inference_seconds": totals.get("inference_seconds", 0.0),
        "one_card_turns": totals.get("one_card_turns", 0.0),
        "two_card_turns": totals.get("two_card_turns", 0.0),
        "random_safe_turns": totals.get("random_safe_turns", 0.0),
        "eligible_no_refills": totals.get(
            "eligible_no_refills", 0.0
        ),
        "selected_no_refills": totals.get(
            "selected_no_refills", 0.0
        ),
        "estimated_uncompressed_bytes_per_game": (
            compressed_bytes / games
        ),
    }


def evaluate(
    checkpoint: str | Path,
    *,
    mode: str,
    games: int,
    seed: int,
    player_index: int = 0,
) -> dict[str, Any]:
    npc = FastValueNpc(checkpoint, mode=mode)
    wins = 0.0
    totals: dict[str, float] = {}
    started = monotonic()
    seed_rng = Random(seed)
    for local_game in range(games):
        initial_seed = seed_rng.randrange(2**63)
        gameplay_seed = seed_rng.randrange(2**63)
        replay, facts = play_one_game(
            npc,
            game_id=local_game,
            seed=seed,
            npc_players=frozenset((player_index,)),
            initial_seed=initial_seed,
            gameplay_seed=gameplay_seed,
        )
        if player_index in replay.winners:
            wins += 1.0 / len(replay.winners)
        _add_totals(totals, facts)
    seconds = monotonic() - started
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "seed": seed,
        "player_index": player_index,
        "mode": mode,
        "seconds": seconds,
        "seconds_per_game": seconds / games,
        "npc_turns": totals.get("npc_turns", 0.0),
        "random_safe_turns": totals.get("random_safe_turns", 0.0),
        "eligible_no_refills": totals.get(
            "eligible_no_refills", 0.0
        ),
        "selected_no_refills": totals.get(
            "selected_no_refills", 0.0
        ),
    }


def evaluate_gate(
    checkpoint: str | Path,
    *,
    games: int,
    seed: int,
    player_index: int = 0,
    minimum_win_rate: float = 0.26,
) -> dict[str, Any]:
    """Try two, then eight, then full until the strength gate passes."""
    results: list[dict[str, Any]] = []
    for mode in (MODE_TWO, MODE_EIGHT, MODE_FULL):
        result = evaluate(
            checkpoint,
            mode=mode,
            games=games,
            seed=seed,
            player_index=player_index,
        )
        results.append(result)
        if result["win_rate"] >= minimum_win_rate:
            break
    selected = results[-1]
    return {
        "checkpoint": str(checkpoint),
        "games_per_mode": games,
        "seed": seed,
        "player_index": player_index,
        "minimum_win_rate": minimum_win_rate,
        "passed": selected["win_rate"] >= minimum_win_rate,
        "selected_mode": selected["mode"],
        "results": results,
    }


def collect_for_duration(
    checkpoint: str | Path,
    *,
    mode: str,
    duration_hours: float,
    seed: int,
    game_id_offset: int,
    output: str | Path,
    shard_games: int = 100,
) -> dict[str, Any]:
    if duration_hours <= 0 or shard_games <= 0:
        raise ValueError("duration and shard size must be positive")
    npc = FastValueNpc(checkpoint, mode=mode)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "collection_manifest.json"
    target_seconds = duration_hours * 3600
    policy_id = f"{POLICY_NAME}:{mode}"
    checkpoint_hash = file_sha256(checkpoint)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "collector": policy_id,
            "seed": seed,
            "game_id_offset": game_id_offset,
            "target_seconds": target_seconds,
            "shard_games": shard_games,
            "checkpoint_sha256": checkpoint_hash,
        }
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
        started_at = manifest["started_at"]
    else:
        games = completed_shards = compressed_bytes = 0
        prior_wall = 0.0
        totals = {}
        started_at = datetime.now().astimezone().isoformat()
    run_started = monotonic()
    while prior_wall + monotonic() - run_started < target_seconds:
        shard_start = game_id_offset + games
        shard: list[ReplayGameV2] = []
        for index in range(shard_games):
            if (
                index > 0
                and prior_wall + monotonic() - run_started >= target_seconds
            ):
                break
            replay, facts = play_one_game(
                npc,
                game_id=shard_start + index,
                seed=seed,
                npc_players=frozenset(range(4)),
            )
            shard.append(replay)
            _add_totals(totals, facts)
        if not shard:
            break
        destination = output_path / f"part_{shard_start:06d}.jsonl.gz"
        if destination.exists():
            raise FileExistsError(f"shard exists: {destination}")
        storage = write_replay_shard(shard, destination)
        games += len(shard)
        completed_shards += 1
        compressed_bytes += int(storage["compressed_bytes"])
        wall = prior_wall + monotonic() - run_started
        manifest = {
            "schema": "yellowstone.replay.v2.fast_value_collection",
            "collector": policy_id,
            "rules_version": RULES_VERSION_V2,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "seed": seed,
            "game_id_offset": game_id_offset,
            "target_hours": duration_hours,
            "target_seconds": target_seconds,
            "shard_games": shard_games,
            "started_at": started_at,
            "updated_at": datetime.now().astimezone().isoformat(),
            "wall_seconds": wall,
            "games": games,
            "completed_shards": completed_shards,
            "compressed_bytes": compressed_bytes,
            "policy_totals": totals,
            "status": "complete" if wall >= target_seconds else "running",
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        print(
            json.dumps(
                {
                    "games": games,
                    "wall_seconds": wall,
                    "completed_shard": str(destination),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return manifest


def _append_history(
    history: list[RecentPlacement],
    before: GameState,
    action: Action,
    after: GameState,
) -> None:
    if not isinstance(action, PlaceCardAction):
        return
    player_index = before.current_player_index
    card = before.players[player_index].hand[action.hand_index]
    history.append(
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
    del history[:-HISTORY_SIZE]


def _add_totals(
    target: dict[str, float], values: dict[str, float]
) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0.0) + float(value)


def _mixed_game_seed(seed: int, game_id: int) -> int:
    value = (game_id + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value ^= seed & ((1 << 64) - 1)
    value ^= value >> 30
    value = (
        value * 0xBF58476D1CE4E5B9
    ) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (
        value * 0x94D049BB133111EB
    ) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("benchmark", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--checkpoint", type=Path, required=True)
        child.add_argument("--mode", choices=SUPPORTED_MODES, default=MODE_TWO)
        child.add_argument("--games", type=int, required=True)
        child.add_argument("--seed", type=int, default=20260725)
        child.add_argument("--output", type=Path, required=True)
    subparsers.choices["benchmark"].add_argument(
        "--all-npc", action="store_true"
    )
    subparsers.choices["evaluate"].add_argument(
        "--player-index", type=int, default=0
    )
    gate = subparsers.add_parser("gate")
    gate.add_argument("--checkpoint", type=Path, required=True)
    gate.add_argument("--games", type=int, required=True)
    gate.add_argument("--seed", type=int, default=20260725)
    gate.add_argument("--player-index", type=int, default=0)
    gate.add_argument("--minimum-win-rate", type=float, default=0.26)
    gate.add_argument("--output", type=Path, required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--checkpoint", type=Path, required=True)
    collect.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    collect.add_argument("--duration-hours", type=float, required=True)
    collect.add_argument("--seed", type=int, default=20260728)
    collect.add_argument("--game-id-offset", type=int, default=1_000_000)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--shard-games", type=int, default=100)
    args = parser.parse_args()

    if args.command == "benchmark":
        result = benchmark(
            args.checkpoint,
            mode=args.mode,
            games=args.games,
            seed=args.seed,
            all_npc=args.all_npc,
        )
    elif args.command == "evaluate":
        result = evaluate(
            args.checkpoint,
            mode=args.mode,
            games=args.games,
            seed=args.seed,
            player_index=args.player_index,
        )
    elif args.command == "gate":
        result = evaluate_gate(
            args.checkpoint,
            games=args.games,
            seed=args.seed,
            player_index=args.player_index,
            minimum_win_rate=args.minimum_win_rate,
        )
    else:
        result = collect_for_duration(
            args.checkpoint,
            mode=args.mode,
            duration_hours=args.duration_hours,
            seed=args.seed,
            game_id_offset=args.game_id_offset,
            output=args.output,
            shard_games=args.shard_games,
        )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.command != "collect":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
