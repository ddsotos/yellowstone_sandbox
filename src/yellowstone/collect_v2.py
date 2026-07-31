"""Generation-0 replay collection with a legacy low-hand teacher."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

from yellowstone.bots import HeuristicBot
from yellowstone.game import (
    apply_known_legal_action,
    create_initial_state,
    legal_actions,
)
from yellowstone.replay_v2 import (
    ReplayGameV2,
    file_sha256,
    write_replay_shard,
)
from yellowstone.serialization import action_to_dict, game_state_to_dict
from yellowstone.types import (
    Action,
    Phase,
    PlaceCardAction,
    RefillAction,
    RefillSource,
)
from yellowstone.value_learning import HISTORY_SIZE, RecentPlacement, ValueRecord
from yellowstone.value_policy import (
    TorchWinValueEstimator,
    TurnCandidate,
    enumerate_turn_end_candidates,
)


CATEGORY_RETAIN = "retain_hand"
CATEGORY_DECK = "deck_refill"
CATEGORY_NEGATIVE = "negative_refill"
CATEGORY_ORDER = (CATEGORY_RETAIN, CATEGORY_DECK, CATEGORY_NEGATIVE)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    category: str
    candidate: TurnCandidate
    score: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class TeacherDecision:
    actions: tuple[Action, ...]
    audit: dict[str, Any]
    sampling_seconds: float


class LegacyLowHandTeacher:
    """Use a frozen V1 model to improve only turns starting with <=2 cards."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        sample_count: int = 20,
        approximate_new_color_neighbor_limit: bool = True,
    ) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self.checkpoint = str(checkpoint)
        self.estimator = TorchWinValueEstimator(str(checkpoint))
        self.sample_count = sample_count
        self.approximate_new_color_neighbor_limit = (
            approximate_new_color_neighbor_limit
        )

    def choose(
        self,
        state,
        history: tuple[RecentPlacement, ...],
        *,
        game_id: int,
        turn_id: int,
        rng: Random,
    ) -> TeacherDecision:
        player_index = state.current_player_index
        starting_negative = len(state.players[player_index].negative_cards)
        normal = enumerate_turn_end_candidates(
            state,
            history=history,
            game_id=game_id,
            approximate_new_color_neighbor_limit=(
                self.approximate_new_color_neighbor_limit
            ),
            collapse_equivalent_frames=True,
        )
        full = (
            enumerate_turn_end_candidates(
                state,
                history=history,
                game_id=game_id,
                approximate_new_color_neighbor_limit=(
                    self.approximate_new_color_neighbor_limit
                ),
                collapse_equivalent_frames=False,
            )
            if starting_negative <= 5
            else normal
        )

        retain = _dedupe_candidates(
            candidate
            for candidate in normal
            if len(candidate.record.state.players[player_index].hand) > 0
        )
        empty_normal = _dedupe_candidates(
            candidate
            for candidate in normal
            if not candidate.record.state.players[player_index].hand
            and candidate.record.state.phase == Phase.REFILL
        )
        empty_negative = _dedupe_candidates(
            candidate
            for candidate in full
            if not candidate.record.state.players[player_index].hand
            and candidate.record.state.phase == Phase.REFILL
            and len(candidate.record.state.players[player_index].negative_cards) >= 6
        )

        sampling_started = perf_counter()
        scored: list[ScoredCandidate] = []
        scored.extend(self._score_direct(CATEGORY_RETAIN, retain))
        scored.extend(
            self._score_sampled(
                CATEGORY_DECK,
                empty_normal,
                RefillSource.DECK,
                rng,
            )
        )
        scored.extend(
            self._score_sampled(
                CATEGORY_NEGATIVE,
                empty_negative,
                RefillSource.NEGATIVE_CARDS,
                rng,
            )
        )
        sampling_seconds = perf_counter() - sampling_started
        winners = {
            category: max(
                (item for item in scored if item.category == category),
                key=lambda item: item.score,
                default=None,
            )
            for category in CATEGORY_ORDER
        }
        available = [category for category in CATEGORY_ORDER if winners[category]]
        if not available:
            raise ValueError("legacy low-hand teacher found no category")
        best_category = max(available, key=lambda category: winners[category].score)
        probabilities = _category_probabilities(available, best_category)
        selected_category = _weighted_choice(probabilities, rng)
        selected = winners[selected_category]
        if selected is None:
            raise AssertionError("weighted category chose an unavailable winner")
        refill_action = {
            CATEGORY_RETAIN: (),
            CATEGORY_DECK: (RefillAction(RefillSource.DECK),),
            CATEGORY_NEGATIVE: (
                RefillAction(RefillSource.NEGATIVE_CARDS),
            ),
        }[selected_category]
        actions = (*selected.candidate.actions, *refill_action)

        audit_candidates = [
            {
                "category": item.category,
                "score": item.score,
                "sample_count": item.sample_count,
                "actions": [
                    action_to_dict(action) for action in item.candidate.actions
                ],
                "resulting_negative_count": len(
                    item.candidate.record.state.players[
                        player_index
                    ].negative_cards
                ),
            }
            for item in scored
        ]
        audit = {
            "game_id": game_id,
            "turn_id": turn_id,
            "player_index": player_index,
            "starting_hand_count": len(state.players[player_index].hand),
            "starting_negative_count": starting_negative,
            "candidate_counts": {
                category: sum(
                    candidate["category"] == category
                    for candidate in audit_candidates
                )
                for category in CATEGORY_ORDER
            },
            "category_winners": {
                category: (
                    None
                    if winners[category] is None
                    else {
                        "score": winners[category].score,
                        "actions": [
                            action_to_dict(action)
                            for action in winners[category].candidate.actions
                        ],
                    }
                )
                for category in CATEGORY_ORDER
            },
            "selection_probabilities": probabilities,
            "selected_category": selected_category,
            "selected_actions": [action_to_dict(action) for action in actions],
            "all_candidates": audit_candidates,
            "sampling_seconds": sampling_seconds,
            "privileged_actual_deck": True,
        }
        return TeacherDecision(actions, audit, sampling_seconds)

    def _score_direct(
        self, category: str, candidates: tuple[TurnCandidate, ...]
    ) -> list[ScoredCandidate]:
        if not candidates:
            return []
        scores = self.estimator.estimate_many(
            tuple(candidate.record for candidate in candidates)
        )
        return [
            ScoredCandidate(category, candidate, float(score), 1)
            for candidate, score in zip(candidates, scores, strict=True)
        ]

    def _score_sampled(
        self,
        category: str,
        candidates: tuple[TurnCandidate, ...],
        source: RefillSource,
        rng: Random,
    ) -> list[ScoredCandidate]:
        if not candidates:
            return []
        records: list[ValueRecord] = []
        owners: list[int] = []
        for candidate_index, candidate in enumerate(candidates):
            for _ in range(self.sample_count):
                sample_seed = rng.randrange(2**63)
                sampled = _sample_refill_state(
                    candidate.record.state,
                    source,
                    seed=sample_seed,
                )
                records.append(
                    ValueRecord(
                        game_id=candidate.record.game_id,
                        perspective_player_index=(
                            candidate.record.perspective_player_index
                        ),
                        state=sampled,
                        history=candidate.record.history,
                        target=0.0,
                    )
                )
                owners.append(candidate_index)
        values = self.estimator.estimate_many(tuple(records))
        totals = [0.0] * len(candidates)
        counts = [0] * len(candidates)
        for owner, value in zip(owners, values, strict=True):
            totals[owner] += float(value)
            counts[owner] += 1
        return [
            ScoredCandidate(
                category,
                candidate,
                totals[index] / counts[index],
                counts[index],
            )
            for index, candidate in enumerate(candidates)
        ]


def collect_generation0_games(
    *,
    game_count: int,
    seed: int,
    checkpoint: str | Path,
    game_id_offset: int = 0,
    sample_count: int = 20,
    _teacher: LegacyLowHandTeacher | None = None,
) -> tuple[tuple[ReplayGameV2, ...], dict[str, float]]:
    if game_count <= 0:
        raise ValueError("game_count must be positive")
    teacher = _teacher or LegacyLowHandTeacher(
        checkpoint, sample_count=sample_count
    )
    checkpoint_hash = file_sha256(checkpoint)
    games: list[ReplayGameV2] = []
    total_sampling_seconds = 0.0
    total_low_hand_decisions = 0
    total_candidate_counts = {category: 0 for category in CATEGORY_ORDER}
    started = perf_counter()

    for local_game_id in range(game_count):
        game_id = game_id_offset + local_game_id
        game_seed_rng = Random(_mixed_game_seed(seed, game_id))
        initial_seed = game_seed_rng.randrange(2**63)
        gameplay_seed = game_seed_rng.randrange(2**63)
        teacher_seed = game_seed_rng.randrange(2**63)
        state = create_initial_state(4, seed=initial_seed)
        initial_state = state
        gameplay_rng = Random(gameplay_seed)
        teacher_rng = Random(teacher_seed)
        heuristic = HeuristicBot()
        actions: list[Action] = []
        decisions: list[dict[str, Any]] = []
        legacy_history: list[RecentPlacement] = []
        planned: list[Action] = []
        planned_player: int | None = None
        turn_id = 0

        while state.phase != Phase.GAME_OVER:
            if planned:
                if state.current_player_index != planned_player:
                    raise AssertionError("planned player changed before plan finished")
                action = planned.pop(0)
            elif (
                state.phase == Phase.PLAY
                and state.cards_played_this_turn == 0
                and 0 < len(state.players[state.current_player_index].hand) <= 2
                and any(
                    isinstance(action, PlaceCardAction)
                    for action in legal_actions(state)
                )
            ):
                decision = teacher.choose(
                    state,
                    tuple(legacy_history),
                    game_id=game_id,
                    turn_id=turn_id,
                    rng=teacher_rng,
                )
                planned = list(decision.actions)
                planned_player = state.current_player_index
                action = planned.pop(0)
                decisions.append(decision.audit)
                total_sampling_seconds += decision.sampling_seconds
                total_low_hand_decisions += 1
                for category, count in decision.audit["candidate_counts"].items():
                    total_candidate_counts[category] += int(count)
            else:
                action = heuristic.choose_action(state)
                if action is None:
                    raise RuntimeError("heuristic returned no action")

            if action not in legal_actions(state):
                raise RuntimeError(f"selected illegal action: {action!r}")
            before = state
            state = apply_known_legal_action(state, action, rng=gameplay_rng)
            actions.append(action)
            _append_legacy_history(legacy_history, before, action, state)
            if (
                before.current_player_index != state.current_player_index
                or state.phase == Phase.GAME_OVER
            ):
                planned = []
                planned_player = None
                turn_id += 1

        games.append(
            ReplayGameV2(
                game_id=game_id,
                initial_seed=initial_seed,
                gameplay_seed=gameplay_seed,
                initial_state=initial_state,
                actions=tuple(actions),
                decisions=tuple(decisions),
                winners=state.winners,
                teacher_checkpoint=str(checkpoint),
                teacher_sha256=checkpoint_hash,
                teacher_generation=0,
                privileged_teacher_deck=True,
            )
        )

    elapsed = perf_counter() - started
    return tuple(games), {
        "games": float(game_count),
        "seconds": elapsed,
        "seconds_per_game": elapsed / game_count,
        "low_hand_decisions": float(total_low_hand_decisions),
        "sampling_seconds": total_sampling_seconds,
        **{
            f"{category}_candidates": float(count)
            for category, count in total_candidate_counts.items()
        },
    }


def collect_generation0_shards(
    *,
    game_count: int,
    seed: int,
    checkpoint: str | Path,
    output: str | Path,
    game_id_offset: int = 0,
    sample_count: int = 20,
    shard_games: int = 100,
) -> dict[str, Any]:
    """Collect restartable generation-0 shards with one frozen teacher."""
    if shard_games <= 0:
        raise ValueError("shard_games must be positive")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    teacher = LegacyLowHandTeacher(checkpoint, sample_count=sample_count)
    empty_totals: dict[str, float] = {
        "games": 0.0,
        "seconds": 0.0,
        "low_hand_decisions": 0.0,
        "sampling_seconds": 0.0,
        **{f"{category}_candidates": 0.0 for category in CATEGORY_ORDER},
    }
    empty_storage = {
        "games": 0,
        "compressed_bytes": 0,
        "uncompressed_bytes": 0,
    }
    manifest_path = output_path / "collection_manifest.json"
    teacher_hash = file_sha256(checkpoint)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "seed": seed,
            "game_id_offset": game_id_offset,
            "requested_games": game_count,
            "sample_count": sample_count,
            "shard_games": shard_games,
            "teacher_sha256": teacher_hash,
        }
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(
                    f"existing collection manifest differs at {key}: "
                    f"{existing.get(key)!r} != {value!r}"
                )
        totals = {
            key: float(existing["collection"].get(key, 0.0))
            for key in empty_totals
        }
        storage_totals = {
            key: int(existing["storage"].get(key, 0))
            for key in empty_storage
        }
        completed_shards = int(existing["completed_shards"])
    else:
        totals = empty_totals
        storage_totals = empty_storage
        completed_shards = 0
    for local_offset in range(0, game_count, shard_games):
        count = min(shard_games, game_count - local_offset)
        absolute_offset = game_id_offset + local_offset
        destination = output_path / f"part_{absolute_offset:06d}.jsonl.gz"
        if destination.exists():
            if local_offset < int(totals["games"]):
                continue
            raise FileExistsError(
                "replay shard exists but is not committed in the progress "
                f"manifest: {destination}"
            )
        games, facts = collect_generation0_games(
            game_count=count,
            seed=seed,
            checkpoint=checkpoint,
            game_id_offset=absolute_offset,
            sample_count=sample_count,
            _teacher=teacher,
        )
        storage = write_replay_shard(games, destination)
        for key in totals:
            totals[key] += float(facts.get(key, 0.0))
        for key in storage_totals:
            storage_totals[key] += int(storage[key])
        completed_shards += 1
        manifest = {
            "schema": "yellowstone.replay.v2.collection",
            "seed": seed,
            "game_id_offset": game_id_offset,
            "requested_games": game_count,
            "sample_count": sample_count,
            "shard_games": shard_games,
            "teacher_checkpoint": str(checkpoint),
            "teacher_sha256": teacher_hash,
            "completed_shards": completed_shards,
            "collection": totals,
            "storage": storage_totals,
        }
        temporary = output_path / "collection_manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(output_path / "collection_manifest.json")
        print(
            json.dumps(
                {
                    "completed_shard": str(destination),
                    "completed_games": int(totals["games"]),
                    "requested_games": game_count,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    totals["seconds_per_game"] = (
        totals["seconds"] / totals["games"] if totals["games"] else 0.0
    )
    return {
        "completed_shards": completed_shards,
        "collection": totals,
        "storage": storage_totals,
    }


def _sample_refill_state(state, source: RefillSource, *, seed: int):
    if source == RefillSource.DECK:
        deck = list(state.deck)
        Random(seed).shuffle(deck)
        state = replace(state, deck=tuple(deck))
        return apply_known_legal_action(
            state,
            RefillAction(RefillSource.DECK),
            rng=Random(seed ^ 0x5DEECE66D),
        )
    if source == RefillSource.NEGATIVE_CARDS:
        return apply_known_legal_action(
            state,
            RefillAction(RefillSource.NEGATIVE_CARDS),
            rng=Random(seed),
        )
    raise ValueError(f"unsupported sampled refill source: {source}")


def _mixed_game_seed(seed: int, game_id: int) -> int:
    value = (game_id + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value ^= seed & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _dedupe_candidates(candidates) -> tuple[TurnCandidate, ...]:
    unique: dict[str, TurnCandidate] = {}
    for candidate in candidates:
        key = json.dumps(
            game_state_to_dict(candidate.record.state),
            sort_keys=True,
            separators=(",", ":"),
        )
        unique.setdefault(key, candidate)
    return tuple(unique.values())


def _category_probabilities(
    available: list[str], best_category: str
) -> dict[str, float]:
    probabilities = {category: 0.0 for category in CATEGORY_ORDER}
    probabilities[best_category] = 0.6
    for category in CATEGORY_ORDER:
        if category == best_category:
            continue
        if category in available:
            probabilities[category] = 0.2
        else:
            probabilities[best_category] += 0.2
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(f"category probabilities do not sum to one: {total}")
    return probabilities


def _weighted_choice(probabilities: dict[str, float], rng: Random) -> str:
    draw = rng.random()
    cumulative = 0.0
    for category in CATEGORY_ORDER:
        cumulative += probabilities[category]
        if draw < cumulative:
            return category
    return CATEGORY_ORDER[-1]


def _append_legacy_history(
    history: list[RecentPlacement],
    before,
    action: Action,
    after,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect V2 generation-0 replays")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--game-id-offset", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shard-games",
        type=int,
        help="write restartable shards to --output directory",
    )
    args = parser.parse_args()

    if args.shard_games is not None:
        result = collect_generation0_shards(
            game_count=args.games,
            seed=args.seed,
            checkpoint=args.checkpoint,
            output=args.output,
            game_id_offset=args.game_id_offset,
            sample_count=args.sample_count,
            shard_games=args.shard_games,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return

    games, facts = collect_generation0_games(
        game_count=args.games,
        seed=args.seed,
        checkpoint=args.checkpoint,
        game_id_offset=args.game_id_offset,
        sample_count=args.sample_count,
    )
    storage = write_replay_shard(games, args.output)
    print(
        json.dumps(
            {"collection": facts, "storage": storage},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
