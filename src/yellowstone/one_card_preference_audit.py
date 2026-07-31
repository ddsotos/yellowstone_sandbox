"""Audit turn starts where value search prefers one-card end over zero-loss two-card plays."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action, create_initial_state
from yellowstone.render import render_state
from yellowstone.serialization import action_to_dict, game_state_to_dict
from yellowstone.types import Action, GameState, Phase, PlaceCardAction
from yellowstone.value_learning import HISTORY_SIZE, RecentPlacement
from yellowstone.value_policy import TorchWinValueEstimator, TurnCandidate, enumerate_turn_end_candidates


@dataclass(frozen=True, slots=True)
class AuditCase:
    game_id: int
    turn_id: int
    state: GameState
    zero_loss_two_card_value: float
    best_one_card_value: float
    value_gap: float
    zero_loss_two_card_negative_delta: int
    best_one_card_negative_delta: int
    candidate_count: int
    zero_loss_two_card: TurnCandidate
    best_one_card: TurnCandidate


def run_one_card_preference_audit(
    *, checkpoint: str | Path, games: int, seed: int, output: str | Path
) -> dict[str, object]:
    """Audit all-heuristic games for turns where one-card end beats zero-loss two-card."""
    if games <= 0:
        raise ValueError("games must be positive")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    estimator = TorchWinValueEstimator(str(checkpoint))
    bot = HeuristicBot()
    seeds = Random(seed)
    cases: list[AuditCase] = []

    for game_id in range(games):
        state = create_initial_state(4, seed=seeds.randrange(2**63))
        rng = Random(seeds.randrange(2**63))
        history: list[RecentPlacement] = []
        turn_id = 0
        while state.phase != Phase.GAME_OVER:
            if state.phase == Phase.PLAY and state.cards_played_this_turn == 0:
                audit_case = _audit_turn(
                    state, estimator, tuple(history), game_id=game_id, turn_id=turn_id
                )
                if audit_case is not None:
                    cases.append(audit_case)
                turn_id += 1
            action = bot.choose_action(state)
            if action is None:
                raise RuntimeError("heuristic bot returned no action before game end")
            before = state
            state = apply_known_legal_action(state, action, rng=rng)
            _append_history(history, before, action, state)
        _write_progress(output_path, games_complete=game_id + 1, matched_turns=len(cases))

    if not cases:
        raise RuntimeError("no matching turn found")
    top_cases = sorted(cases, key=lambda case: case.value_gap, reverse=True)[:10]
    value_gaps = [case.value_gap for case in cases]
    candidate_counts = [case.candidate_count for case in cases]
    summary: dict[str, object] = {
        "games": games,
        "matched_turns": len(cases),
        "checkpoint": str(checkpoint),
        "seed": seed,
        "value_gap": {
            "min": min(value_gaps),
            "median": _quantile(value_gaps, 0.5),
            "p95": _quantile(value_gaps, 0.95),
            "max": max(value_gaps),
        },
        "candidate_count": {
            "mean": sum(candidate_counts) / len(candidate_counts),
            "max": max(candidate_counts),
        },
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    payload = [_case_to_dict(case) for case in top_cases]
    (output_path / "top10_cases.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_path / "top10_cases.md").write_text(
        _render_cases_markdown(top_cases), encoding="utf-8"
    )
    return summary


def _audit_turn(
    state: GameState,
    estimator: TorchWinValueEstimator,
    history: tuple[RecentPlacement, ...],
    *,
    game_id: int,
    turn_id: int,
) -> AuditCase | None:
    candidates = enumerate_turn_end_candidates(state, history=history, game_id=game_id)
    player_index = state.current_player_index
    negative_before = len(state.players[player_index].negative_cards)
    losses = tuple(
        len(candidate.record.state.players[player_index].negative_cards) - negative_before
        for candidate in candidates
    )
    zero_loss_two_card_indexes = [
        index
        for index, candidate in enumerate(candidates)
        if losses[index] == 0 and _count_place_cards(candidate) == 2
    ]
    if not zero_loss_two_card_indexes:
        return None
    one_card_indexes = [
        index
        for index, candidate in enumerate(candidates)
        if _count_place_cards(candidate) == 1
    ]
    if not one_card_indexes:
        return None
    scores = estimator.estimate_many(tuple(candidate.record for candidate in candidates))
    zero_loss_two_card_index = max(zero_loss_two_card_indexes, key=lambda index: scores[index])
    best_one_card_index = max(one_card_indexes, key=lambda index: scores[index])
    if scores[best_one_card_index] <= scores[zero_loss_two_card_index]:
        return None
    return AuditCase(
        game_id=game_id,
        turn_id=turn_id,
        state=state,
        zero_loss_two_card_value=scores[zero_loss_two_card_index],
        best_one_card_value=scores[best_one_card_index],
        value_gap=scores[best_one_card_index] - scores[zero_loss_two_card_index],
        zero_loss_two_card_negative_delta=losses[zero_loss_two_card_index],
        best_one_card_negative_delta=losses[best_one_card_index],
        candidate_count=len(candidates),
        zero_loss_two_card=candidates[zero_loss_two_card_index],
        best_one_card=candidates[best_one_card_index],
    )


def _count_place_cards(candidate: TurnCandidate) -> int:
    return sum(isinstance(action, PlaceCardAction) for action in candidate.actions)


def _append_history(
    history: list[RecentPlacement], before: GameState, action: Action, after: GameState
) -> None:
    if not isinstance(action, PlaceCardAction):
        return
    player_index = before.current_player_index
    history.append(
        RecentPlacement(
            player_index=player_index,
            card=before.players[player_index].hand[action.hand_index],
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


def _case_to_dict(case: AuditCase) -> dict[str, object]:
    return {
        "game_id": case.game_id,
        "turn_id": case.turn_id,
        "zero_loss_two_card_predicted_win_probability": case.zero_loss_two_card_value,
        "best_one_card_predicted_win_probability": case.best_one_card_value,
        "value_gap": case.value_gap,
        "zero_loss_two_card_negative_card_delta": case.zero_loss_two_card_negative_delta,
        "best_one_card_negative_card_delta": case.best_one_card_negative_delta,
        "candidate_count": case.candidate_count,
        "zero_loss_two_card_actions": [
            action_to_dict(action) for action in case.zero_loss_two_card.actions
        ],
        "best_one_card_actions": [
            action_to_dict(action) for action in case.best_one_card.actions
        ],
        "complete_state": game_state_to_dict(case.state),
    }


def _render_cases_markdown(cases: list[AuditCase]) -> str:
    lines = ["# Top 10 one-card-over-zero-loss-two-card cases", ""]
    for index, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"## {index}. game {case.game_id}, turn {case.turn_id}",
                "",
                (
                    f"- predicted win probability: a={case.zero_loss_two_card_value:.4f}, "
                    f"b={case.best_one_card_value:.4f}, gap={case.value_gap:.4f}"
                ),
                (
                    f"- negative delta: a={case.zero_loss_two_card_negative_delta}, "
                    f"b={case.best_one_card_negative_delta}"
                ),
                f"- candidates: {case.candidate_count}",
                (
                    f"- a actions: "
                    f"`{[action_to_dict(action) for action in case.zero_loss_two_card.actions]}`"
                ),
                (
                    f"- b actions: "
                    f"`{[action_to_dict(action) for action in case.best_one_card.actions]}`"
                ),
                "",
                "```text",
                render_state(case.state),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _write_progress(output: Path, *, games_complete: int, matched_turns: int) -> None:
    (output / "progress.json").write_text(
        json.dumps({"games_complete": games_complete, "matched_turns": matched_turns}) + "\n",
        encoding="utf-8",
    )


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one-card preference cases")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/audits/one_card_preference_audit"),
    )
    args = parser.parse_args()
    summary = run_one_card_preference_audit(
        checkpoint=args.checkpoint, games=args.games, seed=args.seed, output=args.output
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
