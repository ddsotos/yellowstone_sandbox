"""Audit an exact win-value search before introducing loss-based pruning."""

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
    baseline_loss: int
    best_loss: int
    loss_gap: int
    best_value: float
    candidate_count: int
    survivors_at_gap: int
    baseline: TurnCandidate
    best: TurnCandidate


def run_pruning_audit(
    *, checkpoint: str | Path, games: int, seed: int, output: str | Path
) -> dict[str, object]:
    """Audit all play-phase turn starts from all-heuristic games.

    ``a`` is the highest-value two-card candidate that adds no negative cards.
    ``b`` is the unpruned candidate with maximum learned win value.  Only turns
    with such an ``a`` are recorded. The report stores complete states for the
    ten greatest positive ``b - a`` gaps.
    """
    if games <= 0:
        raise ValueError("games must be positive")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    estimator = TorchWinValueEstimator(str(checkpoint))
    bot = HeuristicBot()
    seeds = Random(seed)
    cases: list[AuditCase] = []
    gaps: list[int] = []
    candidate_counts: list[int] = []

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
                    gaps.append(audit_case.loss_gap)
                    candidate_counts.append(audit_case.candidate_count)
                turn_id += 1
            action = bot.choose_action(state)
            if action is None:
                raise RuntimeError("heuristic bot returned no action before game end")
            before = state
            state = apply_known_legal_action(state, action, rng=rng)
            _append_history(history, before, action, state)
        _write_progress(output_path, games_complete=game_id + 1, audited_turns=len(cases))

    if not cases:
        raise RuntimeError("no turn had a two-card, zero-negative candidate")
    threshold = max(gaps)
    top_cases = sorted(cases, key=lambda case: case.loss_gap, reverse=True)[:10]
    survival_rates = [
        case.survivors_at_gap / case.candidate_count
        for case in cases
    ]
    summary: dict[str, object] = {
        "games": games,
        "audited_turns": len(cases),
        "checkpoint": str(checkpoint),
        "seed": seed,
        "threshold_max_gap": threshold,
        "gap_distribution": {
            "min": min(gaps),
            "median": _quantile(gaps, 0.5),
            "p95": _quantile(gaps, 0.95),
            "max": threshold,
        },
        "candidate_count": {
            "mean": sum(candidate_counts) / len(candidate_counts),
            "max": max(candidate_counts),
        },
        "survival_at_max_gap": {
            "mean_rate": sum(survival_rates) / len(survival_rates),
            "min_rate": min(survival_rates),
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
        if losses[index] == 0
        and sum(isinstance(action, PlaceCardAction) for action in candidate.actions) == 2
    ]
    if not zero_loss_two_card_indexes:
        return None
    scores = estimator.estimate_many(tuple(candidate.record for candidate in candidates))
    baseline_index = max(zero_loss_two_card_indexes, key=lambda index: scores[index])
    best_index = max(range(len(candidates)), key=lambda index: scores[index])
    baseline_loss = losses[baseline_index]
    best_loss = losses[best_index]
    gap = best_loss - baseline_loss
    return AuditCase(
        game_id=game_id,
        turn_id=turn_id,
        state=state,
        baseline_loss=baseline_loss,
        best_loss=best_loss,
        loss_gap=gap,
        best_value=scores[best_index],
        candidate_count=len(candidates),
        survivors_at_gap=sum(loss <= baseline_loss + gap for loss in losses),
        baseline=candidates[baseline_index],
        best=candidates[best_index],
    )


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
        "baseline_negative_card_delta": case.baseline_loss,
        "best_negative_card_delta": case.best_loss,
        "loss_gap": case.loss_gap,
        "best_predicted_win_probability": case.best_value,
        "candidate_count": case.candidate_count,
        "survivors_at_observed_gap": case.survivors_at_gap,
        "baseline_actions": [action_to_dict(action) for action in case.baseline.actions],
        "best_actions": [action_to_dict(action) for action in case.best.actions],
        "complete_state": game_state_to_dict(case.state),
    }


def _render_cases_markdown(cases: list[AuditCase]) -> str:
    lines = ["# Top 10 pruning-gap cases", ""]
    for index, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"## {index}. game {case.game_id}, turn {case.turn_id}",
                "",
                f"- negative delta: a={case.baseline_loss}, b={case.best_loss}, gap={case.loss_gap}",
                f"- b predicted win probability: {case.best_value:.4f}",
                f"- candidates: {case.candidate_count}",
                f"- a actions: `{[action_to_dict(action) for action in case.baseline.actions]}`",
                f"- b actions: `{[action_to_dict(action) for action in case.best.actions]}`",
                "",
                "```text",
                render_state(case.state),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _write_progress(output: Path, *, games_complete: int, audited_turns: int) -> None:
    (output / "progress.json").write_text(
        json.dumps({"games_complete": games_complete, "audited_turns": audited_turns}) + "\n",
        encoding="utf-8",
    )


def _quantile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit loss-based Yellowstone pruning")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/audits/pruning_audit")
    )
    args = parser.parse_args()
    summary = run_pruning_audit(
        checkpoint=args.checkpoint, games=args.games, seed=args.seed, output=args.output
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
