"""Public action-delta records and candidate policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from yellowstone.bots import HeuristicBot
from yellowstone.game import apply_known_legal_action, legal_actions
from yellowstone.types import Action, Card, GameState, Phase, PlaceCardAction
from yellowstone.value_evaluation_v2_lite import V2LiteTurnChoice
from yellowstone.value_learning import COLOR_ORDER, RecentPlacement
from yellowstone.value_policy import (
    TorchWinValueEstimator,
    TurnCandidate,
    _has_zero_negative_two_card_witness,
    enumerate_turn_end_candidates,
)
from yellowstone.value_v2 import CompletedTurnTracker, PendingRefillSource
from yellowstone.value_v2_lite import ValueRecordV2Lite, canonical_tensors_v2_lite


VALUE_SCHEMA_ACTION_DELTA = "yellowstone.value.action-delta.v1"
CANONICALIZATION_ACTION_DELTA = "strict_residual_v2_lite_plus_action_cards"
HISTORY_SEMANTICS_ACTION_DELTA = "last_two_completed_turns_before_turn"
ACTION_CARD_CONTEXT = 2 * 6
ACTION_DELTA_CONTEXT_SIZE = 138 + ACTION_CARD_CONTEXT


@dataclass(frozen=True, slots=True)
class ProposedCandidate:
    candidate: TurnCandidate
    proposer_score: float
    cards: tuple[Card, ...]
    group_key: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ActionDeltaRecord:
    transition: ValueRecordV2Lite
    cards: tuple[Card, ...]
    target: float


def validate_proposer_checkpoint(path: str | Path) -> None:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "value_schema": "yellowstone.value.v1",
        "input_canonicalization": "fast_lr_ud_color_v1",
        "history_semantics": "rolling_last_two_placements",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"proposer checkpoint differs at {key}")


def propose_top_card_groups(
    state: GameState,
    history: tuple[RecentPlacement, ...],
    proposer: TorchWinValueEstimator,
    *,
    adaptive_pq_pruning: bool = True,
    approximate_new_color_neighbor_limit: bool = True,
    one_limit: int = 3,
    two_limit: int = 5,
) -> tuple[ProposedCandidate, ...]:
    """Deduplicate by unordered played-card multiset and retain 3/5 groups."""
    pruning_limit = None
    if adaptive_pq_pruning and _has_zero_negative_two_card_witness(state):
        player = state.players[state.current_player_index]
        pruning_limit = 4 if len(player.negative_cards) + player.loss_score >= 10 else 8
    candidates = enumerate_turn_end_candidates(
        state,
        history=history,
        max_negative_card_increase=pruning_limit,
        approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
        collapse_equivalent_frames=True,
    )
    if not candidates:
        raise ValueError("no completed-turn candidate")
    scores = proposer.estimate_many(tuple(item.record for item in candidates))
    grouped: dict[tuple[tuple[str, int], ...], ProposedCandidate] = {}
    for candidate, score in zip(candidates, scores, strict=True):
        cards = _candidate_cards(state, candidate)
        key = tuple(sorted((card.color.value, card.rank_index) for card in cards))
        proposed = ProposedCandidate(candidate, float(score), cards, key)
        previous = grouped.get(key)
        if previous is None or proposed.proposer_score > previous.proposer_score:
            grouped[key] = proposed
    ordered = sorted(
        grouped.values(),
        key=lambda item: (-item.proposer_score, item.group_key),
    )
    one = [item for item in ordered if len(item.cards) == 1][:one_limit]
    two = [item for item in ordered if len(item.cards) == 2][:two_limit]
    return tuple((*one, *two))


def enumerate_action_delta_candidates(
    state: GameState,
    history: tuple[RecentPlacement, ...],
    *,
    adaptive_pq_pruning: bool = True,
    approximate_new_color_neighbor_limit: bool = True,
) -> tuple[ProposedCandidate, ...]:
    """Return every retained completed-turn candidate for delta inference."""
    pruning_limit = None
    if adaptive_pq_pruning and _has_zero_negative_two_card_witness(state):
        player = state.players[state.current_player_index]
        pruning_limit = 4 if len(player.negative_cards) + player.loss_score >= 10 else 8
    candidates = enumerate_turn_end_candidates(
        state,
        history=history,
        max_negative_card_increase=pruning_limit,
        approximate_new_color_neighbor_limit=approximate_new_color_neighbor_limit,
        collapse_equivalent_frames=True,
    )
    if not candidates:
        raise ValueError("no completed-turn candidate")
    retained: list[ProposedCandidate] = []
    for candidate in candidates:
        cards = _candidate_cards(state, candidate)
        retained.append(
            ProposedCandidate(
                candidate=candidate,
                proposer_score=0.0,
                cards=cards,
                group_key=tuple(
                    sorted(
                        (card.color.value, card.rank_index)
                        for card in cards
                    )
                ),
            )
        )
    return tuple(retained)


def encode_action_delta(record: ActionDeltaRecord):
    import numpy as np

    board, context, transform = canonical_tensors_v2_lite(record.transition)
    values = list(float(value) for value in context)
    for slot in range(2):
        if slot < len(record.cards):
            card = record.cards[slot]
            old_color = COLOR_ORDER.index(card.color)
            color = transform.old_to_new_color[old_color]
            rank = (
                6 - card.rank_index
                if transform.vertical_reflection
                else card.rank_index
            )
            values.extend([1.0, *_one_hot(color, 4), rank / 6])
        else:
            values.extend([0.0] * 6)
    if len(values) != ACTION_DELTA_CONTEXT_SIZE:
        raise AssertionError("unexpected action-delta context size")
    return board, np.asarray(values, dtype=np.float32)


def build_action_delta_net(*, hidden_channels: int = 64, hidden_size: int = 128):
    import torch
    import torch.nn as nn
    from yellowstone.value_v2_lite import BOARD_CHANNELS_V2_LITE

    class ActionDeltaNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(BOARD_CHANNELS_V2_LITE, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(
                    hidden_channels * 7 * 7 + ACTION_DELTA_CONTEXT_SIZE,
                    hidden_size,
                ),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, board, context):
            encoded = self.board_encoder(board)
            return torch.tanh(
                self.head(self.trunk(torch.cat((encoded, context), dim=1)))
            ).squeeze(-1)

    return ActionDeltaNet()


class TorchActionDeltaEstimator:
    def __init__(self, checkpoint_path: str | Path) -> None:
        import torch

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        expected = {
            "value_schema": VALUE_SCHEMA_ACTION_DELTA,
            "input_canonicalization": CANONICALIZATION_ACTION_DELTA,
            "history_semantics": HISTORY_SEMANTICS_ACTION_DELTA,
            "opponent_private_inputs": False,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"action-delta checkpoint differs at {key}")
        self.torch = torch
        self.model = build_action_delta_net()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def estimate_many(
        self, records: tuple[ActionDeltaRecord, ...]
    ) -> tuple[float, ...]:
        import numpy as np

        encoded = [encode_action_delta(record) for record in records]
        if not encoded:
            return ()
        with self.torch.no_grad():
            values = self.model(
                self.torch.from_numpy(np.stack([item[0] for item in encoded])),
                self.torch.from_numpy(np.stack([item[1] for item in encoded])),
            ).tolist()
        return tuple(float(value) for value in values)


@dataclass(slots=True)
class ActionDeltaPlayer:
    player_index: int
    proposer: TorchWinValueEstimator | None
    estimator: TorchActionDeltaEstimator
    candidate_mode: str = "all"
    history: list[RecentPlacement] = field(default_factory=list)
    completed: CompletedTurnTracker = field(default_factory=CompletedTurnTracker)
    remaining_plan: list[Action] = field(default_factory=list)
    choices: list[V2LiteTurnChoice] = field(default_factory=list)
    heuristic: HeuristicBot = field(default_factory=HeuristicBot)

    def choose_action(self, state: GameState) -> Action:
        actions = legal_actions(state)
        if self.remaining_plan:
            action = self.remaining_plan.pop(0)
            if action not in actions:
                raise RuntimeError("action-delta plan became illegal")
            return action
        if state.phase != Phase.PLAY or not any(
            isinstance(action, PlaceCardAction) for action in actions
        ):
            action = self.heuristic.choose_action(state)
            if action is None:
                raise RuntimeError("no heuristic fallback")
            return action
        if self.candidate_mode == "all":
            proposed = enumerate_action_delta_candidates(
                state, tuple(self.history)
            )
        elif self.candidate_mode == "v1_top_3_5":
            if self.proposer is None:
                raise ValueError("v1_top_3_5 requires a proposer")
            proposed = propose_top_card_groups(
                state, tuple(self.history), self.proposer
            )
        else:
            raise ValueError(
                f"unknown action-delta candidate mode: {self.candidate_mode}"
            )
        records = tuple(
            ActionDeltaRecord(
                transition=ValueRecordV2Lite(
                    game_id=-1,
                    perspective_player_index=self.player_index,
                    state_before_turn=state,
                    state=item.candidate.record.state,
                    history_before_turn=self.completed.snapshot()[-2:],
                    pending_refill_source=PendingRefillSource.NO_PENDING,
                    target=0.0,
                ),
                cards=item.cards,
                target=0.0,
            )
            for item in proposed
        )
        scores = self.estimator.estimate_many(records)
        best = max(range(len(proposed)), key=lambda index: scores[index])
        item = proposed[best]
        self.remaining_plan = list(item.candidate.actions)
        self.choices.append(
            V2LiteTurnChoice(
                actions=item.candidate.actions,
                record=records[best].transition,
                predicted_win_probability=scores[best],
                candidate_count=len(proposed),
            )
        )
        return self.choose_action(state)

    def observe(self, before: GameState, action: Action, after: GameState) -> None:
        self.completed.observe(before, action, after)
        _append_recent(self.history, before, action, after)


def make_transition_record(
    *,
    game_id: int,
    state_before: GameState,
    state_after: GameState,
    history_before: tuple,
    cards: tuple[Card, ...],
    target: float,
) -> ActionDeltaRecord:
    return ActionDeltaRecord(
        transition=ValueRecordV2Lite(
            game_id=game_id,
            perspective_player_index=state_before.current_player_index,
            state_before_turn=state_before,
            state=state_after,
            history_before_turn=history_before[-2:],
            pending_refill_source=PendingRefillSource.NO_PENDING,
            target=target,
        ),
        cards=cards,
        target=target,
    )


def _candidate_cards(
    state: GameState, candidate: TurnCandidate
) -> tuple[Card, ...]:
    working = state
    cards: list[Card] = []
    for action in candidate.actions:
        if isinstance(action, PlaceCardAction):
            cards.append(
                working.players[working.current_player_index].hand[
                    action.hand_index
                ]
            )
        working = apply_known_legal_action(working, action)
    return tuple(cards)


def _append_recent(
    history: list[RecentPlacement],
    before: GameState,
    action: Action,
    after: GameState,
) -> None:
    if not isinstance(action, PlaceCardAction):
        return
    player = before.current_player_index
    history.append(
        RecentPlacement(
            player_index=player,
            card=before.players[player].hand[action.hand_index],
            score_delta=(
                before.players[player].loss_score
                - after.players[player].loss_score
            ),
            negative_card_delta=(
                len(after.players[player].negative_cards)
                - len(before.players[player].negative_cards)
            ),
        )
    )
    del history[:-2]


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if value == index else 0.0 for value in range(size)]
