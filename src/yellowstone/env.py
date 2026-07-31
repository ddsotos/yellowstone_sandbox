"""Reinforcement-learning environment for one learner against heuristic NPCs."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from yellowstone.action_space import action_from_index, legal_action_mask
from yellowstone.bots import BotPolicy, HeuristicBot
from yellowstone.game import (
    apply_action,
    apply_known_legal_action,
    create_initial_state,
    legal_actions,
)
from yellowstone.observation import state_to_observation
from yellowstone.rewards import reward_for_transition
from yellowstone.serialization import actions_to_dicts, game_state_to_dict
from yellowstone.types import GameState, Phase


Info = dict[str, Any]


@dataclass(frozen=True, slots=True)
class EnvReset:
    observation: tuple[int, ...]
    legal_action_mask: tuple[bool, ...]
    info: Info


@dataclass(frozen=True, slots=True)
class EnvStep:
    observation: tuple[int, ...]
    reward: float
    done: bool
    legal_action_mask: tuple[bool, ...]
    info: Info


@dataclass(slots=True)
class YellowstoneEnv:
    """A minimal RL environment for player 0 against heuristic NPCs."""

    player_count: int = 4
    learning_player_index: int = 0
    npc_policy: BotPolicy | None = None
    state: GameState | None = field(init=False, default=None)
    rng: Random = field(init=False, default_factory=Random)
    stopped_reason: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not 0 <= self.learning_player_index < self.player_count:
            raise ValueError("learning_player_index must be within player_count")
        if self.npc_policy is None:
            self.npc_policy = HeuristicBot()

    def reset(self, seed: int | None = None) -> EnvReset:
        """Start a new game and advance NPCs until the learner can act."""
        self.rng = Random(seed)
        self.stopped_reason = None
        self.state = create_initial_state(
            self.player_count,
            seed=seed,
        )
        self.state = self._advance_npcs(self.state)
        return EnvReset(
            observation=state_to_observation(self.state),
            legal_action_mask=legal_action_mask(self.state),
            info=self._info(),
        )

    def step(self, action_index: int) -> EnvStep:
        """Apply one learner action, then auto-advance NPC turns."""
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        if self.state.phase == Phase.GAME_OVER:
            raise RuntimeError("cannot step after game over")
        if self.state.current_player_index != self.learning_player_index:
            raise RuntimeError("environment is not waiting for the learner")

        before = self.state
        action = action_from_index(action_index, self.state)
        self.state = apply_action(self.state, action, rng=self.rng)
        if self.state.phase != Phase.GAME_OVER:
            self.state = self._advance_npcs(self.state)

        done = self.state.phase == Phase.GAME_OVER or self.stopped_reason is not None
        return EnvStep(
            observation=state_to_observation(self.state),
            reward=reward_for_transition(
                before,
                self.state,
                player_index=self.learning_player_index,
            ),
            done=done,
            legal_action_mask=legal_action_mask(self.state),
            info=self._info(),
        )

    def _advance_npcs(self, state: GameState) -> GameState:
        while (
            state.phase != Phase.GAME_OVER
            and state.current_player_index != self.learning_player_index
        ):
            action = self.npc_policy.choose_action(state)
            if action is None:
                self.stopped_reason = "no_legal_action"
                return state
            state = apply_known_legal_action(state, action, rng=self.rng)
        if state.phase != Phase.GAME_OVER and not legal_actions(state):
            self.stopped_reason = "no_legal_action"
        return state

    def _info(self) -> Info:
        if self.state is None:
            raise RuntimeError("reset must be called before info is available")
        info: Info = {
            "state": game_state_to_dict(self.state),
            "legal_actions": actions_to_dicts(legal_actions(self.state)),
            "current_player_index": self.state.current_player_index,
            "learning_player_index": self.learning_player_index,
            "winners": list(self.state.winners),
            "settlement_count": self.state.settlement_count,
        }
        if self.stopped_reason is not None:
            info["stopped_reason"] = self.stopped_reason
        return info
