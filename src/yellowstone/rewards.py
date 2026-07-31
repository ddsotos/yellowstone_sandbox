"""Model-free baseline rewards included with the standalone bundle."""

from __future__ import annotations

from yellowstone.types import GameState, Phase


NEGATIVE_CARD_REWARD_WEIGHT = 0.1
WIN_REWARD = 1.0
LOSS_REWARD = -1.0


def reward_for_transition(
    before: GameState,
    after: GameState,
    *,
    player_index: int,
) -> float:
    """Reward score improvement, negative-card reduction, and final outcome.

    This bundle intentionally has no learned-model reward dependency.
    """
    before_player = before.players[player_index]
    after_player = after.players[player_index]
    reward = before_player.loss_score - after_player.loss_score
    reward += NEGATIVE_CARD_REWARD_WEIGHT * (
        len(before_player.negative_cards) - len(after_player.negative_cards)
    )
    if after.phase == Phase.GAME_OVER:
        reward += WIN_REWARD if player_index in after.winners else LOSS_REWARD
    return float(reward)
