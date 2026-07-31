"""Reference spatial encoders for the standalone RL bundle.

New module: this representation is not part of the copied game-rule API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from yellowstone.observation import state_to_observation
from yellowstone.types import BOARD_SIZE, Card, Color, GameState

if TYPE_CHECKING:
    import numpy as np


COLOR_ORDER = (Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW)
BOARD_CHANNELS = len(COLOR_ORDER) + 1


def board_tensor(state: GameState) -> "np.ndarray":
    """Return float32 [5, 7, 7]: one count plane per color plus total count."""
    np = _numpy()
    tensor = np.zeros((BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    color_indices = {color: index for index, color in enumerate(COLOR_ORDER)}
    for position, stack in state.board.items():
        for card in stack:
            tensor[color_indices[card.color], position.y, position.x] += 1.0
            tensor[-1, position.y, position.x] += 1.0
    return tensor


def global_feature_tensor(state: GameState) -> "np.ndarray":
    """Return the copied fixed-length observation as a float32 feature vector."""
    np = _numpy()
    return np.asarray(state_to_observation(state), dtype=np.float32)


def card_tensor(card: Card) -> "np.ndarray":
    """Return a simple one-hot color and normalized rank feature for a card."""
    np = _numpy()
    values = np.zeros(len(COLOR_ORDER) + 1, dtype=np.float32)
    values[COLOR_ORDER.index(card.color)] = 1.0
    values[-1] = card.rank_index / float(BOARD_SIZE - 1)
    return values


def _numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise ImportError("board tensors require `pip install -e .[tensor]`") from error
    return np
