"""Optional PyTorch policy/value reference network for board tensors.

New module. It is intentionally an encoder reference, not a trained policy or
an experiment runner.
"""

from __future__ import annotations

from yellowstone.action_space import ACTION_SPACE_SIZE
from yellowstone.board_tensor import BOARD_CHANNELS
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.value_learning import RANK_BOARD_CHANNELS, VALUE_CONTEXT_SIZE

DEFAULT_WIN_VALUE_CONVOLUTION_LAYERS = 2
DEFAULT_WIN_VALUE_HIDDEN_CHANNELS = 64
DEFAULT_WIN_VALUE_HIDDEN_SIZE = 128


def win_value_architecture_id(
    *,
    convolution_layers: int,
    hidden_channels: int,
    hidden_size: int,
    board_channels: int = RANK_BOARD_CHANNELS,
    board_size: int = 7,
    board_height: int | None = None,
    board_width: int | None = None,
) -> str:
    height = board_size if board_height is None else board_height
    width = board_size if board_width is None else board_width
    architecture = (
        "yellowstone.win_value.v1."
        f"conv{convolution_layers}_{hidden_channels}_fc{hidden_size}"
    )
    if board_channels != RANK_BOARD_CHANNELS or height != 7 or width != 7:
        architecture += f"_board{board_channels}x{height}x{width}"
    return architecture


def win_value_architecture_metadata(
    *,
    convolution_layers: int = DEFAULT_WIN_VALUE_CONVOLUTION_LAYERS,
    hidden_channels: int = DEFAULT_WIN_VALUE_HIDDEN_CHANNELS,
    hidden_size: int = DEFAULT_WIN_VALUE_HIDDEN_SIZE,
    board_channels: int = RANK_BOARD_CHANNELS,
    board_size: int = 7,
    board_height: int | None = None,
    board_width: int | None = None,
) -> dict[str, int | str]:
    height = board_size if board_height is None else board_height
    width = board_size if board_width is None else board_width
    _validate_win_value_architecture(
        convolution_layers=convolution_layers,
        hidden_channels=hidden_channels,
        hidden_size=hidden_size,
        board_channels=board_channels,
        board_height=height,
        board_width=width,
    )
    metadata: dict[str, int | str] = {
        "model_architecture": win_value_architecture_id(
            convolution_layers=convolution_layers,
            hidden_channels=hidden_channels,
            hidden_size=hidden_size,
            board_channels=board_channels,
            board_size=board_size,
            board_height=height,
            board_width=width,
        ),
        "convolution_layers": convolution_layers,
        "hidden_channels": hidden_channels,
        "hidden_size": hidden_size,
    }
    if board_channels != RANK_BOARD_CHANNELS or height != 7 or width != 7:
        metadata["board_channels"] = board_channels
        if height == width:
            metadata["board_size"] = height
        else:
            metadata["board_height"] = height
            metadata["board_width"] = width
            metadata["board_shape"] = [board_channels, height, width]
    return metadata


def win_value_architecture_from_checkpoint(
    checkpoint: dict[str, object],
) -> dict[str, int | str]:
    """Return validated V1 architecture, defaulting legacy metadata to Conv2."""
    has_architecture = "model_architecture" in checkpoint
    convolution_layers = int(
        checkpoint.get(
            "convolution_layers",
            DEFAULT_WIN_VALUE_CONVOLUTION_LAYERS,
        )
    )
    hidden_channels = int(
        checkpoint.get("hidden_channels", DEFAULT_WIN_VALUE_HIDDEN_CHANNELS)
    )
    hidden_size = int(
        checkpoint.get("hidden_size", DEFAULT_WIN_VALUE_HIDDEN_SIZE)
    )
    board_channels = int(checkpoint.get("board_channels", RANK_BOARD_CHANNELS))
    board_size = int(checkpoint.get("board_size", 7))
    board_height = int(checkpoint.get("board_height", board_size))
    board_width = int(checkpoint.get("board_width", board_size))
    metadata = win_value_architecture_metadata(
        convolution_layers=convolution_layers,
        hidden_channels=hidden_channels,
        hidden_size=hidden_size,
        board_channels=board_channels,
        board_size=board_size,
        board_height=board_height,
        board_width=board_width,
    )
    if has_architecture and checkpoint["model_architecture"] != metadata[
        "model_architecture"
    ]:
        raise ValueError(
            "unsupported or inconsistent win-value architecture: "
            f"{checkpoint['model_architecture']!r}"
        )
    return metadata


def _validate_win_value_architecture(
    *,
    convolution_layers: int,
    hidden_channels: int,
    hidden_size: int,
    board_channels: int = RANK_BOARD_CHANNELS,
    board_size: int = 7,
    board_height: int | None = None,
    board_width: int | None = None,
) -> None:
    height = board_size if board_height is None else board_height
    width = board_size if board_width is None else board_width
    if convolution_layers not in (2, 3):
        raise ValueError("win-value convolution layers must be 2 or 3")
    if hidden_channels <= 0 or hidden_size <= 0:
        raise ValueError("win-value hidden dimensions must be positive")
    if board_channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("win-value board dimensions must be positive")


def torch_available() -> bool:
    """Return whether the optional PyTorch dependency is installed."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def build_policy_value_net(*, hidden_channels: int = 64, hidden_size: int = 128):
    """Build a CPU-compatible CNN with policy logits and scalar value heads."""
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as error:
        raise ImportError("CNN support requires `pip install -e .[cnn]`") from error

    class YellowstonePolicyValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(BOARD_CHANNELS, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(hidden_channels * 7 * 7 + OBSERVATION_SIZE, hidden_size),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden_size, ACTION_SPACE_SIZE)
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, board, global_features):
            encoded = self.board_encoder(board)
            hidden = self.trunk(torch.cat((encoded, global_features), dim=1))
            return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    return YellowstonePolicyValueNet()


def build_win_value_net(
    *,
    hidden_channels: int = DEFAULT_WIN_VALUE_HIDDEN_CHANNELS,
    hidden_size: int = DEFAULT_WIN_VALUE_HIDDEN_SIZE,
    context_size: int = VALUE_CONTEXT_SIZE,
    convolution_layers: int = DEFAULT_WIN_VALUE_CONVOLUTION_LAYERS,
    board_channels: int = RANK_BOARD_CHANNELS,
    board_size: int = 7,
    board_height: int | None = None,
    board_width: int | None = None,
):
    """Build a player-perspective CNN that returns one win-probability logit."""
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as error:
        raise ImportError("win-value support requires `pip install -e .[value]`") from error

    height = board_size if board_height is None else board_height
    width = board_size if board_width is None else board_width
    _validate_win_value_architecture(
        convolution_layers=convolution_layers,
        hidden_channels=hidden_channels,
        hidden_size=hidden_size,
        board_channels=board_channels,
        board_height=height,
        board_width=width,
    )

    class YellowstoneWinValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = [
                nn.Conv2d(
                    board_channels,
                    hidden_channels,
                    3,
                    padding=1,
                ),
                nn.ReLU(),
            ]
            for _ in range(convolution_layers - 1):
                layers.extend(
                    (
                        nn.Conv2d(
                            hidden_channels,
                            hidden_channels,
                            3,
                            padding=1,
                        ),
                        nn.ReLU(),
                    )
                )
            layers.append(nn.Flatten())
            self.board_encoder = nn.Sequential(*layers)
            self.trunk = nn.Sequential(
                nn.Linear(
                    hidden_channels * height * width + context_size,
                    hidden_size,
                ),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, board, context):
            encoded = self.board_encoder(board)
            hidden = self.trunk(torch.cat((encoded, context), dim=1))
            return self.value_head(hidden).squeeze(-1)

    return YellowstoneWinValueNet()


def build_win_value_net_v2(*, hidden_channels: int = 64, hidden_size: int = 128):
    """Build the strict-canonical, refill-conditioned V2 value network."""
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as error:
        raise ImportError("V2 win-value support requires `pip install -e .[value]`") from error
    from yellowstone.value_v2 import BOARD_CHANNELS_V2, VALUE_CONTEXT_SIZE_V2

    class YellowstoneWinValueNetV2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(BOARD_CHANNELS_V2, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(hidden_channels * 7 * 7 + VALUE_CONTEXT_SIZE_V2, hidden_size),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, board, context):
            encoded = self.board_encoder(board)
            hidden = self.trunk(torch.cat((encoded, context), dim=1))
            return self.value_head(hidden).squeeze(-1)

    return YellowstoneWinValueNetV2()


def build_win_value_net_v2_lite(
    *, hidden_channels: int = 64, hidden_size: int = 128
):
    """Build the transition-aware compact V2-lite value network."""
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as error:
        raise ImportError(
            "V2-lite win-value support requires `pip install -e .[value]`"
        ) from error
    from yellowstone.value_v2_lite import (
        BOARD_CHANNELS_V2_LITE,
        VALUE_CONTEXT_SIZE_V2_LITE,
    )

    class YellowstoneWinValueNetV2Lite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(
                    BOARD_CHANNELS_V2_LITE,
                    hidden_channels,
                    3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.trunk = nn.Sequential(
                nn.Linear(
                    hidden_channels * 7 * 7 + VALUE_CONTEXT_SIZE_V2_LITE,
                    hidden_size,
                ),
                nn.ReLU(),
            )
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, board, context):
            encoded = self.board_encoder(board)
            hidden = self.trunk(torch.cat((encoded, context), dim=1))
            return self.value_head(hidden).squeeze(-1)

    return YellowstoneWinValueNetV2Lite()
