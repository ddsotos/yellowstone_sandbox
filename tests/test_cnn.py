import pytest

from yellowstone.action_space import ACTION_SPACE_SIZE
from yellowstone.board_tensor import BOARD_CHANNELS, board_tensor, global_feature_tensor
from yellowstone.cnn import (
    build_policy_value_net,
    build_win_value_net,
    torch_available,
    win_value_architecture_from_checkpoint,
)
from yellowstone.game import create_initial_state
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.types import BOARD_SIZE


def test_cnn_cpu_forward_shapes() -> None:
    # CPU上のCNN forwardがpolicy logitsとvalueを所定shapeで返すことを確認する。
    if not torch_available():
        pytest.skip("PyTorch is not installed")
    import torch

    state = create_initial_state(4, seed=5)
    model = build_policy_value_net(hidden_channels=8, hidden_size=16)
    board = torch.from_numpy(board_tensor(state)).unsqueeze(0)
    global_features = torch.from_numpy(global_feature_tensor(state)).unsqueeze(0)

    policy, value = model(board, global_features)

    assert board.shape == (1, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert global_features.shape == (1, OBSERVATION_SIZE)
    assert policy.shape == (1, ACTION_SPACE_SIZE)
    assert value.shape == (1,)


def test_conv3_win_value_architecture_has_expected_shape_and_size() -> None:
    torch = pytest.importorskip("torch")
    model = build_win_value_net(convolution_layers=3)

    output = model(
        torch.zeros((2, 29, 7, 7)),
        torch.zeros((2, 81)),
    )

    assert output.shape == (2,)
    assert sum(p.numel() for p in model.parameters()) == 502_657


def test_legacy_win_value_checkpoint_defaults_to_conv2() -> None:
    architecture = win_value_architecture_from_checkpoint({})

    assert architecture == {
        "model_architecture": "yellowstone.win_value.v1.conv2_64_fc128",
        "convolution_layers": 2,
        "hidden_channels": 64,
        "hidden_size": 128,
    }


def test_inconsistent_win_value_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported or inconsistent"):
        win_value_architecture_from_checkpoint(
            {
                "model_architecture": (
                    "yellowstone.win_value.v1.unknown"
                ),
                "convolution_layers": 3,
            }
        )
