import pytest

from yellowstone.board_tensor import BOARD_CHANNELS, board_tensor, global_feature_tensor
from yellowstone.game import create_initial_state
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.types import BOARD_SIZE


def test_board_and_global_tensor_shapes() -> None:
    # 盤面tensorと固定長global特徴量のshapeとdtypeを確認する。
    np = pytest.importorskip("numpy")
    state = create_initial_state(4, seed=3)

    board = board_tensor(state)
    global_features = global_feature_tensor(state)

    assert board.shape == (BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert board.dtype == np.float32
    assert global_features.shape == (OBSERVATION_SIZE,)
    assert global_features.dtype == np.float32
    assert board[-1].sum() == 1.0
