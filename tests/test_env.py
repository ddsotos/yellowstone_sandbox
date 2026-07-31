from yellowstone.action_space import action_to_index
from yellowstone.env import YellowstoneEnv
from yellowstone.game import legal_actions
from yellowstone.observation import OBSERVATION_SIZE
from yellowstone.types import PlaceCardAction


def test_env_reset_returns_observation_mask_and_info() -> None:
    # resetで学習用の観測、合法手mask、infoが返ることを確認する。
    env = YellowstoneEnv()

    result = env.reset(seed=1)

    assert len(result.observation) == OBSERVATION_SIZE
    assert any(result.legal_action_mask)
    assert result.info["current_player_index"] == 0
    assert result.info["learning_player_index"] == 0
    assert "state" in result.info
    assert "legal_actions" in result.info


def test_env_step_applies_learner_action_and_returns_next_decision_point() -> None:
    # stepで学習対象の行動を適用し、次の判断点を返すことを確認する。
    env = YellowstoneEnv()
    env.reset(seed=1)
    action = next(
        action for action in legal_actions(env.state) if isinstance(action, PlaceCardAction)
    )

    result = env.step(action_to_index(action))

    assert len(result.observation) == OBSERVATION_SIZE
    assert isinstance(result.reward, float)
    assert result.info["learning_player_index"] == 0
    assert result.done or result.info["current_player_index"] == 0


def test_env_reset_can_advance_to_nonzero_learning_player() -> None:
    # 学習対象が0番以外でもNPCを自動進行して学習対象手番で止まることを確認する。
    env = YellowstoneEnv(learning_player_index=1)

    result = env.reset(seed=1)

    assert result.info["current_player_index"] == 1
    assert result.info["learning_player_index"] == 1


def test_env_step_requires_reset() -> None:
    # reset前のstepを拒否することを確認する。
    env = YellowstoneEnv()

    try:
        env.step(0)
    except RuntimeError as error:
        assert "reset must be called" in str(error)
    else:
        raise AssertionError("RuntimeError was not raised")
