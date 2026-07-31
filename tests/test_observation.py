from yellowstone.heuristic_turn_plan import (
    HeuristicTurnPlan,
    heuristic_played_rank_features_from_plans,
)
from yellowstone.observation import (
    BOARD_OBSERVATION_SIZE,
    HEURISTIC_BONUS_OBSERVATION_SIZE,
    HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE,
    HEURISTIC_PLAYED_RANK_OBSERVATION_SIZE,
    HAND_OBSERVATION_SIZE,
    OBSERVATION_SIZE,
    OPPONENT_LAST_TURN_OBSERVATION_SIZE,
    PLAYERS_OBSERVATION_SIZE,
    observation_metadata,
    state_to_observation,
)
from yellowstone.types import (
    Card,
    Color,
    EndTurnAction,
    Frame,
    GameState,
    PlayerState,
    PlaceCardAction,
    Position,
)


def test_state_to_observation_has_fixed_size() -> None:
    # 観測が常に固定長になることを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    observation = state_to_observation(state)

    assert len(observation) == OBSERVATION_SIZE
    assert observation_metadata()["observation_size"] == OBSERVATION_SIZE


def test_state_to_observation_encodes_compressed_board_counts() -> None:
    # 盤面は3x3枠の基準位置とセルごとの総枚数だけで入ることを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState()),
        board={
            Position(1, 2): (
                Card(Color.RED, 2),
                Card(Color.BLUE, 2),
                Card(Color.RED, 2),
            )
        },
    )

    observation = state_to_observation(state)

    assert observation[:BOARD_OBSERVATION_SIZE] == (
        0,
        0,
        3,
        0,
        0,
        0,
        0,
        1,
        3,
        0,
        0,
    )


def test_state_to_observation_encodes_relative_hand_colors_from_board_columns() -> None:
    # 手札の色は絶対色ではなく、盤面3列に対する相対色one-hotで入ることを確認する。
    state = GameState(
        players=(
            PlayerState(
                hand=(
                    Card(Color.RED, 0),
                    Card(Color.BLUE, 1),
                    Card(Color.GREEN, 2),
                    Card(Color.YELLOW, 3),
                )
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(1, 2): (Card(Color.GREEN, 2),),
            Position(2, 2): (Card(Color.BLUE, 2),),
            Position(3, 2): (Card(Color.RED, 2),),
        },
    )

    observation = state_to_observation(state)
    hand_start = BOARD_OBSERVATION_SIZE

    assert observation[hand_start : hand_start + 24] == (
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        2,
        1,
        0,
        0,
        0,
        1,
        3,
    )


def test_state_to_observation_assigns_unseen_hand_colors_without_absolute_order() -> None:
    # 盤面にない色は手札スロット順で相対色IDへ割り当てられることを確認する。
    state = GameState(
        players=(
            PlayerState(),
            PlayerState(hand=(Card(Color.GREEN, 4),)),
            PlayerState(),
            PlayerState(),
        ),
        current_player_index=1,
    )

    observation = state_to_observation(state)
    hand_start = BOARD_OBSERVATION_SIZE

    assert observation[hand_start : hand_start + 6] == (1, 1, 0, 0, 0, 4)
    assert observation[hand_start + 6 : hand_start + 12] == (0, 0, 0, 0, 0, 0)


def test_state_to_observation_encodes_players_and_scalars() -> None:
    # プレイヤー状態と、山札段階・決算回数だけが末尾に入ることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),), loss_score=5),
            PlayerState(negative_cards=(Card(Color.BLUE, 1),), loss_score=8),
            PlayerState(),
            PlayerState(hand=(Card(Color.GREEN, 1), Card(Color.YELLOW, 2))),
        ),
        deck=tuple(Card(Color.YELLOW, index % 7) for index in range(7)),
        settlement_count=3,
    )

    observation = state_to_observation(state)
    player_start = BOARD_OBSERVATION_SIZE + HAND_OBSERVATION_SIZE
    opponent_last_turn_start = player_start + PLAYERS_OBSERVATION_SIZE
    heuristic_bonus_start = (
        opponent_last_turn_start + OPPONENT_LAST_TURN_OBSERVATION_SIZE
    )
    heuristic_negative_delta_start = (
        heuristic_bonus_start + HEURISTIC_BONUS_OBSERVATION_SIZE
    )
    scalar_start = (
        heuristic_negative_delta_start
        + HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE
        + HEURISTIC_PLAYED_RANK_OBSERVATION_SIZE
    )

    assert observation[player_start : player_start + 12] == (
        1,
        0,
        5,
        0,
        1,
        8,
        0,
        0,
        5,
        2,
        0,
        5,
    )
    assert (
        observation[
            opponent_last_turn_start
            : opponent_last_turn_start + OPPONENT_LAST_TURN_OBSERVATION_SIZE
        ]
        == (0, 0, 0)
    )
    assert (
        observation[
            heuristic_bonus_start
            : heuristic_bonus_start + HEURISTIC_BONUS_OBSERVATION_SIZE
        ]
        == (0, 0)
    )
    assert (
        observation[
            heuristic_negative_delta_start
            : heuristic_negative_delta_start
            + HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE
        ]
        == (0, 0)
    )
    assert observation[scalar_start : scalar_start + 2] == (2, 3)


def test_state_to_observation_encodes_opponent_last_turn_counts() -> None:
    # 相手3人の直前ターンのプレイ枚数が現在プレイヤーからの相対順で入る。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState()),
        current_player_index=2,
        last_turn_play_counts=(2, 1, 0, 2),
    )

    observation = state_to_observation(state)
    start = BOARD_OBSERVATION_SIZE + HAND_OBSERVATION_SIZE + PLAYERS_OBSERVATION_SIZE

    assert observation[start : start + OPPONENT_LAST_TURN_OBSERVATION_SIZE] == (
        2,
        2,
        1,
    )


def test_state_to_observation_encodes_heuristic_turn_bonus_scores() -> None:
    # heuristicの1枚出し/2枚出しで得られる即時ボーナス点を入れる。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.BLUE, 2), Card(Color.GREEN, 2))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(0, 0): (Card(Color.RED, 0),),
            Position(1, 0): (Card(Color.GREEN, 0),),
            Position(2, 0): (Card(Color.BLUE, 0),),
            Position(0, 1): (Card(Color.RED, 1),),
            Position(1, 1): (Card(Color.GREEN, 1),),
            Position(2, 1): (Card(Color.BLUE, 1),),
            Position(0, 2): (Card(Color.RED, 2),),
        },
    )

    observation = state_to_observation(state)
    start = (
        BOARD_OBSERVATION_SIZE
        + HAND_OBSERVATION_SIZE
        + PLAYERS_OBSERVATION_SIZE
        + OPPONENT_LAST_TURN_OBSERVATION_SIZE
    )

    assert observation[start : start + HEURISTIC_BONUS_OBSERVATION_SIZE] == (1, 4)


def test_state_to_observation_accepts_precomputed_heuristic_bonuses() -> None:
    # 事前計算した一枚・二枚のボーナス点を観測へそのまま利用できる。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    observation = state_to_observation(state, heuristic_bonuses=(3, 4))
    start = (
        BOARD_OBSERVATION_SIZE
        + HAND_OBSERVATION_SIZE
        + PLAYERS_OBSERVATION_SIZE
        + OPPONENT_LAST_TURN_OBSERVATION_SIZE
    )

    assert observation[start : start + HEURISTIC_BONUS_OBSERVATION_SIZE] == (3, 4)


def test_state_to_observation_accepts_precomputed_negative_deltas() -> None:
    # 1枚案・2枚案の即時マイナス増加量を観測へそのまま利用できる。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState())
    )

    observation = state_to_observation(
        state,
        heuristic_bonuses=(0, 0),
        heuristic_negative_deltas=(2, 5),
    )
    start = (
        BOARD_OBSERVATION_SIZE
        + HAND_OBSERVATION_SIZE
        + PLAYERS_OBSERVATION_SIZE
        + OPPONENT_LAST_TURN_OBSERVATION_SIZE
        + HEURISTIC_BONUS_OBSERVATION_SIZE
    )

    assert (
        observation[start : start + HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE]
        == (2, 5)
    )


def test_state_to_observation_accepts_precomputed_played_ranks() -> None:
    # 事前計算した1枚案・2枚案の数字を観測へそのまま利用できる。
    state = GameState(
        players=(
            PlayerState(
                hand=(
                    Card(Color.BLUE, 4),
                    Card(Color.GREEN, 1),
                    Card(Color.RED, 6),
                )
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
    )

    observation = state_to_observation(
        state,
        heuristic_bonuses=(0, 0),
        heuristic_negative_deltas=(0, 0),
        heuristic_played_ranks=(4, 1, 6),
    )
    start = (
        BOARD_OBSERVATION_SIZE
        + HAND_OBSERVATION_SIZE
        + PLAYERS_OBSERVATION_SIZE
        + OPPONENT_LAST_TURN_OBSERVATION_SIZE
        + HEURISTIC_BONUS_OBSERVATION_SIZE
        + HEURISTIC_NEGATIVE_DELTA_OBSERVATION_SIZE
    )

    assert observation[start : start + HEURISTIC_PLAYED_RANK_OBSERVATION_SIZE] == (
        4,
        1,
        6,
    )


def test_heuristic_played_rank_features_sorts_two_card_ranks() -> None:
    # 2枚案の数字特徴は実際のプレイ順ではなく昇順で入る。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.BLUE, 5), Card(Color.GREEN, 2))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
    )
    one_card_plan = HeuristicTurnPlan(
        actions=(
            PlaceCardAction(0, Position(0, 5), Frame(0, 3)),
            EndTurnAction(),
        ),
        bonus_score=0,
        negative_card_delta=0,
    )
    two_card_plan = HeuristicTurnPlan(
        actions=(
            PlaceCardAction(0, Position(0, 5), Frame(0, 3)),
            PlaceCardAction(0, Position(1, 2), Frame(0, 2)),
        ),
        bonus_score=0,
        negative_card_delta=0,
    )

    ranks = heuristic_played_rank_features_from_plans(
        state,
        one_card_plan=one_card_plan,
        two_card_plan=two_card_plan,
    )

    assert ranks == (5, 2, 5)
