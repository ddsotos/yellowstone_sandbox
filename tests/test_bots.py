from dataclasses import replace

from yellowstone.bots import (
    BotPolicy,
    FixedFrameHandSixOneOffMinLossOneCardBot,
    HeuristicBot,
    choose_heuristic_action,
    placement_sort_key,
)
from yellowstone.game import apply_known_legal_action, legal_actions
from yellowstone.types import (
    Card,
    Color,
    EndTurnAction,
    Frame,
    GameState,
    Phase,
    PlaceCardAction,
    PlayerState,
    Position,
    RefillAction,
    RefillSource,
)


class _FixedRandom:
    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


def test_heuristic_bot_implements_policy_interface() -> None:
    # HeuristicBotが行動ルールの共通インターフェースとして使えることを確認する。
    policy: BotPolicy = HeuristicBot()
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    )

    assert isinstance(policy.choose_action(state), PlaceCardAction)


def test_choose_heuristic_action_returns_none_for_game_over() -> None:
    # ゲーム終了中は行動を返さないことを確認する。
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState()),
        phase=Phase.GAME_OVER,
    )

    assert choose_heuristic_action(state) is None


def test_heuristic_bot_refills_empty_hand_at_turn_start() -> None:
    state = GameState(
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState()),
        deck=(Card(Color.RED, 0),),
    )

    assert choose_heuristic_action(state) == RefillAction(RefillSource.DECK)


def test_refill_prefers_negative_cards_when_hand_empty_and_enough_negative_cards() -> None:
    # 手札0枚かつマイナスカード6枚以上ならマイナスカード補充を選ぶことを確認する。
    state = GameState(
        players=(
            PlayerState(
                hand=(),
                negative_cards=tuple(Card(Color.RED, rank) for rank in range(6)),
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    assert choose_heuristic_action(state) == RefillAction(RefillSource.NEGATIVE_CARDS)


def test_refill_chooses_deck_without_six_negative_cards() -> None:
    # マイナスカード補充できない補充フェーズでは山札補充を選ぶことを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),), negative_cards=()),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    assert choose_heuristic_action(state) == RefillAction(RefillSource.DECK)


def test_common_placement_prefers_far_rank_before_stacking() -> None:
    # 真ん中から遠い数字の優先が重ね置き優先より強いことを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 2), Card(Color.BLUE, 0))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(0, 2): (Card(Color.RED, 2),),
        },
    )
    stacking_action = PlaceCardAction(
        hand_index=0,
        position=Position(0, 2),
        frame=Frame(0, 0),
    )
    far_rank_action = PlaceCardAction(
        hand_index=1,
        position=Position(1, 0),
        frame=Frame(0, 0),
    )

    assert placement_sort_key(state, far_rank_action) < placement_sort_key(
        state,
        stacking_action,
    )


def test_common_placement_prefers_stacking_when_rank_priority_ties() -> None:
    # 数字優先度が同じなら重ね置きを優先することを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0), Card(Color.BLUE, 6))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(0, 0): (Card(Color.RED, 0),),
            Position(5, 5): (Card(Color.GREEN, 5),),
        },
    )

    action = choose_heuristic_action(state)

    assert isinstance(action, PlaceCardAction)
    assert action.hand_index == 0
    assert action.position == Position(0, 0)


def test_choose_heuristic_action_prefers_no_damage_two_card_pair() -> None:
    # 2枚ノーダメージのペアがある場合、ペア内で重要な遠い数字のカードを優先する。
    state = GameState(
        players=(
            PlayerState(
                hand=(
                    Card(Color.RED, 3),
                    Card(Color.BLUE, 0),
                    Card(Color.GREEN, 6),
                )
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={},
    )

    action = choose_heuristic_action(state)

    assert isinstance(action, PlaceCardAction)
    assert action.hand_index == 1


def test_choose_heuristic_action_ends_after_one_card_when_second_would_damage() -> None:
    # 1枚配置後、2枚目がダメージありでプラスポイントもない場合は終了する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.BLUE, 6),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(0, 0): (Card(Color.RED, 0),),
            Position(0, 1): (Card(Color.RED, 1),),
            Position(0, 2): (Card(Color.RED, 2),),
        },
        cards_played_this_turn=1,
    )

    assert choose_heuristic_action(state) == EndTurnAction()


def _fixed_frame_hand_six_state(
    hand: tuple[Card, ...],
) -> GameState:
    return GameState(
        players=(
            PlayerState(hand=hand),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(0, 2): (Card(Color.RED, 2),),
            Position(2, 4): (Card(Color.BLUE, 4),),
            Position(0, 3): (Card(Color.RED, 3),),
            Position(1, 3): (Card(Color.BLUE, 3),),
            Position(2, 3): (Card(Color.RED, 3),),
        },
    )


def test_board_five_hand_six_one_off_two_plus_uses_sixty_percent() -> None:
    state = GameState(
        players=(
            PlayerState(
                hand=(
                    Card(Color.RED, 0),
                    Card(Color.BLUE, 1),
                    Card(Color.GREEN, 2),
                    Card(Color.YELLOW, 3),
                    Card(Color.RED, 4),
                    Card(Color.BLUE, 5),
                )
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={
            Position(0, 2): (Card(Color.RED, 2),),
            Position(2, 4): (Card(Color.BLUE, 4),),
            Position(0, 3): (Card(Color.RED, 3),),
            Position(1, 3): (Card(Color.BLUE, 3),),
            Position(2, 3): (Card(Color.RED, 3),),
        },
    )
    bot = FixedFrameHandSixOneOffMinLossOneCardBot(rng=_FixedRandom(0.59))

    first = bot.choose_action(state)

    assert isinstance(first, PlaceCardAction)
    assert bot.last_branch is not None
    assert bot.last_branch.bucket == "2_plus"
    assert first == min(
        [
            action
            for action in legal_actions(state)
            if isinstance(action, PlaceCardAction)
        ],
        key=lambda action: placement_sort_key(state, action),
    )
    after_first = apply_known_legal_action(state, first)
    assert bot.choose_action(after_first) == EndTurnAction()
    assert bot.last_branch is None


def test_forced_one_card_uses_heuristic_first_card_choice() -> None:
    state = _fixed_frame_hand_six_state(
        (
            Card(Color.RED, 0),
            Card(Color.BLUE, 1),
            Card(Color.GREEN, 2),
            Card(Color.YELLOW, 3),
            Card(Color.RED, 4),
            Card(Color.BLUE, 5),
        )
    )

    assert choose_heuristic_action(
        state, force_one_card_turn=True
    ) == min(
        (
            action
            for action in legal_actions(state)
            if isinstance(action, PlaceCardAction)
        ),
        key=lambda action: placement_sort_key(state, action),
    )


def test_board_five_hand_six_one_off_one_uses_forty_percent() -> None:
    state = _fixed_frame_hand_six_state(
        (
            Card(Color.RED, 1),
            Card(Color.RED, 2),
            Card(Color.RED, 2),
            Card(Color.RED, 3),
            Card(Color.RED, 3),
            Card(Color.RED, 4),
        )
    )
    hit = FixedFrameHandSixOneOffMinLossOneCardBot(rng=_FixedRandom(0.39))
    miss = FixedFrameHandSixOneOffMinLossOneCardBot(rng=_FixedRandom(0.40))

    assert isinstance(hit.choose_action(state), PlaceCardAction)
    assert miss.choose_action(state) == choose_heuristic_action(state)
    assert hit.last_branch is not None
    assert hit.last_branch.bucket == "1"
    assert miss.last_branch is not None
    assert miss.last_branch.bucket == "1"


def test_board_five_hand_six_one_off_zero_uses_ten_percent() -> None:
    state = _fixed_frame_hand_six_state(
        (
            Card(Color.RED, 2),
            Card(Color.RED, 2),
            Card(Color.RED, 3),
            Card(Color.RED, 3),
            Card(Color.RED, 4),
            Card(Color.RED, 4),
        )
    )
    hit = FixedFrameHandSixOneOffMinLossOneCardBot(rng=_FixedRandom(0.09))
    miss = FixedFrameHandSixOneOffMinLossOneCardBot(rng=_FixedRandom(0.10))

    assert isinstance(hit.choose_action(state), PlaceCardAction)
    assert miss.choose_action(state) == choose_heuristic_action(state)
    assert hit.last_branch is not None
    assert hit.last_branch.bucket == "0"
    assert miss.last_branch is not None
    assert miss.last_branch.bucket == "0"


def test_board_five_hand_six_one_card_requires_board_five_cards() -> None:
    state = GameState(
        players=(
            PlayerState(
                hand=(
                    Card(Color.RED, 0),
                    Card(Color.BLUE, 1),
                    Card(Color.GREEN, 2),
                    Card(Color.YELLOW, 3),
                    Card(Color.RED, 4),
                    Card(Color.BLUE, 5),
                )
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={Position(3, 3): (Card(Color.RED, 3),)},
    )
    bot = FixedFrameHandSixOneOffMinLossOneCardBot(rng=_FixedRandom(0.0))

    assert bot.choose_action(state) == choose_heuristic_action(state)
    assert bot.last_branch is None
