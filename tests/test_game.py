from dataclasses import replace

import pytest

from yellowstone.game import (
    apply_action,
    apply_known_legal_action,
    board_fits_in_some_frame,
    can_place_card_at,
    create_deck,
    create_initial_state,
    legal_actions,
)
from yellowstone.types import (
    BOARD_SIZE,
    DEFAULT_LOSS_SCORE,
    HAND_SIZE,
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


def test_create_deck_has_two_of_each_card() -> None:
    # デッキが4色x7数字x各2枚で構成されることを確認する。
    deck = create_deck()

    assert len(deck) == 56
    for color in Color:
        for rank_index in range(BOARD_SIZE):
            assert deck.count(Card(color, rank_index)) == 2


def test_create_initial_state_deals_hands_and_places_initial_card() -> None:
    # 初期化で各プレイヤーに6枚配り、初期カードを対応行の中央に置くことを確認する。
    state = create_initial_state(4, seed=1)

    assert len(state.players) == 4
    assert all(len(player.hand) == HAND_SIZE for player in state.players)
    assert all(player.loss_score == DEFAULT_LOSS_SCORE for player in state.players)
    assert len(state.board) == 1
    position, stack = next(iter(state.board.items()))
    assert position == Position(x=BOARD_SIZE // 2, y=stack[0].rank_index)
    assert len(state.deck) == 56 - 4 * HAND_SIZE - 1
    assert state.current_player_index == 0
    assert state.phase == Phase.PLAY


def test_create_initial_state_sorts_hands_by_rank() -> None:
    # 初期手札は観測時に並べ替えず、配札時点で数字昇順に保つ。
    state = create_initial_state(4, seed=1)

    for player in state.players:
        assert player.hand == tuple(
            sorted(player.hand, key=lambda card: (card.rank_index, card.color.value))
        )


def test_create_initial_state_rejects_unsupported_player_count() -> None:
    # 初期実装では4〜5人以外を拒否することを確認する。
    with pytest.raises(ValueError):
        create_initial_state(3, seed=1)


def test_can_place_card_at_uses_rank_index_as_row() -> None:
    # rank_index と y が一致する横列だけに配置できることを確認する。
    card = Card(Color.RED, 2)

    assert can_place_card_at({}, card, Position(0, 2))
    assert not can_place_card_at({}, card, Position(0, 3))


def test_can_place_card_at_prevents_second_column_for_existing_color() -> None:
    # 盤面上にある色は既存列にしか置けないことを確認する。
    board = {Position(3, 2): (Card(Color.RED, 2),)}

    assert can_place_card_at(board, Card(Color.RED, 4), Position(3, 4))
    assert not can_place_card_at(board, Card(Color.RED, 4), Position(2, 4))


def test_apply_place_card_stacks_without_increasing_occupied_cells() -> None:
    # 既存カードのあるマスへの配置は重ね置きになり、占有マス数は増えないことを確認する。
    red_three = Card(Color.RED, 2)
    player = PlayerState(hand=(red_three,), loss_score=5)
    state = GameState(
        players=(player, PlayerState(), PlayerState(), PlayerState()),
        board={Position(3, 2): (Card(Color.RED, 2),)},
        deck=(),
    )

    next_state = apply_action(
        state,
        PlaceCardAction(
            hand_index=0,
            position=Position(3, 2),
            frame=Frame(2, 0),
        ),
    )

    assert next_state.board[Position(3, 2)] == (Card(Color.RED, 2), red_three)
    assert len(next_state.board) == 1
    assert next_state.players[0].loss_score == 5


def test_apply_known_legal_action_matches_apply_action_for_legal_action() -> None:
    # 検証済み合法手用の高速適用が通常適用と同じ結果になることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={},
        deck=(),
    )
    action = PlaceCardAction(
        hand_index=0,
        position=Position(0, 0),
        frame=Frame(0, 0),
    )

    assert apply_known_legal_action(state, action) == apply_action(state, action)


def test_end_turn_advances_after_one_card() -> None:
    # 1枚配置後に1枚で終了すると次プレイヤーへ進むことを確認する。
    state = replace(create_initial_state(4, seed=2), cards_played_this_turn=1)

    next_state = apply_action(state, EndTurnAction())

    assert next_state.current_player_index == 1
    assert next_state.cards_played_this_turn == 0
    assert next_state.phase == Phase.PLAY
    assert next_state.last_turn_play_counts == (1, 0, 0, 0)


def test_end_turn_with_empty_hand_requires_refill_before_next_player() -> None:
    # 1枚配置後に手札が空なら、ターン終了ではなく補充フェーズに入ることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=()),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        deck=tuple(Card(Color.RED, rank_index) for rank_index in range(HAND_SIZE)),
        cards_played_this_turn=1,
    )

    next_state = apply_action(state, EndTurnAction())

    assert next_state.current_player_index == 0
    assert next_state.cards_played_this_turn == 1
    assert next_state.phase == Phase.REFILL
    assert RefillAction(RefillSource.DECK) in legal_actions(next_state)
    assert RefillAction(RefillSource.NONE) not in legal_actions(next_state)


def test_refill_after_one_card_empty_hand_advances_after_deck_draw() -> None:
    # 1枚終了後の補充で山札から手札を戻してから次プレイヤーへ進むことを確認する。
    state = GameState(
        players=(
            PlayerState(hand=()),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        deck=tuple(Card(Color.RED, rank_index) for rank_index in range(HAND_SIZE)),
        phase=Phase.REFILL,
        cards_played_this_turn=1,
    )

    next_state = apply_action(state, RefillAction(RefillSource.DECK))

    assert len(next_state.players[0].hand) == HAND_SIZE
    assert next_state.current_player_index == 1
    assert next_state.cards_played_this_turn == 0
    assert next_state.phase == Phase.PLAY
    assert next_state.last_turn_play_counts == (1, 0, 0, 0)


def test_refill_sorts_hand_after_draw() -> None:
    # 補充で引いたカードは手札に追加した後、数字昇順に整列される。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 5), Card(Color.BLUE, 1))),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        deck=(
            Card(Color.GREEN, 4),
            Card(Color.YELLOW, 0),
            Card(Color.RED, 2),
            Card(Color.BLUE, 6),
        ),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    next_state = apply_action(state, RefillAction(RefillSource.DECK))

    assert next_state.players[0].hand == (
        Card(Color.YELLOW, 0),
        Card(Color.BLUE, 1),
        Card(Color.RED, 2),
        Card(Color.GREEN, 4),
        Card(Color.RED, 5),
        Card(Color.BLUE, 6),
    )


def test_end_turn_with_empty_hand_can_recover_from_negative_cards() -> None:
    # 手札が空でマイナスカードが6枚以上あれば、失点側からの回復を選べることを確認する。
    negative_cards = tuple(Card(Color.BLUE, rank_index) for rank_index in range(HAND_SIZE))
    state = GameState(
        players=(
            PlayerState(hand=(), negative_cards=negative_cards),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        deck=tuple(Card(Color.RED, rank_index) for rank_index in range(HAND_SIZE)),
        cards_played_this_turn=1,
    )

    refill_state = apply_action(state, EndTurnAction())
    next_state = apply_action(
        refill_state,
        RefillAction(RefillSource.NEGATIVE_CARDS),
    )

    assert RefillAction(RefillSource.NEGATIVE_CARDS) in legal_actions(refill_state)
    assert len(next_state.players[0].hand) == HAND_SIZE
    assert next_state.players[0].negative_cards == ()
    assert next_state.current_player_index == 1


def test_empty_hand_at_turn_start_offers_refill_actions() -> None:
    state = GameState(
        players=(
            PlayerState(hand=()),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        deck=tuple(Card(Color.RED, rank_index) for rank_index in range(HAND_SIZE)),
    )

    assert RefillAction(RefillSource.DECK) in legal_actions(state)


def test_refill_none_advances_after_two_cards() -> None:
    # 2枚配置後の補充しない選択で次プレイヤーへ進むことを確認する。
    state = replace(
        create_initial_state(4, seed=3),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    next_state = apply_action(state, RefillAction(RefillSource.NONE))

    assert next_state.current_player_index == 1
    assert next_state.cards_played_this_turn == 0
    assert next_state.phase == Phase.PLAY
    assert next_state.last_turn_play_counts == (2, 0, 0, 0)


def test_refill_deck_exhaustion_settles_negative_cards() -> None:
    # 山札不足の補充で決算し、マイナスカード枚数ぶん失点することを確認する。
    current_player = PlayerState(
        hand=(Card(Color.RED, 0),),
        negative_cards=(Card(Color.BLUE, 0), Card(Color.GREEN, 0)),
        loss_score=5,
    )
    state = GameState(
        players=(
            current_player,
            PlayerState(negative_cards=(Card(Color.YELLOW, 0),), loss_score=5),
            PlayerState(loss_score=5),
            PlayerState(loss_score=5),
        ),
        deck=(Card(Color.RED, 1),),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    next_state = apply_action(state, RefillAction(RefillSource.DECK))

    assert next_state.players[0].loss_score == 7
    assert next_state.players[1].loss_score == 6
    assert all(not player.negative_cards for player in next_state.players)
    assert next_state.current_player_index == 1
    assert next_state.settlement_count == 1
    assert board_fits_in_some_frame(next_state.board)


def test_refill_that_draws_the_last_deck_card_still_settles() -> None:
    current_player = PlayerState(
        hand=(Card(Color.RED, 0),) * 5,
        negative_cards=(Card(Color.BLUE, 0),),
        loss_score=5,
    )
    state = GameState(
        players=(
            current_player,
            PlayerState(negative_cards=(Card(Color.GREEN, 0),), loss_score=5),
            PlayerState(loss_score=5),
            PlayerState(loss_score=5),
        ),
        deck=(Card(Color.YELLOW, 0),),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    next_state = apply_action(
        state,
        RefillAction(RefillSource.DECK),
    )

    assert len(next_state.players[0].hand) == 6
    assert next_state.players[0].loss_score == 6
    assert next_state.players[1].loss_score == 6
    assert all(not player.negative_cards for player in next_state.players)
    assert next_state.settlement_count == 1
    assert len(next_state.deck) == 2


def test_legacy_exact_deck_exhaustion_can_be_replayed_without_settlement() -> None:
    state = GameState(
        players=(
            PlayerState(
                hand=(Card(Color.RED, 0),) * 5,
                negative_cards=(Card(Color.BLUE, 0),),
                loss_score=5,
            ),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        deck=(Card(Color.YELLOW, 0),),
        phase=Phase.REFILL,
        cards_played_this_turn=2,
    )

    next_state = apply_known_legal_action(
        state,
        RefillAction(RefillSource.DECK),
        settle_on_empty_deck=False,
    )

    assert len(next_state.deck) == 0
    assert len(next_state.players[0].hand) == 6
    assert len(next_state.players[0].negative_cards) == 1
    assert next_state.players[0].loss_score == 5
    assert next_state.settlement_count == 0


def test_legal_actions_include_place_actions_for_current_hand() -> None:
    # 手番プレイヤーの手札から配置アクションが生成されることを確認する。
    state = GameState(
        players=(
            PlayerState(hand=(Card(Color.RED, 0),)),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        ),
        board={},
        deck=(),
    )

    actions = legal_actions(state)

    assert any(isinstance(action, PlaceCardAction) for action in actions)
