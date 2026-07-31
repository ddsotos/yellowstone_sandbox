"""JSON-friendly serialization helpers for game state and actions."""

from __future__ import annotations

from typing import Any

from yellowstone.types import (
    Action,
    Board,
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


JsonDict = dict[str, Any]


def card_to_dict(card: Card) -> JsonDict:
    return {
        "color": card.color.value,
        "rank_index": card.rank_index,
    }


def card_from_dict(data: JsonDict) -> Card:
    return Card(
        color=Color(data["color"]),
        rank_index=int(data["rank_index"]),
    )


def position_to_dict(position: Position) -> JsonDict:
    return {
        "x": position.x,
        "y": position.y,
    }


def position_from_dict(data: JsonDict) -> Position:
    return Position(x=int(data["x"]), y=int(data["y"]))


def frame_to_dict(frame: Frame) -> JsonDict:
    return {
        "x": frame.x,
        "y": frame.y,
    }


def frame_from_dict(data: JsonDict) -> Frame:
    return Frame(x=int(data["x"]), y=int(data["y"]))


def player_state_to_dict(player: PlayerState) -> JsonDict:
    return {
        "hand": [card_to_dict(card) for card in player.hand],
        "negative_cards": [card_to_dict(card) for card in player.negative_cards],
        "loss_score": player.loss_score,
    }


def player_state_from_dict(data: JsonDict) -> PlayerState:
    return PlayerState(
        hand=tuple(card_from_dict(card) for card in data["hand"]),
        negative_cards=tuple(
            card_from_dict(card) for card in data["negative_cards"]
        ),
        loss_score=int(data["loss_score"]),
    )


def board_to_dict(board: Board) -> list[JsonDict]:
    return [
        {
            "position": position_to_dict(position),
            "stack": [card_to_dict(card) for card in stack],
        }
        for position, stack in sorted(board.items(), key=lambda item: item[0])
    ]


def board_from_dict(data: list[JsonDict]) -> Board:
    return {
        position_from_dict(cell["position"]): tuple(
            card_from_dict(card) for card in cell["stack"]
        )
        for cell in data
    }


def game_state_to_dict(state: GameState) -> JsonDict:
    return {
        "players": [player_state_to_dict(player) for player in state.players],
        "board": board_to_dict(state.board),
        "deck": [card_to_dict(card) for card in state.deck],
        "current_player_index": state.current_player_index,
        "phase": state.phase.value,
        "cards_played_this_turn": state.cards_played_this_turn,
        "winners": list(state.winners),
        "settlement_count": state.settlement_count,
        "last_turn_play_counts": list(state.last_turn_play_counts),
    }


def game_state_from_dict(data: JsonDict) -> GameState:
    return GameState(
        players=tuple(player_state_from_dict(player) for player in data["players"]),
        board=board_from_dict(data["board"]),
        deck=tuple(card_from_dict(card) for card in data["deck"]),
        current_player_index=int(data["current_player_index"]),
        phase=Phase(data["phase"]),
        cards_played_this_turn=int(data["cards_played_this_turn"]),
        winners=tuple(int(winner) for winner in data["winners"]),
        settlement_count=int(data["settlement_count"]),
        last_turn_play_counts=tuple(
            int(count) for count in data.get("last_turn_play_counts", ())
        ),
    )


def action_to_dict(action: Action) -> JsonDict:
    if isinstance(action, PlaceCardAction):
        return {
            "type": "place_card",
            "hand_index": action.hand_index,
            "position": position_to_dict(action.position),
            "frame": frame_to_dict(action.frame),
        }
    if isinstance(action, EndTurnAction):
        return {"type": "end_turn"}
    if isinstance(action, RefillAction):
        return {
            "type": "refill",
            "source": action.source.value,
        }
    raise TypeError(f"unsupported action: {action!r}")


def action_from_dict(data: JsonDict) -> Action:
    action_type = data["type"]
    if action_type == "place_card":
        return PlaceCardAction(
            hand_index=int(data["hand_index"]),
            position=position_from_dict(data["position"]),
            frame=frame_from_dict(data["frame"]),
        )
    if action_type == "end_turn":
        return EndTurnAction()
    if action_type == "refill":
        return RefillAction(source=RefillSource(data["source"]))
    raise ValueError(f"unsupported action type: {action_type}")


def actions_to_dicts(actions: tuple[Action, ...]) -> list[JsonDict]:
    return [action_to_dict(action) for action in actions]


def actions_from_dicts(data: list[JsonDict]) -> tuple[Action, ...]:
    return tuple(action_from_dict(action) for action in data)
