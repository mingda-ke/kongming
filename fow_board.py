from typing import Dict

import chess

from fow_rules import FoWRulesBoard
from fow_view import PiecePlacement


def combined_board(own: PiecePlacement, other: PiecePlacement, turn: chess.Color) -> chess.Board:
    """The full board of both sides' placements, with ``turn`` to move."""
    board = FoWRulesBoard(fen=None)
    other.place_on(board=board)
    own.place_on(board=board)
    board.turn = turn
    board.castling_rights = board.clean_castling_rights()
    return board


def apply_move_to_board(board: chess.Board, move: chess.Move) -> chess.Board:
    """``board`` after ``move``."""
    result = board.copy(stack=False)
    result.push(move)
    return result


def pieces_of(board: chess.Board, color: chess.Color) -> Dict[chess.Square, chess.PieceType]:
    """The squares of ``color``'s pieces on ``board`` and their types."""
    return {square: board.piece_type_at(square) for square in chess.SquareSet(board.occupied_co[color])}
