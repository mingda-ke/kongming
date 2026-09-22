from __future__ import annotations
from logging import log

import chess
from dataclasses import dataclass
from typing import Dict, Set

from fow_view import BoardView
from static import INVISIBLE_PIECE


@dataclass(frozen=True)
class SpecificPiece:
    """Represent a specific piece on the board."""

    color: chess.Color
    piece_type: chess.PieceType
    file: int | None = None

    def __post_init__(self) -> None:
        allowed_files = {
            chess.PAWN: set(range(8)),
            chess.KNIGHT: {chess.FILE_NAMES.index("b"), chess.FILE_NAMES.index("g")},
            chess.ROOK: {chess.FILE_NAMES.index("a"), chess.FILE_NAMES.index("h")},
            chess.BISHOP: {chess.FILE_NAMES.index("c"), chess.FILE_NAMES.index("f")},
        }
        if self.piece_type in allowed_files and self.file is None:
            raise ValueError("A pawn, knight, or rook identity requires its file")
        if self.file is not None and not 0 <= self.file < 8:
            raise ValueError("file must be between 0 and 7")
        if self.piece_type in allowed_files and self.file not in allowed_files[self.piece_type]:
            piece_name = chess.piece_name(self.piece_type)
            valid_files = ", ".join(chess.FILE_NAMES[file] for file in sorted(allowed_files[self.piece_type]))
            raise ValueError(f"A {piece_name} must use file {valid_files}")



def compare_views(before: BoardView, after: BoardView) -> Set[chess.Square]:
    """Return the squares whose visible state changed between two views."""
    return {
        square
        for square in chess.SQUARES
        if before.values[square] != after.values[square]
    }


class PieceProbabilityModel:
    """Track possible positions of hidden pieces from color perspective based on observed views."""

    def __init__(self, color: chess.Color):
        """
        piece_possible_positions is a dictionary mapping SpecificPiece to a dictionary of age to set of possible squares.
        only store opponent's pieces, i.e. if color is white, only store black pieces.
        """
        self.color = color
        self.current_view = BoardView(self.color, [INVISIBLE_PIECE] * 64)
        self.piece_possible_positions = self._initialize_piece_positions()

    def _initialize_piece_positions(self) -> Dict[SpecificPiece, Dict[int, Set[chess.Square]]]:
        """
        Initial piece positions are determined by the standard starting position of a chess game. 
        """
        starting_board = chess.Board()
        positions: Dict[SpecificPiece, Dict[int, Set[chess.Square]]] = {}
        for square, piece in starting_board.piece_map().items():
            if piece.color == self.color:
                continue
            file = (
                chess.square_file(square)
                if piece.piece_type in {
                    chess.PAWN,
                    chess.KNIGHT,
                    chess.BISHOP,
                    chess.ROOK,
                }
                else None
            )
            specific_piece = SpecificPiece(piece.color, piece.piece_type, file)
            positions[specific_piece] = {0: {square}}
        return positions

    def _reset_seen_pieces(self, view: BoardView) -> None:
        """
        When a visible piece is re-observed, reset its age map to the current square.
        We need to specify which specific piece it is, e.g. which specific knight or rook, by looking up their possible positions. However, when both specific pieces are possible, we should not randomly pick one because later on it may not reconcile.
        For simplicity, we will pick one of them for now.
        """
        for square in chess.SQUARES:
            # update for visible and opponent pieces
            if isinstance(view.values[square], chess.Piece) and view.values[square].color != self.color:
                possible_specific_piece = list(piece for piece, possible_squares in self.piece_possible_positions.items() if piece.piece_type == view.values[square].piece_type and square in set().union(*possible_squares.values()))
                if len(possible_specific_piece) == 1:
                    specific_piece = possible_specific_piece[0]
                elif len(possible_specific_piece) > 1:
                    # TODO: deal with this situation smartly.
                    log.warning(f"Multiple possible specific pieces found for visible piece at {chess.square_name(square)}. Picking one arbitrarily.")
                    specific_piece = possible_specific_piece[0]
                else:
                    log.warning(f"Visible piece at {chess.square_name(square)} not found in possible positions.")
                self.piece_possible_positions[specific_piece] = {0: {square}}

    def update(self, view: BoardView, turn_color: chess.Color) -> None:
        """
        Visible pieces are always reset to their observed square, regardless of whose turn it is.
        When the opponent moves, compare the new view against the previous one and infer the destination square.
        """
        self._reset_seen_pieces(view)

        if turn_color != self.color:
            changed = compare_views(self.current_view, view)
            for square in changed:
                value = view.values[square]
                if not isinstance(value, chess.Piece):
                    continue
                for piece, history in list(self.piece_possible_positions.items()):
                    if value.color != piece.color or value.piece_type != piece.piece_type:
                        continue
                    if square in history.get(0, set()):
                        continue
                    self.piece_possible_positions[piece] = {0: {square}}
                    break

        self.current_view = view

    def _next_hidden_piece_squares(
        self,
        board: chess.Board,
        piece: SpecificPiece,
        origin: chess.Square,
        visible: Set[chess.Square],
        invisible: Set[chess.Square],
    ) -> Set[chess.Square]:
        """Return legal destinations under the visible-piece assumption."""
        assumed_board = board.copy()

        for square in set(assumed_board.piece_map()) - visible:
            assumed_board.remove_piece_at(square)
        for square in invisible:
            assumed_board.remove_piece_at(square)

        assumed_board.remove_piece_at(origin)
        assumed_board.set_piece_at(
            origin,
            chess.Piece(piece.piece_type, piece.color),
        )
        assumed_board.turn = piece.color

        return {
            move.to_square
            for move in assumed_board.generate_legal_moves(
                from_mask=chess.BB_SQUARES[origin],
            )
        }

    

if __name__ == "__main__":
    model = PieceProbabilityModel(chess.WHITE)
    print(model.piece_possible_positions)