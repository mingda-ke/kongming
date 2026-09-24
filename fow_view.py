import chess
from dataclasses import dataclass
from chess import BB_FILE_H, BB_SQUARES, H1, SQUARES_180, Piece
from typing import List, Set

from static import (
    BLOCKED_PIECE,
    BLOCKED_SQUARE_UNICODE,
    DARK_SQUARE_UNICODE,
    EMPTY_SQUARE,
    EMPTY_SQUARE_UNICODE,
    INVISIBLE_PIECE,
)


@dataclass(frozen=True)
class FogMove:
    """Move derived from consecutive FogView states, allowing unknown from/to squares and piece types."""

    from_square: chess.Square | None = None
    to_square: chess.Square | None = None
    piece: chess.Piece | None = None

    @property
    def is_invisible(self) -> bool:
        """The move happened entirely out of sight: nothing about it is known."""
        return self.from_square is None and self.to_square is None and self.piece is None


class FogView:
    """A per-side FoW view represent INVISIBLE_PIECE, EMPTY_SQUARE, BLOCKED_PIECE or visible Piece for each square."""

    def __init__(self, color: chess.Color, values: List[int | Piece]):
        """color as White means from white's perspective."""
        self.color = color
        self.values: List[int | Piece] = values

    def display(self) -> str:
        builder: List[str] = []
        for square in SQUARES_180:
            value = self.values[square]

            if value == BLOCKED_PIECE:
                builder.append(BLOCKED_SQUARE_UNICODE)
            elif value == EMPTY_SQUARE:
                builder.append(EMPTY_SQUARE_UNICODE)
            elif value == INVISIBLE_PIECE:
                builder.append(DARK_SQUARE_UNICODE)
            else:
                builder.append(value.unicode_symbol())

            if BB_SQUARES[square] & BB_FILE_H:
                if square != H1:
                    builder.append("\n")
            else:
                builder.append(" ")

        return "".join(builder)


def deduce_opponent_move(before: FogView, after: FogView) -> FogMove:
    """
    Deduce opponent move from two consecutive FogView snapshots by opponent move, i.e. opposite color of both FogView color.
    This seems intuitive when human looks at this, but it's actually tricky to implement. For example, discovered move (one piece move and the piece behind it becomes visible).
    So we need to find a "cause" move that results in all changes of visibility. If a knight in front of bishop moves, the bishop becomes visible and knight becomes invisible, we need to deduce that the knight moves away instead of bishop moves in.
    """
    # TODO: need a bit refactor. further distinguish piece -> empty vs piece -> invisible (and the other way around).
    changed = {
        square
        for square in chess.SQUARES
        if before.values[square] != after.values[square]
    }
    if not changed:
        return FogMove()

    # Opponent capture move, i.e. our own piece is lost.
    our_own_lost = [
        square
        for square in changed
        if isinstance(before.values[square], chess.Piece) and before.values[square].color == before.color
        and not (isinstance(after.values[square], chess.Piece) and after.values[square].color == before.color)
    ]
    if our_own_lost:
        lost_square = our_own_lost[0]
        view_of_lost_square = after.values[lost_square]
        return FogMove(
            from_square=None,
            to_square=lost_square,
            piece=view_of_lost_square if isinstance(view_of_lost_square, chess.Piece) else None
        )

    # Non-capture move
    known_from = {
        square: before.values[square]
        for square in changed
        if isinstance(before.values[square], chess.Piece)
    }
    known_to = {
        square: after.values[square]
        for square in changed
        if isinstance(after.values[square], chess.Piece)
    }

    blocked_from = [square for square in changed if before.values[square] == BLOCKED_PIECE]
    blocked_to = [square for square in changed if after.values[square] == BLOCKED_PIECE]

    # same piece observed to move from one square to another
    known_moved_piece = list(set(known_from.values()).intersection(set(known_to.values())))
    if len(known_moved_piece) == 1:
        known_moved_piece = known_moved_piece[0]
        from_square = [square for square, piece in known_from.items() if piece == known_moved_piece][0]
        to_square = [square for square, piece in known_to.items() if piece == known_moved_piece][0]
        return FogMove(
            from_square=from_square,
            to_square=to_square,
            piece=known_moved_piece,
        )

    # only see one square becomes invisible while no other visible changes, then it must be a piece moved out of visibility.
    if known_from and not known_to:
        from_square = list(known_from.keys())[0]
        return FogMove(
            from_square=from_square,
            to_square=blocked_to[0] if blocked_to else None,
            piece=before.values[from_square],
        )

    # only see one square becomes visible while no other visible changes, then it must be a piece moved into visibility.
    if not known_from and known_to:
        to_square = list(known_to.keys())[0]
        return FogMove(
            from_square=blocked_from[0] if blocked_from else None,
            to_square=to_square,
            piece=after.values[to_square],
        )

    # only see blocked squares change, then no piece type information (can be further derived in possibility model).
    if not known_from and not known_to and (blocked_from or blocked_to):
        return FogMove(
            from_square=blocked_from[0] if blocked_from else None,
            to_square=blocked_to[0] if blocked_to else None,
            piece=None
        )

    # Different pieces are observed to move in and out. Place the candidate opponent pieces together
    # with all our known pieces on a temporary board and ask which candidate square is under attack by us.
    # The under-attacked piece is the one that actually moved; the other square is the reveal/blocking effect.
    temp = chess.Board(None)

    for square, piece in known_from.items():
        temp.set_piece_at(square, piece)

    for square, piece in known_to.items():
        temp.set_piece_at(square, piece)

    for square, piece in enumerate(before.values):
        if isinstance(piece, chess.Piece) and piece.color == before.color:
            temp.set_piece_at(square, piece)

    attacked_squares = set([move.to_square for move in temp.generate_pseudo_legal_moves()])
    known_moved_piece = attacked_squares.intersection(set(known_from.keys()).union(set(known_to.keys())))
    known_moved_piece = list(known_moved_piece)[0]
    return FogMove(
        from_square=[square for square, piece in known_from.items() if piece == known_moved_piece][0] if known_moved_piece in known_from.values() else None,
        to_square=[square for square, piece in known_to.items() if piece == known_moved_piece][0] if known_moved_piece in known_to.values() else None,
        piece=known_moved_piece
    )