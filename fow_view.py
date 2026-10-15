import random

import chess
from dataclasses import dataclass
from chess import BB_FILE_H, BB_SQUARES, H1, SQUARES_180, Piece
from typing import Dict, List, Set

from fow_canvas import show_view
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


@dataclass(frozen=True)
class PiecePlacement:
    """Where one side's pieces are: their squares and types."""

    color: chess.Color
    pieces: Dict[chess.Square, chess.PieceType]

    def place_on(self, board: chess.Board) -> None:
        """Set the pieces on ``board``."""
        for square, piece_type in self.pieces.items():
            board.set_piece_at(square=square, piece=Piece(piece_type=piece_type, color=self.color))

    def override_with(self, other: "PiecePlacement") -> "PiecePlacement":
        """
        This placement with ``other``'s pieces put on their squares. Each piece of ``other`` takes the place of a piece
        of its type here: the one already on its square if any, else a random one, preferring one ``other`` displaces.
        Pieces left on a square of ``other`` are dropped.
        """
        if other.color != self.color:
            raise ValueError("Cannot override a placement with the other side's pieces")
        remaining = {square: piece_type for square, piece_type in self.pieces.items() if other.pieces.get(square) != piece_type}
        for square, piece_type in other.pieces.items():
            if self.pieces.get(square) == piece_type:
                continue
            candidates = [candidate for candidate, candidate_type in remaining.items() if candidate_type == piece_type]
            displaced = [candidate for candidate in candidates if candidate in other.pieces]
            if candidates:
                del remaining[random.choice(seq=displaced or candidates)]
        kept = {square: piece_type for square, piece_type in remaining.items() if square not in other.pieces}
        return PiecePlacement(color=self.color, pieces={**kept, **other.pieces})


class FogView:
    """A per-side FoW view represent INVISIBLE_PIECE, EMPTY_SQUARE, BLOCKED_PIECE or visible Piece for each square."""

    def __init__(self, color: chess.Color, values: List[int | Piece]):
        """color as White means from white's perspective."""
        self.color = color
        self.values: List[int | Piece] = values

    @classmethod
    def initial(cls, color: chess.Color) -> "FogView":
        """
        ``color``'s view of the standard starting position: his own pieces, and the two empty ranks in front of them
        that his pawns (and knights) see. The rest is invisible.
        """
        values: List[int | Piece] = [INVISIBLE_PIECE] * 64
        for square, piece in chess.Board().piece_map().items():
            if piece.color == color:
                values[square] = piece
        for rank in ((2, 3) if color == chess.WHITE else (5, 4)):
            for file in range(8):
                values[chess.square(file, rank)] = EMPTY_SQUARE
        return cls(color=color, values=values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FogView) and self.color == other.color and self.values == other.values

    def __hash__(self) -> int:
        return hash((self.color, tuple(self.values)))

    def generate_piece_placement(self) -> PiecePlacement:
        """The placement of the view's own side, whose pieces are always visible to it."""
        return PiecePlacement(color=self.color, pieces={
            square: value.piece_type for square, value in enumerate(self.values)
            if isinstance(value, Piece) and value.color == self.color
        })

    def display(self, title: str | None = None) -> None:
        """Show the view in a window (titled by its side by default), returning when it is closed."""
        show_view(self, title or f"{chess.COLOR_NAMES[self.color].capitalize()}'s view")

    def __str__(self) -> str:
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


def is_piece_move(piece: Piece, from_square: chess.Square, to_square: chess.Square) -> bool:
    """
    Whether ``piece`` can move from ``from_square`` to ``to_square`` without capturing, on an otherwise empty board: a
    pawn only pushes forward (two squares from its starting rank), since its diagonal moves are captures. A king may
    also castle from its starting square.
    """
    if piece.piece_type == chess.KING and from_square == (chess.E1 if piece.color == chess.WHITE else chess.E8):
        if to_square in {from_square - 2, from_square + 2}:
            return True
    board = chess.Board(None)
    board.set_piece_at(from_square, piece)
    board.turn = piece.color
    # compare squares only, so a promoting push counts too
    return any(move.to_square == to_square for move in board.generate_pseudo_legal_moves(from_mask=BB_SQUARES[from_square]))


def can_piece_capture(piece: Piece, from_square: chess.Square, to_square: chess.Square) -> bool:
    """Whether ``piece`` can capture from ``from_square`` on ``to_square`` on an otherwise empty board."""
    board = chess.Board(None)
    board.set_piece_at(from_square, piece)
    board.set_piece_at(to_square, Piece(chess.PAWN, not piece.color))
    board.turn = piece.color
    return any(move.to_square == to_square for move in board.generate_pseudo_legal_moves(from_mask=BB_SQUARES[from_square]))


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
        piece = view_of_lost_square if isinstance(view_of_lost_square, chess.Piece) else None
        from_squares = [
            square for square in changed
            if isinstance(before.values[square], chess.Piece) and before.values[square].color != before.color
            and after.values[square] == EMPTY_SQUARE
            and (piece is None or before.values[square] == piece)
            and can_piece_capture(before.values[square], square, lost_square)
        ]
        from_square = from_squares[0] if len(from_squares) == 1 else None
        return FogMove(
            from_square=from_square,
            to_square=lost_square,
            piece=piece or (before.values[from_square] if from_square is not None else None),
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

    # same piece observed to move from one square to another, by a move its type can make
    known_moves = [
        (from_square, to_square, piece)
        for from_square, piece in known_from.items()
        for to_square, other in known_to.items()
        if piece == other and is_piece_move(piece, from_square, to_square)
    ]
    if len(known_moves) == 1:
        from_square, to_square, piece = known_moves[0]
        return FogMove(
            from_square=from_square,
            to_square=to_square,
            piece=piece,
        )

    # only see one square becomes invisible while no other visible changes, then it must be a piece moved out of visibility.
    if known_from and not known_to:
        from_square = list(known_from.keys())[0]
        piece = before.values[from_square]
        # a newly blocked square is its destination only if the piece can move there
        to_squares = [square for square in blocked_to if is_piece_move(piece, from_square, square)]
        return FogMove(
            from_square=from_square,
            to_square=to_squares[0] if to_squares else None,
            piece=piece,
        )

    # only see one square becomes visible while no other visible changes, then it must be a piece moved into visibility.
    if not known_from and known_to:
        to_square = list(known_to.keys())[0]
        piece = after.values[to_square]
        # a newly unblocked square is its origin only if the piece can move from there
        from_squares = [square for square in blocked_from if is_piece_move(piece, square, to_square)]
        return FogMove(
            from_square=from_squares[0] if from_squares else None,
            to_square=to_square,
            piece=piece,
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
    # the squares we attack
    temp.turn = before.color

    attacked_squares = set([move.to_square for move in temp.generate_pseudo_legal_moves()])
    moved_squares = attacked_squares.intersection(set(known_from.keys()).union(set(known_to.keys())))
    if not moved_squares:
        # no telling which piece moved: an unseen move, while the observed squares still update the tracking
        return FogMove()
    moved_square = sorted(moved_squares)[0]
    if moved_square in known_from:
        return FogMove(from_square=moved_square, to_square=None, piece=known_from[moved_square])
    return FogMove(from_square=None, to_square=moved_square, piece=known_to[moved_square])