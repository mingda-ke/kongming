import chess
from contextlib import contextmanager
from chess import BB_FILE_H, BB_SQUARES, H1, SQUARES_180, BB_RANK_2, BB_RANK_7
from typing import Iterator, List

EMPTY_SQUARE_UNICODE = "⭘"
DARK_SQUARE_UNICODE = "■"
BLOCKED_SQUARE_UNICODE = "□"


class FoWChessBoard(chess.Board):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fog_of_war = True

    def _turn_as(self, color: chess.Color) -> Iterator[None]:
        previous_turn = self.turn
        self.turn = color
        try:
            yield
        finally:
            self.turn = previous_turn

    def _visibility_mask(self, color: chess.Color) -> chess.Bitboard:
        """Return the squares visible to ``color`` in the current position."""
        visible = self.occupied_co[color]

        for square in chess.scan_forward(self.occupied_co[color]):
            visible |= self.attacks_mask(square)

        with self._turn_as(color):
            for move in self.generate_pseudo_legal_moves():
                visible |= chess.BB_SQUARES[move.to_square]

        return visible

    def is_visible(
        self, square: chess.Square, color: chess.Color | None = None
    ) -> bool:
        """Whether ``square`` can be seen from ``color``'s point of view."""
        if color is None:
            color = not self.turn
        return bool(self._visibility_mask(color) & chess.BB_SQUARES[square])

    def _blocked_mask(self, color: chess.Color) -> chess.Bitboard:
        """Return opposing squares that block a legal pawn advance for ``color``."""
        pawn_mask = self.pieces_mask(chess.PAWN, color)
        double_move_mask = (pawn_mask & BB_RANK_2) << 16 if color == chess.WHITE else (pawn_mask & BB_RANK_7) >> 16
        single_move_mask = (pawn_mask & ~BB_RANK_2) << 8 if color == chess.WHITE else (pawn_mask & ~BB_RANK_7) >> 8
        pawn_advances = double_move_mask | single_move_mask
        opponent_mask = self.occupied_co[not color]
        # TODO: rare in practice, but need to consider when both double move and single move advances are blocked, need to only show single move.
        return pawn_advances & opponent_mask

    def is_blocked(
        self, square: chess.Square, color: chess.Color | None = None
    ) -> bool:
        """Whether an opposing piece blocks a legal pawn advance to ``square``."""
        if color is None:
            color = not self.turn
        return bool(self._blocked_mask(color) & chess.BB_SQUARES[square])

    def display(self, fow_color: chess.Color) -> str:
        """Display the board from the perspective of ``fow_color``."""
        builder: List[str] = []

        for square in SQUARES_180:
            piece = self.piece_at(square)

            if self.is_blocked(square, fow_color):
                builder.append(BLOCKED_SQUARE_UNICODE)
            elif piece and self.is_visible(square, fow_color):
                builder.append(piece.unicode_symbol())
            elif piece is None and self.is_visible(square, fow_color):
                builder.append(EMPTY_SQUARE_UNICODE)
            else:
                builder.append(DARK_SQUARE_UNICODE)

            if BB_SQUARES[square] & BB_FILE_H:
                if square != H1:
                    builder.append("\n")
            else:
                builder.append(" ")

        return "".join(builder)


if __name__ == "__main__":
    board = FoWChessBoard()
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    board.push(chess.Move.from_uci("f1c4"))
    board.push(chess.Move.from_uci("d8h4"))
    print(board.display(chess.WHITE))