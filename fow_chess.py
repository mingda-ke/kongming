import chess
from contextlib import contextmanager
from chess import BB_FILE_H, BB_RANK_2, BB_RANK_7, BB_SQUARES, H1, SQUARES_180
from typing import Dict, Iterator

from fow_view import FogView
from static import (
    BLOCKED_PIECE,
    BLOCKED_SQUARE_UNICODE,
    DARK_SQUARE_UNICODE,
    EMPTY_SQUARE,
    EMPTY_SQUARE_UNICODE,
    INVISIBLE_PIECE,
)


class FoWRulesBoard(chess.Board):
    """
    A board with the move rules of FoW chess, which differ from chess in two ways: there is no check, so a king may
    move into check (pseudo-legal moves are the legal moves), and castling is allowed through attacked squares.
    ``generate_view`` gives what each side sees: the squares it can move to or attack, and its blocked pawn pushes.
    """

    def generate_castling_moves(
        self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL
    ) -> Iterator[chess.Move]:
        """Castling as in chess, but the king may be in check or pass through or land on attacked squares."""
        if self.is_variant_end():
            return
        backrank = chess.BB_RANK_1 if self.turn == chess.WHITE else chess.BB_RANK_8
        king = self.occupied_co[self.turn] & self.kings & ~self.promoted & backrank & from_mask
        king &= -king
        if not king:
            return

        for candidate in chess.scan_reversed(self.clean_castling_rights() & backrank & to_mask):
            rook = chess.BB_SQUARES[candidate]
            a_side = rook < king
            king_to = (chess.BB_FILE_C if a_side else chess.BB_FILE_G) & backrank
            rook_to = (chess.BB_FILE_D if a_side else chess.BB_FILE_F) & backrank
            king_path = chess.between(chess.msb(king), chess.msb(king_to))
            rook_path = chess.between(candidate, chess.msb(rook_to))
            if not (self.occupied ^ king ^ rook) & (king_path | rook_path | king_to | rook_to):
                yield self._from_chess960(self.chess960, chess.msb(king), candidate)

    def generate_view(self, color: chess.Color) -> FogView:
        """Generate the FoW view for one side from the current board state."""
        visible_mask = self._get_visibility_mask(color)
        blocked_mask = self._get_blockage_mask(color)
        view = [INVISIBLE_PIECE] * 64

        for square in chess.SQUARES:
            if visible_mask & chess.BB_SQUARES[square]:
                piece = self.piece_at(square)
                view[square] = piece if piece is not None else EMPTY_SQUARE
            if blocked_mask & chess.BB_SQUARES[square]:
                view[square] = BLOCKED_PIECE

        return FogView(color=color, values=view)

    @contextmanager
    def _turn_as(self, color: chess.Color) -> Iterator[None]:
        previous_turn = self.turn
        self.turn = color
        try:
            yield
        finally:
            self.turn = previous_turn

    def _get_visibility_mask(self, color: chess.Color) -> chess.Bitboard:
        """Return the squares visible to ``color`` in the current position."""
        visible = self.occupied_co[color]

        for square in chess.scan_forward(self.occupied_co[color]):
            visible |= self.attacks_mask(square)

        with self._turn_as(color):
            for move in self.generate_pseudo_legal_moves():
                visible |= chess.BB_SQUARES[move.to_square]

        return visible

    def _get_blockage_mask(self, color: chess.Color) -> chess.Bitboard:
        """Return opposing squares that block a legal pawn advance for ``color``."""
        pawn_mask = self.pieces_mask(chess.PAWN, color)
        double_move_mask = (pawn_mask & BB_RANK_2) << 16 if color == chess.WHITE else (pawn_mask & BB_RANK_7) >> 16
        single_move_mask = (pawn_mask & ~BB_RANK_2) << 8 if color == chess.WHITE else (pawn_mask & ~BB_RANK_7) >> 8
        pawn_advances = double_move_mask | single_move_mask
        opponent_mask = self.occupied_co[not color]
        return pawn_advances & opponent_mask


class FoWChessBoard(FoWRulesBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fog_of_war = True
        self.views: Dict[chess.Color, FogView] = {
            chess.WHITE: FogView(chess.WHITE, [INVISIBLE_PIECE] * 64),
            chess.BLACK: FogView(chess.BLACK, [INVISIBLE_PIECE] * 64),
        }
        self.update_board_views()

    def push(self, move: chess.Move) -> None:
        super().push(move)
        self.update_board_views()

    def update_board_views(self) -> None:
        """Refresh the cached views for both sides."""
        for color in [chess.WHITE, chess.BLACK]:
            self.views[color] = self.generate_view(color)

    def display(self, fow_color: chess.Color) -> str:
        """Display the board from the perspective of ``fow_color``."""
        return self.views[fow_color].display()


if __name__ == "__main__":
    import chess.pgn
    from fow_view import deduce_opponent_move
    # move_sequence = [
    #     chess.Move.from_uci("e2e4"),
    #     chess.Move.from_uci("e7e5"),
    #     chess.Move.from_uci("f1c4"),
    #     chess.Move.from_uci("d8h4"),
    # ]

    move_sequence = []
    with open("mingdak_vs_antichessbob.pgn") as pgn_file:
        for raw_line in pgn_file:
            line = "".join([c for c in raw_line.strip().split(".", 1)[1].strip() if not c.isupper()])
            move_sequence.extend(line.split())

    board = FoWChessBoard()
    for move in move_sequence:
        move_obj = chess.Move.from_uci(move)
        print(board.turn)
        print(move_obj)
        current_view = board.generate_view(chess.WHITE)
        board.push(move_obj)
        new_view = board.generate_view(chess.WHITE)
        if board.turn == chess.WHITE:
            fog_move = deduce_opponent_move(current_view, new_view)
            print(fog_move)
        print(board.display(chess.WHITE))
