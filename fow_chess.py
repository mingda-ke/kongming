import chess
from contextlib import contextmanager
from chess import BB_FILE_H, BB_RANK_2, BB_RANK_7, BB_SQUARES, H1, SQUARES_180
from typing import Dict, Iterator

from fow_view import BoardView
from probability_model import PieceProbabilityModel
from static import (
    BLOCKED_PIECE,
    BLOCKED_SQUARE_UNICODE,
    DARK_SQUARE_UNICODE,
    EMPTY_SQUARE,
    EMPTY_SQUARE_UNICODE,
    INVISIBLE_PIECE,
)


class FoWChessBoard(chess.Board):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fog_of_war = True
        self.views: Dict[chess.Color, BoardView] = {
            chess.WHITE: BoardView(chess.WHITE, [INVISIBLE_PIECE] * 64),
            chess.BLACK: BoardView(chess.BLACK, [INVISIBLE_PIECE] * 64),
        }
        self.update_board_views()

    def push(self, move: chess.Move) -> None:
        super().push(move)
        self.update_board_views()

    def generate_view(self, color: chess.Color) -> BoardView:
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

        return BoardView(color=color, values=view)

    def update_board_views(self) -> None:
        """Refresh the cached views for both sides."""
        for color in [chess.WHITE, chess.BLACK]:
            self.views[color] = self.generate_view(color)

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

    def display(self, fow_color: chess.Color) -> str:
        """Display the board from the perspective of ``fow_color``."""
        return self.views[fow_color].display()


if __name__ == "__main__":
    import chess.pgn
    from fow_view import derive_opponent_move
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
            fog_move = derive_opponent_move(current_view, new_view)
            print(fog_move)
        print(board.display(chess.WHITE))
