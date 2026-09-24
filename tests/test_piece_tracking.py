from pathlib import Path
from typing import Iterable

import chess


from piece_tracking import PieceTracking, PieceTrackingModel, SpecificPiece
from fow_chess import FoWChessBoard


def replay_moves(moves: Iterable[chess.Move | str], color: chess.Color = chess.WHITE) -> PieceTrackingModel:
    """
    Play ``moves`` (``chess.Move`` or UCI strings) from the starting position and return the belief state model of
    ``color`` updated after every move.
    """
    board = FoWChessBoard()
    model = PieceTrackingModel(color, board.generate_view(color))
    for move in moves:
        move = chess.Move.from_uci(move) if isinstance(move, str) else move
        mover = board.turn
        board.push(move)
        view = board.generate_view(color)
        if mover == color:
            model.update_own_move(move, view)
        else:
            model.update_opponent_move(view)
    return model


def test_full_consumed_move_counts():
    """
    After these moves, as black, we know exactly what white played.
    """
    moves = ["d2d4", "e7e6", "c2c4", "c7c5"]
    bsm = replay_moves(moves, color=chess.BLACK)
    assert bsm.piece_tracking.invisible_move_count==0
    assert all(len(bsm.piece_tracking.possible_squares(piece))==1 for piece in bsm.piece_tracking.evolutions.keys())


def test_sliding_piece_clears_its_path():
    """
    As black, the white queen is seen landing on h5 from d1: the squares between must have been empty, so no white
    piece can be on e2, f3 or g4.
    """
    bsm = replay_moves(["e2e4", "e7e5", "d1h5"], color=chess.BLACK)
    tracking = bsm.piece_tracking
    assert tracking.determined_squares()[SpecificPiece(chess.WHITE, chess.QUEEN)] == chess.H5
    for square in (chess.E2, chess.F3, chess.G4):
        assert not tracking.candidates_at(None, square)


def test_cleared_path_only_for_sliding_pieces():
    rook = SpecificPiece(chess.WHITE, chess.ROOK, 0)
    bishop = SpecificPiece(chess.WHITE, chess.BISHOP, 5)
    knight = SpecificPiece(chess.WHITE, chess.KNIGHT, 6)
    assert PieceTracking.cleared_path(rook, chess.A1, chess.A4) == {chess.A2, chess.A3}
    assert PieceTracking.cleared_path(bishop, chess.F1, chess.B5) == {chess.E2, chess.D3, chess.C4}
    assert PieceTracking.cleared_path(knight, chess.G1, chess.F3) == set()


if __name__=="__main__":
    test_full_consumed_move_counts()
    test_sliding_piece_clears_its_path()
    test_cleared_path_only_for_sliding_pieces()

def test_place_piece_consumes_moves():
    """Black made one unseen move; placing the g-knight on f6 uses it, so every other piece is back home."""
    model = replay_moves(["e2e4", "g8f6"], color=chess.WHITE)
    tracking = model.piece_tracking
    knight = SpecificPiece(chess.BLACK, chess.KNIGHT, 6)
    assert tracking.invisible_move_count == 1
    model.place_piece(knight, chess.F6)
    assert tracking.invisible_move_count == 0
    assert all(len(tracking.possible_squares(piece)) == 1 for piece in tracking.evolutions)


def test_allows_boards_with_tracked_pieces_on_possible_squares():
    model = replay_moves(["e2e4", "g8f6"], color=chess.WHITE)
    real = chess.Board()
    real.push_uci("e2e4")
    real.push_uci("g8f6")
    assert model.allows(real)
    queen_jumped = real.copy()
    queen_jumped.remove_piece_at(chess.D8)
    queen_jumped.set_piece_at(chess.A3, chess.Piece(chess.QUEEN, chess.BLACK))
    assert not model.allows(queen_jumped)
