import chess

from fow_chess import FoWChessBoard, FoWRulesBoard

KINGSIDE = chess.Move.from_uci("e1g1")
QUEENSIDE = chess.Move.from_uci("e1c1")


def test_castling_through_attacked_squares_is_allowed():
    # the black rook on f5 attacks f1, the square the king passes through
    board = FoWRulesBoard("4k3/8/8/5r2/8/8/8/R3K2R w KQ - 0 1")
    assert KINGSIDE not in set(chess.Board(board.fen()).generate_pseudo_legal_moves())
    assert KINGSIDE in set(board.generate_pseudo_legal_moves())


def test_castling_out_of_and_into_check_is_allowed():
    board = FoWRulesBoard("4k3/8/8/4r3/8/8/8/R3K2R w KQ - 0 1")  # king in check on the e-file
    assert {KINGSIDE, QUEENSIDE} <= set(board.generate_pseudo_legal_moves())
    board = FoWRulesBoard("4k3/8/8/6r1/8/8/8/R3K2R w KQ - 0 1")  # g1 attacked
    assert KINGSIDE in set(board.generate_pseudo_legal_moves())


def test_castling_still_needs_rights_and_an_empty_path():
    assert KINGSIDE not in set(FoWRulesBoard("4k3/8/8/8/8/8/8/R3K1NR w KQ - 0 1").generate_pseudo_legal_moves())
    assert KINGSIDE not in set(FoWRulesBoard("4k3/8/8/8/8/8/8/R3K2R w Q - 0 1").generate_pseudo_legal_moves())
    assert QUEENSIDE not in set(FoWRulesBoard("4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1").generate_pseudo_legal_moves())


def test_copies_and_views_keep_fow_rules():
    board = FoWChessBoard("4k3/8/8/5r2/8/8/8/R3K2R w KQ - 0 1")
    assert isinstance(board.copy(), FoWChessBoard)
    # visibility is the squares we can move to, so the castling square is visible
    assert board.generate_view(chess.WHITE).values[chess.G1] != -1
