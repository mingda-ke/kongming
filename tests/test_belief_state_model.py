

import chess

from belief_state_model import BeliefStateModel
from config import BeliefStateConfig
from fow_board import pieces_of
from fow_chess import FoWChessBoard
from piece_tracking import SpecificPiece


def test_e4_e6_Bc4_Nf6():
    # Test the e4 move and its effect on the belief state model
    board = FoWChessBoard()
    model = BeliefStateModel(color=chess.WHITE, seed=123, config=BeliefStateConfig(sizes=(10, 3), num_of_top_moves=5, opponent_strength=100))
    
    # white: e4
    e4_move = chess.Move.from_uci("e2e4")
    board.push(e4_move)
    model.update_own_move(move=e4_move, view=board.generate_view(color=chess.WHITE))
    
    assert model.view.values[chess.E4].piece_type == chess.PAWN, "Pawn should be at e4 after the move"
    assert len(model.simulations) == 1, "Opponent black should have one simulation after the move"
    assert len(model.simulations[0].simulations) == 3, "There should be 3 simulations on opponent BSM according to sizes"
    assert model.simulations[0].piece_tracking.invisible_move_count == 1, "There should be one invisible move counted in the opponent's piece tracking"
    assert model.simulations[0].piece_tracking.evolutions[SpecificPiece(piece_type=chess.PAWN, color=chess.WHITE, file=4)][1] == {chess.E3, chess.E4}

    # black: e6
    e6_move = chess.Move.from_uci("e7e6")
    board.push(e6_move)
    model.update_opponent_move(view=board.generate_view(color=chess.WHITE))
    assert model.piece_tracking.invisible_move_count == 1, "There should be one invisible move counted in the opponent's piece tracking after e6"
    assert model.piece_tracking.evolutions[SpecificPiece(piece_type=chess.PAWN, color=chess.BLACK, file=4)][1] == {chess.E6}, "E pawn should only be possible to advance to e6 after the move as e5 is empty"
    assert model.piece_tracking.evolutions[SpecificPiece(piece_type=chess.PAWN, color=chess.BLACK, file=3)][1] == {chess.D6}, "D pawn should only be possible to advance to d6 after the move as d5 is empty"
    assert all(simulation.color == chess.BLACK for simulation in model.simulations), "All simulations should be for black"

    # white: Bc4
    bc4_move = chess.Move.from_uci("f1c4")
    board.push(bc4_move)
    model.update_own_move(move=bc4_move, view=board.generate_view(color=chess.WHITE))
    assert len(model.simulations) == 1, "There should one black simulation left as Bc4 sees the black pawn on e6"
    assert model.piece_tracking.invisible_move_count == 0, "Invisible move count reduced to 0"
    assert max(age for evolution in model.piece_tracking.evolutions.values() for age in evolution.keys())==0, "All evolutions should be age 0 after Bc4 as the black pawn on e6 is seen"
    assert model.piece_tracking.evolutions[SpecificPiece(piece_type=chess.PAWN, color=chess.BLACK, file=4)][0] == {chess.E6}, "E pawn should only be possible to advance to e6 after the move as e5 is empty"
    assert len(model.simulations[0].simulations)==3, "There should be 3 simulations on opponent BSM according to sizes"

    # black: Nf6
    nf6_move = chess.Move.from_uci("g8f6")
    board.push(nf6_move)
    model.update_opponent_move(view=board.generate_view(color=chess.WHITE))

    assert all(
        simulation.view.values[chess.E7] != chess.Piece(piece_type=chess.KING, color=chess.BLACK)
        for simulation in model.simulations
    ), "A black king on e7 is bad move and should not appear in any simulation"

    assert all(
        simulation.view.values[chess.G5] != chess.Piece(piece_type=chess.PAWN, color=chess.BLACK)
        for simulation in model.simulations
    ), "A black pawn on g5 is bad move and should not appear in any simulation"

    c4_blockers = {chess.Piece(piece_type=chess.BISHOP, color=chess.WHITE), chess.Piece(piece_type=chess.PAWN, color=chess.WHITE)}
    assert all(
        white_simulation.view.values[chess.C4] in c4_blockers
        for simulation in model.simulations
        if simulation.view.values[chess.C5] == chess.Piece(piece_type=chess.PAWN, color=chess.BLACK)
        for white_simulation in simulation.simulations
    ), "A black simulation with a pawn on c5 sees c4 blocked, so its white simulations should have a bishop or pawn on c4"

    # White: Nf3
    nf3_move = chess.Move.from_uci("g1f3")
    board.push(nf3_move)
    model.update_own_move(move=nf3_move, view=board.generate_view(color=chess.WHITE))
    assert len(model.simulations[0].simulations) == 3, "There should be 3 white simulations from black perspective"

    # Black: g6
    g6_move = chess.Move.from_uci("g7g6")
    board.push(g6_move)
    model.update_opponent_move(view=board.generate_view(color=chess.WHITE))
    assert len(model.simulations) == 10, "There should be 10 black simulations"

    e6_only_board = chess.Board()
    e6_only_board.push(chess.Move.from_uci("e2e4"))
    e6_only_board.push(chess.Move.from_uci("e7e6"))
    e6_only_pieces = pieces_of(board=e6_only_board, color=chess.BLACK)
    assert all(simulation.pieces != e6_only_pieces for simulation in model.simulations), \
        "A retreating move (e.g. Ng8 back from f6) wastes a tempo and should not leave a black simulation with only e6 played"

    # Push remaining moves before revealing king is castled (normal invisible moves and one capture move)
    remaining_moves = ["e1g1", "f8g7", "d2d3", "e8g8", "b1c3", "d7d5", "e4d5", "e6d5", "c4b3", "c7c6"]
    for uci in remaining_moves:
        move = chess.Move.from_uci(uci)
        board.push(move)
        if board.turn == chess.BLACK:
            model.update_own_move(move=move, view=board.generate_view(color=chess.WHITE))
        else:
            model.update_opponent_move(view=board.generate_view(color=chess.WHITE))
    assert len(model.simulations) == 10, "There should be 10 black simulations after the 20th ply"

    # Moves that reveal king is not on e file (castled)
    # "f1e1", "b8d7", "c1g5", "h7h6"

if __name__ == "__main__":
    test_e4_e6_Bc4_Nf6()
    print("All tests passed.")