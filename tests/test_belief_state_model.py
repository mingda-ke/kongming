

import chess

from belief_state_model import BeliefStateModel
from fow_chess import FoWChessBoard
from piece_tracking import SpecificPiece


def test_e4_e6_Bc4():
    # Test the e4 move and its effect on the belief state model
    board = FoWChessBoard()
    model = BeliefStateModel(color=chess.WHITE, sizes=(10, 3))
    
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
    assert len(model.simulations) == 10, "There should be 10 black simulations after e6"
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
    assert len(model.simulations) == 10, "There should be 10 black simulations after Nf6"

    assert all(
        simulation.view.values[chess.E7] != chess.Piece(piece_type=chess.KING, color=chess.BLACK)
        for simulation in model.simulations
    ), "A black king on e7 is bad move and should not appear in any simulation"

    assert all(
        simulation.view.values[chess.G5] != chess.Piece(piece_type=chess.PAWN, color=chess.BLACK)
        for simulation in model.simulations
    ), "A black pawn on g5 is bad move and should not appear in any simulation"



if __name__ == "__main__":
    test_e4_e6_Bc4()
    print("All tests passed.")