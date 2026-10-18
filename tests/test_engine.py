import chess

from belief_state_model import BeliefStateModel
from engine import Engine
from evaluator import Evaluator, aggregate, search
from fow_board import combined_board
from fow_chess import FoWChessBoard
from tests.helpers import FakeEngine, requires_stockfish
from tests.test_belief_state_model import follow


def test_aggregate_mean_and_percentile():
    assert aggregate([0, 100, 200, 500], chess.WHITE) == 200
    assert aggregate([0, 100, 200, 500], chess.WHITE, percentile=25) == 75
    assert aggregate([0, 100, 200, 500], chess.BLACK, percentile=25) == 275
    assert aggregate([0, 100, 200, 500], chess.WHITE, percentile=0) == 0


def test_capturing_the_king_ends_the_game():
    board = FoWChessBoard("4k3/8/8/8/8/8/8/4RK2 w - - 0 1")
    model = BeliefStateModel(chess.WHITE, board.generate_view(chess.WHITE), Evaluator(FakeEngine()), sizes=(1, 1),
                             board=board)
    result = search(model, num_top_moves=3)
    assert result.move == chess.Move.from_uci("e1e8")
    assert result.value == 10_000


@requires_stockfish
def test_search_captures_a_visible_king():
    evaluator = Evaluator()
    model = BeliefStateModel(chess.WHITE, FoWChessBoard().generate_view(chess.WHITE), evaluator, sizes=(3, 2), seed=0)
    follow(model, ["e2e4", "f7f6", "d1h5", "g7g5"])
    result = Engine(evaluator, num_top_moves=3).search(model)
    assert result.move == chess.Move.from_uci("h5e8")
    assert result.value == 10_000


@requires_stockfish
def test_search_returns_a_move_we_can_play_and_leaves_the_model_untouched():
    evaluator = Evaluator()
    model = BeliefStateModel(chess.WHITE, FoWChessBoard().generate_view(chess.WHITE), evaluator, sizes=(3, 2), seed=0)
    board = follow(model, ["e2e4", "g8f6"])
    before = (model.display(), [simulation.pieces for simulation in model.simulations])
    result = Engine(evaluator, num_top_moves=2).search(model)
    assert result.move in set(board.generate_pseudo_legal_moves())
    assert len(result.move_values) == 2
    assert (model.display(), [simulation.pieces for simulation in model.simulations]) == before


@requires_stockfish
def test_move_value_is_the_percentile_over_our_simulations_of_the_opponent_search():
    evaluator = Evaluator()
    model = BeliefStateModel(chess.WHITE, FoWChessBoard().generate_view(chess.WHITE), evaluator, sizes=(4,), seed=0)
    follow(model, ["e2e4", "g8f6", "d2d4", "b8c6"])
    engine = Engine(evaluator, num_top_moves=3, aggressiveness=25)
    result = engine.search(model)
    for move, value in result.move_values.items():
        scores = []
        for simulation in model.simulations:
            board = combined_board(own=simulation.placement, other=model.current_view.generate_piece_placement(), turn=chess.WHITE)
            board.push(move)
            opponent = simulation.branch()
            opponent.update_opponent_move(board.generate_view(chess.BLACK))
            scores.append(search(opponent, evaluator, None, engine.num_top_moves).value)
        assert value == aggregate(scores, chess.WHITE, engine.aggressiveness)
