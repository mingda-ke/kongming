import chess

from belief_state_model import BeliefStateModel
from engine import Engine
from evaluator import Evaluator
from fow_chess import FoWChessBoard
from tests.helpers import FakeEngine, requires_stockfish
from tests.test_belief_state_model import follow


def test_aggregate_mean_and_percentile():
    evaluator = Evaluator(FakeEngine())
    assert Engine(evaluator).aggregate([0, 100, 200, 500]) == 200
    assert Engine(evaluator, percentile=25).aggregate([0, 100, 200, 500]) == 75
    assert Engine(evaluator, percentile=0).aggregate([0, 100, 200, 500]) == 0


def test_capturing_the_king_ends_the_game():
    engine = Engine(Evaluator(FakeEngine()))
    world = chess.Board("4k3/8/8/8/8/8/8/4RK2 w - - 0 1")
    assert engine.world_value(None, None, world, chess.Move.from_uci("e1e8"), plies=3) == 10_000
    assert next(iter(engine.evaluator.top_moves(world, 1))) == chess.Move.from_uci("e1e8")


@requires_stockfish
def test_search_captures_a_visible_king():
    evaluator = Evaluator()
    model = BeliefStateModel(chess.WHITE, FoWChessBoard().generate_view(chess.WHITE), evaluator, sizes=(3, 2), seed=0)
    follow(model, ["e2e4", "f7f6", "d1h5", "g7g5"])
    result = Engine(evaluator, depth=2, width=3).search(model)
    assert result.move == chess.Move.from_uci("h5e8")
    assert result.value == 10_000


@requires_stockfish
def test_search_returns_a_move_we_can_play_and_leaves_the_model_untouched():
    evaluator = Evaluator()
    model = BeliefStateModel(chess.WHITE, FoWChessBoard().generate_view(chess.WHITE), evaluator, sizes=(3, 2), seed=0)
    board = follow(model, ["e2e4", "g8f6"])
    before = (model.display(), [simulation.board.fen() for simulation in model.simulations])
    result = Engine(evaluator, depth=2, width=2).search(model)
    assert result.move in set(board.generate_pseudo_legal_moves())
    assert len(result.move_values) == 2
    assert (model.display(), [simulation.board.fen() for simulation in model.simulations]) == before


@requires_stockfish
def test_one_ply_move_value_is_the_percentile_over_our_simulations():
    evaluator = Evaluator()
    model = BeliefStateModel(chess.WHITE, FoWChessBoard().generate_view(chess.WHITE), evaluator, sizes=(4, 2), seed=0)
    follow(model, ["e2e4", "g8f6", "d2d4", "b8c6"])
    engine = Engine(evaluator, depth=1, width=3, percentile=25)
    result = engine.search(model)
    for move, value in result.move_values.items():
        scores = []
        for simulation in model.simulations:
            board = simulation.board.copy(stack=False)
            board.push(move)
            scores.append(evaluator.score(board, chess.WHITE))
        assert value == engine.aggregate(scores)
