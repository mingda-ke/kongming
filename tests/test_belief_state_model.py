import chess

from belief_state_model import BeliefStateModel, Simulation, fits, convert_score_to_probability
from evaluator import Evaluator
from fow_chess import FoWChessBoard
from fow_view import FogMove
from piece_tracking import SpecificPiece
from tests.helpers import FakeEngine


def follow(model: BeliefStateModel, moves) -> FoWChessBoard:
    """Play ``moves`` and update ``model`` with each view of its side."""
    board = FoWChessBoard()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        mover = board.turn
        board.push(move)
        view = board.generate_view(model.color)
        if mover == model.color:
            model.update_own_move(move, view)
        else:
            model.update_opponent_move(view)
    return board


def new_model(**options) -> BeliefStateModel:
    return BeliefStateModel(chess.WHITE, FoWChessBoard().generate_view(chess.WHITE), Evaluator(FakeEngine()), seed=0,
                            **options)


def test_simulations_are_allowed_by_the_tracking_and_keep_the_real_board():
    # with the fake engine all moves score equal, so consider all of them
    model = new_model(sizes=(20, 3), top_k=40)
    board = follow(model, ["e2e4", "g8f6", "d2d4", "b8c6"])
    assert 0 < len(model.simulations) <= 20
    assert all(model.tracking.allows(simulation.board) for simulation in model.simulations)
    assert board.board_fen() in {simulation.board.board_fen() for simulation in model.simulations}


def test_every_simulation_carries_the_opponent_belief_until_the_recursion_ends():
    model = new_model(sizes=(4, 2, 1))
    follow(model, ["e2e4", "g8f6"])
    for simulation in model.simulations:
        theirs = simulation.opponent_bsm
        assert theirs is not None and theirs.color == chess.BLACK
        assert all(inner.opponent_bsm is not None and inner.opponent_bsm.color == chess.WHITE for inner in theirs.simulations)
        assert all(inner.opponent_bsm.simulations[0].opponent_bsm is None for inner in theirs.simulations)
    assert new_model(sizes=(4,)).simulations[0].opponent_bsm is None


def test_opponent_belief_does_not_know_our_unseen_move():
    """White's Nh3 is out of black's sight, so on the real board his belief keeps the knight on g1 possible."""
    model = new_model(sizes=(20, 3), top_k=40)
    board = follow(model, ["g1h3", "a7a6"])
    real = next(simulation for simulation in model.simulations if simulation.board.board_fen() == board.board_fen())
    knight = SpecificPiece(chess.WHITE, chess.KNIGHT, 6)
    assert chess.G1 in real.opponent_bsm.piece_tracking.possible_squares(knight)


def test_tracking_reseeds_the_simulations_when_they_run_dry():
    model = new_model(sizes=(3, 2))
    follow(model, ["e2e4", "g8f6"])
    model.simulations = []
    model.refill_simulations(turn=chess.WHITE)
    assert model.simulations
    for simulation in model.simulations:
        assert model.tracking.allows(simulation.board)
        assert simulation.board.turn == chess.WHITE
        assert simulation.opponent_bsm is not None


def test_refill_keeps_the_boards_stronger_for_the_opponent_with_the_closest_donor():
    model = new_model(sizes=(3, 2))
    follow(model, ["e2e4", "g8f6"])
    kept = model.simulations[0]
    model.simulations = [kept]
    # score boards by the opponent's material so the boards differ in strength
    model.evaluator.score = lambda board, color: len(board.pieces(chess.KNIGHT, color)) + board.fullmove_number
    candidates = []
    sample_boards = model.sample_boards
    model.sample_boards = lambda count, turn: candidates.extend(sample_boards(count, turn)) or candidates
    model.refill_simulations(turn=chess.WHITE)
    assert len(candidates) <= 4 and len(model.simulations) == 1 + min(2, len(candidates))
    likelihoods = [simulation.likelihood for simulation in model.simulations]
    assert likelihoods == sorted(likelihoods, reverse=True)
    for simulation in model.simulations:
        if simulation is not kept:
            assert simulation.opponent_bsm is not kept.opponent_bsm
            assert simulation.opponent_bsm.display() == kept.opponent_bsm.display()


def test_branch_leaves_the_model_untouched():
    model = new_model(sizes=(5, 2))
    follow(model, ["e2e4", "g8f6"])
    before = (model.display(), [simulation.board.fen() for simulation in model.simulations])
    branch = model.branch()
    board = model.simulations[0].board.copy()
    board.push_uci("d2d4")
    branch.update_own_move(chess.Move.from_uci("d2d4"), board.generate_view(chess.WHITE))
    assert (model.display(), [simulation.board.fen() for simulation in model.simulations]) == before


def test_moves_must_fit_what_we_saw_of_them():
    board = chess.Board()
    board.push_uci("e2e4")
    seen = FogMove(to_square=chess.F6, piece=chess.Piece(chess.KNIGHT, chess.BLACK))
    assert [move for move in board.generate_pseudo_legal_moves() if fits(board, move, seen)] == [
        chess.Move.from_uci("g8f6")
    ]


def test_move_probabilities_follow_the_opponent_scores():
    board = chess.Board()
    board.push_uci("e2e4")
    best, middle, worst = (chess.Move.from_uci(uci) for uci in ("e7e5", "h7h6", "a7a6"))
    model = new_model(sizes=(1,), top_k=3)
    model.evaluator.top_moves = lambda board, count, moves=None: {best: 50, middle: 0, worst: -100}
    probabilities = model.move_probabilities(Simulation(board, 1.0, None), [best, middle, worst])
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    assert probabilities[best] > probabilities[middle] > probabilities[worst] > 0


def test_strength_doubles_the_likelihood_every_10000_over_strength_centipawns():
    assert convert_score_to_probability([50, 0, -100], 0) == [1 / 3] * 3
    better, worse = convert_score_to_probability([100, 0], 100)
    assert abs(better / worse - 2) < 1e-9
    better, worse = convert_score_to_probability([100, 0], 200)
    assert abs(better / worse - 4) < 1e-9
    assert convert_score_to_probability([], 100) == []
