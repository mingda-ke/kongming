from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import chess
from stockfish import Stockfish

from config import BeliefStateConfig
from fow_board import combined_board

if TYPE_CHECKING:
    from belief_state_model import BeliefStateModel


# centipawn value of a forced mate, and of capturing the king (which ends a FoW game)
MATE_SCORE = 10_000

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# board statuses the engine may not survive; such boards are scored by material instead
UNSAFE_FOR_ENGINE = (
    chess.STATUS_NO_WHITE_KING | chess.STATUS_NO_BLACK_KING | chess.STATUS_TOO_MANY_KINGS
    | chess.STATUS_PAWNS_ON_BACKRANK | chess.STATUS_OPPOSITE_CHECK | chess.STATUS_TOO_MANY_CHECKERS
)


class Evaluator:
    """Engine scores of full boards in centipawns, cached by position, shared by the belief states and the search."""

    def __init__(self, engine: Stockfish | None = None, depth: int = 5):
        self.engine = engine or Stockfish(shutil.which("stockfish") or "stockfish", depth=depth)
        self._cache: Dict[str, int] = {}

    def score(self, board: chess.Board) -> int:
        """
        White's score of ``board``, with whichever side is to move. A missing king has been captured, and a king the
        side to move can capture is lost. Boards the engine cannot take are scored by material.
        """
        fen = board.fen()
        if fen not in self._cache:
            score = self._mover_score(board)
            self._cache[fen] = score if board.turn == chess.WHITE else -score
        return self._cache[fen]

    def top_moves(
        self, board: chess.Board, count: int
    ) -> Dict[chess.Move, int]:
        """
        The side to move's ``count`` best of ``moves`` on ``board`` (all its FoW moves if None), best first, each with
        White's score of the board after it. Capturing the king scores as a win, so it comes first.
        """
        sign = 1 if board.turn == chess.WHITE else -1
        scores = {}
        for move in board.generate_pseudo_legal_moves():
            after = board.copy(stack=False)
            after.push(move)
            scores[move] = self.score(after)
        return dict(sorted(scores.items(), key=lambda item: -sign * item[1])[:count])

    def _mover_score(self, board: chess.Board) -> int:
        """The side to move's score of ``board``."""
        mover = board.turn
        if board.king(mover) is None:
            return -MATE_SCORE
        if board.king(not mover) is None or board.is_attacked_by(mover, board.king(not mover)):
            return MATE_SCORE
        if board.status() & UNSAFE_FOR_ENGINE:
            return material(board, mover)
        self.engine.set_fen_position(board.fen())
        return centipawns(self.engine.get_evaluation())


def centipawns(evaluation: Dict[str, str | int]) -> int:
    """Convert an engine evaluation of the side to move to centipawns, with mates at the extremes."""
    value = int(evaluation["value"])
    if evaluation["type"] != "mate":
        return max(-MATE_SCORE, min(MATE_SCORE, value))
    if value == 0:
        # the side to move is mated
        return -MATE_SCORE
    return (MATE_SCORE - abs(value)) * (1 if value > 0 else -1)


def material(board: chess.Board, color: chess.Color) -> int:
    """``color``'s material balance in centipawns."""
    return sum(
        PIECE_VALUES[piece.piece_type] * (1 if piece.color == color else -1) for piece in board.piece_map().values()
    )


@dataclass
class SearchResult:
    """``color``'s White scores of each candidate move on each simulation, and the likelihoods of the simulations."""

    color: chess.Color
    score_matrix: Dict[chess.Move, List[float]] = field(default_factory=dict)
    simulation_likelihoods: List[float] = field(default_factory=list)


@dataclass
class MoveDecision:
    """The move decided, its score, and how likely each candidate move is to be played."""

    best_move: chess.Move | None
    score: float
    move_probabilities: Dict[chess.Move, float] = field(default_factory=dict)


def search(
    model: BeliefStateModel,
    evaluator: Evaluator | None = None,
    moves: List[chess.Move] | None = None,
    num_top_moves: int = 4,
) -> SearchResult:
    """
    
    If ``moves`` are provided, consider these moves in addition to the top moves.
    """
    evaluator = evaluator or model.evaluator

    scores_by_move_simulation = {}
    if model.simulations:
        board = combined_board(own=model.simulations[0].placement, other=model.current_view.generate_piece_placement(), turn=model.color)
        # legal moves are indifferent across model.simulations (same view gives same legal moves)
        # use most likely board to get top moves so not to iterate all legal moves.
        candidate_moves = list(evaluator.top_moves(board, num_top_moves).keys())
        if moves is not None:
            candidate_moves = list(set(candidate_moves + moves))
        for move in candidate_moves:
            scores = []
            for simulation in model.simulations:
                board_after_move = combined_board(own=simulation.placement, other=model.current_view.generate_piece_placement(),
                                                  turn=model.color)
                board_after_move.push(move)
                # a leaf (end of recursion) is scored directly on the board
                # of our pieces and its pieces
                if simulation.is_leaf or board_after_move.king(simulation.color) is None:
                    scores.append(evaluator.score(board_after_move))
                    continue
                opponent_bsm = simulation.branch()
                opponent_bsm.update_opponent_move(board_after_move.generate_view(opponent_bsm.color))
                search_result = search(opponent_bsm, evaluator, None, num_top_moves)
                scores.append(decide_move(search_result=search_result, config=opponent_bsm.config).score)
            scores_by_move_simulation[move] = scores
    if not scores_by_move_simulation:
        return SearchResult(color=model.color)

    return SearchResult(color=model.color, score_matrix=scores_by_move_simulation, simulation_likelihoods=[sim.likelihood for sim in model.simulations])


def decide_move(search_result: SearchResult, config: BeliefStateConfig = BeliefStateConfig()) -> MoveDecision:
    """
    Decide best move to play based on score matrix (score by move and simulation):

    Per simulation, take the ``config.num_of_moves_per_sim`` top scored moves and softmax them, so that a move
    100 centipawns worse than the top one is ``config.softmax_prob_per_100_cp`` times less likely. Multiply these by
    the simulation likelihood and sum them up per move, then pick the move with the highest summed likelihood.

    Future enhancement: based on maximizing winning probability.
    """
    score_matrix, color = search_result.score_matrix, search_result.color
    if not score_matrix:
        return MoveDecision(best_move=None, score=-MATE_SCORE if color == chess.WHITE else MATE_SCORE)
    likelihoods = search_result.simulation_likelihoods
    if sum(likelihoods) <= 0:
        likelihoods = [1.0] * len(likelihoods)
    sign = 1 if color == chess.WHITE else -1
    move_probabilities = dict.fromkeys(score_matrix, 0.0)
    for sim_index, sim_likelihood in enumerate(likelihoods):
        top_moves = sorted(score_matrix, key=lambda move: -sign * score_matrix[move][sim_index])[:config.num_of_moves_per_sim]
        top_score = sign * score_matrix[top_moves[0]][sim_index]
        weights = {move: config.softmax_prob_per_100_cp ** ((sign * score_matrix[move][sim_index] - top_score) / 100)
                   for move in top_moves}
        weight_sum = sum(weights.values())
        for move, weight in weights.items():
            move_probabilities[move] += sim_likelihood * weight / weight_sum
    total = sum(likelihoods)
    move_probabilities = {move: likelihood / total for move, likelihood in move_probabilities.items()}
    best_move = max(move_probabilities, key=move_probabilities.get)
    avg_score = sum(likelihood * score for likelihood, score in zip(likelihoods, score_matrix[best_move])) / total
    return MoveDecision(best_move=best_move, score=avg_score, move_probabilities=move_probabilities)
