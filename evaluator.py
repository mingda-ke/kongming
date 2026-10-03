from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import chess
from stockfish import Stockfish

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

    def __init__(self, engine: Stockfish | None = None, depth: int = 4):
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
        self, board: chess.Board, count: int, moves: List[chess.Move] | None = None
    ) -> Dict[chess.Move, int]:
        """
        The side to move's ``count`` best of ``moves`` on ``board`` (all its FoW moves if None), best first, each with
        White's score of the board after it. Capturing the king scores as a win, so it comes first.
        """
        sign = 1 if board.turn == chess.WHITE else -1
        scores = {}
        for move in board.generate_pseudo_legal_moves() if moves is None else moves:
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
    """The best move found, its value, and the value of every candidate searched."""

    move: chess.Move | None
    value: float
    move_values: Dict[chess.Move, float] = field(default_factory=dict)


def search(
    model: BeliefStateModel,
    evaluator: Evaluator | None = None,
    moves: List[chess.Move] | None = None,
    num_top_moves: int = 4,
    aggressiveness: float | None = None,
) -> SearchResult:
    """
    The best move of ``model``'s side, with the value of every candidate, all as White's scores: White picks the
    highest and Black the lowest. The candidates are its ``num_top_moves`` top moves (of ``moves`` if given) on its
    most likely board, each worth the ``aggressiveness`` percentile (mean if None) of its score over the side's
    simulations, counted from the side's own end: higher picks moves more optimistically. On each simulation the move
    is played, and the score is the engine's (``model``'s by default) unless the simulation has an opponent belief
    state: then he sees the move and the score is his ``search`` value, so the recursion goes as deep as the nested
    belief states.
    """
    evaluator = evaluator or model.evaluator
    worst = -math.inf if model.color == chess.WHITE else math.inf
    board = model.simulations[0].board.copy(stack=False)
    board.turn = model.color
    values = {}
    for move in evaluator.top_moves(board, num_top_moves, moves):
        scores = []
        for simulation in model.simulations:
            board_after_move = simulation.board.copy(stack=False)
            board_after_move.turn = model.color
            board_after_move.push(move)
            opponent_bsm = simulation.opponent_bsm
            # capturing the king ends the game
            if opponent_bsm is None or board_after_move.king(opponent_bsm.color) is None:
                scores.append(evaluator.score(board_after_move))
                continue
            opponent_bsm = opponent_bsm.branch()
            opponent_bsm.update_opponent_move(board_after_move.generate_view(opponent_bsm.color))
            scores.append(search(opponent_bsm, evaluator, None, num_top_moves, aggressiveness).value)
        values[move] = aggregate(scores, model.color, aggressiveness)
    if not values:
        return SearchResult(move=None, value=worst)
    best = (max if model.color == chess.WHITE else min)(values, key=values.get)
    return SearchResult(move=best, value=values[best], move_values=values)


def aggregate(scores: List[float], color: chess.Color, percentile: float | None = None) -> float:
    """
    The mean of ``scores`` (White's), or their ``percentile`` if set (linear interpolation) counted from ``color``'s
    end: from the lowest for White and from the highest for Black.
    """
    if percentile is None:
        return sum(scores) / len(scores)
    if color == chess.BLACK:
        percentile = 100 - percentile
    ordered = sorted(scores)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
