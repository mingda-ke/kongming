from __future__ import annotations

import shutil
from typing import Dict, List

import chess
from stockfish import Stockfish

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

    def score(self, board: chess.Board, color: chess.Color) -> int:
        """
        ``color``'s score of ``board``, with whichever side is to move. A missing king has been captured, and a king
        the side to move can capture is lost. Boards the engine cannot take are scored by material.
        """
        fen = board.fen()
        if fen not in self._cache:
            self._cache[fen] = self._mover_score(board)
        score = self._cache[fen]
        return score if board.turn == color else -score

    def top_moves(
        self, board: chess.Board, count: int, moves: List[chess.Move] | None = None
    ) -> Dict[chess.Move, int]:
        """
        The side to move's ``count`` best of ``moves`` on ``board`` (all its FoW moves if None), best first, each with
        its score of the board after it. Capturing the king scores as a win, so it comes first.
        """
        mover = board.turn
        scores = {}
        for move in board.generate_pseudo_legal_moves() if moves is None else moves:
            after = board.copy(stack=False)
            after.push(move)
            scores[move] = self.score(after, mover)
        return dict(sorted(scores.items(), key=lambda item: -item[1])[:count])

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
