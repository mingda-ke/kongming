from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List

import chess

from belief_state_model import BeliefStateModel
from evaluator import MATE_SCORE, Evaluator

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """The best move found, its value, and the value of every root move searched."""

    move: chess.Move | None
    value: float
    move_values: Dict[chess.Move, float] = field(default_factory=dict)


class Engine:
    """
    Minimax over belief states, the same for both sides. The side to move searches with its belief state: a move is
    worth the ``percentile`` (mean if None) of its value over the side's simulated boards. On each board, taken as the
    real one, the move is played, and the other side replies by searching with its own belief state on that board
    (the opponent belief of the simulation); at the end of the recursion (no belief state), it plays the engine's best
    move on the real board. A leaf is the engine's score of a board, which its own search extends a few plies.

    ``depth`` counts plies from our move; ``width`` bounds the moves considered per node.
    """

    def __init__(self, evaluator: Evaluator, depth: int = 2, width: int = 4, percentile: float | None = None):
        self.evaluator = evaluator
        self.depth = depth
        self.width = width
        self.percentile = percentile

    def search(self, model: BeliefStateModel, plies: int | None = None) -> SearchResult:
        """
        The best move of ``model``'s side with ``plies`` left (``depth`` by default), with the value of every
        candidate: its top moves on its most likely board, valued over all its simulated boards.
        """
        plies = self.depth if plies is None else plies
        side = model.color
        board = model.simulations[0].board.copy(stack=False)
        board.turn = side
        values = {}
        for move in self.evaluator.top_moves(board, self.width):
            world_values = [value for simulation in model.simulations
                            if (value := self.world_value(model, simulation.opponent_bsm, simulation.board, move, plies))
                            is not None]
            values[move] = self.aggregate(world_values) if world_values else -math.inf
        if not values:
            return SearchResult(move=None, value=-math.inf)
        best = max(values, key=values.get)
        return SearchResult(move=best, value=values[best], move_values=values)

    def world_value(
        self,
        mover: BeliefStateModel | None,
        other: BeliefStateModel | None,
        world: chess.Board,
        move: chess.Move,
        plies: int,
    ) -> float | None:
        """
        The value of ``move`` to the side to move if ``world`` is the real board, ``plies`` left including this move.
        ``mover`` and ``other`` are both sides' belief states there (None: plays by the engine on the real board). A
        side's moves are the same on every board consistent with what it sees, so ``move`` can be played there. None
        if the belief states cannot follow the moves there.
        """
        side = world.turn if mover is None else mover.color
        board = world.copy(stack=False)
        board.turn = side
        if move.to_square == board.king(not side):
            return MATE_SCORE
        board.push(move)
        if plies == 1:
            return self.evaluator.score(board, side)
        try:
            if other is not None:
                other = other.branch()
                other.update_opponent_move(board.generate_view(not side))
            # the mover's belief is only needed again if it moves again below
            if mover is not None and plies > 2:
                mover = mover.branch()
                mover.update_own_move(move, board.generate_view(side))
        except Exception as error:
            logger.warning(f"Skipping world for {move}: {error}")
            return None
        if other is not None:
            reply = self.search(other, plies - 1).move
        else:
            reply = next(iter(self.evaluator.top_moves(board, 1)), None)
        if reply is None:
            return self.evaluator.score(board, side)
        value = self.world_value(other, mover, board, reply, plies - 1)
        return None if value is None else -value

    def aggregate(self, scores: List[float]) -> float:
        """The mean of ``scores``, or their ``percentile`` if set (linear interpolation)."""
        if self.percentile is None:
            return sum(scores) / len(scores)
        ordered = sorted(scores)
        position = (len(ordered) - 1) * self.percentile / 100
        lower = math.floor(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


if __name__ == "__main__":
    import time

    from fow_chess import FoWChessBoard

    evaluator = Evaluator()
    board = FoWChessBoard()
    model = BeliefStateModel(chess.WHITE, board.generate_view(chess.WHITE), evaluator, seed=0)
    for uci in ["e2e4", "g8f6", "d2d4", "b8c6", "d4d5", "c6e5", "f2f4", "e5g6"]:
        move = chess.Move.from_uci(uci)
        mover = board.turn
        board.push(move)
        view = board.generate_view(chess.WHITE)
        if mover == model.color:
            model.update_own_move(move, view)
        else:
            model.update_opponent_move(view)

    print(board.display(chess.WHITE))
    for depth in (1, 2, 3):
        engine = Engine(evaluator, depth=depth, width=4, percentile=25)
        start = time.perf_counter()
        result = engine.search(model)
        values = ", ".join(
            f"{move.uci()}: {value:.0f}" for move, value in sorted(result.move_values.items(), key=lambda item: -item[1])
        )
        print(f"depth {depth}: {time.perf_counter() - start:.1f}s  best {result.move}  ({values})")
