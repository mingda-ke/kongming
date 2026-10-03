from __future__ import annotations

import chess

from belief_state_model import BeliefStateModel
from evaluator import Evaluator, SearchResult, search


class Engine:
    """
    Minimax over belief states, the same for both sides (see ``search``): it looks as many plies ahead as the belief
    states are nested. ``num_top_moves`` bounds the moves considered per node, and ``aggressiveness`` is the
    percentile of a move's scores over the simulations it is worth (mean if None).
    """

    def __init__(self, evaluator: Evaluator, num_top_moves: int = 4, aggressiveness: float | None = None):
        self.evaluator = evaluator
        self.num_top_moves = num_top_moves
        self.aggressiveness = aggressiveness

    def search(self, model: BeliefStateModel) -> SearchResult:
        """The best move of ``model``'s side."""
        return search(model, self.evaluator, num_top_moves=self.num_top_moves, aggressiveness=self.aggressiveness)


if __name__ == "__main__":
    import time

    from fow_chess import FoWChessBoard

    evaluator = Evaluator()
    engine = Engine(evaluator, num_top_moves=4, aggressiveness=25)
    for sizes in ((20,), (20, 3), (20, 3, 2)):
        board = FoWChessBoard()
        model = BeliefStateModel(chess.WHITE, board.generate_view(chess.WHITE), evaluator, sizes=sizes, seed=0)
        for uci in ["e2e4", "g8f6", "d2d4", "b8c6", "d4d5", "c6e5", "f2f4", "e5g6"]:
            move = chess.Move.from_uci(uci)
            mover = board.turn
            board.push(move)
            view = board.generate_view(chess.WHITE)
            if mover == model.color:
                model.update_own_move(move, view)
            else:
                model.update_opponent_move(view)

        start = time.perf_counter()
        result = engine.search(model)
        values = ", ".join(
            f"{move.uci()}: {value:.0f}" for move, value in sorted(result.move_values.items(), key=lambda item: -item[1])
        )
        print(f"sizes {sizes}: {time.perf_counter() - start:.1f}s  best {result.move}  ({values})")
