from __future__ import annotations

import re
from pathlib import Path
from typing import List

import chess

from belief_state_model import BeliefStateModel
from engine import Engine
from fow_rules import FoWRulesBoard


class FoWChessBoard(FoWRulesBoard):
    """
    A game of FoW chess. With an ``engine``, it plays ``engine_color``: it keeps that side's belief state up to date
    with every move pushed, and ``engine_move`` plays its move.
    """

    def __init__(
        self,
        fen: str | None = chess.STARTING_FEN,
        *,
        engine: Engine | None = None,
        engine_color: chess.Color = chess.BLACK,
        seed: int | None = None,
    ):
        super().__init__(fen)
        self.engine = engine
        self.model = None
        if engine is not None:
            self.model = BeliefStateModel(engine_color, self.generate_view(engine_color), engine.evaluator, seed=seed,
                                          board=FoWRulesBoard(self.fen()))

    def push(self, move: chess.Move) -> None:
        mover = self.turn
        super().push(move)
        if self.model is not None:
            view = self.generate_view(self.model.color)
            if mover == self.model.color:
                self.model.update_own_move(move, view)
            else:
                self.model.update_opponent_move(view)

    def engine_move(self) -> chess.Move:
        """Search and play the engine's move."""
        move = self.engine.search(self.model).move
        self.push(move)
        return move


GAMES_DIR = Path(__file__).parent / "games"

# a move as chess.com writes FoW games: an optional piece letter, the from and to squares, and an optional promotion
PGN_MOVE = re.compile(r"\b[KQRBN]?([a-h][1-8])([a-h][1-8])(?:=?([QRBN]))?")


def load_pgn(name: str) -> List[str]:
    """
    The UCI moves of the FoW game in the PGN file ``name`` of the games folder, written in long algebraic notation
    (e.g. ``Ng1f3``).
    """
    with open(GAMES_DIR / name) as pgn_file:
        moves = "\n".join(line for line in pgn_file if not line.startswith("["))
    return [
        from_square + to_square + (promotion or "").lower()
        for from_square, to_square, promotion in PGN_MOVE.findall(moves)
    ]


if __name__ == "__main__":
    from fow_view import deduce_opponent_move
    # move_sequence = [
    #     chess.Move.from_uci("e2e4"),
    #     chess.Move.from_uci("e7e5"),
    #     chess.Move.from_uci("f1c4"),
    #     chess.Move.from_uci("d8h4"),
    # ]

    move_sequence = load_pgn("mingdak_vs_antichessbob.pgn")

    board = FoWChessBoard()
    for move in move_sequence:
        move_obj = chess.Move.from_uci(move)
        print(board.turn)
        print(move_obj)
        current_view = board.generate_view(chess.WHITE)
        board.push(move_obj)
        new_view = board.generate_view(chess.WHITE)
        if board.turn == chess.WHITE:
            fog_move = deduce_opponent_move(current_view, new_view)
            print(fog_move)
        print(board.generate_view(chess.WHITE))
