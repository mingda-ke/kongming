from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import chess

from evaluator import Evaluator, search
from fow_rules import FoWRulesBoard
from fow_view import FogMove, FogView
from piece_tracking import PieceTracking, PieceTrackingModel


@dataclass(frozen=True)
class Simulation:
    """
    A full board ``color``'s side considers possible, its likelihood, and the opponent's belief state on it (None at
    the end of the recursion).
    """

    board: chess.Board
    likelihood: float
    opponent_bsm: BeliefStateModel | None


class BeliefStateModel:
    """
    ``color``'s belief about the opponent's pieces, from two parts updated together:

    - ``tracking``: the possible squares of each opponent piece, deduced logically from the views.
    - ``simulations``: at most ``sizes[0]`` full boards simulated move by move, most likely first, each with the
      opponent's own belief state on that board (sized by ``sizes[1:]``, None once they run out). The opponent moves
      by his belief: on each board, his ``num_of_top_moves`` moves from a ``search`` of his belief state, weighted
      by how likely a player of ``opponent_strength`` is to play each (see ``convert_score_to_probability``).

    The tracking validates the simulations: boards it does not allow are dropped with their opponent belief, and the
    dropped ones are refilled with placements sampled from it.
    """

    def __init__(
        self,
        color: chess.Color,
        initial_view: FogView,
        evaluator: Evaluator,
        sizes: Sequence[int] = (20, 3),
        num_of_top_moves: int = 10,
        opponent_strength: float = 50.0,
        seed: int | None = None,
        board: chess.Board | None = None,
    ):
        """``board`` is the real board at the start (the starting position by default)."""
        self.color = color
        self.opponent = not color
        self.evaluator = evaluator
        self.sizes = tuple(sizes)
        self.num_of_top_moves = num_of_top_moves
        self.opponent_strength = opponent_strength
        self.random = random.Random(seed)
        self.tracking = PieceTrackingModel(color, initial_view)
        board = board or FoWRulesBoard()
        self.simulations = [Simulation(board, 1.0, self.opponent_model(board))]

    @property
    def current_view(self) -> FogView:
        return self.tracking.current_view

    @property
    def piece_tracking(self) -> PieceTracking:
        return self.tracking.piece_tracking

    def opponent_model(self, board: chess.Board) -> BeliefStateModel | None:
        """A new belief state of the opponent on ``board``, None at the end of the recursion."""
        if len(self.sizes) < 2:
            return None
        return BeliefStateModel(
            color=self.opponent,
            initial_view=board.generate_view(self.opponent),
            evaluator=self.evaluator,
            sizes=self.sizes[1:],
            num_of_top_moves=self.num_of_top_moves,
            opponent_strength=self.opponent_strength,
            seed=self.random.randrange(2**32),
            board=board,
        )

    def branch(self) -> BeliefStateModel:
        """
        A copy to play hypothetical moves on, e.g. in a search, since updates change a belief state in place: the
        tracking and every simulation's opponent belief are copied in turn (boards are never changed in place).
        """
        model = copy.copy(self)
        model.tracking = self.tracking.hypothetical_copy()
        model.simulations = [
            Simulation(simulation.board, simulation.likelihood,
                       simulation.opponent_bsm.branch() if simulation.opponent_bsm else None)
            for simulation in self.simulations
        ]
        return model

    def display(self) -> str:
        return self.tracking.display()

    # ------------------------------------------------------------------ updates

    def update_own_move(self, move: chess.Move, view: FogView) -> None:
        """
        We made ``move``; ``view`` is our view after it. It is played on every simulated board (removing what it
        captured), the boards the updated tracking no longer allows are dropped, the opponent's belief on each
        remaining one sees the move as an opponent move with his view there, and the simulations are refilled.
        """
        self.tracking.update_own_move(move, view)
        self.filter_valid_simulations()

        # update opponent belief state model as we make a move
        simulations = []
        for simulation in self.simulations:
            board = apply_move_to_board(simulation.board, move, self.color)
            if simulation.opponent_bsm:
                simulation.opponent_bsm.update_opponent_move(board.generate_view(self.opponent))
            simulations.append(Simulation(board, simulation.likelihood, simulation.opponent_bsm))
        self.simulations = simulations

        self.refill_simulations(turn=self.opponent)

    def update_opponent_move(self, view: FogView) -> None:
        """
        The opponent moved; ``view`` is our view after it. Every simulated board branches into one board per move of
        his ``num_of_top_moves`` that fit what we deduced of his move, with the board's likelihood times the move's
        probability (summing to 1 per board) and a copy of the board's opponent belief seeing his own move. The
        branches the updated tracking allows are kept, the ``sizes[0]`` most likely of them, and the simulations are
        refilled.
        """
        fog_move = self.tracking.update_opponent_move(view)

        # explode every simulation by the opponent's moves aligned with what we saw
        simulations = []
        for simulation in self.simulations:
            board = simulation.board
            moves = [move for move in board.generate_pseudo_legal_moves() if is_fog_move_aligned(board, move, fog_move)]
            for move, probability in self.move_probabilities(simulation, moves).items():
                child = apply_move_to_board(board, move, self.opponent)
                opponent_bsm = simulation.opponent_bsm.branch() if simulation.opponent_bsm else None
                if opponent_bsm:
                    opponent_bsm.update_own_move(move, child.generate_view(self.opponent))
                simulations.append(Simulation(child, simulation.likelihood * probability, opponent_bsm))
        self.simulations = simulations

        self.filter_valid_simulations()
        self.simulations = sorted(self.simulations, key=lambda simulation: -simulation.likelihood)[:self.sizes[0]]

        self.refill_simulations(turn=self.color)

    def move_probabilities(self, simulation: Simulation, moves: List[chess.Move]) -> Dict[chess.Move, float]:
        """
        The opponent's ``num_of_top_moves`` of ``moves`` on the simulated board and their probabilities, from their
        values in his search of his belief state (the scores on the simulated board without one).
        """
        if simulation.opponent_bsm is None:
            top_moves_and_probs = self.evaluator.top_moves(simulation.board, self.num_of_top_moves, moves)
        else:
            top_moves_and_probs = search(simulation.opponent_bsm, self.evaluator, moves, self.num_of_top_moves).move_values
        probabilities = convert_score_to_probability(list(top_moves_and_probs.values()), self.opponent, self.opponent_strength)
        return dict(zip(top_moves_and_probs, probabilities))

    def filter_valid_simulations(self) -> None:
        """Keep the simulations the tracking allows (it is updated with our view)."""
        self.simulations = [simulation for simulation in self.simulations if self.tracking.allows(simulation.board)]

    def refill_simulations(self, turn: chess.Color) -> None:
        """
        If ``n`` < ``sizes[0]`` simulations remain, sample 2 * (``sizes[0]`` - ``n``) new boards from the tracking,
        with ``turn`` to move, weigh them by how strong they are for the opponent, and add the stronger half. Each new
        board gets a copy of the opponent belief of the remaining simulation whose board looks closest to him.
        """
        missing = self.sizes[0] - len(self.simulations)
        if missing <= 0:
            return
        boards = self.sample_boards(2 * missing, turn)
        likelihoods = convert_score_to_probability([self.evaluator.score(board) for board in boards], self.opponent,
                                                   self.opponent_strength)
        best = sorted(zip(boards, likelihoods), key=lambda pair: -pair[1])[:missing]
        refill = [Simulation(board, likelihood, self.donor_bsm(board)) for board, likelihood in best]
        self.simulations = sorted(self.simulations + refill, key=lambda simulation: -simulation.likelihood)

    def sample_boards(self, count: int, turn: chess.Color) -> List[chess.Board]:
        """Up to ``count`` boards sampled from the tracking, with ``turn`` to move, new to the simulations."""
        seen = {simulation.board.board_fen() for simulation in self.simulations}
        boards = []
        for _ in range(count):
            board = self.sample_board(turn)
            if board is not None and board.board_fen() not in seen:
                seen.add(board.board_fen())
                boards.append(board)
        return boards

    def sample_board(self, turn: chess.Color) -> chess.Board | None:
        """
        A board with our pieces as in our view and the opponent's on a placement sampled from the tracking, with
        ``turn`` to move; None when the sampling hits a dead end.
        """
        placement = self.tracking.simulate(self.random)
        if placement is None:
            return None
        board = FoWRulesBoard(None)
        for square, value in enumerate(self.current_view.values):
            if isinstance(value, chess.Piece) and value.color == self.color:
                board.set_piece_at(square, value)
        for piece, square in placement.items():
            board.set_piece_at(square, chess.Piece(piece.piece_type, piece.color))
        board.turn = turn
        board.castling_rights = board.clean_castling_rights()
        return board

    def donor_bsm(self, board: chess.Board) -> BeliefStateModel | None:
        """
        The opponent belief for a new ``board``: a copy of the one on the simulation whose board the opponent sees
        most like ``board`` (a new one if no simulation remains), with his current view set to his view of ``board``.
        """
        if not self.simulations:
            return self.opponent_model(board)
        view = board.generate_view(self.opponent)
        donor = min(self.simulations,
                    key=lambda simulation: view_distance(view, simulation.board.generate_view(self.opponent)))
        if donor.opponent_bsm is None:
            return None
        bsm = donor.opponent_bsm.branch()
        bsm.tracking.current_view = view
        return bsm


def apply_move_to_board(board: chess.Board, move: chess.Move, mover: chess.Color | None = None) -> chess.Board:
    """``board`` after ``move``, played by ``mover`` (the side to move by default)."""
    result = board.copy(stack=False)
    if mover is not None:
        result.turn = mover
    result.push(move)
    return result


def is_fog_move_aligned(board: chess.Board, move: chess.Move, fog_move: FogMove) -> bool:
    """Whether ``move`` on ``board`` agrees with what we saw of it: its squares and piece, where seen."""
    if fog_move.from_square is not None and move.from_square != fog_move.from_square:
        return False
    if fog_move.to_square is not None and move.to_square != fog_move.to_square:
        return False
    return fog_move.piece is None or board.piece_type_at(move.from_square) == fog_move.piece.piece_type


def convert_score_to_probability(scores: List[float], color: chess.Color, strength: float) -> List[float]:
    """
    How likely the ``color`` player of ``strength`` (non-negative) is to pick each option of ``scores`` (White's
    centipawns, so Black prefers the lower): a softmax
    at temperature 10000 / (strength * ln 2) centipawns, so every ``10000 / strength`` centipawns more doubles the
    likelihood: 0 strength makes all options equally likely and 100 strength makes an option 100 centipawns better
    twice as likely.
    """
    if not scores:
        return []
    if color == chess.BLACK:
        scores = [-score for score in scores]
    best = max(scores)
    weights = [math.exp(strength * math.log(2) * (score - best) / 10000) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def view_distance(first: FogView, second: FogView) -> int:
    """The number of squares on which two views differ."""
    return sum(a != b for a, b in zip(first.values, second.values))


if __name__ == "__main__":
    import time

    from fow_chess import FoWChessBoard

    board = FoWChessBoard()
    model = BeliefStateModel(chess.WHITE, board.generate_view(chess.WHITE), Evaluator(), seed=0)
    for uci in ["e2e4", "g8f6", "d2d4", "b8c6", "d4d5", "c6e5", "f2f4", "e5g6"]:
        move = chess.Move.from_uci(uci)
        mover = board.turn
        board.push(move)
        view = board.generate_view(chess.WHITE)
        start = time.perf_counter()
        if mover == model.color:
            model.update_own_move(move, view)
        else:
            model.update_opponent_move(view)
        print(f"{uci}: {time.perf_counter() - start:.2f}s, {len(model.simulations)} simulations")

    print(board.generate_view(chess.WHITE))
    for simulation in model.simulations[:3]:
        real = " (real)" if simulation.board.board_fen() == board.board_fen() else ""
        print(f"\n{simulation.likelihood:.3f}{real}")
        print(simulation.board.unicode(invert_color=True, empty_square="⭘"))
        if simulation.opponent_bsm:
            print("as the opponent believes it:")
            print(simulation.opponent_bsm.simulations[0].board.unicode(invert_color=True, empty_square="⭘"))
