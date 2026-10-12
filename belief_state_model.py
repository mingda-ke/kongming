from __future__ import annotations

import copy
import logging
import math
import random
from typing import Dict, List, Sequence

import chess

from evaluator import Evaluator, SearchResult, search
from fow_rules import FoWRulesBoard
from fow_board import apply_move_to_board, combined_board, pieces_of
from fow_view import FogView, PiecePlacement, deduce_opponent_move
from piece_tracking import PieceTracking, PieceTrackingModel
from utils import load_pgn

logger = logging.getLogger(__name__)


class BeliefStateModel:
    """
    ``color``'s belief about the opponent's pieces, from two parts updated together:

    - ``tracking``: the possible squares of each opponent piece, deduced logically from the views.
    - ``simulations``: at most ``sizes[0]`` boards simulated move by move, most likely first, each being the
      opponent's own belief state on that board (sized by ``sizes[1:]``): his ``view`` holds all of his pieces and ours
      he can see, so it cannot leak where our unseen pieces are, and ``likelihood`` is how likely we believe it is.
      Once ``sizes`` runs out, a belief state is a leaf that ends the recursion: it keeps only its view (no
      tracking nor simulations), and is scored on the board of its pieces and its believer's. The opponent moves
      by his belief: on each board, his ``num_of_top_moves`` moves from a ``search`` of his belief state, weighted
      by how likely a player of ``opponent_strength`` is to play each (see ``convert_score_to_probability``).

    The tracking validates the simulations: boards it does not allow are dropped with their opponent belief, and the
    dropped ones are refilled with placements sampled from it.
    """

    def __init__(
        self,
        color: chess.Color,
        view: FogView | None = None,
        tracking: PieceTrackingModel | None = None,
        evaluator: Evaluator | None = None,
        sizes: Sequence[int] = (20, 3),
        num_of_top_moves: int = 10,
        opponent_strength: float = 80.0,
        seed: int | None = None,
        depth: int = 0,
        path: tuple[str] = ("actual",),
        likelihood: float = 1.0,
    ):
        """
        ``board`` is the real board at the start (the starting position by default), where the opponent knows where all
        our pieces are; its simulations are generated from the tracking (see ``generate_simulations``). ``depth`` is
        the belief state's
        level in the tree of nested beliefs: 0 for the root, one more for each opponent belief in a simulation, and
        ``path`` locates it in the tree, and ``likelihood`` is how likely its parent believes it is (1 for the root).
        A belief started
        mid-game passes its ``tracking`` and starts without simulations, to be generated from it (see
        ``regenerate_opponent_model``).
        """
        # input arguments
        self.color = color
        self.opponent = not color
        view = view or FogView.initial(color=self.color)
        self.likelihood = likelihood

        self.depth = depth
        self.path = path or f"{chess.COLOR_NAMES[color]} actual"
        self.evaluator = evaluator or Evaluator()
        self.sizes = tuple(sizes) or (0,)
        # a leaf keeps only its view; any other belief state keeps it in its tracking
        self._view = view
        self.tracking = None if self.is_leaf else tracking or PieceTrackingModel(color=self.color, view=view)
        self.num_of_top_moves = num_of_top_moves
        self.opponent_strength = opponent_strength
        self.random = random.Random(seed)
        self.simulation_count = 0

        # two information to keep updating as moves are played
        # - opponent_knowledge is the opponent's tracking of our pieces with 100% certainty, used to generate new simulations for what opponent knows.
        #       - as conservative side, assuming all opponent pieces stay at initial stage until we know deterministic placement of opponent pieces.
        # - simulations is a list of opponent piece placements with their own opponent belief state (BeliefStateModel contains both info)
        self.opponent_knowledge = None if self.is_leaf else PieceTrackingModel(color=self.opponent)
        self.simulations: List[BeliefStateModel] = self.generate_simulations(num_sims=self.sizes[0])

    def __repr__(self) -> str:
        return (f"<BSM {self.name} likelihood={self.likelihood:.4f} depth={self.depth} "
                f"simulations={len(self.simulations)} "
                f"unseen_moves={None if self.is_leaf else self.piece_tracking.invisible_move_count}>")

    def __eq__(self, other: object) -> bool:
        """Belief states are the same simulation when they see the same board."""
        return isinstance(other, BeliefStateModel) and self.view == other.view

    def __hash__(self) -> int:
        """By the view, like ``__eq__``; a belief updated in place must not be kept in a set meanwhile."""
        return hash(self.view)

    @property
    def name(self) -> str:
        return "-".join(self.path)

    @property
    def is_leaf(self) -> bool:
        """Whether ``sizes`` ran out, so this belief state ends the recursion with only its view."""
        return self.sizes[0] == 0

    @property
    def view(self) -> FogView:
        """Our current view: all of our pieces and the opponent's we can see."""
        return self._view if self.is_leaf else self.tracking.current_view

    def update_view(self, view: FogView) -> None:
        """Set our current view to ``view`` without observing anything from it."""
        if self.is_leaf:
            self._view = view
        else:
            self.tracking.update_view(view=view)

    @property
    def placement(self) -> PiecePlacement:
        """Where our pieces are."""
        return self.view.generate_piece_placement()

    @property
    def pieces(self) -> Dict[chess.Square, chess.PieceType]:
        """The squares of our pieces and their types."""
        return self.placement.pieces

    def create_board(self, other: PiecePlacement, turn: chess.Color) -> chess.Board:
        """The full board of our pieces and the opponent's ``other``, with ``turn`` to move."""
        return combined_board(own=self.placement, other=other, turn=turn)

    def next_simulation_name(self) -> str:
        self.simulation_count += 1
        return f"sim {self.simulation_count}"

    def simulation_path(self, name: str) -> tuple[str]:
        """The path of this belief state's simulation ``name``."""
        return (*self.path, f"{chess.COLOR_NAMES[self.opponent]} {name}")

    def relocate(self, path: tuple[str]) -> None:
        """Move this belief state to ``path``, with its own simulations' paths in turn."""
        self.path = path
        for simulation in self.simulations:
            simulation.relocate(path=(*path, simulation.path[-1]))

    def log(self, message: str, *args) -> None:
        """Log a debug ``message`` of this belief state, indented by its depth."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s%s %s: " + message, "    " * self.depth, chess.COLOR_NAMES[self.color], self.depth,
                         *args)

    @property
    def current_view(self) -> FogView:
        return self.view

    @property
    def piece_tracking(self) -> PieceTracking:
        return self.tracking.piece_tracking

    def branch(self) -> BeliefStateModel:
        """
        A copy to play hypothetical moves on, e.g. in a search, since updates change a belief state in place: the
        tracking, the random generator and every simulation's opponent belief are copied in turn (placements are never
        changed in place).
        """
        model = copy.copy(self)
        model.random = copy.copy(self.random)
        if not self.is_leaf:
            model.tracking = self.tracking.hypothetical_copy()
        model.simulations = [simulation.branch() for simulation in self.simulations]
        return model

    def display(self) -> None:
        self.current_view.display()

    # ------------------------------------------------------------------ updates

    def update_own_move(self, move: chess.Move, view: FogView) -> None:
        """
        We made ``move``; ``view`` is our view after it. It is played on every simulated board (removing what it
        captured), the placements the updated tracking no longer allows are dropped, the opponent's belief on each
        remaining one sees the move as an opponent move with his view there, and the simulations are refilled. A leaf
        only takes ``view``.
        """
        if self.is_leaf:
            self.update_view(view=view)
            return
        own_piece_placement_before_move = self.placement
        # update our belief state on opponent's pieces after our move which could observe new info.
        self.tracking.update_own_move(move=move, view=view)

        # update simulations: filter consistent simulations
        own_piece_placement = view.generate_piece_placement()
        filtered_simulations = []
        for simulation in self.simulations:
            board_after_move = simulation.create_board(other=own_piece_placement, turn=self.color)
            if board_after_move.generate_view(color=self.color) == view:
                opponent_view_after_move = board_after_move.generate_view(color=self.opponent)
                simulation.update_opponent_move(view=opponent_view_after_move)
                filtered_simulations.append(simulation)

        dropped = len(self.simulations) - len(filtered_simulations)
        generated_simulations = []
        if dropped: # only regenerate simulations when some were dropped.
            self.log("filtered %d simulations: no longer allowed by the tracking", dropped)
            generated_simulations = self.generate_simulations(num_sims=self.sizes[0] - len(filtered_simulations) if dropped else 0)
        self.simulations = self.combine_simulations(simulation_set1=filtered_simulations, simulation_set2=generated_simulations)

        # update opponent determined belief state (opponent_knowledge) no matter which simulation is the real.  
        opponent_minimum_piece_placement = self.opponent_knowledge.current_view.generate_piece_placement().override_with(self.tracking.determined_opponent_placement())
        opponent_minimum_knowledge_board_ex_move = combined_board(
            own=own_piece_placement_before_move,
            other=opponent_minimum_piece_placement,
            turn=self.opponent)
        self.opponent_knowledge.update_view(view=opponent_minimum_knowledge_board_ex_move.generate_view(color=self.opponent))
        opponent_minimum_knowledge_board_post_move = combined_board(
            own=own_piece_placement, # our piece placement after the move.
            other=opponent_minimum_piece_placement,
            turn=self.opponent)
        self.opponent_knowledge.update_opponent_move(view=opponent_minimum_knowledge_board_post_move.generate_view(color=self.opponent))

        self.update_view(view=view)

    def update_opponent_move(self, view: FogView) -> None:
        """
        The opponent moved; ``view`` is our view after it. Before the move, the simulations where no move of his fits
        our view are dropped, and refilled with half as many generated from the tracking with him to move. Then our tracking sees his move, and every simulation branches into one simulation
        per move of his ``num_of_top_moves`` that fits our view, with its likelihood times the move's probability
        (summing to 1 per simulation), seeing his own move. The branches the updated tracking allows are kept, the
        ``sizes[0]`` most likely of them. A leaf only takes ``view``.
        """
        if self.is_leaf:
            self.update_view(view=view)
            return
        # first update new info exposed by the opponent's move.
        our_view = self.current_view
        our_piece_placement = our_view.generate_piece_placement()
        fog_move = deduce_opponent_move(before=our_view, after=view)
        self.tracking.observe_discovered_squares(view=view, fog_move=fog_move)  

        # filter the simulations before the move: some moves of his must fit our view
        valid_simulation_move_combinations: List[tuple[BeliefStateModel, chess.Move]] = [(simulation, valid_move) for simulation in self.simulations for valid_move in get_view_aligned_moves(simulation=simulation, view=our_view)]
        simulations = list({simulation for simulation, _ in valid_simulation_move_combinations})

        # refill simluations when there's deficit
        dropped_simulations = set(self.simulations) - {simulation for simulation, _ in valid_simulation_move_combinations}
        if len(dropped_simulations) > 0 and len(valid_simulation_move_combinations) < self.sizes[0]:
            generated_simulations = self.generate_simulations(num_sims=len(dropped_simulations))
            combined_simulations = self.combine_simulations(simulation_set1=simulations, simulation_set2=generated_simulations)
            valid_simulation_move_combinations += [(simulation, valid_move) for simulation in set(combined_simulations).difference(simulations) for valid_move in get_view_aligned_moves(simulation=simulation, view=our_view)]
            simulations = combined_simulations

        # for each simulation, assign probability of valid moves and explode into new simulations
        # equivalence of Bayesian update.
        simulations_post_move = []
        for simulation in simulations:
            valid_moves = [valid_move for sim, valid_move in valid_simulation_move_combinations if sim == simulation]
            probabilities_by_move = self.move_probabilities(simulation=simulation, moves=valid_moves)
            board_of_simulation = simulation.create_board(other=our_piece_placement, turn=simulation.color)
            for move, move_probability in probabilities_by_move.items():
                if move == chess.Move.from_uci("c7c5"):
                    pass
                simulation_after_move = simulation.branch()
                view_after_move = apply_move_to_board(board_of_simulation, move).generate_view(color=self.opponent)
                simulation_after_move.update_own_move(move=move, view=view_after_move)
                simulation_after_move.likelihood = simulation.likelihood * move_probability
                simulations_post_move.append(simulation_after_move)

        simulations_post_move = group_simulations(simulations_post_move)
        self.simulations = sorted(normalize_simulation_likelihood(simulations=simulations_post_move), key=lambda simulation: -simulation.likelihood)[:self.sizes[0]]

        self.tracking.apply_fog_move(fog_move=fog_move)

        self.update_view(view=view)

    def generate_simulation(self, seed: int, number: int) -> BeliefStateModel | None:
        """
        A new simulation on a placement of the opponent's pieces sampled from the tracking with a generator seeded by
        ``seed`` (see ``simulation_on``). None when the sampling hits a dead end.
        """
        # TODO: there is a known gap using piece tracking to simulate placement - piece dependency, e.g. Bc4 requires e pawn to not be on e2.
        opponent_placement = self.tracking.simulate_placement(rng=random.Random(seed))
        if opponent_placement is None:
            return None
        opponent_view = combined_board(own=opponent_placement, other=self.current_view.generate_piece_placement(),
                                       turn=self.opponent).generate_view(color=self.opponent)
        opponent_piece_tracking = None
        # a leaf opponent belief keeps only its view
        if any(self.sizes[1:2]):
            opponent_piece_tracking = PieceTrackingModel(
                color=self.opponent, view=opponent_view,
                piece_tracking=self.opponent_knowledge.hypothetical_copy().piece_tracking)
        new_path = (*self.path, f"{chess.COLOR_NAMES[self.opponent]} sim {number}")
        return BeliefStateModel(
                    color=self.opponent,
                    view=opponent_view,
                    tracking=opponent_piece_tracking,
                    evaluator=self.evaluator,
                    sizes=self.sizes[1:],
                    num_of_top_moves=self.num_of_top_moves,
                    opponent_strength=self.opponent_strength,
                    seed=seed,
                    depth=self.depth + 1,
                    path=new_path,
                    likelihood=1.0,
                )

    def generate_simulations(self, num_sims: int) -> List[BeliefStateModel]:
        """
        New simulations (with ``turn`` to move, see ``simulation_on``) on up to ``num_sims`` placements sampled from the
        tracking, each by a fresh seed. Placements already simulated or sampled are skipped before their opponent
        belief is built, and a fully determined tracking (e.g. at the start) is sampled only once, having a single
        placement.
        """
        if num_sims == 0:
            return []

        # starting position
        if all(len(self.piece_tracking.possible_squares(piece)) == 1 for piece in self.piece_tracking.evolutions):
            num_sims = 1
        
        self.log("generating up to %d simulations", num_sims)
        simulations = [sim for i in range(num_sims*2)
                       if (sim := self.generate_simulation(seed=self.random.randrange(2**i), number=i)) is not None]
        probabilities = self.simulation_probabilities(simulations=[sim for sim in simulations if sim is not None])
        
        for simulation in simulations:
            simulation.likelihood = probabilities[simulation]
        return sorted(simulations, key=lambda sim: sim.likelihood, reverse=True)[:num_sims]

    def combine_simulations(self, simulation_set1: List[BeliefStateModel],
                            simulation_set2: List[BeliefStateModel]) -> List[BeliefStateModel]:
        """
        Both sets of simulations together, most likely first. Each set keeps its relative likelihoods, and the sets
        are realigned against each other by the score-to-probability of their most likely simulations (see
        ``simulation_score``). Simulations seeing the same board are then grouped (see ``group_simulations``) and the
        likelihoods normalized to sum to 1.
        """
        if simulation_set1 and simulation_set2:
            top1 = max(simulation_set1, key=lambda simulation: simulation.likelihood)
            top2 = max(simulation_set2, key=lambda simulation: simulation.likelihood)
            score1, score2 = self.simulation_score(simulation=top1), self.simulation_score(simulation=top2)
            weight1, weight2 = convert_score_to_probability(scores=[score1, score2], color=self.opponent,
                                                            strength=self.opponent_strength)
            rescale_simulation_likelihood(simulations=simulation_set1, top=weight1)
            rescale_simulation_likelihood(simulations=simulation_set2, top=weight2)
            self.log("realigned %s (score %.1f) against %s (score %.1f): %.4f vs %.4f",
                     top1.name, score1, top2.name, score2, weight1, weight2)
        simulations = group_simulations(simulations=simulation_set1 + simulation_set2)
        if not simulations:
            return []
        return sorted(simulations, key=lambda simulation: -simulation.likelihood)

    def search_simulation(self, simulation: BeliefStateModel, moves: List[chess.Move] | None = None) -> SearchResult:
        """
        The opponent's ``search`` of his belief on ``simulation``; a leaf (end of recursion) is scored on the board of
        our pieces and its pieces instead.
        """
        if simulation.is_leaf:
            board = simulation.create_board(other=self.placement, turn=simulation.color)
            scores = self.evaluator.top_moves(board=board, count=self.num_of_top_moves)
            if not scores:
                return SearchResult(best_move=None, best_score=-math.inf if simulation.color == chess.WHITE else math.inf)
            best_move = next(iter(scores))
            return SearchResult(best_move=best_move, best_score=scores[best_move], candidate_scores=scores)
        return search(model=simulation, evaluator=self.evaluator, moves=moves, num_top_moves=self.num_of_top_moves)

    def simulation_score(self, simulation: BeliefStateModel) -> float:
        """White's value of ``simulation``: the value of the opponent's best move by a search of his belief on it."""
        return self.search_simulation(simulation=simulation).best_score

    def move_probabilities(self, simulation: BeliefStateModel, moves: List[chess.Move]) -> Dict[chess.Move, float]:
        """
        How likely the opponent (of ``opponent_strength``) is to play each of his ``num_of_top_moves`` best ``moves``
        on ``simulation``, from their values in a search of his own belief state (see ``search_simulation``).
        """
        scores = self.search_simulation(simulation=simulation, moves=moves).candidate_scores
        probabilities = convert_score_to_probability(scores=list(scores.values()), color=self.opponent,
                                                     strength=self.opponent_strength)
        return {move: prob for move, prob in zip(scores.keys(), probabilities) if move in moves}

    def simulation_probabilities(self, simulations: List[BeliefStateModel]) -> Dict[BeliefStateModel, float]:
        """
        The probabilities of each simulation in ``simulations`` being the correct one, by the opponent's (of
        ``opponent_strength``) best score on each (see ``simulation_score``).
        """
        if len(simulations) <= 1:
            return dict.fromkeys(simulations, 1.0)
        scores = [self.simulation_score(simulation=simulation) for simulation in simulations]
        probabilities = convert_score_to_probability(scores=scores, color=self.opponent,
                                                     strength=self.opponent_strength)
        return dict(zip(simulations, probabilities))


def get_view_aligned_moves(simulation: BeliefStateModel, view: FogView) -> List[chess.Move]:
    """Return the list of moves that leave ``view``'s side with exactly ``view``."""
    board = combined_board(own=view.generate_piece_placement(), other=simulation.current_view.generate_piece_placement(), turn=simulation.color)
    aligned_moves = []
    for move in board.generate_pseudo_legal_moves():
        board_after_move = apply_move_to_board(board=board, move=move)
        if board_after_move.generate_view(color=view.color).values == view.values:
            aligned_moves.append(move)
    return aligned_moves


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


def rescale_simulation_likelihood(simulations: List[BeliefStateModel], top: float) -> List[BeliefStateModel]:
    """
    ``simulations``, rescaled in place to keep their relative likelihoods with the most likely one at ``top``; all
    equally likely if their likelihoods vanished.
    """
    highest = max(simulation.likelihood for simulation in simulations)
    for simulation in simulations:
        simulation.likelihood = top * simulation.likelihood / highest if highest > 0 else top
    return simulations


def group_simulations(simulations: List[BeliefStateModel]) -> List[BeliefStateModel]:
    """
    ``simulations`` with those seeing the same board combined into the first of them (in order), its likelihood the
    sum of theirs; the others are assumed alike in everything else, and are dropped.
    """
    grouped: Dict[FogView, BeliefStateModel] = {}
    for simulation in simulations:
        first = grouped.setdefault(simulation.view, simulation)
        if first is not simulation:
            first.likelihood += simulation.likelihood
    return list(grouped.values())


def normalize_simulation_likelihood(simulations: List[BeliefStateModel]) -> List[BeliefStateModel]:
    """``simulations``, their likelihoods scaled in place to sum to 1 (equal if they all vanished)."""
    total = sum(simulation.likelihood for simulation in simulations)
    for simulation in simulations:
        simulation.likelihood = simulation.likelihood / total if total > 0 else 1 / len(simulations)
    return simulations


if __name__ == "__main__":
    import time

    from fow_chess import FoWChessBoard

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    # DEBUG logs every simulation created, updated or filtered, including in hypothetical searches; INFO hides them
    logger.setLevel(logging.DEBUG)

    board = FoWChessBoard()
    model = BeliefStateModel(color=chess.WHITE, view=board.generate_view(color=chess.WHITE),
                             evaluator=Evaluator(), seed=0)
    moves = load_pgn(name="mingdak vs 1938 Fog of War 2026-10-05.pgn")
    for uci in moves:
        move = chess.Move.from_uci(uci)
        mover = board.turn
        board.push(move)
        view = board.generate_view(color=chess.WHITE)
        start = time.perf_counter()
        if mover == model.color:
            model.update_own_move(move=move, view=view)
        else:
            model.update_opponent_move(view=view)
        print(f"{uci}: {time.perf_counter() - start:.2f}s, {len(model.simulations)} simulations")

    print(board.generate_view(color=chess.WHITE))
    for simulation in model.simulations[:3]:
        real = " (real)" if simulation.pieces == pieces_of(board=board, color=model.opponent) else ""
        print(f"\n{simulation.likelihood:.3f}{real}")
        ours = model.current_view.generate_piece_placement()
        print(simulation.create_board(other=ours, turn=board.turn).unicode(invert_color=True, empty_square="⭘"))
        theirs = simulation
        if theirs.simulations:
            print("as the opponent believes it:")
            believed = theirs.simulations[0].create_board(other=theirs.current_view.generate_piece_placement(),
                                                          turn=board.turn)
            print(believed.unicode(invert_color=True, empty_square="⭘"))
