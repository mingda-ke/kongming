from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

import chess

from fow_view import FogMove, FogView, deduce_opponent_move
from static import BLOCKED_PIECE, EMPTY_SQUARE, INVISIBLE_PIECE

logger = logging.getLogger(__name__)

# age (number of moves since the piece was last seen) -> possible squares
PieceEvolution = Dict[int, Set[chess.Square]]

# pieces whose move requires every square between its origin and destination to be empty
SLIDING_PIECE_TYPES = {chess.ROOK, chess.BISHOP, chess.QUEEN}


@dataclass(frozen=True)
class SpecificPiece:
    """Represent a specific piece on the board."""

    color: chess.Color
    piece_type: chess.PieceType
    file: int | None = None

    def __post_init__(self) -> None:
        allowed_files = {
            chess.PAWN: set(range(8)),
            chess.KNIGHT: {chess.FILE_NAMES.index("b"), chess.FILE_NAMES.index("g")},
            chess.ROOK: {chess.FILE_NAMES.index("a"), chess.FILE_NAMES.index("h")},
            chess.BISHOP: {chess.FILE_NAMES.index("c"), chess.FILE_NAMES.index("f")},
        }
        if self.piece_type in allowed_files and self.file is None:
            raise ValueError("A pawn, knight, bishop, or rook identity requires its file")
        if self.file is not None and not 0 <= self.file < 8:
            raise ValueError("file must be between 0 and 7")
        if self.piece_type in allowed_files and self.file not in allowed_files[self.piece_type]:
            piece_name = chess.piece_name(self.piece_type)
            valid_files = ", ".join(chess.FILE_NAMES[file] for file in sorted(allowed_files[self.piece_type]))
            raise ValueError(f"A {piece_name} must use file {valid_files}")

    def __repr__(self) -> str:
        symbol = chess.piece_symbol(self.piece_type)
        symbol = symbol.upper() if self.color == chess.WHITE else symbol
        return symbol if self.file is None else f"{symbol}{chess.FILE_NAMES[self.file]}"


@dataclass
class PieceConstraint:
    """
    Exactly one of ``candidates`` moved to ``square``, but we cannot tell which one.
    ``candidates`` maps each candidate to the age in its tracking evolution at which ``square`` was appended,
    so the evolution can be trimmed once the constraint is resolved.
    """

    square: chess.Square
    piece_type: chess.PieceType | None
    candidates: Dict[SpecificPiece, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EmptySquare:
    """A visible square observed to be empty."""

    square: chess.Square


@dataclass(frozen=True)
class BlockedSquare:
    """A square observed to hold a piece of unknown type (BLOCKED_PIECE)."""

    square: chess.Square


@dataclass(frozen=True)
class VisiblePiece:
    """A visible square observed to hold ``piece``, of either color."""

    square: chess.Square
    piece: chess.Piece


@dataclass(frozen=True)
class CaptureMove:
    """Our move captured the opponent piece of ``piece_type`` on ``square``."""

    square: chess.Square
    piece_type: chess.PieceType


Observation = EmptySquare | BlockedSquare | VisiblePiece
Evidence = CaptureMove | Observation | FogMove | None


def observe_square(square: chess.Square, value: int | chess.Piece) -> Observation | None:
    """Turn a FogView value into the observation it represents, or None for an invisible square."""
    if value == EMPTY_SQUARE:
        return EmptySquare(square)
    if value == BLOCKED_PIECE:
        return BlockedSquare(square)
    if isinstance(value, chess.Piece):
        return VisiblePiece(square, value)
    return None

def compare_views(before: FogView, after: FogView) -> Set[chess.Square]:
    """Return the squares whose visible state changed between two views."""
    return {
        square
        for square in chess.SQUARES
        if before.values[square] != after.values[square]
    }


@dataclass
class PieceTracking:
    """
    Possible squares of each specific opponent piece: SpecificPiece -> {age: set of possible squares}, where age is
    the number of moves since the piece was last seen. ``invisible_move_count`` bounds the total number of unseen
    moves the pieces still in the dark can have made, so no piece's age can exceed it. ``constraints`` records
    ambiguous moves or sightings where several specific pieces could be on a square.
    """

    evolutions: Dict[SpecificPiece, PieceEvolution] = field(default_factory=dict)
    invisible_move_count: int = 0
    constraints: List[PieceConstraint] = field(default_factory=list)
    prerequisites: Dict[SpecificPiece, Set[chess.Square]] = field(default_factory=dict)

    @classmethod
    def initial(cls, color: chess.Color) -> PieceTracking:
        """Initial piece positions of ``color`` are determined by the standard starting position of a chess game."""
        evolutions: Dict[SpecificPiece, PieceEvolution] = {}
        for square, piece in chess.Board().piece_map().items():
            if piece.color != color:
                continue
            file = chess.square_file(square) if piece.piece_type not in {chess.KING, chess.QUEEN} else None
            evolutions[SpecificPiece(piece.color, piece.piece_type, file)] = {0: {square}}
        return cls(evolutions)

    def __contains__(self, piece: SpecificPiece) -> bool:
        return piece in self.evolutions

    def __getitem__(self, piece: SpecificPiece) -> PieceEvolution:
        return self.evolutions[piece]

    # ------------------------------------------------------------------ queries

    def possible_squares(self, piece: SpecificPiece) -> Set[chess.Square]:
        """All squares a piece could be on: it may have made any number of moves up to its max age."""
        evolution = self.evolutions[piece]
        return set().union(*evolution.values()) if evolution else set()

    def latest_squares(self, piece: SpecificPiece) -> Set[chess.Square]:
        """Squares of the most recent step in the evolution, from which the piece advances its next move."""
        evolution = self.evolutions[piece]
        return evolution[max(evolution)] if evolution else set()

    def determined_squares(self) -> Dict[SpecificPiece, chess.Square]:
        """Pieces with a single possible square, mapped to that square."""
        return {
            piece: next(iter(squares))
            for piece in self.evolutions
            if len(squares := self.possible_squares(piece)) == 1
        }

    def candidates_at(self, piece_type: chess.PieceType | None, square: chess.Square) -> List[SpecificPiece]:
        """Specific pieces of ``piece_type`` (any type if None) that could be on ``square``."""
        return [
            piece for piece in self.evolutions
            if (piece_type is None or piece.piece_type == piece_type) and square in self.possible_squares(piece)
        ]

    def required_absent_squares(self, piece: SpecificPiece, destination: chess.Square) -> Set[chess.Square]:
        """
        Lightweight prerequisite model: a specific piece may need a particular square to be empty before it can
        reasonably reach ``destination``. This is intentionally simple: we model one piece not on one square rather
        than richer multi-square path dependencies.
        """
        required = set(self.prerequisites.get(piece, set()))
        if piece.piece_type == chess.BISHOP and piece.file == chess.FILE_NAMES.index("f") and piece.color == chess.WHITE:
            if destination == chess.C4:
                required.add(chess.E2)
        return required

    @staticmethod
    def cleared_path(piece: SpecificPiece, origin: chess.Square, square: chess.Square) -> Set[chess.Square]:
        """
        Squares a sliding piece (rook, bishop, queen) must have passed through to reach ``square`` from ``origin``
        along one of its lines. Other pieces jump or step, so they need no cleared path.
        """
        if piece.piece_type not in SLIDING_PIECE_TYPES:
            return set()
        # the piece's lines through origin on an empty board
        lines = 0
        if piece.piece_type != chess.BISHOP:
            lines |= chess.BB_RANK_ATTACKS[origin][0] | chess.BB_FILE_ATTACKS[origin][0]
        if piece.piece_type != chess.ROOK:
            lines |= chess.BB_DIAG_ATTACKS[origin][0]
        if not lines & chess.BB_SQUARES[square]:
            return set()
        return set(chess.SquareSet(chess.between(origin, square)))

    def next_squares(self, piece: SpecificPiece, origins: Set[chess.Square], view: FogView) -> Set[chess.Square]:
        """
        Return pseudo-legal destinations of ``piece`` from any of ``origins``. Only determined pieces are obstacles:
        the other side's pieces from ``view``, tracked pieces with a single possible square, and blocked squares (an
        unknown tracked piece). Pseudo-legal moves are used because FoW allows moving into check.
        """
        base = chess.Board(None)
        for square, value in enumerate(view.values):
            if isinstance(value, chess.Piece) and value.color != piece.color:
                base.set_piece_at(square, value)
            elif value == BLOCKED_PIECE:
                base.set_piece_at(square, chess.Piece(chess.PAWN, piece.color))
        for other, square in self.determined_squares().items():
            if other != piece:
                base.set_piece_at(square, chess.Piece(other.piece_type, other.color))
        base.castling_rights = chess.BB_EMPTY
        base.turn = piece.color

        destinations: Set[chess.Square] = set()
        for origin in origins:
            board = base.copy(stack=False)
            board.set_piece_at(origin, chess.Piece(piece.piece_type, piece.color))
            for move in board.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[origin]):
                required_absent = self.required_absent_squares(piece, move.to_square)
                if any(
                    square in required_absent and board.piece_at(square) is not None and board.piece_at(square).color == piece.color
                    for square in required_absent
                ):
                    continue
                destinations.add(move.to_square)
        return destinations

    # ------------------------------------------------------------------ updates

    def advance(self, piece: SpecificPiece, squares: Set[chess.Square]) -> int:
        """Append ``squares`` as the next step of ``piece``'s evolution and return its age."""
        evolution = self.evolutions[piece]
        age = max(evolution, default=-1) + 1
        evolution[age] = squares
        return age

    def append_square(self, piece: SpecificPiece, square: chess.Square) -> int:
        """Add ``square`` to the latest step of ``piece``'s evolution without advancing its age, and return the age."""
        evolution = self.evolutions[piece]
        age = max(evolution, default=0)
        evolution.setdefault(age, set()).add(square)
        return age

    def consume_moves(self, candidates: Iterable[SpecificPiece], squares: Set[chess.Square]) -> None:
        """
        One of ``candidates`` is observed on one of ``squares``: it made at least the minimum age, across all
        candidates, at which it could be there of the invisible moves. Deduct it from the invisible move count; the
        other pieces' ages are trimmed to the remaining count by ``refresh``.
        """
        min_age = min(
            (age for piece in candidates for age, step in self.evolutions[piece].items() if step & squares),
            default=0,
        )
        self.invisible_move_count = max(0, self.invisible_move_count - min_age)

    def reset(self, piece: SpecificPiece, square: chess.Square) -> None:
        """Identify ``piece`` on ``square``: no other piece can be there."""
        self.evolutions[piece] = {0: {square}}
        for other, evolution in self.evolutions.items():
            if other != piece:
                for squares in evolution.values():
                    squares.discard(square)

    def trim(self, piece: SpecificPiece, age: int, square: chess.Square) -> None:
        """``piece`` is known to have moved to ``square`` at ``age``: discard its older evolution."""
        evolution = self.evolutions[piece]
        if evolution.get(age) and square in evolution[age]:
            self.evolutions[piece] = {step - age: squares for step, squares in evolution.items() if step >= age}
            self.evolutions[piece][0] = {square}

    def capture(self, piece_type: chess.PieceType, square: chess.Square) -> None:
        """
        Identify the specific piece captured on ``square`` and remove it. If several specific pieces could be there,
        remove one of them and hand its possible squares to the others, since any of them may be the survivor
        standing on the removed piece's squares.
        """
        candidates = self.candidates_at(piece_type, square)
        if not candidates:
            raise Exception(f"Captured {chess.piece_name(piece_type)} at {chess.square_name(square)} is not tracked.")
        if len(candidates) == 1:
            del self.evolutions[candidates[0]]
            return

        removed, survivors = candidates[0], candidates[1:]
        logger.info(f"Ambiguous capture at {chess.square_name(square)}: removing {removed}, merging into {survivors}.")
        removed_evolution = self.evolutions.pop(removed)
        for survivor in survivors:
            evolution = self.evolutions[survivor]
            for age, squares in removed_evolution.items():
                evolution.setdefault(age, set()).update(squares)
            for squares in evolution.values():
                squares.discard(square)

    def remove_squares(self, removed: Set[chess.Square]) -> None:
        """No tracked piece can stand on ``removed``, so drop them from every piece's evolution."""
        for evolution in self.evolutions.values():
            for squares in evolution.values():
                squares -= removed

    def add_constraint(
        self,
        square: chess.Square,
        piece_type: chess.PieceType | None,
        candidates: Dict[SpecificPiece, int],
    ) -> None:
        """Record that exactly one of the ``candidates`` is on ``square``, replacing an older constraint there."""
        self.constraints = [constraint for constraint in self.constraints if constraint.square != square]
        self.constraints.append(PieceConstraint(square=square, piece_type=piece_type, candidates=candidates))

    def refresh(self) -> None:
        """
        Bring the tracking back to a consistent state after an evidence is applied.

        No piece can have made more moves than the invisible move count, so older ages are trimmed, except the age at
        which a constraint candidate reached its constraint square: beyond the count, it can only be there.

        A piece with a single possible square is determined to occupy it, so no other piece can be there.
        Constraint candidates that are captured or determined on another square are removed from the constraint;
        once a single candidate remains, it is assigned to the constraint square and the constraint is dropped.
        Either may determine another piece in turn, so repeat until nothing changes.
        """
        constraint_ages: Dict[SpecificPiece, int] = {}
        for constraint in self.constraints:
            for piece, age in constraint.candidates.items():
                constraint_ages[piece] = max(age, constraint_ages.get(piece, 0))
        for piece, evolution in self.evolutions.items():
            max_age = max(self.invisible_move_count, constraint_ages.get(piece, 0))
            for age in [age for age in evolution if age > max_age]:
                del evolution[age]
        for constraint in self.constraints:
            for piece, age in constraint.candidates.items():
                if age > self.invisible_move_count and age in self.evolutions.get(piece, {}):
                    self.evolutions[piece][age] &= {constraint.square}

        prev_determined: Set[chess.Square] = set()
        while True:
            remaining: List[PieceConstraint] = []
            for constraint in self.constraints:
                determined_pieces = {
                    piece for piece, square in self.determined_squares().items() if square != constraint.square
                }
                candidates = {
                    piece: age for piece, age in constraint.candidates.items()
                    if piece in self and piece not in determined_pieces
                }
                if len(candidates) > 1:
                    constraint.candidates = candidates
                    remaining.append(constraint)
                elif candidates:
                    piece, age = next(iter(candidates.items()))
                    self.trim(piece, age, constraint.square)
            self.constraints = remaining

            determined = {square: piece for piece, square in self.determined_squares().items()}
            new_determined = set(determined) - prev_determined
            if not new_determined:
                return
            for piece, evolution in self.evolutions.items():
                for squares in evolution.values():
                    squares -= {square for square in new_determined if determined[square] != piece}
            prev_determined |= new_determined

    def display(self) -> str:
        lines = []
        for piece, evolution in self.evolutions.items():
            ages = ", ".join(
                f"{age}: {{{', '.join(sorted(chess.square_name(s) for s in squares))}}}"
                for age, squares in sorted(evolution.items())
            )
            lines.append(f"{piece!r}: {{{ages}}}")
        return "\n".join(lines)


class PieceTrackingModel:
    """
    Track the possible squares of each opponent piece from ``color``'s side, by logical deduction from the views.

    piece_tracking holds the possible squares of each opponent piece and the constraints between them.
    """

    def __init__(self, color: chess.Color, initial_view: FogView | None = None):
        self.color = color
        self.opponent = not color
        self.current_view = initial_view or FogView(self.color, [INVISIBLE_PIECE] * 64)
        # the view before the latest move, e.g. to know which squares blocked the mover
        self.piece_tracking = PieceTracking.initial(self.opponent)
        # evidences queued for the current apply_evidences pass; derived evidences are appended during the pass
        self.evidences: List[Evidence] = []

    def hypothetical_copy(self) -> PieceTrackingModel:
        """A copy to change hypothetically: only the piece tracking is copied, the views are shared (never modified)."""
        model = copy.copy(self)
        model.piece_tracking = copy.deepcopy(self.piece_tracking)
        model.evidences = []
        return model

    def allows(self, board: chess.Board) -> bool:
        """
        Whether the opponent pieces on ``board`` fit the tracking: each on a square possible for a tracked piece of its
        type. Captured pieces are removed from the boards as from the tracking, so the counts need no check.
        """
        tracking = self.piece_tracking
        for piece_type in chess.PIECE_TYPES:
            tracked = [piece for piece in tracking.evolutions if piece.piece_type == piece_type]
            possible = set().union(*(tracking.possible_squares(piece) for piece in tracked))
            if not set(board.pieces(piece_type, self.opponent)) <= possible:
                return False
        return True

    def update_own_move(self, move: chess.Move, view: FogView) -> None:
        """
        Our move is fully known; view is the new FogView after we make our move.
        """
        capture_move = self._deduce_capture(self.current_view, move)
        self.apply_evidences([
            capture_move,
            *(observe_square(square, value) for square, value in enumerate(view.values)),
        ])
        self.current_view = view

    def update_opponent_move(self, view: FogView) -> FogMove:
        """
        The opponent move is deduced from the view change; view is the new FogView after the opponent's move.
        Every visible square is observed except the squares the FogMove already accounts for, so the moved piece is
        not reset before the move is applied. Other squares may still be discovered by the move.
        Returns the deduced FoW move so existing callers keep the same interface.
        """
        fog_move = deduce_opponent_move(self.current_view, view)
        move_squares = {fog_move.from_square, fog_move.to_square}
        self.apply_evidences([
            fog_move,
            *(observe_square(square, value) for square, value in enumerate(view.values) if square not in move_squares),
        ])
        self.current_view = view
        return fog_move

    # ------------------------------------------------------------------ public API

    @staticmethod
    def _evidence_priority(evidence: Evidence) -> int:
        """Higher priority means more deterministic evidence and should be applied first."""
        if isinstance(evidence, CaptureMove):
            return 3
        if isinstance(evidence, (EmptySquare, BlockedSquare, VisiblePiece)):
            return 2
        if isinstance(evidence, FogMove):
            return 1
        return 0

    def apply_evidence(self, evidence: Evidence) -> None:
        """
        Apply one atomic fact to the belief state. Derived evidences are queued on ``self.evidences`` so they can be
        processed after the current evidence pass completes.
        """
        if isinstance(evidence, CaptureMove):
            self._apply_capture_move_evidence(evidence)
        elif isinstance(evidence, EmptySquare):
            self._apply_empty_square_evidence(evidence)
        elif isinstance(evidence, BlockedSquare):
            self._apply_blocked_square_evidence(evidence)
        elif isinstance(evidence, VisiblePiece):
            self._apply_visible_piece_evidence(evidence)
        elif isinstance(evidence, FogMove) and evidence.from_square is None and evidence.to_square is None and evidence.piece is None:
            self._apply_invisible_fog_move_evidence()
        elif isinstance(evidence, FogMove) and evidence.to_square is not None:
            self._apply_observed_fog_move(evidence)
        elif isinstance(evidence, FogMove):
            self._apply_unseen_destination_fog_move(evidence)
        elif evidence is not None:
            raise TypeError(f"Unsupported evidence type: {type(evidence)!r}")
        self.piece_tracking.refresh()

    def apply_evidences(self, evidences: Iterable[Evidence]) -> None:
        """
        Apply evidence in deterministic order: specific observations before uncertain FoW moves. Derived evidences (visible piece could indicate no other pieces blocking the path)
        are appended to ``self.evidences`` on-the-fly.
        """
        self.evidences = [
            evidence for evidence in sorted(evidences, key=self._evidence_priority, reverse=True)
            if evidence is not None
        ]
        while self.evidences:
            evidence = self.evidences.pop(0)
            self.apply_evidence(evidence)

    def _apply_capture_move_evidence(self, evidence: CaptureMove) -> None:
        self.piece_tracking.capture(evidence.piece_type, evidence.square)

    def _apply_empty_square_evidence(self, evidence: EmptySquare) -> None:
        self.piece_tracking.remove_squares({evidence.square})
        self._drop_vacated_constraint(evidence.square, EMPTY_SQUARE)

    def _apply_blocked_square_evidence(self, evidence: BlockedSquare) -> None:
        # Blocked can be observed after we make our own move
        self._apply_occupied_square_evidence(None, evidence.square)

    def _apply_visible_piece_evidence(self, evidence: VisiblePiece) -> None:
        self._apply_occupied_square_evidence(evidence.piece.piece_type, evidence.square)
        self._drop_vacated_constraint(evidence.square, evidence.piece)

    def _apply_invisible_fog_move_evidence(self) -> None:
        """Every piece may have made the move, so every piece advances one step within invisible space."""
        tracking = self.piece_tracking
        updates: Dict[SpecificPiece, Set[chess.Square]] = {}
        # hold off updating tracking piece-by-piece until last because we want to use determined piece at current stage as it may block other pieces.
        for piece in list(tracking.evolutions):
            possible = tracking.next_squares(piece, tracking.latest_squares(piece), self.current_view)
            next_evolution = {square for square in possible if self.current_view.values[square] == INVISIBLE_PIECE}
            if next_evolution:
                updates[piece] = next_evolution
        for piece, squares in updates.items():
            tracking.advance(piece, squares)
        tracking.invisible_move_count += 1

    def _apply_observed_fog_move(self, fog_move: FogMove) -> None:
        """
        Known move-to square, with known or unknown piece. Derive from candidates based on tracking and append the square to latest evolution.
        """
        candidates = self._candidates(fog_move.piece.piece_type if fog_move.piece else None, fog_move.from_square, fog_move.to_square)
        if not candidates:
            raise Exception(f"No tracked piece can explain opponent move {fog_move}.")

        if len(candidates) == 1:
            piece = candidates[0]
            self.piece_tracking.reset(piece, fog_move.to_square)
            self._extend_cleared_path_evidence(piece, fog_move.to_square, origins={fog_move.from_square} if fog_move.from_square else None)

        else:
            for piece in candidates:
                self.piece_tracking.append_square(piece, fog_move.to_square)
            # the earliest age at which each candidate could have reached the square
            ages_by_candidate = {
                candidate: min(
                    age for age, squares in self.piece_tracking[candidate].items() if fog_move.to_square in squares
                )
                for candidate in candidates
            }
            piece_type = fog_move.piece.piece_type if fog_move.piece else None
            self.piece_tracking.add_constraint(fog_move.to_square, piece_type, ages_by_candidate)

    def _apply_unseen_destination_fog_move(self, fog_move: FogMove) -> None:
        """
        The mover went into the dark (to square unknown). A single candidate seen leaving a known from square restarts
        its evolution from the invisible squares it can reach, at age 0 since the move itself was observed. Otherwise
        every candidate advances one step within invisible space, as for an invisible move.
        """
        tracking = self.piece_tracking
        piece_type = fog_move.piece.piece_type if fog_move.piece else None
        destinations = {
            piece: reachable
            for piece in self._candidates(piece_type, fog_move.from_square, None)
            if (reachable := {
                square for square in tracking.next_squares(
                    piece,
                    {fog_move.from_square} if fog_move.from_square is not None else tracking.latest_squares(piece),
                    self.current_view,
                )
                if self.current_view.values[square] == INVISIBLE_PIECE
            })
        }
        if not destinations:
            raise Exception(f"No tracked piece can explain opponent move {fog_move}.")

        if len(destinations) == 1 and fog_move.from_square is not None:
            piece, reachable = next(iter(destinations.items()))
            tracking.evolutions[piece] = {0: reachable}
        else:
            for piece, reachable in destinations.items():
                tracking.advance(piece, reachable)
            tracking.invisible_move_count += 1

        if fog_move.from_square is not None and self.current_view.values[fog_move.from_square] == EMPTY_SQUARE:
            self.evidences.append(EmptySquare(fog_move.from_square))


    def simulate(self, rng: random.Random) -> Dict[SpecificPiece, chess.Square] | None:
        """
        One random placement of the tracked pieces, without any evaluation, on a single hypothetical copy: pieces from
        the most determined to the least, each on a uniformly random square still possible for it. Placing a piece
        consumes the moves it needed and narrows the others. None on a dead end, where a piece has no square left.
        """
        model = self.hypothetical_copy()
        tracking = model.piece_tracking
        placement: Dict[SpecificPiece, chess.Square] = {}
        for piece in sorted(tracking.evolutions, key=lambda piece: len(tracking.possible_squares(piece))):
            squares = tracking.possible_squares(piece)
            if not squares:
                return None
            placement[piece] = rng.choice(sorted(squares))
            model.place_piece(piece, placement[piece])
        return placement

    def place_piece(self, piece: SpecificPiece, square: chess.Square) -> None:
        """
        Hypothetically observe ``piece`` on ``square``, e.g. to simulate a full board on a copy of the model. It is
        applied like a visible piece with a single candidate: the moves it needed are consumed, its cleared path is
        emptied, and the tracking is refreshed, which trims the other pieces to the remaining move count.
        """
        self.evidences = []
        self._apply_occupied_square_single_candidate_evidence(piece, square)
        self.piece_tracking.refresh()
        while self.evidences:
            self.apply_evidence(self.evidences.pop(0))

    def possible_pieces_at(self, square: chess.Square) -> Set[SpecificPiece]:
        """Specific pieces that could currently be on ``square``, e.g. to identify a blocked square."""
        return set(self.piece_tracking.candidates_at(None, square))

    # ------------------------------------------------------------------ opponent move

    def _candidates(
        self,
        piece_type: chess.PieceType | None,
        from_square: chess.Square | None,
        to_square: chess.Square | None,
    ) -> List[SpecificPiece]:
        """
        Specific pieces of ``piece_type`` that could currently be on ``from_square`` and could move to ``to_square``
        with one more move (from ``from_square`` if known, otherwise from any of their possible squares). A None
        argument places no restriction.
        """
        tracking = self.piece_tracking
        candidates = []
        for piece in tracking.evolutions:
            possible = tracking.possible_squares(piece)
            origins = {from_square} if from_square is not None else possible
            type_matches = piece_type is None or piece.piece_type == piece_type
            from_matches = from_square is None or from_square in possible
            to_matches = to_square is None or to_square in tracking.next_squares(piece, origins, self.current_view)
            if type_matches and from_matches and to_matches:
                candidates.append(piece)
        return candidates

    def _apply_castling_rook(
        self,
        king: SpecificPiece,
        from_square: chess.Square | None,
        to_square: chess.Square | None,
    ) -> None:
        """Castling also moves the rook, which may happen out of sight."""
        if king.piece_type != chess.KING or from_square is None or to_square is None:
            return
        if abs(chess.square_file(to_square) - chess.square_file(from_square)) != 2:
            return
        rank = chess.square_rank(from_square)
        kingside = chess.square_file(to_square) > chess.square_file(from_square)
        rook = SpecificPiece(king.color, chess.ROOK, 7 if kingside else 0)
        if rook in self.piece_tracking:
            self.piece_tracking.reset(rook, chess.square(5 if kingside else 3, rank))

    # ------------------------------------------------------------------ visibility updates

    def _deduce_capture(self, before: FogView, move: chess.Move) -> CaptureMove | None:
        """
        Our move captures the opponent piece on its target square, which is always visible before the move.
        En passant is a diagonal pawn move onto an empty square: the captured pawn is on the square behind it,
        which may be invisible (a pawn does not see the square beside it).
        """
        target = before.values[move.to_square]
        if isinstance(target, chess.Piece) and target.color == self.opponent:
            return CaptureMove(move.to_square, target.piece_type)

        # En passant capture
        mover = before.values[move.from_square]
        is_diagonal = chess.square_file(move.from_square) != chess.square_file(move.to_square)
        if mover == chess.Piece(chess.PAWN, self.color) and is_diagonal and target == EMPTY_SQUARE:
            forward = 8 if self.color == chess.WHITE else -8
            return CaptureMove(move.to_square - forward, chess.PAWN)
        return None

    def _apply_occupied_square_evidence(self, piece_type: chess.PieceType | None, square: chess.Square) -> None:
        """
        When a visible or blocked opponent piece is observed, reset its evolution to the observed square.
        Knights, rooks, bishops and pawns are indistinguishable, so the specific piece is deduced from the tracking.
        A blocked piece is of unknown type (``piece_type`` None), so any specific piece that could be on the square
        is a candidate. If multiple specific pieces match, exactly one of them is on the square, which is recorded as
        a constraint (unless one already exists for the square, e.g. from the opponent move that landed there).
        """
        candidates = self._candidates(piece_type, square, None)
        if len(candidates) == 1:
            self._apply_occupied_square_single_candidate_evidence(candidates[0], square)
        else:
            self._apply_occupied_square_multiple_candidate_evidence(candidates, piece_type, square)

    def _extend_cleared_path_evidence(
        self,
        piece: SpecificPiece,
        square: chess.Square,
        origins: Set[chess.Square] | None = None,
    ) -> None:
        """
        Queue the path ``piece`` cleared to reach ``square`` as empty square evidences. Without ``origins`` (the move
        was not seen), the path is only known to be clear if the piece reached the square with the last invisible
        moves (no other piece moved since), from the squares of the step before its arrival. The path is only known
        when the origin is determined.
        """
        tracking = self.piece_tracking
        if origins is None:
            evolution = tracking[piece]
            arrival = min(age for age, squares in evolution.items() if square in squares)
            if not 0 < arrival == tracking.invisible_move_count:
                return
            origins = evolution[arrival - 1]
        if len(origins) != 1:
            return
        self.evidences.extend(
            EmptySquare(path_square) for path_square in tracking.cleared_path(piece, next(iter(origins)), square)
        )

    def _apply_occupied_square_single_candidate_evidence(self, piece: SpecificPiece, square: chess.Square) -> None:
        """
        Only ``piece`` can be on ``square``, so reset it there. A sliding piece that reached the square with the last
        invisible moves (no other piece moved since) left its path empty, which is queued as empty square evidences.
        """
        # deduct total invisible moves by observed minimum moves
        self.piece_tracking.consume_moves([piece], {square})
        
        self._extend_cleared_path_evidence(piece, square)

        self.piece_tracking.reset(piece, square)

    def _apply_occupied_square_multiple_candidate_evidence(
        self,
        candidates: List[SpecificPiece],
        piece_type: chess.PieceType | None,
        square: chess.Square,
    ) -> None:
        """Exactly one of ``candidates`` is on ``square``, which is recorded as a constraint."""
        tracking = self.piece_tracking
        # deduct total invisible moves by observed minimum moves
        tracking.consume_moves(candidates, {square})

        # the earliest age at which each candidate could have reached the square
        ages_by_candidate = {
            candidate: min(age for age, squares in tracking[candidate].items() if square in squares)
            for candidate in candidates
        }
        tracking.add_constraint(square, piece_type, ages_by_candidate)

    # ------------------------------------------------------------------ constraints

    def _drop_vacated_constraint(self, square: chess.Square, value: int | chess.Piece) -> None:
        """
        A constraint disappears once its square is observed without the moved piece: the piece moved away (or was
        captured) to an unknown square, so the piece tracking simply keeps evolving.
        """
        def holds_mover(constraint: PieceConstraint) -> bool:
            return value in (INVISIBLE_PIECE, BLOCKED_PIECE) or (
                isinstance(value, chess.Piece) and value.color == self.opponent
                and (constraint.piece_type is None or value.piece_type == constraint.piece_type)
            )

        tracking = self.piece_tracking
        tracking.constraints = [
            constraint for constraint in tracking.constraints if constraint.square != square or holds_mover(constraint)
        ]

    def display(self) -> str:
        return self.piece_tracking.display()


if __name__ == "__main__":
    from fow_chess import FoWChessBoard

    board = FoWChessBoard()
    model = PieceTrackingModel(chess.WHITE, board.generate_view(chess.WHITE))
    for uci in ["e2e4", "g8f6", "d2d4", "b8c6", "d4d5", "c6e5"]:
        move = chess.Move.from_uci(uci)
        mover = board.turn
        board.push(move)
        view = board.generate_view(chess.WHITE)
        if mover == model.color:
            model.update_own_move(move, view)
        else:
            print(uci, model.update_opponent_move(view))
    print(board.display(chess.WHITE))
    print(model.display())
