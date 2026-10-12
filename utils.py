import re
from pathlib import Path
from typing import List

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
