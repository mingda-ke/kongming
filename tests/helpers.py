import shutil

import pytest

requires_stockfish = pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish binary not installed")


class FakeEngine:
    """Stands in for Stockfish: every position scores equal, and there are no top moves (so moves are uniform)."""

    def set_fen_position(self, fen: str) -> None:
        pass

    def get_evaluation(self) -> dict:
        return {"type": "cp", "value": 0}

    def get_top_moves(self, count: int) -> list:
        return []
