from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING, Collection

import chess

from static import BLOCKED_PIECE, INVISIBLE_PIECE

if TYPE_CHECKING:
    from fow_view import FogView

MARGIN = 30
SQUARE_SIZE = 72
BOARD_SIZE = 2 * MARGIN + 8 * SQUARE_SIZE

BACKGROUND = "#111827"
LIGHT_SQUARE = "#e7dfcf"
DARK_SQUARE = "#77917f"
FOG_LIGHT = "#485568"
FOG_DARK = "#303b4c"
SELECTED = "#e9c46a"
MOVE_DOT = "#244b3c"
CAPTURE = "#c75b55"


def square_at_grid(color: chess.Color, col: int, row: int) -> chess.Square:
    """The square at column ``col`` and row ``row`` of the board drawn from ``color``'s side."""
    if color == chess.WHITE:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def draw_view(
    canvas: tk.Canvas,
    view: FogView,
    selected: chess.Square | None = None,
    destinations: Collection[chess.Square] = (),
) -> None:
    """
    Draw ``view`` on ``canvas`` from its side, with the ``selected`` square outlined and dots (rings on captures) on
    the ``destinations``.
    """
    canvas.delete("all")
    for row in range(8):
        for col in range(8):
            square = square_at_grid(view.color, col, row)
            x1, y1 = MARGIN + col * SQUARE_SIZE, MARGIN + row * SQUARE_SIZE
            x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            value = view.values[square]
            light = (col + row) % 2 == 0
            visible = value != INVISIBLE_PIECE and value != BLOCKED_PIECE
            if visible:
                background = LIGHT_SQUARE if light else DARK_SQUARE
            else:
                background = FOG_LIGHT if light else FOG_DARK
            canvas.create_rectangle(x1, y1, x2, y2, fill=background, outline=background)

            if square == selected:
                canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2, outline=SELECTED, width=5)

            if isinstance(value, chess.Piece):
                canvas.create_text(
                    center_x, center_y - 1, text=value.unicode_symbol(),
                    fill="#fffdf6" if value.color == chess.WHITE else "#202938", font=("Arial", 43),
                )
            elif value == BLOCKED_PIECE:
                canvas.create_text(center_x, center_y, text="△", fill="#f4c86a", font=("Arial", 29, "bold"))

            if square in destinations:
                if isinstance(value, chess.Piece) and value.color != view.color:
                    canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, outline=CAPTURE, width=5)
                else:
                    radius = 10
                    canvas.create_oval(
                        center_x - radius, center_y - radius, center_x + radius, center_y + radius,
                        fill=MOVE_DOT, outline="",
                    )

            foreground = "#d6dce5" if not visible else (DARK_SQUARE if light else LIGHT_SQUARE)
            if col == 0:
                canvas.create_text(
                    x1 + 7, y1 + 7, text=str(chess.square_rank(square) + 1),
                    anchor="nw", fill=foreground, font=("Arial", 8, "bold"),
                )
            if row == 7:
                canvas.create_text(
                    x2 - 7, y2 - 6, text=chess.FILE_NAMES[chess.square_file(square)],
                    anchor="se", fill=foreground, font=("Arial", 8, "bold"),
                )

    canvas.create_rectangle(MARGIN, MARGIN, MARGIN + 8 * SQUARE_SIZE, MARGIN + 8 * SQUARE_SIZE,
                            outline="#1b2635", width=2)


def show_view(view: FogView, title: str = "Fog of War Chess") -> None:
    """Show ``view`` in a window, returning when it is closed."""
    root = tk.Tk()
    root.title(title)
    canvas = tk.Canvas(root, width=BOARD_SIZE, height=BOARD_SIZE, bg=BACKGROUND, highlightthickness=0)
    canvas.pack()
    # show the window before drawing: on macOS, a canvas drawn before its window is shown can stay blank
    root.lift()
    root.update()
    draw_view(canvas, view)
    root.mainloop()
