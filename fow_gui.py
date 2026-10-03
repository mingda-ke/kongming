from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import chess

from engine import Engine
from evaluator import Evaluator
from fow_chess import FoWChessBoard
from fow_canvas import BACKGROUND, BOARD_SIZE, MARGIN, SQUARE_SIZE, draw_view, square_at_grid


class FogOfWarChessApp:
    """A Tkinter interface for a human playing ``human_color`` against the engine on a ``FoWChessBoard``."""

    WINDOW_BG = BACKGROUND
    PANEL_BG = "#1f2937"
    TEXT = "#f3f4f6"
    MUTED = "#aab4c3"

    def __init__(self, root: tk.Tk, human_color: chess.Color = chess.WHITE):
        self.root = root
        self.human_color = human_color
        self.engine = Engine(Evaluator())
        self.root.title("Fog of War Chess")
        self.root.configure(bg=self.WINDOW_BG)
        self.root.minsize(910, 700)

        self._configure_style()
        self._build_layout()
        self._start_game()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", font=("Arial", 11, "bold"), padding=(12, 9))
        style.configure("Status.TLabel", background=self.PANEL_BG, foreground=self.TEXT, font=("Arial", 16, "bold"))
        style.configure("Body.TLabel", background=self.PANEL_BG, foreground=self.MUTED, font=("Arial", 10))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        content = tk.Frame(self.root, bg=self.WINDOW_BG, padx=24, pady=20)
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        header = tk.Frame(content, bg=self.WINDOW_BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        tk.Label(
            header, text="FOG OF WAR", bg=self.WINDOW_BG, fg=self.TEXT,
            font=("Arial", 23, "bold"),
        ).pack(side="left")
        tk.Label(
            header, text="CHESS", bg=self.WINDOW_BG, fg="#89b99b",
            font=("Arial", 11, "bold"), padx=12,
        ).pack(side="left", pady=(8, 0))

        self.canvas = tk.Canvas(
            content, width=BOARD_SIZE, height=BOARD_SIZE, bg=self.WINDOW_BG,
            highlightthickness=0, cursor="hand2",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_board_click)

        self.panel = tk.Frame(content, bg=self.PANEL_BG, padx=20, pady=20, width=250)
        self.panel.grid(row=1, column=1, sticky="ns", padx=(20, 0))
        self.panel.grid_propagate(False)

        tk.Label(
            self.panel, text="YOUR TURN", bg=self.PANEL_BG, fg=self.MUTED,
            font=("Arial", 9, "bold"),
        ).pack(anchor="w")
        self.status_label = ttk.Label(self.panel, style="Status.TLabel", wraplength=205, justify="left")
        self.status_label.pack(anchor="w", pady=(7, 20))

        self.turn_indicator = tk.Canvas(self.panel, width=38, height=38, bg=self.PANEL_BG, highlightthickness=0)
        self.turn_indicator.pack(anchor="w", pady=(0, 18))

        ttk.Button(self.panel, text="New game", command=self._new_game).pack(fill="x", pady=(0, 20))

        tk.Frame(self.panel, bg="#3a4657", height=1).pack(fill="x", pady=(0, 17))
        tk.Label(
            self.panel, text="HOW TO PLAY", bg=self.PANEL_BG, fg=self.TEXT,
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        tk.Label(
            self.panel,
            text=(
                "Only squares your pieces can see are shown. Select one of your pieces, then choose a highlighted "
                "destination.\n\nThere is no check or checkmate. Capture the opposing king to win.\n\n"
                "A triangle marks a pawn push blocked by an unseen piece."
            ),
            bg=self.PANEL_BG, fg=self.MUTED, font=("Arial", 10),
            justify="left", wraplength=205,
        ).pack(anchor="w")

        tk.Label(
            self.panel, text="HUMAN VS ENGINE · LOCAL GAME", bg=self.PANEL_BG,
            fg="#778397", font=("Arial", 8, "bold"),
        ).pack(side="bottom", anchor="w")

    def _refresh(self) -> None:
        self._draw_board()
        if self.game_over:
            winner = "White" if self.board.king(chess.WHITE) is not None else "Black"
            self.status_label.configure(text=f"{winner} wins")
        elif self.board.turn == self.human_color:
            self.status_label.configure(text="Your move")
        else:
            self.status_label.configure(text="Engine is thinking...")
        self._draw_turn_indicator()

    def _draw_turn_indicator(self) -> None:
        self.turn_indicator.delete("all")
        fill = "#f4f1e8" if self.board.turn == chess.WHITE else "#202938"
        outline = "#d0d5dd" if self.board.turn == chess.WHITE else "#778397"
        self.turn_indicator.create_oval(3, 3, 35, 35, fill=fill, outline=outline, width=2)

    def _draw_board(self) -> None:
        draw_view(self.canvas, self.board.generate_view(self.human_color), self.selected, self._destination_squares())

    def _destination_squares(self) -> set[chess.Square]:
        return {move.to_square for move in self.selected_moves}

    def _on_board_click(self, event: tk.Event) -> None:
        if self.game_over or self.board.turn != self.human_color:
            return
        col = (event.x - MARGIN) // SQUARE_SIZE
        row = (event.y - MARGIN) // SQUARE_SIZE
        if not (0 <= col < 8 and 0 <= row < 8):
            return
        square = square_at_grid(self.human_color, col, row)

        if self.selected is not None:
            candidates = [move for move in self.selected_moves if move.to_square == square]
            if candidates:
                self.board.push(candidates[0])
                self._after_move()
                return

        piece = self.board.piece_at(square)
        if piece is not None and piece.color == self.board.turn:
            self.selected = square
            self.selected_moves = list(self.board.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[square]))
        else:
            self.selected = None
            self.selected_moves = []
        self._refresh()

    def _after_move(self) -> None:
        self.selected = None
        self.selected_moves = []
        self.game_over = self.board.king(chess.WHITE) is None or self.board.king(chess.BLACK) is None
        self._refresh()
        if not self.game_over and self.board.turn != self.human_color:
            # let the board redraw before the engine's search blocks the window
            self.root.after(50, self._engine_move)

    def _engine_move(self) -> None:
        self.board.engine_move()
        self._after_move()

    def _start_game(self) -> None:
        self.board = FoWChessBoard(engine=self.engine, engine_color=not self.human_color)
        self._after_move()

    def _new_game(self) -> None:
        if self.game_over or messagebox.askyesno("New game", "Start a new game?", parent=self.root):
            self._start_game()


def main() -> None:
    root = tk.Tk()
    FogOfWarChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
