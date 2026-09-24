# kongming
Fog of War chess engine. Kong Ming (aka Zhuge Liang) is the greatest military strategist in ancient China, Three Kingdoms period. FoW chess is like a real war that you can only see where you can move or attach. The goal is to directly capture opponent's king, so no checkmate or walk into checkmate.

### Guideline
The algorithm resembles how human thinks when playing Fog of War chess, but it has greater searching breadth, depth and mostly importantly higher accuracy than human (even the best human blunders), and therefore aims to achieve a higher Elo than best human.

### Algo
#### 1. Rules exception. 

FoW has two exception in legal moves than chess, so need to override these two functions.

   1. checkmate: no more checkmate, need to capture king directly. So king has legal move into a check.
   2. castle: allowed even when the king castling path is under attack.

#### 2. FoW view and FoW move

We only see what we can attack, i.e. *visibility* equals our legal move.

If pawn advancement is blocked (*blockage*), we can see but don't know what exact piece.

*FoW View*: we use a list of 64 items (corresponding to 64 squares) to represent what we see. Each item could be a Piece, empty square, invisible square, or blocked square.

After opponent makes a move, the *FoW View* will change or maybe not. We can deduce direct information about opponent move, which contains following information:

*FoW Move*:
- from_square: a certain square or unknown square
- to_square: a certain square or unknown square
- piece(type): Piece or unknown

The logic to deduce FoW Move is to scan all changes in squares and determine which piece is the "cause" move, since one piece move will expose another piece behind.

#### 3. Belief State Model

Before we make a move, we always try to guess where opponent pieces could be. There could be trillions of possible positions, but we don't need to simulate all of them, so the correct question for this model to answer is - what are top n opponent positions that I need to consider?

We approach it from the most deterministic and simple deduction to the most uncertain deduction which needs some weighting parameters as input or trained from played games.

   - *Piece Tracking* model (deterministic): use logical deduction to track possible squares of each specific piece, and apply certain constraints when there's ambiguity.

   *piece evolution*: stores a dictionary per *specific piece* a sequence of possible squares over number of its move. initialize with starting position. update with following rule:
      
      - Unknown opponent *FoW move*: advance all pieces with one step and record its next legal squares (within invisible space). Next legal squares for a piece should base on setting up other determined pieces to possible squares. After a few moves, the data should be like {Knight1: {0: set(b1), 1: set(a3, c3), 2: set(a4, b5, c4, d5, e4)}, PawnA: {0: set(a2), 1: set(a3, a4), 2: set(a4, a5)}, ...}. 1 and 2 are move counts from last time seen the piece.
      
      - opponent *FoW move* with known move-to square: even if we know the piece type, we still don't know which specific piece (knights, rooks and pawns are undistinguishable), so deduce it from current *piece tracking* dictionary. If only one specific piece matches, reset the dict of the piece to {0: set(observed square)}. But if multiple speicific piece matches, we need to use another *constraint* dictionary to record it for later simulation.

      - visibility update from after either our own move or opponent move:

         - similar to opponent move, for single candidate, reset the observed piece to the observed square; for multiple candidates, use *constraint*.

         - empty square: remove that square from all possible squares within the tracking dictionary

      - we capture opponent piece: identify the specific piece and remove it from tracking dictionary. When there's ambiguity in determining which specific piece (usually knight or rook), we remove either one and assign its possible squares to the other.

      - trimming: when one piece is determined to occupy a square, remove that square from all other possible squares, as well as ambiguity constraint.

      - Total move count: keep track of total invisible move count (age), and once observe a piece, deduct the minimum move from that age from the total invisible move count. Trim all other pieces age to no greater than remaining invisible move count.

Missing components:

   - piece dependency: current possibility model will include some scenarios that are not possible (due to dependency). This is especially common during opening stage. Consider using full simulation instead of piece-wise simulation during opening stage.


2. Probability model. The most important part of the game is to deduct and guess where opponent pieces are based on known information. After a few moves, the invisible parts can have billions of possible states. So instead of storing probability/possibility of each board state, we store probability/possibility of each piece and calculate probability/possibility of states that we need later.
   1. Initial known state
   2. After each opponent move, for pieces in dark, store next legal move states from previous possible states (increment). After a few moves, the data should be like {Knight1: {0: set(b1), 1: set(a3, c3), 2: set(a4, b5, c4, d5, e4)}, PawnA: {0: set(a2), 1: set(a3, a4), 2: set(a4, a5)}, ...}. 1 and 2 are move counts from last time seen the piece.
   3. Whenever see the piece, reset the dict of the piece to {0: set(observed square)}, and no need to increment other pieces.
   4. Dependency between pieces: for example Q/B/R requires path is cleared from its own pieces. Take it as consideration.
   5. When opponent recaptures our piece but we don't know what recaptured, only increment pieces that are possible to do so.

3. Optimization (minimax model with alpha-beta pruning). The basic tree based minimax algorithm works the same for FoW chess.
   1. heuristics (as terminal evaluation): besides regular piece activity, information is also important factor.
   2. minimax considers opponent playing their best moves, but we need to model from their visibility. Here comes bluffing: if we throwing a bishop to attack opponent's queen, he probably won't take it if he doesn't know whether it's protected or not.
      However, if under opponent's probability that our queen is somewhere else (for example just took a pawn when only Queen is left on the board), he would take the bishop. This model behavior makes sure how much we should expose our information.
   3. risk aversion. regular chess minimax is targeting on one objective function but FoW has multiple outcomes (distribution). There should be a function to convert an outcome distribution into one scalar for optimization. This function is basically how aggressive we want to play.
