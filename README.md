# kongming
Fog of War chess engine. Kong Ming (aka Zhuge Liang) is the greatest military strategist in ancient China, Three Kingdoms period. FoW chess is like a real war that you can only see where you can move or attach. The goal is to directly capture opponent's king, so no checkmate or walk into checkmate.

### Development Plan
1. Rules. FoW rules extension to chess python package. All rules are the same as regular chess except:
   - a. move representation from one side point of view: e.g. ? (no information), *xe4 (something takes e4 square), *h4 (something occupies h4 square)
      - not needed as board state representation is enough.
   - b. board state representation from each side point of view: empty string (still for known empty square), ? (unknown), * (something occupies this square)
   - c. checkmate: no more checkmate, need to capture king directly
   - d. castle: always allowed, no more restriction

2. Probability model. The most important part of the game is to deduct and guess where opponent pieces are based on known information. After a few moves, the invisible parts can have billions of possible states. So instead of storing probability/possibility of each board state, we store probability/possibility of each piece and calculate probability/possibility of states that we need later.

Build Algorithm:
   - Initial known state
   - After each opponent move, for pieces in dark, store next legal move states from previous possible states (increment). After a few moves, the data should be like {Knight1: {0: set(b1), 1: set(a3, c3), 2: set(a4, b5, c4, d5, e4)}, PawnA: {0: set(a2), 1: set(a3, a4), 2: set(a4, a5)}, ...}. 1 and 2 are move counts from last time seen the piece.
   - Whenever see the piece, reset the dict of the piece to {0: set(observed square)}, and no need to increment other pieces.
   - Dependency between pieces: for example Q/B/R requires path is cleared from its own pieces. Take it as consideration.
   - When opponent recaptures our piece but we don't know what recaptured, only increment pieces that are possible to do so.

3. Optimization (minimax model with alpha-beta pruning). The basic tree based minimax algorithm works the same for FoW chess.
   - a. heuristics (as terminal evaluation): besides regular piece activity, information is also important factor.
   - b. minimax considers opponent playing their best moves, but we need to model from their visibility. Here comes bluffing: if we throwing a bishop to attack opponent's queen, he probably won't take it if he doesn't know whether it's protected or not./
       However, if under opponent's probability that our queen is somewhere else (for example just took a pawn when only Queen is left on the board), he would take the bishop. This model behavior makes sure how much we should expose our information.
   - c. risk aversion. regular chess minimax is targeting on one objective function but FoW has multiple outcomes (distribution). There should be a function to convert an outcome distribution into one scalar for optimization. This function is basically how aggressive we want to play. 
