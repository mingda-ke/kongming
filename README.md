# kongming
Fog of War chess engine. Kong Ming (aka Zhuge Liang) is the greatest military strategist in ancient China, Three Kingdoms period. FoW chess is like a real war that you can only see where you can move or attach. The goal is to directly capture opponent's king, so no checkmate or walk into checkmate.

### Development Plan
1. Rules. FoW rules extension to chess python package. All rules are the same as regular chess except:
   - a. move representation from one side point of view: e.g. ? (no information), *xe4 (something takes e4 square), *h4 (something occupies h4 square)
   - b. board state representation from each side point of view: empty string (still for known empty square), ? (unknown), * (something occupies this square)
   - c. checkmate: no more checkmate, need to capture king directly
   - d. castle: always allowed, no more restriction

2. Probability model. The most important part of the game is to guess where opponent pieces are based on known information. It's depending on historical move stacks. For unknown opponent's move, we need to assign a probability score on each move based on minmax evaluation score or heuristics.
   - a. as there are so many possible states after a few moves, is there a way to prune or optimize like alpha-beta pruning?

3. Optimization (minimax model with alpha-beta pruning). The basic tree based minimax algorithm works the same for FoW chess.
   - a. heuristics (as terminal evaluation): besides regular piece activity, information is also important factor.
   - b. minimax considers opponent playing their best moves, but we need to model from their visibility. Here comes bluffing: if we throwing a bishop to attack opponent's queen, he probably won't take it if he doesn't know whether it's protected or not./
       However, if under opponent's probability that our queen is somewhere else (for example just took a pawn when only Queen is left on the board), he would take the bishop. This model behavior makes sure how much we should expose our information.
   - c. risk aversion. regular chess minimax is targeting on one objective function but FoW has multiple outcomes (distribution). There should be a function to convert an outcome distribution into one scalar for optimization. This function is basically how aggressive we want to play. 
