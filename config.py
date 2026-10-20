from dataclasses import dataclass, field


@dataclass
class StockfishConfig:
    depth: int = 6


@dataclass
class BeliefStateConfig:
    # simulations kept per level of nested belief states; a belief state is a leaf once they run out
    sizes: tuple[int, ...] = (20, 3)
    num_of_top_moves: int = 10 # how many top moves the opponent considers on each simulated board
    opponent_strength: float = 100.0 # see convert_score_to_probability: every 5000 / strength centipawns doubles the likelihood

    # config for opponent's move decision assumption
    num_of_moves_per_sim: int = 5 # how many top moves to consider per his piece simulation
    softmax_prob_per_100_cp: float = 3 # likelihood of top move vs move 100 centipawn score below it.


@dataclass
class ModelConfig:
    StockfishConfig: StockfishConfig = field(default_factory=StockfishConfig)
    BeliefStateConfig: BeliefStateConfig = field(default_factory=BeliefStateConfig)