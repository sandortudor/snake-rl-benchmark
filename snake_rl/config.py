from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    board_size: int = 10
    max_steps_without_food: int = 100
    train_episodes: int = 300
    eval_episodes: int = 30
    seeds: tuple[int, ...] = (1, 2, 3)
    target_score: float = 10.0

    gamma: float = 0.95
    learning_rate: float = 0.001
    batch_size: int = 32
    replay_capacity: int = 20_000
    warmup_steps: int = 100
    target_update: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000

    food_reward: float = 10.0
    death_penalty: float = -10.0
    step_penalty: float = -0.01
    closer_reward: float = 0.10
    farther_penalty: float = -0.10

    q_learning_alpha: float = 0.10
    q_learning_epsilon_decay: float = 0.995

    def to_dict(self) -> dict:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        return data


MODEL_ORDER = [
    "random",
    "greedy",
    "q_learning",
    "dqn",
    "double_dqn",
    "dueling_dqn",
    "lightweight_snake_ddqn",
]
