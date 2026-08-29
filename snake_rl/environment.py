from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import random
from typing import Iterable

import numpy as np


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DIR_VECTORS = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class SnakeEnv:
    """Deterministic Snake with a compact feature vector for simple RL agents.

    Actions are relative to the current direction:
    0 = turn left, 1 = go straight, 2 = turn right.
    """

    action_count = 3

    def __init__(
        self,
        board_size: int = 10,
        max_steps_without_food: int = 100,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        step_penalty: float = -0.01,
        closer_reward: float = 0.10,
        farther_penalty: float = -0.10,
    ):
        self.board_size = board_size
        self.max_steps_without_food = max_steps_without_food
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.step_penalty = step_penalty
        self.closer_reward = closer_reward
        self.farther_penalty = farther_penalty
        self.rng = random.Random()
        self.snake: list[tuple[int, int]] = []
        self.direction = Direction.RIGHT
        self.food = (0, 0)
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        c = self.board_size // 2
        self.snake = [(c, c), (c - 1, c), (c - 2, c)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.food = self._spawn_food()
        return self.observation()

    def step(self, action: int) -> StepResult:
        old_distance = self._distance_to_food(self.snake[0])
        self.direction = self._turn(self.direction, action)
        dx, dy = DIR_VECTORS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        self.steps += 1
        self.steps_without_food += 1

        if self._hits_wall(new_head) or self._hits_self(new_head):
            return StepResult(self.observation(), self.death_penalty, True, self.info("dead"))

        self.snake.insert(0, new_head)
        ate_food = new_head == self.food
        if ate_food:
            self.score += 1
            self.steps_without_food = 0
            self.food = self._spawn_food()
            reward = self.food_reward
        else:
            self.snake.pop()
            new_distance = self._distance_to_food(new_head)
            reward = self.step_penalty
            reward += self.closer_reward if new_distance < old_distance else self.farther_penalty

        done = self.steps_without_food >= self.max_steps_without_food
        if done:
            reward -= 2.0
        return StepResult(self.observation(), reward, done, self.info("ok"))

    def observation(self) -> np.ndarray:
        head_x, head_y = self.snake[0]
        left_dir = self._turn(self.direction, 0)
        straight_dir = self.direction
        right_dir = self._turn(self.direction, 2)

        features = [
            self._danger(left_dir),
            self._danger(straight_dir),
            self._danger(right_dir),
            float(self.food[0] < head_x),
            float(self.food[0] > head_x),
            float(self.food[1] < head_y),
            float(self.food[1] > head_y),
            float(self.direction == Direction.UP),
            float(self.direction == Direction.RIGHT),
            float(self.direction == Direction.DOWN),
            float(self.direction == Direction.LEFT),
            len(self.snake) / float(self.board_size * self.board_size),
        ]
        return np.array(features, dtype=np.float32)

    def state_key(self) -> tuple[int, ...]:
        """Discretized state for tabular Q-learning."""
        return tuple(int(x > 0.5) for x in self.observation()[:-1]) + (len(self.snake) // 4,)

    def valid_actions(self) -> Iterable[int]:
        return range(self.action_count)

    def info(self, status: str) -> dict:
        return {
            "score": self.score,
            "steps": self.steps,
            "status": status,
            "snake": list(self.snake),
            "food": self.food,
            "direction": int(self.direction),
        }

    def render_ascii(self) -> str:
        cells = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]
        fx, fy = self.food
        cells[fy][fx] = "F"
        for i, (x, y) in enumerate(self.snake):
            cells[y][x] = "H" if i == 0 else "S"
        return "\n".join(" ".join(row) for row in cells)

    def _spawn_food(self) -> tuple[int, int]:
        free = [
            (x, y)
            for y in range(self.board_size)
            for x in range(self.board_size)
            if (x, y) not in self.snake
        ]
        return self.rng.choice(free)

    def _distance_to_food(self, pos: tuple[int, int]) -> int:
        return abs(pos[0] - self.food[0]) + abs(pos[1] - self.food[1])

    def _danger(self, direction: Direction) -> float:
        dx, dy = DIR_VECTORS[direction]
        head_x, head_y = self.snake[0]
        pos = (head_x + dx, head_y + dy)
        return float(self._hits_wall(pos) or self._hits_self(pos))

    def _hits_wall(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return x < 0 or y < 0 or x >= self.board_size or y >= self.board_size

    def _hits_self(self, pos: tuple[int, int]) -> bool:
        return pos in self.snake[:-1]

    @staticmethod
    def _turn(direction: Direction, action: int) -> Direction:
        if action == 0:
            return Direction((int(direction) - 1) % 4)
        if action == 2:
            return Direction((int(direction) + 1) % 4)
        return direction
