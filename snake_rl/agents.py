from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import pickle
import random
from typing import Deque

import numpy as np

from .environment import DIR_VECTORS, SnakeEnv


class RandomAgent:
    name = "random"
    trainable = False

    def act(self, env: SnakeEnv, observation: np.ndarray, training: bool = False) -> int:
        return env.rng.randrange(env.action_count)


class GreedyAgent:
    name = "greedy"
    trainable = False

    def act(self, env: SnakeEnv, observation: np.ndarray, training: bool = False) -> int:
        best_action = 1
        best_distance = 10**9
        for action in env.valid_actions():
            direction = env._turn(env.direction, action)
            dx, dy = DIR_VECTORS[direction]
            head_x, head_y = env.snake[0]
            pos = (head_x + dx, head_y + dy)
            if env._hits_wall(pos) or env._hits_self(pos):
                continue
            distance = abs(pos[0] - env.food[0]) + abs(pos[1] - env.food[1])
            if distance < best_distance:
                best_action = action
                best_distance = distance
        return best_action


class TabularQAgent:
    name = "q_learning"
    trainable = True

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 1.0, decay: float = 0.995):
        self.q = defaultdict(lambda: np.zeros(3, dtype=np.float32))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay

    def act(self, env: SnakeEnv, observation: np.ndarray, training: bool = False) -> int:
        if training and env.rng.random() < self.epsilon:
            return env.rng.randrange(env.action_count)
        return int(np.argmax(self.q[env.state_key()]))

    def learn(self, state, action: int, reward: float, next_state, done: bool):
        future = 0.0 if done else float(np.max(self.q[next_state]))
        target = reward + self.gamma * future
        self.q[state][action] += self.alpha * (target - self.q[state][action])

    def end_episode(self):
        self.epsilon = max(0.05, self.epsilon * self.decay)

    def model_size(self) -> int:
        return len(self.q) * 3

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q = defaultdict(lambda: np.zeros(3, dtype=np.float32), data)
        self.epsilon = 0.0


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.items: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.items, batch_size)

    def __len__(self) -> int:
        return len(self.items)


class DQNAgent:
    trainable = True

    def __init__(
        self,
        name: str,
        model_kind: str,
        input_size: int,
        action_count: int,
        gamma: float,
        learning_rate: float,
        batch_size: int,
        replay_capacity: int,
        warmup_steps: int,
        target_update: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        double: bool = False,
    ):
        from .models import build_model, choose_device
        import torch
        from torch import nn

        self.name = name
        self.device = choose_device()
        self.action_count = action_count
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.double = double
        self.steps = 0
        self.loss = 0.0

        self.policy = build_model(model_kind, input_size, action_count).to(self.device)
        self.target = build_model(model_kind, input_size, action_count).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(replay_capacity)

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.steps / float(self.epsilon_decay_steps))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def act(self, env: SnakeEnv, observation: np.ndarray, training: bool = False) -> int:
        import torch

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_count)
        with torch.no_grad():
            x = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy(x).argmax(dim=1).item())

    def learn(self, transition: Transition):
        import torch

        self.steps += 1
        self.replay.push(transition)
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return

        batch = self.replay.sample(self.batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.policy(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            if self.double:
                next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = float(loss.item())

        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def end_episode(self):
        pass

    def model_size(self) -> int:
        from .models import parameter_count

        return parameter_count(self.policy)

    def save(self, path: str):
        import torch

        torch.save({"name": self.name, "state_dict": self.policy.state_dict()}, path)

    def load(self, path: str):
        import torch

        payload = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(payload["state_dict"])
        self.target.load_state_dict(payload["state_dict"])


def make_agent(model_name: str, config, observation_size: int, action_count: int):
    if model_name == "random":
        return RandomAgent()
    if model_name == "greedy":
        return GreedyAgent()
    if model_name == "q_learning":
        return TabularQAgent(
            alpha=config.q_learning_alpha,
            gamma=config.gamma,
            decay=config.q_learning_epsilon_decay,
        )
    if model_name == "dqn":
        return DQNAgent(model_name, "dqn", observation_size, action_count, config.gamma, config.learning_rate, config.batch_size, config.replay_capacity, config.warmup_steps, config.target_update, config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)
    if model_name == "double_dqn":
        return DQNAgent(model_name, "dqn", observation_size, action_count, config.gamma, config.learning_rate, config.batch_size, config.replay_capacity, config.warmup_steps, config.target_update, config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps, double=True)
    if model_name == "dueling_dqn":
        return DQNAgent(model_name, "dueling_dqn", observation_size, action_count, config.gamma, config.learning_rate, config.batch_size, config.replay_capacity, config.warmup_steps, config.target_update, config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)
    if model_name == "lightweight_snake_ddqn":
        return DQNAgent(model_name, "lightweight_snake_ddqn", observation_size, action_count, config.gamma, config.learning_rate, config.batch_size, config.replay_capacity, config.warmup_steps, config.target_update, config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps, double=True)
    raise ValueError(f"Unknown model: {model_name}")
