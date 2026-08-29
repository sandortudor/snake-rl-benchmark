from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import GreedyAgent, RandomAgent, TabularQAgent
from .environment import SnakeEnv
from .train import run_episode


def evaluate_simple(agent_name: str, episodes: int, seed: int, board_size: int) -> dict:
    env = SnakeEnv(board_size=board_size)
    agents = {
        "random": RandomAgent(),
        "greedy": GreedyAgent(),
        "q_learning_untrained": TabularQAgent(epsilon=0.0),
    }
    agent = agents[agent_name]
    scores = []
    for episode in range(episodes):
        result = run_episode(env, agent, seed=seed + episode, training=False)
        scores.append(result["score"])
    return {
        "agent": agent_name,
        "episodes": episodes,
        "avg_score": round(sum(scores) / len(scores), 4),
        "best_score": max(scores),
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate simple Snake agents.")
    parser.add_argument("--agent", choices=["random", "greedy", "q_learning_untrained"], default="greedy")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--board-size", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args()

    result = evaluate_simple(args.agent, args.episodes, args.seed, args.board_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

