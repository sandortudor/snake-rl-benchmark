from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .agents import make_agent
from .config import ExperimentConfig, MODEL_ORDER
from .environment import SnakeEnv
from .train import run_episode


def load_config(path: Path) -> ExperimentConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["seeds"] = tuple(data["seeds"])
    return ExperimentConfig(**data)


def evaluate_run(run_dir: Path, model_name: str, episodes: int, eval_seed_start: int) -> dict:
    config = load_config(run_dir / "config.json")
    env = SnakeEnv(
        board_size=config.board_size,
        max_steps_without_food=config.max_steps_without_food,
        food_reward=config.food_reward,
        death_penalty=config.death_penalty,
        step_penalty=config.step_penalty,
        closer_reward=config.closer_reward,
        farther_penalty=config.farther_penalty,
    )
    observation_size = len(env.reset(seed=eval_seed_start))
    agent = make_agent(model_name, config, observation_size, env.action_count)

    if model_name == "q_learning":
        agent.load(str(run_dir / "checkpoint.pkl"))
    elif hasattr(agent, "load") and (run_dir / "checkpoint.pt").exists():
        agent.load(str(run_dir / "checkpoint.pt"))

    scores = []
    for episode in range(episodes):
        result = run_episode(env, agent, seed=eval_seed_start + episode, training=False)
        scores.append(result["score"])
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return {
        "model": model_name,
        "train_seed": int(run_dir.name.split("_")[1]),
        "eval_episodes": episodes,
        "eval_mean_score": round(mean, 4),
        "eval_std_score": round(variance**0.5, 4),
        "eval_best_score": max(scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Snake agents without exploration.")
    parser.add_argument("--runs", type=Path, default=Path("runs/full_2000_tuned"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=900_000)
    parser.add_argument("--out", type=Path, default=Path("reports/full_2000_tuned/evaluation_summary.csv"))
    args = parser.parse_args()

    rows = []
    for model_name in MODEL_ORDER:
        for run_dir in sorted((args.runs / model_name).glob("seed_*")):
            rows.append(evaluate_run(run_dir, model_name, args.episodes, args.seed_start))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
