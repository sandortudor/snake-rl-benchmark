from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import random
import time

import numpy as np

from .agents import Transition, make_agent
from .config import ExperimentConfig, MODEL_ORDER
from .environment import SnakeEnv
from .models import torch_available


def run_episode(env: SnakeEnv, agent, seed: int, training: bool) -> dict:
    observation = env.reset(seed=seed)
    total_reward = 0.0
    done = False

    while not done:
        state_key = env.state_key()
        action = agent.act(env, observation, training=training)
        result = env.step(action)
        total_reward += result.reward

        if training and getattr(agent, "trainable", False):
            if agent.name == "q_learning":
                agent.learn(state_key, action, result.reward, env.state_key(), result.done)
            else:
                agent.learn(Transition(observation, action, result.reward, result.observation, result.done))

        observation = result.observation
        done = result.done

    if training and hasattr(agent, "end_episode"):
        agent.end_episode()

    return {
        "score": env.score,
        "steps": env.steps,
        "reward": round(total_reward, 4),
        "epsilon": round(float(getattr(agent, "epsilon", 0.0)), 5),
        "loss": round(float(getattr(agent, "loss", 0.0)), 6),
    }


def train_model(model_name: str, config: ExperimentConfig, seed: int, out_dir: Path) -> dict:
    if model_name in {"dqn", "double_dqn", "dueling_dqn", "lightweight_snake_ddqn"} and not torch_available():
        raise RuntimeError("PyTorch is required for neural agents. Install requirements.txt first.")

    set_global_seed(seed)
    env = SnakeEnv(
        board_size=config.board_size,
        max_steps_without_food=config.max_steps_without_food,
        food_reward=config.food_reward,
        death_penalty=config.death_penalty,
        step_penalty=config.step_penalty,
        closer_reward=config.closer_reward,
        farther_penalty=config.farther_penalty,
    )
    observation_size = len(env.reset(seed=seed))
    agent = make_agent(model_name, config, observation_size, env.action_count)

    run_dir = out_dir / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    start = time.perf_counter()
    rows = []

    for episode in range(1, config.train_episodes + 1):
        metrics = run_episode(env, agent, seed=seed * 100_000 + episode, training=True)
        elapsed = time.perf_counter() - start
        row = {
            "episode": episode,
            "model": model_name,
            "seed": seed,
            "elapsed_seconds": round(elapsed, 4),
            "moving_avg_score": 0.0,
            "model_size": agent.model_size() if hasattr(agent, "model_size") else 0,
            "device": getattr(agent, "device", "cpu"),
            **metrics,
        }
        recent = [r["score"] for r in rows[-49:]] + [row["score"]]
        row["moving_avg_score"] = round(sum(recent) / len(recent), 4)
        rows.append(row)

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows, config.target_score)
    summary.update(
        {
            "model": model_name,
            "seed": seed,
            "device": getattr(agent, "device", "cpu"),
            "model_size": agent.model_size() if hasattr(agent, "model_size") else 0,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        }
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    if hasattr(agent, "save"):
        suffix = ".pkl" if model_name == "q_learning" else ".pt"
        agent.save(str(run_dir / f"checkpoint{suffix}"))
    return summary


def summarize_rows(rows: list[dict], target_score: float) -> dict:
    final_window = rows[-50:] if len(rows) >= 50 else rows
    best_score = max(r["score"] for r in rows)
    final_avg = sum(r["score"] for r in final_window) / len(final_window)
    target_time = None
    target_episode = None
    for row in rows:
        if row["moving_avg_score"] >= target_score:
            target_time = row["elapsed_seconds"]
            target_episode = row["episode"]
            break
    return {
        "final_avg_score": round(final_avg, 4),
        "best_score": best_score,
        "total_train_seconds": rows[-1]["elapsed_seconds"],
        "target_score": target_score,
        "target_episode": target_episode,
        "target_seconds": target_time,
    }


def run_benchmark(models: list[str], config: ExperimentConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model_name in models:
        for seed in config.seeds:
            print(f"training {model_name} seed={seed}", flush=True)
            summaries.append(train_model(model_name, config, seed, out_dir))

    summary_path = out_dir / "benchmark_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {summary_path}", flush=True)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch_available():
        import torch

        torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare Snake RL agents.")
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--board-size", type=int, default=10)
    parser.add_argument("--max-steps-without-food", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--epsilon-decay-steps", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        board_size=args.board_size,
        max_steps_without_food=args.max_steps_without_food,
        train_episodes=args.episodes,
        seeds=tuple(args.seeds),
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        epsilon_decay_steps=args.epsilon_decay_steps,
        learning_rate=args.learning_rate,
    )
    try:
        run_benchmark(args.models, config, args.out)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
