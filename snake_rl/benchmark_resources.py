from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path
import statistics
import time

from .agents import make_agent
from .config import MODEL_ORDER
from .environment import SnakeEnv
from .evaluate_benchmark import load_config


def _synchronize(agent) -> None:
    device = getattr(agent, "device", "cpu")
    if device == "cuda":
        import torch

        torch.cuda.synchronize()
    elif device == "mps":
        import torch

        torch.mps.synchronize()


def _checkpoint_path(run_dir: Path, model_name: str) -> Path | None:
    if model_name == "q_learning":
        return run_dir / "checkpoint.pkl"
    if model_name in {"random", "greedy"}:
        return None
    return run_dir / "checkpoint.pt"


def benchmark_model(
    runs_dir: Path,
    model_name: str,
    train_seed: int,
    warmup: int,
    iterations: int,
) -> dict:
    run_dir = runs_dir / model_name / f"seed_{train_seed}"
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
    observation = env.reset(seed=900_000)
    agent = make_agent(model_name, config, len(observation), env.action_count)
    checkpoint = _checkpoint_path(run_dir, model_name)
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        agent.load(str(checkpoint))

    for _ in range(warmup):
        agent.act(env, observation, training=False)
    _synchronize(agent)

    samples_us = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        agent.act(env, observation, training=False)
        _synchronize(agent)
        samples_us.append((time.perf_counter_ns() - start) / 1_000.0)

    ordered = sorted(samples_us)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "model": model_name,
        "train_seed": train_seed,
        "device": getattr(agent, "device", "cpu"),
        "model_size": agent.model_size() if hasattr(agent, "model_size") else 0,
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint is not None else 0,
        "latency_median_us": round(statistics.median(samples_us), 3),
        "latency_p95_us": round(ordered[p95_index], 3),
        "iterations": iterations,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure checkpoint size and action-selection latency."
    )
    parser.add_argument("--runs", type=Path, default=Path("runs/full_2000_tuned"))
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/full_2000_tuned/deployment_metrics.csv"),
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("warmup must be non-negative and iterations must be positive")

    rows = [
        benchmark_model(
            args.runs, model, args.train_seed, args.warmup, args.iterations
        )
        for model in args.models
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
