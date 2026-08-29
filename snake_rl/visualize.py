from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from .agents import make_agent
from .config import MODEL_ORDER
from .environment import SnakeEnv
from .evaluate_benchmark import load_config


MODEL_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "q_learning": "Q-learning",
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "lightweight_snake_ddqn": "Lightweight DDQN",
}


def grid_from_env(env: SnakeEnv) -> np.ndarray:
    """Return display cells: 0=empty, 1=body, 2=head, 3=food."""
    grid = np.zeros((env.board_size, env.board_size), dtype=np.uint8)
    for index, (x, y) in enumerate(env.snake):
        grid[y, x] = 2 if index == 0 else 1
    food_x, food_y = env.food
    grid[food_y, food_x] = 3
    return grid


def load_trained_agent(runs_dir: Path, model_name: str, train_seed: int):
    run_dir = runs_dir / model_name / f"seed_{train_seed}"
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No saved run at {run_dir}. Pass --runs with the checkpoint directory."
        )

    config = load_config(config_path)
    env = SnakeEnv(
        board_size=config.board_size,
        max_steps_without_food=config.max_steps_without_food,
        food_reward=config.food_reward,
        death_penalty=config.death_penalty,
        step_penalty=config.step_penalty,
        closer_reward=config.closer_reward,
        farther_penalty=config.farther_penalty,
    )
    observation = env.reset(seed=train_seed)
    agent = make_agent(model_name, config, len(observation), env.action_count)

    if model_name == "q_learning":
        agent.load(str(run_dir / "checkpoint.pkl"))
    elif model_name not in {"random", "greedy"}:
        agent.load(str(run_dir / "checkpoint.pt"))
    return env, agent


def draw_frame(ax, env: SnakeEnv, model_name: str, colors, message: str = ""):
    ax.clear()
    ax.imshow(
        grid_from_env(env), cmap=colors, vmin=0, vmax=3, interpolation="nearest"
    )
    ax.set_xticks(np.arange(-0.5, env.board_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.board_size, 1), minor=True)
    ax.grid(which="minor", color="#334155", linewidth=0.7)
    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    suffix = f"  |  {message}" if message else ""
    ax.set_title(
        f"{MODEL_LABELS[model_name]}  |  score {env.score}  |  step {env.steps}{suffix}",
        color="white" if not message else "#fca5a5",
        fontsize=15,
        pad=12,
    )


def watch(
    runs_dir: Path,
    model_name: str,
    train_seed: int,
    games: int,
    delay: float,
    eval_seed_start: int,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    env, agent = load_trained_agent(runs_dir, model_name, train_seed)
    colors = ListedColormap(["#101827", "#22c55e", "#86efac", "#ef4444"])

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#101827")
    try:
        fig.canvas.manager.set_window_title("Snake RL Thesis Benchmark")
    except AttributeError:
        pass

    game_index = 0
    while games == 0 or game_index < games:
        if not plt.fignum_exists(fig.number):
            break

        observation = env.reset(seed=eval_seed_start + game_index)
        done = False
        while not done and plt.fignum_exists(fig.number):
            draw_frame(ax, env, model_name, colors)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(delay)

            action = agent.act(env, observation, training=False)
            result = env.step(action)
            observation = result.observation
            done = result.done

        if plt.fignum_exists(fig.number):
            draw_frame(ax, env, model_name, colors, "GAME OVER")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(max(0.4, delay * 4))
        game_index += 1

    if plt.fignum_exists(fig.number):
        plt.ioff()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained Snake agent play.")
    parser.add_argument(
        "--runs", type=Path, default=Path("runs/full_2000_tuned")
    )
    parser.add_argument(
        "--model", choices=MODEL_ORDER, default="lightweight_snake_ddqn"
    )
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="Number of games; 0 keeps playing until the window closes.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.08, help="Seconds between frames."
    )
    parser.add_argument("--eval-seed-start", type=int, default=900_000)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        watch(
            args.runs,
            args.model,
            args.train_seed,
            args.games,
            max(0.001, args.delay),
            args.eval_seed_start,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
