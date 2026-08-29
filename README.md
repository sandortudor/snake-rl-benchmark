# Snake RL Benchmark

A complete, reproducible reinforcement-learning benchmark for the game of
Snake. The project compares seven agents under the same environment and
training budget, preserves the canonical trained checkpoints, and includes
evaluation, reporting, deployment measurements, tests, and a graphical player.

## Quick start

Requirements: Python 3.10 or newer. The recorded benchmark used
Python 3.12, PyTorch 2.11, and CPU execution on an Apple M3 Pro.

```bash
cd /path/to/snake_rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m snake_rl.visualize --model lightweight_snake_ddqn --train-seed 1 --games 3
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. The final command opens a window and plays three
games using the included trained checkpoint; it does not retrain the agent.

If `make` is available, the common commands are:

```bash
make help
make install
make test
make demo
```

## What is included

| Identifier | Agent | Learning | Saved checkpoint |
|---|---|---:|---:|
| `random` | Uniform random baseline | No | No |
| `greedy` | Hand-written food-seeking baseline | No | No |
| `q_learning` | Tabular Q-learning | Yes | Yes |
| `dqn` | Deep Q-Network | Yes | Yes |
| `double_dqn` | Double DQN | Yes | Yes |
| `dueling_dqn` | Dueling DQN | Yes | Yes |
| `lightweight_snake_ddqn` | Compact Dueling Double DQN | Yes | Yes |

Random and Greedy are reference controllers. Greedy embeds task-specific
knowledge and is not a directly comparable learned policy.

## Watch a trained agent

The repository includes all checkpoints from the canonical three-seed run.

```bash
# Compact model
python -m snake_rl.visualize \
  --model lightweight_snake_ddqn \
  --train-seed 1 \
  --games 3

# Highest-scoring learned model
python -m snake_rl.visualize \
  --model dueling_dqn \
  --train-seed 2 \
  --games 3 \
  --delay 0.04
```

Use `--games 0` to continue until the window is closed. The visual player
requires a desktop session; it will not open on a headless server.

## Run a safe smoke experiment

Use a new output directory for experiments. The following command is only a
functional check and does not reproduce the canonical results.

```bash
python -m snake_rl.train \
  --models random greedy q_learning \
  --episodes 20 \
  --seeds 1 \
  --out runs/smoke

python -m snake_rl.plots \
  --runs runs/smoke \
  --reports reports/smoke
```

Do not use `runs/full_2000_tuned` as the output of an exploratory run. It is the
preserved evidence directory for the benchmark.

## Reproduce the complete benchmark

The full run trains seven agents for 2,000 episodes over seeds 1, 2, and 3. On
the recorded CPU it took approximately 16 minutes in total, but duration varies
by hardware and software version.

```bash
python -m snake_rl.train \
  --episodes 2000 \
  --seeds 1 2 3 \
  --out runs/reproduction_2000

python -m snake_rl.evaluate_benchmark \
  --runs runs/reproduction_2000 \
  --episodes 100 \
  --out reports/reproduction_2000/evaluation_summary.csv

python -m snake_rl.plots \
  --runs runs/reproduction_2000 \
  --reports reports/reproduction_2000
```

Using a new directory keeps the canonical evidence unchanged and makes
the reproduction directly comparable.

## Command reference

After installation, either module commands or console commands may be used.

| Task | Module command | Installed command |
|---|---|---|
| Train agents | `python -m snake_rl.train` | `snake-train` |
| Evaluate checkpoints | `python -m snake_rl.evaluate_benchmark` | `snake-evaluate` |
| Generate reports | `python -m snake_rl.plots` | `snake-report` |
| Watch an agent | `python -m snake_rl.visualize` | `snake-watch` |
| Measure resources | `python -m snake_rl.benchmark_resources` | `snake-resources` |

Every command supports `--help`, for example:

```bash
python -m snake_rl.train --help
```

## Canonical benchmark result

Post-training evaluation uses 100 greedy episodes per trained checkpoint and
aggregates the three checkpoint-level means.

| Learned agent | Evaluation score | Parameters |
|---|---:|---:|
| Q-learning | 16.18 +/- 0.07 | 3,894 stored table entries (maximum) |
| DQN | 17.69 +/- 0.65 | 18,563 |
| Double DQN | 18.13 +/- 0.48 | 18,563 |
| Dueling DQN | **19.76 +/- 0.29** | 35,204 |
| Lightweight DDQN | 18.85 +/- 0.93 | **9,412** |

Lightweight DDQN uses 73.3% fewer parameters than Dueling DQN while retaining
95.4% of its mean evaluation score. It is the smallest neural model, not the
fastest to train. See [docs/RESULTS.md](docs/RESULTS.md) for the complete evidence
and [docs/EXPERIMENT_SPEC.md](docs/EXPERIMENT_SPEC.md) for the protocol.

## Outputs

Each trained run contains:

```text
config.json       exact experiment configuration
metrics.csv       one row per training episode
summary.json      final score, best score, time, device, and model size
checkpoint.pt     neural checkpoint, when applicable
checkpoint.pkl    Q-learning table, when applicable
```

Aggregated CSV and SVG reports are written under `reports/`. Deployment
measurement additionally reports checkpoint bytes and median/p95 inference
latency. Latency and wall-clock results are hardware-specific.

## Repository layout

```text
snake_rl/       Environment, agents, models, training, evaluation, and viewer
tests/          Environment, agent, architecture, and production tests
runs/           Canonical raw metrics and trained checkpoints
reports/        Aggregated result and deployment reports
docs/           Protocol, results, deployment notes, demo guide, troubleshooting
```

## Reproducibility and scope

- Training seeds are 1, 2, and 3.
- Evaluation uses the shared seed range beginning at 900,000.
- The environment, rewards, 12-feature state, and hyperparameters are recorded
  in every run and summarized in `docs/EXPERIMENT_SPEC.md`.
- The benchmark studies local-compute trade-offs. It does not claim
  state-of-the-art Snake performance.
- Three seeds provide descriptive replication, not strong statistical evidence.

## Help

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common installation,
checkpoint, and display issues. For a demonstration runbook, see
[docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md).

## License

The software is released under the MIT License.
