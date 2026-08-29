# Defense demo guide

This runbook keeps the live demonstration short, deterministic, and independent
of training time.

## Before the presentation

1. Open a terminal in the repository.
2. Activate the virtual environment: `source .venv/bin/activate`.
3. Run `make test`.
4. Confirm the compact checkpoint exists at
   `runs/full_2000_tuned/lightweight_snake_ddqn/seed_1/checkpoint.pt`.
5. Run the exact demo command once on the presentation laptop.
6. Keep a screen recording of the demo available as an offline backup.

## Recommended live demonstration

```bash
python -m snake_rl.visualize \
  --model lightweight_snake_ddqn \
  --train-seed 1 \
  --games 3 \
  --delay 0.04
```

Explain these points while it runs:

- The model sees 12 engineered features, not the complete image.
- Its three actions are relative: left, straight, and right.
- Exploration is disabled during the demonstration.
- The loaded seed-1 checkpoint is one of three independently trained models.
- The compact network has 9,412 parameters.

## Optional comparison

Run the highest-scoring learned model with the same evaluation seeds:

```bash
python -m snake_rl.visualize \
  --model dueling_dqn \
  --train-seed 1 \
  --games 3 \
  --delay 0.04
```

The comparison illustrates the benchmark result: Dueling DQN has the highest
mean learned score, while Lightweight DDQN retains most of that score with a
substantially smaller checkpoint.

## Short training demonstration

Do not retrain the full benchmark during the defense. If the committee asks to
see training, use a separate output directory:

```bash
python -m snake_rl.train \
  --models q_learning \
  --episodes 50 \
  --seeds 7 \
  --out runs/defense_demo
```

State clearly that 50 episodes demonstrate the pipeline only; they are not a
scientific reproduction of the 2,000-episode experiment.

## Files to know

| Question | File |
|---|---|
| Environment and rewards | `snake_rl/environment.py` |
| Agent behavior and learning | `snake_rl/agents.py` |
| Neural architectures | `snake_rl/models.py` |
| Training loop | `snake_rl/train.py` |
| Fixed-policy evaluation | `snake_rl/evaluate_benchmark.py` |
| Visual player | `snake_rl/visualize.py` |
| Exact protocol | `docs/EXPERIMENT_SPEC.md` |
| Canonical results | `docs/RESULTS.md` |

## Backup plan

If the graphical window fails, show the backup recording and run `make test` in
the terminal. The tests demonstrate that the environment, model shapes,
documented parameter counts, and visualization state conversion are functional.
