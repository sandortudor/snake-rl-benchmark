# Troubleshooting

## `No module named snake_rl`

Activate the repository virtual environment and install the package:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## `PyTorch is required for neural agents`

Install the project dependencies inside the active environment:

```bash
python -m pip install -e .
```

Verify the installation with:

```bash
python -c "import torch; print(torch.__version__)"
```

## A checkpoint cannot be found

Run commands from the repository root. The default visualizer expects
`runs/full_2000_tuned`. For another experiment, pass its directory explicitly:

```bash
python -m snake_rl.visualize --runs runs/my_experiment --model dqn --train-seed 1
```

Trainable agents require `checkpoint.pt` or `checkpoint.pkl` inside the selected
model and seed directory.

## The visual window does not open

The viewer needs a graphical desktop session and a Matplotlib GUI backend. It
will not display over a headless SSH session. Run it from Terminal on the local
presentation laptop, not from a CI runner.

## Training is slow

Neural training is CPU-intensive and scales with the number of models, seeds,
and episodes. Start with a smoke command and a fresh output directory:

```bash
python -m snake_rl.train --models q_learning dqn --episodes 20 --seeds 1 --out runs/debug
```

The canonical checkpoint files are already included; retraining is not required
to run the visual demo.

## Generated results do not match the canonical benchmark exactly

Check the Python, PyTorch, device, seeds, board size, episode count, and reward
configuration. The canonical settings are in `docs/EXPERIMENT_SPEC.md` and the
exact configuration of each run is stored in its `config.json`.

Wall-clock time and inference latency are hardware-specific. Scores can also
vary if a different software version changes numerical behavior.

## Avoid overwriting the canonical evidence

Never use `runs/full_2000_tuned` as the `--out` directory for a test. Use a new
name such as `runs/smoke`, `runs/debug`, or `runs/reproduction_2000`.

The `.gitignore` excludes these exploratory directories while preserving the
canonical evidence.
