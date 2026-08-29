# Experiment specification

This document is the machine-adjacent source of truth for the canonical benchmark.

## Environment

- Board: 10 × 10
- Actions: turn left, continue straight, turn right
- State: 12 engineered features (three danger, four food direction, four heading,
  one normalized length)
- Food reward: +10
- Collision penalty: −10
- Ordinary step penalty: −0.01
- Distance shaping: +0.10 when Manhattan distance decreases, −0.10 otherwise
- Starvation limit: 100 steps without food

The distance signal is heuristic distance-based shaping. It is not the formal
potential-based expression `γΦ(s′) − Φ(s)` and must not be described as such.

## Neural training

- Episodes: 2,000
- Seeds: 1, 2, 3
- Device in recorded artifacts: CPU
- Discount factor: 0.95
- Learning rate: 0.001
- Batch size: 32
- Replay capacity: 20,000 transitions
- Warm-up: 100 transitions
- Target-network update: every 250 environment steps
- Epsilon: 1.0 to 0.05 over 5,000 steps
- Loss: Smooth L1 (Huber)

## Evaluation

- 100 greedy episodes per trained checkpoint
- Shared evaluation seed range begins at 900,000
- Report mean score per checkpoint
- Aggregate checkpoint means across the three training seeds
- Report standard deviation across checkpoint means

Using identical evaluation seeds creates a paired comparison and reduces
environmental noise. The non-learning Random and Greedy agents have no distinct
trained checkpoints, so repeated identical rows are redundant but retained for
the historical report format.

## Artifact contract

Every training run must contain:

- `config.json`
- `metrics.csv`
- `summary.json`
- `checkpoint.pt` or `checkpoint.pkl` for trainable agents

The canonical evidence directory is `runs/full_2000_tuned`. Exploratory runs
must use another directory so they cannot silently overwrite canonical evidence.
