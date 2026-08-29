# Canonical recorded results

These values are generated from `runs/full_2000_tuned` and
`reports/full_2000_tuned/evaluation_summary.csv`. They supersede contradictory
values in older exports.

## Training

Final score is the mean score over the last 50 training episodes, aggregated
across three training seeds. Standard deviation is calculated across the three
seed-level means.

| Agent | Final score | Best observed | Mean time (s) | Model size |
|---|---:|---:|---:|---:|
| Random | 0.22 +/- 0.04 | 3 | 0.33 | 0 |
| Greedy | 20.03 +/- 0.40 | 45 | 3.77 | 0 |
| Q-learning | 9.70 +/- 0.76 | 22 | 2.44 | 3,894 stored table entries (maximum) |
| DQN | 11.67 +/- 0.59 | 39 | 66.33 | 18,563 parameters |
| Double DQN | 11.79 +/- 0.89 | 34 | 65.34 | 18,563 parameters |
| Dueling DQN | 12.10 +/- 0.39 | 42 | 94.98 | 35,204 parameters |
| Lightweight DDQN | 11.19 +/- 0.57 | 36 | 82.22 | 9,412 parameters |

## Post-training evaluation

Each checkpoint is evaluated greedily on the same 100 environment seeds. The
mean and standard deviation below are calculated across three checkpoint-level
means, not across all individual episodes.

| Agent | Mean score | SD across training seeds | Best observed |
|---|---:|---:|---:|
| Random | 0.13 | 0.00 | 2 |
| Greedy | 19.10 | 0.00 | 33 |
| Q-learning | 16.18 | 0.07 | 23 |
| DQN | 17.69 | 0.65 | 38 |
| Double DQN | 18.13 | 0.48 | 37 |
| Dueling DQN | 19.76 | 0.29 | 41 |
| Lightweight DDQN | 18.85 | 0.93 | 35 |

## Defensible conclusion

- Dueling DQN is the strongest learned model by mean evaluation score.
- Double DQN has the shortest recorded neural training time.
- Lightweight DDQN is the smallest neural model, not the fastest.
- Lightweight DDQN uses approximately 73.3% fewer parameters than Dueling DQN
  and retains approximately 95.4% of its evaluation score.
- Greedy is a hand-written reference and should not be presented as a fair
  learned-model comparison.
- Wall-clock and latency measurements are hardware-specific.
