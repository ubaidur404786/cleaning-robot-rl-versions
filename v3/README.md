# Cleaning Robot RL (v3)

Version 3 runs the cleaning robot on a 15x15 apartment map and compares:

- Q-Learning
- SARSA
- DQN

## Quick start

```bash
conda activate torch_gpu
pip install -r requirements.txt
python main.py
```

## Main files

```text
v3/
  main.py
  environment.py
  env/phase2_cleaning_env.py
  agent/                  # core agents
  agents/                 # wrapper imports used by app
  ui/launcher.py
  ui/renderer.py
  utils/plotting.py
  models/
  results/
```

## Main plots

In `v3/results/`:

- `comparison_learning_curves.png`
- `comparison_optimal_paths.png`
- `comparison_bars.png`
- `comparison_smart_analysis.png`
- `optimal_path_qlearning.png`
- `optimal_path_sarsa.png`
- `optimal_path_dqn.png`
- `q_learning_training_*.png`
- `sarsa_training_*.png`
- `dqn_training_*.png`

## Latest results we got

From `results/results_20260314_024545.json` (last 100 episodes):

| Metric                    | Q-Learning |  SARSA |         DQN |
| ------------------------- | ---------: | -----: | ----------: |
| Avg reward                |     430.02 | 366.71 | **1229.17** |
| Reward std (lower better) | **145.43** | 250.95 |      210.59 |
| Avg tiles cleaned         |      66.86 |  68.52 |  **104.61** |
| Avg steps                 |     600.00 | 600.00 |      600.00 |
| Avg battery efficiency    |   **0.88** |   0.28 |        0.18 |
| Success rate              |       0.0% |   0.0% |        0.0% |
| Best reward seen          |     721.40 | 882.20 | **1492.80** |

Quick note: in this saved run, DQN gives the highest reward and cleaned tiles,
but full-task completion is still 0% for all three (episodes hit the step cap).

## Outputs

- JSON: `results/results_*.json`
- Models: `models/q_learning_phase2.pkl`, `models/sarsa_phase2.pkl`, `models/dqn_phase2.pth`

## Requirements

`gymnasium`, `numpy`, `matplotlib`, `pygame`, `torch`
