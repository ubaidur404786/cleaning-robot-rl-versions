# Cleaning Robot RL (v2)

This version compares three agents on the same cleaning task:

- Q-Learning
- SARSA
- DQN

The robot moves in an 8x6 house and must clean 23 tiles.

## Quick setup

Use Python 3.8+.

Install deps:

```bash
pip install -r requirements.txt
```

## Run

Main menu:

```bash
python main.py
```

Standalone scripts:

```bash
python train.py
python test.py
```

## Environment

- Grid: 8x6
- Cleanable tiles: 23
- Actions: Forward, Backward, Left, Right, Wait, Clean
- State size (tabular): 2300

Room rewards:

- Kitchen: +50
- Living Room: +35
- Hallway: +20
- Finish all tiles: +200

Penalties:

- Clean tile revisit: -5
- Clean action on clean tile: -10
- Hit wall: -5
- Wait: -3
- Step cost: -0.1

## Files

```text
v2/
   main.py
   train.py
   test.py
   agent/
      q_learning_agent.py
      sarsa_agent.py
      dqn_agent.py
   env/
      cleaning_env.py
   utils/
      helpers.py
      plotting.py
   models/
   plots/
```

## Notes

- `main.py` can train/test each agent and generate comparison plots.
- Models are saved in `models/`.
- Figures are saved in `plots/`.

## Main plots

In `v2/plots/`:

- `comparison_learning_curves.png`
- `comparison_optimal_paths.png`
- `comparison_bars.png`
- `comparison_smart_analysis.png`
- `optimal_path_qlearning.png`
- `optimal_path_sarsa.png`
- `optimal_path_dqn.png`

## Latest results (from saved plots)

Last 100 episodes summary:

| Metric                    | Q-Learning |      SARSA |    DQN |
| ------------------------- | ---------: | ---------: | -----: |
| Avg reward                |      984.2 | **1014.7** | 1011.2 |
| Avg tiles cleaned         |       23.0 |       23.0 |   23.0 |
| Success rate              |     100.0% |     100.0% | 100.0% |
| Avg steps                 |       29.0 |   **24.0** |   24.0 |
| Reward std (lower better) |       49.8 |    **6.3** |    9.3 |
| Training time (s)         |       10.1 |    **7.0** |  468.9 |

Path quality:

- Q-Learning: 23/23 tiles in 27 steps
- SARSA: 23/23 tiles in 23 steps
- DQN: 23/23 tiles in 23 steps

Short takeaway: all 3 solve this map, SARSA is the most stable and fastest in this setup.
