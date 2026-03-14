# Cleaning Robot RL — Phase 2

A reinforcement learning project where a robot vacuum learns to clean a
15×15 grid-world apartment. Three algorithms are implemented and compared:
**Q-Learning**, **SARSA**, and **DQN** (Deep Q-Network).

---

## Quick Start

```bash
# 1. Activate the conda environment
conda activate torch_gpu

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Run the program
python main.py
```

A settings popup appears. Pick your options and click **Start**.

---

## What the Popup Does

| Option   | Values                     | Notes                         |
| -------- | -------------------------- | ----------------------------- |
| Mode     | Train / Test / Compare All | Compare trains all 3 agents   |
| Agent    | q_learning / sarsa / dqn   | Ignored in Compare mode       |
| Render   | ON / OFF                   | ON opens a live pygame window |
| Episodes | 50 – 5000                  | Spinner, per agent            |

---

## File Structure

```
main.py                   entry point — run this file
environment.py            environment entrypoint (re-exports Phase2CleaningEnv)
agents/                   requested agent package layout
  q_learning_agent.py     wrapper to tabular Q-Learning class
  sarsa_agent.py          wrapper to tabular SARSA class
  dqn_agent.py            wrapper to DQN class
ui/
  launcher.py             tkinter settings popup
  renderer.py             pygame live renderer
env/
  phase2_cleaning_env.py  full Phase-2 15x15 environment implementation
agent/
  q_learning_agent.py     core tabular Q-Learning implementation
  sarsa_agent.py          core tabular SARSA implementation
  dqn_agent.py            core CNN DQN implementation
utils/
  plotting.py             matplotlib chart saving + comparison plots
  __init__.py             plotting exports
models/                   saved model files (.pkl / .pth)
results/                  saved PNG charts and JSON data
sprites/                  optional PNG sprites for the renderer
```

---

## Apartment Layout

The environment is a 15×15 grid with six rooms:

| Room        | Cleaning reward |
| ----------- | --------------- |
| Kitchen     | +20             |
| Bathroom    | +20             |
| Living room | +10             |
| Hallway     | +10             |
| Bedroom     | +5              |
| Storage     | +5              |

The robot starts at the charger `(0, 0)` facing North.
Battery capacity is **80** units. The mission completes when all dirt is
cleaned and the robot returns to the charger.

---

## Agents

### Q-Learning (`agent/q_learning_agent.py`)

Off-policy tabular method. Learns the best action from any state regardless
of what the policy would actually do. Tends to learn bold strategies.

### SARSA (`agent/sarsa_agent.py`)

On-policy tabular method. Updates using the action the policy actually took,
so it learns safer, more conservative routes.

### DQN (`agent/dqn_agent.py`)

CNN neural network maps the 8-channel 15×15 grid directly to Q-values.
Requires PyTorch. Uses experience replay and a target network for stability.

---

## Output Files

After training, two types of files are saved automatically:

**PNG charts** in `results/`:

- `<agent>_training_<timestamp>.png` — reward / steps-to-clean-100% / battery for one agent
- `comparison_<timestamp>.png` — all three agents on the same axes

**JSON data** in `results/`:

- `results_<timestamp>.json` — raw per-episode numbers for replotting

**Model files** in `models/`:

- `q_learning_phase2.pkl`
- `sarsa_phase2.pkl`
- `dqn_phase2.pth`

---

## Optional Sprites

Put 42×42 PNG files in a `sprites/` folder to get nicer graphics.
If any are missing the renderer falls back to coloured shapes.

```
sprites/
  wall.png
  furniture.png
  charger.png
  dirt.png
  robot.png      # should face up (North) by default
```

---

## Requirements

```
gymnasium
numpy
matplotlib
pygame
torch          # only needed for DQN
```

Install with:

```bash
pip install -r requirements.txt
```
