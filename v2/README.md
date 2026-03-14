# 🤖 Cleaning Robot — Reinforcement Learning

A Reinforcement Learning project that trains a virtual cleaning robot to navigate a multi-room house and clean all dirty tiles, using **three algorithms**: **Q-Learning**, **SARSA**, and **DQN (Deep Q-Network)**. The robot starts with zero knowledge and learns an optimal cleaning strategy purely through trial and error.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Pygame](https://img.shields.io/badge/Pygame-2.5%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Technology Stack](#-technology-stack)
- [The Environment](#-the-environment)
  - [House Layout](#house-layout)
  - [Action Space](#action-space)
  - [State Representation](#state-representation)
  - [Reward Structure](#reward-structure)
- [The Algorithms](#-the-algorithms)
  - [What All Three Share](#what-all-three-share-similarities)
  - [Q-Learning](#1-q-learning-off-policy-tabular)
  - [SARSA](#2-sarsa-on-policy-tabular)
  - [DQN](#3-dqn--deep-q-network-off-policy-neural)
  - [Optimization Formulas Compared](#-optimization-formulas-compared)
  - [Full Comparison Table](#full-comparison-table)
  - [Which Algorithm for Which Problem?](#-which-algorithm-for-which-problem)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Training Process](#-training-process)
- [Results & Comparison](#-results--comparison)
  - [Latest Experiment Setup](#-latest-experiment-setup)
  - [Learning Curves](#1-learning-curves)
  - [Optimal Paths](#2-optimal-paths)
  - [Performance Bar Charts](#3-performance-bar-charts)
  - [Smart Analysis Radar](#4-smart-analysis-radar)
  - [Why SARSA Wins This Case](#-why-sarsa-wins-this-case)
  - [When Others Will Outperform](#-when-others-will-outperform)
- [Key Concepts](#-key-concepts)
- [References](#-references)

---

## 🧹 Project Overview

This project implements a complete Reinforcement Learning pipeline around a custom **8×6 grid house** environment. A robot agent must discover — through pure trial and error — how to navigate between a Kitchen, Living Room, and Hallway and clean every dirty tile as efficiently as possible.

**Why three algorithms?** To directly compare classical tabular RL (Q-Learning, SARSA) against deep RL (DQN) on the same task, and to demonstrate concretely how on-policy vs off-policy learning and neural-network function approximation affect final performance.

**Key design decisions:**

- The robot receives **no hardcoded rules** — it learns entirely from the reward signal.
- A **DNUT compass** (Detection of Nearest Uncleaned Tile) is embedded in the state, giving the agent a relative direction hint to the closest dirty tile, but the robot still has to learn how to use it.
- **Pygame** renders the live cleaning visually; **Matplotlib** generates all comparison charts.
- Code was developed with the assistance of AI code generation tools (**partially vibe coded**).

---

## 🛠 Technology Stack

| Tool / Library | Version | Role                                            |
| -------------- | ------- | ----------------------------------------------- |
| **Python**     | 3.8+    | Core language                                   |
| **Gymnasium**  | 0.29+   | RL environment framework (custom `CleaningEnv`) |
| **NumPy**      | latest  | Q-table storage, numerical operations           |
| **PyTorch**    | 2.0+    | Neural network for DQN (`QNetwork`)             |
| **Pygame**     | 2.5+    | 2D real-time visualization of the robot         |
| **Matplotlib** | latest  | Training plots and comparison dashboard         |
| **Pickle**     | stdlib  | Save/load Q-tables between runs                 |

---

## 🏠 The Environment

The environment (`env/cleaning_env.py`) is a **custom Gymnasium environment** — it extends `gymnasium.Env` and implements `reset()`, `step()`, and `render()`.

### House Layout

The house is an **8 × 6 grid** containing three rooms and perimeter walls.

```
    Col:  0     1     2     3     4     5     6     7
         ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
Row 0:   │WALL │WALL │WALL │WALL │WALL │WALL │WALL │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 1:   │WALL │ 🟡  │ 🟡  │ 🟡  │WALL │ 🔵  │ 🔵  │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 2:   │WALL │ 🟡  │ 🟡  │ 🟡  │ ⬜  │ 🔵  │ 🔵  │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 3:   │WALL │ 🟡  │ 🟡  │ 🟡  │ ⬜  │ 🔵  │ 🔵  │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 4:   │WALL │ ⬜  │ ⬜  │ ⬜  │ ⬜  │ ⬜  │ ⬜  │WALL │
         ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
Row 5:   │WALL │WALL │WALL │WALL │WALL │WALL │WALL │WALL │
         └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

🟡 Kitchen      — 9 tiles  — +50 reward each
🔵 Living Room  — 6 tiles  — +35 reward each
⬜ Hallway       — 8 tiles  — +20 reward each
   WALL         — boundary — robot cannot enter
```

**Total cleanable tiles: 23**

### Action Space

The robot has **6 discrete actions** at every step:

| ID  | Name     | Effect                        |
| --- | -------- | ----------------------------- |
| 0   | Forward  | Move up (row − 1)             |
| 1   | Backward | Move down (row + 1)           |
| 2   | Left     | Move left (col − 1)           |
| 3   | Right    | Move right (col + 1)          |
| 4   | Wait     | Stay in place                 |
| 5   | Clean    | Attempt to clean current tile |

### State Representation

The state is a single integer that encodes four components:

```
State = position × dirt_status × movement_history × DNUT_direction
      =    23    ×      2       ×        5         ×      10
      = 2,300 possible states
```

| Component            | Values                   | Meaning                                  |
| -------------------- | ------------------------ | ---------------------------------------- |
| **Position**         | 0–22 (23 tiles)          | Which cleanable tile the robot is on     |
| **Dirt Status**      | 0 or 1                   | Is the current tile dirty?               |
| **Movement History** | N / S / E / W / None (5) | Direction the robot came from            |
| **DNUT Direction**   | 3×3 compass + None (10)  | Relative direction to nearest dirty tile |

> **DNUT** = Detection of Nearest Uncleaned Tile — a compass-like hint embedded in the state that points toward the closest unvisited dirty tile by Manhattan distance. It reduces wandering without hardcoding any policy.

### Reward Structure

The reward function shapes what the robot learns to do:

| Event                        | Reward         | Purpose                     |
| ---------------------------- | -------------- | --------------------------- |
| Clean dirty Kitchen tile     | **+50**        | High-priority room          |
| Clean dirty Living Room tile | **+35**        | Medium priority             |
| Clean dirty Hallway tile     | **+20**        | Low priority                |
| All 23 tiles cleaned         | **+200 bonus** | Incentivise full completion |
| Step on already-clean tile   | **−5**         | Avoid redundant movement    |
| Use Clean on clean tile      | **−10**        | Waste of an action          |
| Hit a wall                   | **−5**         | Invalid move penalty        |
| Wait (do nothing)            | **−3**         | Time is costly              |
| Every step taken             | **−0.1**       | Encourages efficiency       |

---

## 🧠 The Algorithms

### What All Three Share (Similarities)

Despite being architecturally different, all three algorithms operate on **the exact same environment** with **the same interface**:

| Shared Property       | Value                                                         |
| --------------------- | ------------------------------------------------------------- |
| Environment           | `CleaningEnv` — 8×6 grid, 23 cleanable tiles                  |
| Action space          | 6 discrete actions (move × 4, wait, clean)                    |
| State space           | 2,300 discrete states (tabular) / 40-dim feature vector (DQN) |
| Epsilon-greedy policy | All three explore with ε-greedy and decay ε over time         |
| Episode structure     | Reset → loop(observe → act → reward → update) → done          |
| Discount factor γ     | 0.99 for all three                                            |
| Epsilon start / end   | 1.0 → 0.02 for all three                                      |
| Goal                  | Maximize cumulative reward = clean all 23 tiles efficiently   |
| Bellman foundation    | All three approximate the Bellman optimality equation         |

This shared ground makes the algorithm comparison **fair and meaningful**: any performance difference comes purely from the learning mechanism, not from the task or action set.

---

### 1. Q-Learning (Off-Policy, Tabular)

Q-Learning stores a **Q-table** mapping every (state, action) pair to an expected cumulative reward. It is **off-policy**: the update always bootstraps from the best possible next action, regardless of what the agent actually does next.

**Update rule (Bellman optimality):**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma \cdot \underbrace{\max_{a'} Q(s', a')}_{\text{best future action}} - Q(s, a) \Big]$$

**Hyperparameters used:**

| Parameter         | Value                     |
| ----------------- | ------------------------- |
| Learning rate α   | 0.15                      |
| Discount factor γ | 0.99                      |
| Epsilon decay     | 0.998 per episode         |
| Episodes          | 5,000                     |
| Q-table size      | 2,300 × 6 = 13,800 values |

**Intuition:** Q-Learning is the _optimist_ — it always imagines the best possible future when updating its estimates. This makes it converge quickly to the **theoretically optimal policy**, but it ignores the risk that exploration noise might accidentally land the agent in a bad state.

**Training loop:**

```
s ← reset()
while not done:
    a ← ε-greedy(Q[s])
    s', r, done ← step(a)
    Q[s][a] += α * (r + γ * max(Q[s']) - Q[s][a])   # off-policy update
    s ← s'
```

---

### 2. SARSA (On-Policy, Tabular)

SARSA shares Q-Learning's Q-table structure but updates using the **actual next action chosen by the policy**, not the hypothetical best one. The name comes from the five-tuple used: **(S, A, R, S', A')**.

**Update rule (Bellman expectation):**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ r + \gamma \cdot \underbrace{Q(s', a')}_{\text{actual next action}} - Q(s, a) \Big]$$

**Key difference from Q-Learning:** `a'` is selected by ε-greedy _before_ the update, so the Q-value reflects what the agent will _actually_ do — including random exploration moves. This is the **on-policy** property.

**Hyperparameters used:**

| Parameter         | Value                     |
| ----------------- | ------------------------- |
| Learning rate α   | 0.15                      |
| Discount factor γ | 0.99                      |
| Epsilon decay     | 0.998 per episode         |
| Episodes          | 5,000                     |
| Q-table size      | 2,300 × 6 = 13,800 values |

**Intuition:** SARSA is the _realist_ — it knows that during training it sometimes takes random steps (exploration), so it builds that cost into its value estimates. States near walls get penalised because the agent knows it might randomly walk into one. This makes SARSA more **conservative and cautious**.

**Training loop:**

```
s ← reset()
a ← ε-greedy(Q[s])
while not done:
    s', r, done ← step(a)
    a' ← ε-greedy(Q[s'])         # next action chosen NOW (on-policy)
    Q[s][a] += α * (r + γ * Q[s'][a'] - Q[s][a])   # on-policy update
    s ← s', a ← a'
```

---

### 3. DQN — Deep Q-Network (Off-Policy, Neural)

DQN replaces the Q-table with a **neural network** `Q(s; θ) → [Q(s,a₀), ..., Q(s,a₅)]`. It is off-policy like Q-Learning, but adds two stability mechanisms: **experience replay** and a **target network**.

**Network architecture:**

```
Input (40 features)
  [robot_row_norm, robot_col_norm,
   tile_0_dirty?, tile_1_dirty?, ..., tile_22_dirty?,   ← 23 dirt bits
   came_from_one_hot (5 dims),
   DNUT_one_hot (10 dims)]
        │
  Linear(40 → 64) → ReLU
        │
  Linear(64 → 64) → ReLU
        │
  Linear(64 → 6)   ← output: Q-value for each of 6 actions
```

**Update rule (TD with frozen target network):**

$$\mathcal{L}(\theta) = \mathbb{E}\Big[ \Big( r + \gamma \cdot \max_{a'} Q(s', a';\, \theta^{-}) - Q(s, a;\, \theta) \Big)^2 \Big]$$

Where $\theta^{-}$ are the **frozen target network weights** (synced every 250 steps).

**Three innovations that stabilise DQN:**

| Component                                 | Problem Solved                                                                                |
| ----------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Experience Replay** (buffer = 50,000)   | Sequential transitions are correlated — random mini-batch sampling breaks this                |
| **Target Network** (sync every 250 steps) | Without it, the network chases its own shifting predictions (moving target problem)           |
| **Feature Vector** (40-dim)               | Richer input than a single integer state-ID; allows generalisation across tile configurations |

**Hyperparameters used:**

| Parameter              | Value              |
| ---------------------- | ------------------ |
| Learning rate α (Adam) | 0.001              |
| Discount factor γ      | 0.99               |
| Epsilon decay          | 0.9984 per episode |
| Episodes               | 3,000              |
| Batch size             | 64                 |
| Replay buffer          | 50,000             |
| Target update interval | every 250 steps    |

---

### 📐 Optimization Formulas Compared

|                      | Q-Learning                       | SARSA                       | DQN                                         |
| -------------------- | -------------------------------- | --------------------------- | ------------------------------------------- |
| **Bootstrap target** | $r + \gamma \max_{a'} Q(s', a')$ | $r + \gamma\, Q(s', a')$    | $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ |
| **Policy**           | Off-policy (greedy target)       | On-policy (ε-greedy target) | Off-policy (greedy target via frozen net)   |
| **Q estimator**      | Table entry $Q[s][a]$            | Table entry $Q[s][a]$       | Neural network $Q(s,a;\theta)$              |
| **Loss**             | TD error (mean squared)          | TD error (mean squared)     | MSE loss, gradient descent via Adam         |
| **Update trigger**   | Every step                       | Every step                  | After each step (mini-batch from replay)    |

The only difference between Q-Learning and SARSA is **one symbol**: `max Q(s', a')` vs `Q(s', a')`. That single change — from greedy to actual-policy bootstrapping — shifts the agent from optimistic-aggressive to conservative-cautious.

---

### Full Comparison Table

| Feature              | Q-Learning        | SARSA             | DQN                       |
| -------------------- | ----------------- | ----------------- | ------------------------- |
| **Policy type**      | Off-policy        | On-policy         | Off-policy                |
| **Q estimator**      | Table             | Table             | Neural network            |
| **State space**      | Discrete, small   | Discrete, small   | Can be continuous / large |
| **Memory**           | Grows with states | Grows with states | Fixed (network weights)   |
| Epsilon decay        | 0.998             | 0.998             | 0.9984                    |
| **Safety / caution** | Aggressive        | Conservative      | Depends on tuning         |
| **Training speed**   | Fast              | Medium            | Slower (gradient descent) |
| Replay buffer        | —                 | —                 | 50,000                    |
| Target update        | —                 | —                 | Every 250 steps           |
| **Hardware needs**   | CPU only          | CPU only          | Benefits from GPU         |

---

### 🎯 Which Algorithm for Which Problem?

### 🧪 Latest Experiment Setup

The README analysis below is based on the **latest regenerated artifacts** after retraining and testing again:

- **Training run used for plots:** `2000` episodes for **Q-Learning**, **SARSA**, and **DQN**
- **Evaluation pass after training:** `10` test episodes for qualitative verification
- **Quantitative plot statistics:** computed from the **last 100 training episodes** saved in `models/*_history.pkl`
- **Environment used:** `conda activate torch_gpu`
- **Current environment check during this README update:** `torch_gpu` resolves to `torch 2.10.0+cpu` with `cuda_available = False` on this machine

### Latest numeric summary from the current saved models

| Metric (last 100 episodes unless noted) | Q-Learning |       SARSA |         DQN |
| --------------------------------------- | ---------: | ----------: | ----------: |
| Average reward                          |     984.22 | **1014.70** |     1011.24 |
| Reward std $\sigma$                     |      49.80 |    **6.29** |        9.30 |
| Average tiles cleaned                   |      23.00 |       23.00 |       23.00 |
| Success rate                            |     100.0% |      100.0% |      100.0% |
| Average steps                           |      29.43 |   **23.57** |       24.09 |
| Best reward ever                        |    1007.50 | **1017.70** | **1017.70** |
| Training time                           |    10.12 s |  **6.97 s** |    468.88 s |
| Reward per second                       |      97.22 |  **145.58** |        2.16 |
| 90% success reached at episode          |        110 |         107 |     **106** |

**High-level takeaway:** all three agents now solve the task perfectly by the end of training, but **SARSA is the best final overall performer** because it combines the **highest final reward**, **lowest variance**, and **fewest steps**, while **DQN is the fastest to hit 90% success** by a tiny margin.

#### By Problem Complexity

| Problem Size     | State Space                      | Best Algorithm      | Reason                                                   |
| ---------------- | -------------------------------- | ------------------- | -------------------------------------------------------- |
| **Tiny**         | < 1,000 states                   | Q-Learning          | Converges fastest, zero overhead                         |
| **Small–Medium** | 1,000–100,000 states             | Q-Learning or SARSA | Table still fits in memory; choose based on safety needs |
| **Large**        | 100,000+ states                  | DQN                 | Table would be too large; network generalises            |
| **Continuous**   | Infinite (e.g., pixels, sensors) | DQN / Actor-Critic  | Only function approximation scales                       |

#### By Task Characteristics

**Use Q-Learning when:**

- State space is small and discrete (like our 2,300-state house).
- You want the theoretically optimal policy fast.
- Safety is not critical — risky states near penalties are acceptable.
- Quick prototyping or benchmarking needed.

**Use SARSA when:**

- The environment has **cliffs, traps, or costly penalties** near the agent's path.
- You want a policy that is robust to exploration noise during training.
- The learned policy will be deployed in a **safety-critical** setting (robot navigation, medical control).
- You need stable, predictable behaviour near boundaries.

**Use DQN when:**

- State space is **too large** for a table (robotics, games with image inputs).
- You need **generalisation** — the agent should handle unseen-but-similar states.
- Raw features (pixels, sensor readings) rather than discrete IDs.
- Examples: Atari games, self-driving simulations, complex robot manipulation.

#### Concrete Scenario Guide

| Scenario                                 | Algorithm                     |
| ---------------------------------------- | ----------------------------- |
| 5×5 grid maze, 50 states                 | Q-Learning                    |
| Same maze but with a lava cliff edge     | SARSA                         |
| 100×100 grid, 10,000 states              | Q-Learning (table still fine) |
| 1,000-tile house + obstacle avoidance    | DQN                           |
| Atari Breakout (84×84 pixels)            | DQN                           |
| Real robotic arm with joint angles       | DQN / SAC                     |
| Bank fraud detection (discrete features) | Q-Learning / SARSA            |

---

## 📁 Project Structure

```
cleaning-robot-rl/
│
├── main.py              # Interactive menu: train, test, compare all algorithms
├── train.py             # Standalone Q-Learning training script
├── test.py              # Standalone Q-Learning testing/visualisation script
├── requirements.txt     # Python dependencies
│
├── env/
│   ├── __init__.py
│   └── cleaning_env.py  # Custom Gymnasium environment (8×6 grid, rewards, render)
│
├── agent/
│   ├── __init__.py
│   ├── q_learning_agent.py   # Q-Learning (tabular, off-policy)
│   ├── sarsa_agent.py        # SARSA (tabular, on-policy)
│   └── dqn_agent.py          # DQN (neural network, off-policy)
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py       # Progress bars, time formatting, console utilities
│   └── plotting.py      # Matplotlib training curves and export helpers
│
├── models/              # Saved trained models (auto-generated after training)
│   ├── q_table.pkl          # Trained Q-Learning Q-table
│   ├── sarsa_table.pkl      # Trained SARSA Q-table
│   ├── dqn_model.pth        # Trained DQN neural network weights
│   └── *_history.pkl        # Training history for comparison dashboard
│
└── plots/               # Generated comparison charts
    ├── comparison_learning_curves.png
    ├── comparison_optimal_paths.png
    ├── comparison_bars.png
    ├── comparison_smart_analysis.png
    ├── optimal_path_qlearning.png
    ├── optimal_path_sarsa.png
    └── optimal_path_dqn.png
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A display (required for Pygame visualisation)

### Installation

```bash
git clone https://github.com/ubaidur404786/cleaning-robot-rl.git
cd cleaning-robot-rl
pip install -r requirements.txt
```

**Dependencies installed:**

| Package      | Purpose                               |
| ------------ | ------------------------------------- |
| `gymnasium`  | RL environment framework              |
| `numpy`      | Q-table storage, numerical operations |
| `matplotlib` | Training plots and comparison charts  |
| `pygame`     | 2D live visualisation of the robot    |
| `torch`      | Neural network for DQN                |

### Running the Project

#### Interactive Menu (Recommended)

```bash
python main.py
```

```
  +--------------------------------------------------+
  |                   MAIN MENU                      |
  +--------------------------------------------------+
  |  [1]  Train Q-Learning Agent                     |
  |  [2]  Train SARSA Agent                          |
  |  [3]  Train DQN Agent                            |
  |  [4]  Test  Q-Learning Agent (Pygame UI)         |
  |  [5]  Test  SARSA Agent (Pygame UI)              |
  |  [6]  Test  DQN Agent (Pygame UI)                |
  |  [7]  Show Optimal Path — Q-Learning             |
  |  [8]  Show Optimal Path — SARSA                  |
  |  [9]  Show Optimal Path — DQN                    |
  |  [10] Compare All Algorithms (Dashboard)         |
  |  [11] Quick Train & Compare All                  |
  |  [0]  Exit                                       |
  +--------------------------------------------------+
```

#### Train from Python Code

```python
from main import train

history = train(algo='qlearning', num_episodes=5000)
history = train(algo='sarsa',     num_episodes=5000)
history = train(algo='dqn',       num_episodes=3000)
```

#### Standalone Q-Learning Training

```bash
python train.py
```

---

## 📈 Training Process

Each training episode:

```
1. Reset:   All 23 tiles become dirty; robot spawns on a random tile
2. Loop until done:
   a. Observe current state (position + dirt + history + DNUT)
   b. Choose action via ε-greedy policy
   c. Execute action in environment
   d. Receive reward and next state
   e. Update Q-table (or neural network)
   f. Move to next state
3. End:     All tiles clean (success) OR max steps reached (timeout)
4. Decay ε: epsilon × 0.998 (tabular) or × 0.9984 (DQN)
```

### Hyperparameters Summary

| Parameter         | Q-Learning | SARSA | DQN             |
| ----------------- | ---------- | ----- | --------------- |
| Learning rate α   | 0.15       | 0.15  | 0.001 (Adam)    |
| Discount factor γ | 0.99       | 0.99  | 0.99            |
| Epsilon start     | 1.0        | 1.0   | 1.0             |
| Epsilon end       | 0.02       | 0.02  | 0.02            |
| Epsilon decay     | 0.998      | 0.998 | 0.9984          |
| Default episodes  | 5,000      | 5,000 | 5,000           |
| Batch size        | —          | —     | 64              |
| Replay buffer     | —          | —     | 50,000          |
| Target update     | —          | —     | Every 250 steps |

---

## 📊 Results & Comparison

All plots below are generated automatically by the comparison dashboard (`main.py → [10]`) and saved to the `plots/` folder. They are served directly from the GitHub repository.

---

### 1. Learning Curves

![Learning Curves](https://raw.githubusercontent.com/ubaidur404786/cleaning-robot-rl/main/plots/comparison_learning_curves.png)

**What this shows:**
This 4-panel figure tracks each algorithm across the full 2000-episode training run.

- **Top-left — Episode Reward:** SARSA finishes highest at **1014.70 ± 6.29**, DQN is close behind at **1011.24 ± 9.30**, and Q-Learning ends lower at **984.22 ± 49.80**. The big story here is not just reward level, but reward **stability**: Q-Learning's band is much wider, showing noticeably larger variance in late training.
- **Top-right — Tiles Cleaned per Episode:** all three algorithms end at **23.00 / 23.00 tiles cleaned**, so the competition is no longer about whether they can finish the job — it is about **how efficiently and consistently** they do it.
- **Bottom-left — Success Rate:** all three reach **100% success** in the last 100 episodes. The convergence markers show that **DQN reaches 90% success first at episode 106**, SARSA follows at **107**, and Q-Learning at **110**. So the fastest learner early on is DQN by a hair, but the strongest finisher is SARSA.
- **Bottom-right — Epsilon Decay:** Q-Learning and SARSA decay exploration faster toward the minimum, while DQN keeps a slightly higher exploration level longer because of its `0.9984` decay schedule. That helps DQN avoid premature overconfidence, but it also keeps training computationally heavier.

**Comments on this plot:**

- The three curves prove the environment is now **fully learnable** by all agents within 2000 episodes.
- SARSA has the **cleanest late-stage plateau**, which is exactly why it wins the stability-oriented metrics later.
- Q-Learning learns the task, but it remains the noisiest in reward because its off-policy optimism makes it less cautious near penalty-heavy boundaries.
- DQN learns surprisingly well on this small problem, but the neural network is doing extra work for a state space that tabular methods already handle very well.

---

### 2. Optimal Paths

![Optimal Paths Comparison](https://raw.githubusercontent.com/ubaidur404786/cleaning-robot-rl/main/plots/comparison_optimal_paths.png)

**What this shows:**
Side-by-side visualisation of the cleaning path each trained algorithm takes on the house grid. The path is overlaid on the room layout (yellow = Kitchen, blue = Living Room, grey = Hallway). Each numbered dot marks the **first visit** to a tile, arrows show transitions, `S` is the start, and `E` is the end.

| Individual Path         | Link                                                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Q-Learning optimal path | ![Q-Learning](https://raw.githubusercontent.com/ubaidur404786/cleaning-robot-rl/main/plots/optimal_path_qlearning.png) |
| SARSA optimal path      | ![SARSA](https://raw.githubusercontent.com/ubaidur404786/cleaning-robot-rl/main/plots/optimal_path_sarsa.png)          |
| DQN optimal path        | ![DQN](https://raw.githubusercontent.com/ubaidur404786/cleaning-robot-rl/main/plots/optimal_path_dqn.png)              |

**Current path results from the latest models:**

- **Q-Learning:** cleans **23/23 tiles in 27 steps**
- **SARSA:** cleans **23/23 tiles in 23 steps**
- **DQN:** cleans **23/23 tiles in 23 steps**

**Comments on this plot:**

- **SARSA and DQN are tied for the shortest extracted path length** in the current saved models, both finishing in **23 steps**, which is almost a perfect room-to-room sweep.
- **Q-Learning takes 27 steps**, so it still solves the task completely but wastes a few more moves through extra revisits or less compact routing.
- The most important visual cue is how **SARSA's path looks systematic**: it tends to enter and clear room regions with less unnecessary wandering.
- DQN's path is also very strong now, showing that the network has learned the room structure well even though the problem is small enough that deep learning is not strictly necessary.
- These path plots are now **greedy evaluation paths** (`eval_epsilon = 0.0`), so unlike an exploratory rollout, they represent the learned policy directly rather than random test-time noise.

---

### 3. Performance Bar Charts

![Performance Bars](https://raw.githubusercontent.com/ubaidur404786/cleaning-robot-rl/main/plots/comparison_bars.png)

**What this shows:**
Six bar-chart panels, each measuring one metric across the three algorithms. Winner badges (🥇) highlight which algorithm leads in each category using the **last 100 episodes**.

| Metric                | What It Measures                                       | Typical Winner         |
| --------------------- | ------------------------------------------------------ | ---------------------- |
| **Average Reward**    | Mean episode reward over last 100 episodes             | **SARSA**              |
| **Tiles Cleaned**     | Average tiles cleaned per episode                      | **Tie (all at 23.00)** |
| **Success Rate**      | % of episodes where all 23 tiles cleaned               | **Tie (all at 100%)**  |
| **Steps to Complete** | Average steps per successful episode (lower is better) | **SARSA**              |
| **Reward Stability**  | Lower reward standard deviation is better              | **SARSA**              |
| **Training Time**     | Wall-clock seconds to complete training                | **SARSA**              |

**Comments on this plot:**

- This figure makes one thing very clear: **the winner is decided by efficiency and stability, not by completion**, because all three already complete the cleaning task perfectly.
- **SARSA dominates the practical metrics**: best reward, best stability, fewest steps, and even the fastest wall-clock training time in the latest run.
- **Q-Learning is no longer the fastest overall** in this run; it trains quickly, but SARSA finished even faster (**6.97 s vs 10.12 s**) while also producing better behavior.
- **DQN is massively more expensive computationally** here (**468.88 s**) for only a tiny reward gap versus SARSA. That is the clearest sign that this problem is too small to justify deep RL from a compute-efficiency perspective.

---

### 4. Smart Analysis Radar

![Smart Analysis](https://raw.githubusercontent.com/ubaidur404786/cleaning-robot-rl/main/plots/comparison_smart_analysis.png)

**What this shows:**
A 4-panel figure with deeper analysis:

- **Top-left — Convergence Speed:** the current run shows an extremely tight race. **DQN reaches 90% success first at episode 106**, SARSA at **107**, and Q-Learning at **110**. So if you only care about the first moment the agent becomes competent, DQN has a microscopic edge.
- **Top-right — Reward Stability Over Time:** SARSA should have the lowest late-stage reward volatility, matching its final **$\sigma = 6.29$**, versus DQN's **9.30** and Q-Learning's **49.80**. This is where SARSA separates itself from the others.
- **Bottom-left — Training Efficiency:** this panel exposes the real compute story. SARSA delivers **145.58 reward/second**, Q-Learning **97.22**, and DQN only **2.16**. In other words, DQN is effective but wildly inefficient for such a compact state space.
- **Bottom-right — Radar / Spider Chart:** SARSA should cover the broadest, most balanced area because it is simultaneously strong in reward, steps, stability, and compute efficiency. DQN scores well on convergence and final reward, but its compute cost drags the polygon inward.

**Comments on this plot:**

- This figure is the best summary of the whole experiment because it separates **"learns fast"** from **"finishes best"** from **"uses compute wisely."**
- DQN deserves credit: it is not bad here at all — it actually becomes successful slightly earlier than the others.
- But SARSA is the better engineering choice for this project because it gives nearly optimal paths and top rewards **without** the enormous training-time penalty of DQN.
- Q-Learning remains a solid baseline, but its reward variance makes it the least polished final policy in the current run.

---

### 🏆 Why SARSA Wins This Case

On the latest 2000-episode run, **SARSA is the best final overall algorithm** for this environment for one core reason:

> **SARSA is on-policy — it learns the value of the policy it actually executes, including exploration noise. In an environment full of wall penalties and room boundaries, this makes it inherently more conservative and better adapted to the actual training dynamics.**

The precise mechanism:

1. **Wall penalties are frequent during exploration.** Early in training, the agent frequently bumps walls (ε is high). Under Q-Learning, wall-adjacent states are still valued optimistically (it assumes the next move will be perfect). Under SARSA, the wall penalty propagates into the Q-value of wall-adjacent states, because the actual ε-greedy policy sometimes walks into walls from there.

2. **The robot learns safer room entry/exit strategies.** Because SARSA discounts states near risky transitions, it naturally learns to approach room doorways (the two hallway tunnel tiles at (2,4) and (3,4)) from the centre rather than hugging walls — leading to smoother navigation.

3. **Higher stability after convergence.** Once SARSA's conservative policy has settled, it rarely deviates from the learned path. That is exactly what appears in the latest metrics: **1014.70 average reward with only 6.29 standard deviation**, compared with Q-Learning's much noisier **49.80**.

4. **Reward signal is dense.** The cleaning reward structure (+50/+35/+20 per tile, −5 for revisits, −0.1 per step) provides rich feedback. SARSA exploits this dense signal more reliably because on-policy updates keep Q-values calibrated to actual experience.

**In short:** the DNUT compass hint + dense rewards + small state space make this an ideal "cautious navigation" problem — exactly SARSA's domain. DQN learns it too, and even hits 90% success a touch earlier, but SARSA gives the **best final balance** of path quality, reward quality, and compute cost.

---

### 🔮 When Others Will Outperform

| Algorithm      | Will Win When...                                                                                                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Q-Learning** | The environment has no dangerous boundary states; you need the fastest convergence; reward is sparse and you need aggressive optimistic exploration to ever find it                                    |
| **DQN**        | State space grows beyond ~100,000 states; tiles or rooms increase significantly; raw sensor readings replace discrete state IDs; you need the agent to generalise to novel configurations at test time |
| **SARSA**      | (This case) — small state space, dense rewards, wall/cliff penalties near agent paths, training and deployment policy are the same                                                                     |

Q-Learning would likely win on a **cliff-walking task with no intermediate penalties** (it finds the shortest path); DQN would be necessary if the house were scaled to a **20×20 grid** with pixel rendering.

---

## 📚 Key Concepts

| Term                  | Meaning                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------- |
| **Agent**             | The robot (learner that takes actions)                                                      |
| **Environment**       | The house grid (everything outside the agent)                                               |
| **State**             | A snapshot of the current situation (position + dirt + DNUT)                                |
| **Action**            | Something the agent can do (move, clean, wait)                                              |
| **Reward**            | Feedback signal — positive = good, negative = bad                                           |
| **Episode**           | One complete run (start → all clean or time-out)                                            |
| **Policy**            | The agent's strategy (what action to take in each state)                                    |
| **Q-value**           | Expected total future reward for taking an action in a state                                |
| **Epsilon (ε)**       | Probability of taking a random action (exploration rate)                                    |
| **Off-policy**        | Learns optimal behavior regardless of what actions are actually taken                       |
| **On-policy**         | Learns the value of the behavior it actually follows                                        |
| **TD Learning**       | Temporal Difference — update estimates from other estimates without waiting for episode end |
| **Experience Replay** | Store past transitions and train on random samples (DQN)                                    |
| **Target Network**    | Frozen copy of Q-network used for stable DQN updates                                        |
| **Discount γ**        | How much future rewards are worth compared to immediate ones                                |

---

## 📎 References

The following resources were referenced and consulted while building this project:

1. **Q-Learning — Beginner's Guide with GridWorld**
   [A Beginner's Guide to Q-Learning: Understanding with a Simple Gridworld Example](https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9) — _Golden Grisha, Medium_

2. **SARSA — Temporal Difference Learning Explained**
   [SARSA: A Beginner's Guide to Temporal Difference Learning](https://shivang-ahd.medium.com/sarsa-a-beginners-guide-to-temporal-difference-learning-3d72b1011fd8) — _Shivang, Medium_

3. **SARSA — GeeksForGeeks Overview**
   [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/machine-learning/sarsa-reinforcement-learning/) — _GeeksForGeeks_

4. **DQN — Deep Q-Learning Beginner's Guide**
   [What is DQN? A Beginner's Guide to Deep Q-Learning](https://ujangriswanto08.medium.com/what-is-dqn-a-beginners-guide-to-deep-q-learning-db99bf4de688) — _Ujang Riswanto, Medium_

5. **Gymnasium Documentation**
   [gymnasium.farama.org](https://gymnasium.farama.org) — Official RL environment framework

6. **PyTorch Documentation**
   [pytorch.org](https://pytorch.org) — Neural network implementation for DQN

> **Note:** Parts of the code in this project were generated with the assistance of AI tools (**partially vibe coded**). All logic, hyperparameters, and algorithm structures were verified, understood, and intentionally applied.

---

<p align="center">
  <b>Built for the Master 2 Reinforcement Learning course — Université Côte d'Azur, 2025-2026</b><br>
  <i>From random chaos to optimal cleaning — one episode at a time</i>
</p>
