# Cleaning Robot RL Versions (v1, v2, v3)

This repo has 3 versions of the same project idea (robot cleaning with RL).
Each version changes the environment design and state/action setup.

## V1 vs V2 vs V3: Environment, Actions, States, Agents

| Version | Environment                                                                                                                                | Actions                                                                 | States                                                                                                          | Agents compared                                       |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **v1**  | `10x10` open grid (Phase 1 baseline), plus harder `15x15` apartment experiments in notebooks (corner/hallway charger), battery-constrained | **5** discrete actions: Up, Down, Left, Right, Charge                   | Tabular tuple `(row, col, battery_bin)` with `BATTERY_BINS=5` (Phase 1: `10*10*5=500` states)                   | Q-Learning (ε-greedy / UCB / optimistic), SARSA (UCB) |
| **v2**  | `8x6` house map with room structure and dirt tiles to clean                                                                                | **6** discrete actions: Forward, Backward, Left, Right, Wait, Clean     | Encoded tabular state space of **2300 states**                                                                  | Q-Learning, SARSA, DQN                                |
| **v3**  | `15x15` apartment with walls/furniture, orientation dynamics, charger + battery, objective = clean and return to charger                   | **4** discrete actions: Move Forward, Rotate Left, Rotate Right, Charge | Tabular state `(row, col, orientation, battery_bin, is_apartment_clean)`; DQN uses `9x15x15` tensor observation | Q-Learning, SARSA, DQN                                |

## Results comparison

### v1 (from notebooks)

Source notebooks:

- `v1/Qlearning_VS_SARSA/notebook.ipynb`
- `v1/Qlearning_VS_SARSA/notebook_dirt_patterns.ipynb`

#### Full comparison: Phase 1 vs Phase 2 (Corner) vs Phase 2 (Hallway)

Last 500 training episodes:

##### Phase 1 (10x10 open grid, battery=50, 99 dirty tiles, 5000 episodes)

| Agent                |    Reward |  Coverage | Steps | Death% | Battery |
| -------------------- | --------: | --------: | ----: | -----: | ------: |
| Q-Learn + ε-Greedy   |     263.6 |     32.2% |  50.4 | 100.0% |     0.0 |
| Q-Learn + UCB        |     365.4 |     39.6% | 230.0 |  60.0% |    18.0 |
| Q-Learn + Optimistic |     284.3 |     34.3% |  50.1 | 100.0% |     0.0 |
| SARSA + UCB          | **404.6** | **44.9%** | 140.0 |  80.0% |     4.0 |

##### Phase 2: Corner Charger (15x15 apartment, battery=80, 154 dirty tiles, 10000 episodes)

| Agent                |    Reward |  Coverage | Steps | Death% | Battery |
| -------------------- | --------: | --------: | ----: | -----: | ------: |
| Q-Learn + ε-Greedy   |     264.9 |     20.4% | 264.4 |  58.0% |    18.2 |
| Q-Learn + UCB        |     375.3 |     26.2% | 332.0 |  40.0% |    28.8 |
| Q-Learn + Optimistic |     290.8 |     22.8% | 118.7 |  95.4% |     1.6 |
| SARSA + UCB          | **404.9** | **28.7%** | 248.0 |  60.0% |    13.4 |

##### Phase 2: Hallway Charger (15x15 apartment, battery=80, 154 dirty tiles, 10000 episodes)

| Agent                |    Reward |  Coverage | Steps | Death% | Battery |
| -------------------- | --------: | --------: | ----: | -----: | ------: |
| Q-Learn + ε-Greedy   |     303.6 |     23.3% | 142.1 |  86.5% |     5.7 |
| Q-Learn + UCB        |     315.7 |     22.5% | 332.0 |  40.0% |    27.0 |
| Q-Learn + Optimistic |     267.2 |     21.3% | 109.5 |  94.0% |     3.1 |
| SARSA + UCB          | **397.9** | **28.3%** | 248.0 |  60.0% |    12.0 |

#### Dirt-pattern experiment summary (`notebook_dirt_patterns.ipynb`)

| Agent                      |     Reward |    Cleans |   Death% | Batt | Kitchen | Living Room | Bedroom | Bathroom | Hallway | Storage |
| -------------------------- | ---------: | --------: | -------: | ---: | ------: | ----------: | ------: | -------: | ------: | ------: |
| Expected (dirt rate)       |          - |         - |        - |    - |   53.5% |       28.1% |    7.5% |     7.5% |    2.4% |    1.0% |
| SARSA+UCB (Dirt-Trained)   | **3470.0** | **346.4** | **0.0%** | 30.0 |    0.0% |       65.7% |    0.0% |     0.0% |   34.3% |    0.0% |
| SARSA+UCB (Static-Trained) |      420.0 |      47.3 |   100.0% |  0.0 |    0.0% |       62.0% |    0.0% |     0.0% |   38.0% |    0.0% |

### v2

Latest results (from saved plots), last 100 episodes:

| Metric                    | Q-Learning |      SARSA |    DQN |
| ------------------------- | ---------: | ---------: | -----: |
| Avg reward                |      984.2 | **1014.7** | 1011.2 |
| Avg tiles cleaned         |       23.0 |       23.0 |   23.0 |
| Success rate              |     100.0% |     100.0% | 100.0% |
| Avg steps                 |       29.0 |   **24.0** |   24.0 |
| Reward std (lower better) |       49.8 |    **6.3** |    9.3 |
| Training time (s)         |       10.1 |    **7.0** |  468.9 |

### v3

From `v3/results/results_20260314_024545.json` (last 100 episodes):

| Metric                    | Q-Learning |  SARSA |         DQN |
| ------------------------- | ---------: | -----: | ----------: |
| Avg reward                |     430.02 | 366.71 | **1229.17** |
| Reward std (lower better) | **145.43** | 250.95 |      210.59 |
| Avg tiles cleaned         |      66.86 |  68.52 |  **104.61** |
| Avg steps                 |     600.00 | 600.00 |      600.00 |
| Avg battery efficiency    |   **0.88** |   0.28 |        0.18 |
| Success rate              |       0.0% |   0.0% |        0.0% |
| Best reward seen          |     721.40 | 882.20 | **1492.80** |

## High-level takeaway

- **v1**: best for validating tabular RL behavior and exploration strategies; SARSA+UCB is strongest in the reported notebook runs.
- **v2**: all agents solve the task reliably; SARSA is the most stable (low variance) and efficient.
- **v3**: map/objective complexity is much higher; DQN leads on reward/coverage, but full mission completion is still unsolved in this snapshot.


## Main plots by version

- **v1**: notebook plots in `v1/Qlearning_VS_SARSA/notebook.ipynb` and `notebook_dirt_patterns.ipynb`
- **v2**: `v2/plots/comparison_learning_curves.png`, `comparison_optimal_paths.png`, `comparison_bars.png`, `comparison_smart_analysis.png`
- **v3**: `v3/results/comparison_learning_curves.png`, `comparison_optimal_paths.png`, `comparison_bars.png`, `comparison_smart_analysis.png`


## References

1. Path Planning of Cleaning Robot with Reinforcement Learning.  
   https://arxiv.org/pdf/2208.08211

2. A Beginner’s Guide to Q-Learning: Understanding with a Simple Gridworld Example.  
   https://medium.com/@goldengrisha/a-beginners-guide-to-q-learning-understanding-with-a-simple-gridworld-example-2b6736e7e2c9

3. SARSA: A Beginner’s Guide to Temporal Difference Learning.  
   https://shivang-ahd.medium.com/sarsa-a-beginners-guide-to-temporal-difference-learning-3d72b1011fd8

4. Deep Q-Learning Explained.  
   https://www.geeksforgeeks.org/deep-learning/deep-q-learning/

5. Reinforcement Learning Lecture Notes from course material.
