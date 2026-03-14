"""
================================================================================
MAIN ENTRY POINT - Unified RL Training, Testing & Comparison Dashboard
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning
FILE: main.py
PURPOSE: Interactive menu for Q-Learning, SARSA & DQN — train, test, visualise
         optimal paths, and compare all algorithms side by side.

================================================================================
FEATURES
================================================================================

 --- TRAINING ---
 [1]  Train Q-Learning Agent
 [2]  Train SARSA Agent
 [3]  Train DQN Agent

 --- TESTING (Pygame UI) ---
 [4]  Test  Q-Learning Agent
 [5]  Test  SARSA Agent
 [6]  Test  DQN Agent

 --- OPTIMAL PATHS ---
 [7]  Show Optimal Path - Q-Learning
 [8]  Show Optimal Path - SARSA
 [9]  Show Optimal Path - DQN

 --- COMPARISON ---
 [10] Compare All Algorithms (Dashboard)
 [11] Quick Train & Compare All
 [0]  Exit

================================================================================
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
import pickle
import numpy as np
from datetime import datetime

# ============================================================================
# IMPORT PROJECT MODULES
# ============================================================================

try:
    from env.cleaning_env import (
        CleaningEnv, GRID_WIDTH, GRID_HEIGHT,
        EMPTY, KITCHEN, LIVING_ROOM, HALLWAY
    )
    from agent.q_learning_agent import QLearningAgent
    from agent.sarsa_agent import SarsaAgent
    from agent.dqn_agent import DQNAgent
    from utils.helpers import format_time, format_duration, print_header, Timer
except ImportError as e:
    print(f"\n  ERROR: Could not import required modules: {e}")
    print("  Make sure you're running from the project root directory.")
    sys.exit(1)

# Matplotlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = "models"
PLOTS_DIR  = "plots"

# --- Shared hyper-parameters (tabular agents) --------------------------------
DEFAULT_EPISODES      = 5000
DEFAULT_LEARNING_RATE = 0.15
DEFAULT_DISCOUNT      = 0.99
DEFAULT_EPS_START     = 1.0
DEFAULT_EPS_END       = 0.02
DEFAULT_EPS_DECAY     = 0.998

# --- DQN-specific defaults ---------------------------------------------------
DQN_LEARNING_RATE    = 0.001
DQN_EPS_DECAY        = 0.9984    # reaches eps_end ≈ 0.02 in ~5000 episodes
DQN_DEFAULT_EPISODES = 5000
DQN_BATCH_SIZE       = 64
DQN_MEMORY_SIZE      = 50000
DQN_TARGET_UPDATE    = 250
DQN_TRAIN_EVERY      = 1
DQN_INPUT_SIZE       = 40        # 2 position + 23 dirt + 5 history + 10 DNUT

# --- File paths ---------------------------------------------------------------
QL_MODEL_PATH      = os.path.join(MODELS_DIR, "q_table.pkl")
SARSA_MODEL_PATH   = os.path.join(MODELS_DIR, "sarsa_table.pkl")
DQN_MODEL_PATH     = os.path.join(MODELS_DIR, "dqn_model.pth")

QL_HISTORY_PATH    = os.path.join(MODELS_DIR, "ql_history.pkl")
SARSA_HISTORY_PATH = os.path.join(MODELS_DIR, "sarsa_history.pkl")
DQN_HISTORY_PATH   = os.path.join(MODELS_DIR, "dqn_history.pkl")

# --- Algorithm info registry --------------------------------------------------
ALGO_INFO = {
    "qlearning": {"label": "Q-Learning",  "model": QL_MODEL_PATH,
                   "history": QL_HISTORY_PATH,    "color": "#2ECC71"},
    "sarsa":     {"label": "SARSA",        "model": SARSA_MODEL_PATH,
                   "history": SARSA_HISTORY_PATH, "color": "#E74C3C"},
    "dqn":       {"label": "DQN",          "model": DQN_MODEL_PATH,
                   "history": DQN_HISTORY_PATH,   "color": "#3498DB"},
}

# Room colours for matplotlib grid (RGB 0-1)
_ROOM_COLORS = {
    EMPTY:       [0.31, 0.31, 0.31],
    KITCHEN:     [1.00, 1.00, 0.70],
    LIVING_ROOM: [0.70, 0.78, 1.00],
    HALLWAY:     [0.78, 0.78, 0.78],
}

ACTION_NAMES = {0: "Forward", 1: "Backward", 2: "Left", 3: "Right",
                4: "Wait", 5: "Clean"}


# ============================================================================
# UTILITY HELPERS
# ============================================================================

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def wait_for_enter(msg="Press Enter to continue..."):
    input(f"\n  {msg}")


def get_int(prompt, default, lo=1, hi=100000):
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"    Please enter a value between {lo} and {hi}")
        except ValueError:
            print("    Please enter a valid integer")


def yn(prompt, default="y"):
    ds = "Y/n" if default == "y" else "y/N"
    while True:
        raw = input(f"  {prompt} [{ds}]: ").strip().lower()
        if raw == "":
            return default == "y"
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("    Enter y or n")


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _algo_label(algo):
    return ALGO_INFO[algo]["label"]


def _algo_model_path(algo):
    return ALGO_INFO[algo]["model"]


def _algo_history_path(algo):
    return ALGO_INFO[algo]["history"]


def _algo_color(algo):
    return ALGO_INFO[algo]["color"]


# ============================================================================
# DQN FEATURE EXTRACTION
# ============================================================================

def _env_to_features(env):
    """
    Extract a feature vector from the current environment state for DQN.

        Feature vector (40 elements):
      [0]     : robot_row  (normalised 0-1)
      [1]     : robot_col  (normalised 0-1)
      [2:25]  : dirt status of each of the 23 cleanable tiles (0.0 or 1.0)
            [25:30] : one-hot encoded movement history (came_from direction)
            [30:40] : one-hot encoded DNUT direction to nearest dirty tile

    Parameters
    ----------
    env : CleaningEnv
        Environment instance (after reset or step).

    Returns
    -------
    numpy.ndarray, shape (40,), dtype float32
    """
    features = np.zeros(DQN_INPUT_SIZE, dtype=np.float32)
    features[0] = env.robot_row / max(GRID_HEIGHT - 1, 1)
    features[1] = env.robot_col / max(GRID_WIDTH - 1, 1)

    for i, (row, col) in enumerate(env.cleanable_tiles):
        features[2 + i] = 1.0 if env.dirt_map[row][col] == 1 else 0.0

    # Movement history: 5 bins (N, S, E, W, none)
    features[25 + int(env.last_direction)] = 1.0

    # Direction to nearest dirty tile: 10 bins (3x3 relative direction + none)
    dnut_direction = int(env._get_nearest_dirty_direction())
    features[30 + dnut_direction] = 1.0

    return features


# ============================================================================
# BANNER / MENU
# ============================================================================

def print_banner():
    print("\n")
    print("+" + "=" * 66 + "+")
    print("|                                                                  |")
    print("|     CLEANING ROBOT - Reinforcement Learning Comparison           |")
    print("|                                                                  |")
    print("|  Q-Learning (off-policy) . SARSA (on-policy) . DQN (deep RL)    |")
    print("|                                                                  |")
    print("+" + "=" * 66 + "+")


def print_menu():
    print("\n  +--------------------------------------------------+")
    print("  |                   MAIN MENU                      |")
    print("  +--------------------------------------------------+")
    print("  |  --- TRAINING ---                                |")
    print("  |  [1]  Train Q-Learning Agent                     |")
    print("  |  [2]  Train SARSA Agent                          |")
    print("  |  [3]  Train DQN Agent                            |")
    print("  |  --- TESTING (Pygame UI) ---                     |")
    print("  |  [4]  Test  Q-Learning Agent                     |")
    print("  |  [5]  Test  SARSA Agent                          |")
    print("  |  [6]  Test  DQN Agent                            |")
    print("  |  --- OPTIMAL PATHS ---                           |")
    print("  |  [7]  Show Optimal Path - Q-Learning             |")
    print("  |  [8]  Show Optimal Path - SARSA                  |")
    print("  |  [9]  Show Optimal Path - DQN                    |")
    print("  |  --- COMPARISON ---                              |")
    print("  |  [10] Compare All Algorithms (Dashboard)         |")
    print("  |  [11] Quick Train & Compare All                  |")
    print("  |  [0]  Exit                                       |")
    print("  +--------------------------------------------------+")


# ############################################################################
#                            AGENT FACTORY
# ############################################################################

def _make_agent(algo, env):
    """
    Instantiate the correct agent class.

    For tabular agents (Q-Learning, SARSA): uses env.observation_space.n
    For DQN: uses DQN_INPUT_SIZE (25-dim feature vector)
    """
    action_size = env.action_space.n

    if algo == "dqn":
        return DQNAgent(
            input_size=DQN_INPUT_SIZE,
            action_size=action_size,
            learning_rate=DQN_LEARNING_RATE,
            discount_factor=DEFAULT_DISCOUNT,
            epsilon_start=DEFAULT_EPS_START,
            epsilon_end=DEFAULT_EPS_END,
            epsilon_decay=DQN_EPS_DECAY,
            batch_size=DQN_BATCH_SIZE,
            memory_size=DQN_MEMORY_SIZE,
            target_update=DQN_TARGET_UPDATE,
            train_every=DQN_TRAIN_EVERY,
        )

    cls = QLearningAgent if algo == "qlearning" else SarsaAgent
    return cls(
        state_size=env.observation_space.n,
        action_size=action_size,
        learning_rate=DEFAULT_LEARNING_RATE,
        discount_factor=DEFAULT_DISCOUNT,
        epsilon_start=DEFAULT_EPS_START,
        epsilon_end=DEFAULT_EPS_END,
        epsilon_decay=DEFAULT_EPS_DECAY,
    )


# ############################################################################
#                            TRAINING
# ############################################################################

def train(algo="qlearning", num_episodes=None, render_every=0,
          save_path=None, silent=False):
    """
    Train a Q-Learning, SARSA, or DQN agent.

    Training loop differences
    -------------------------
    Q-Learning: learn(s, a, r, s', done)         -> max Q(s',.)
    SARSA:      learn(s, a, r, s', a', done)     -> Q(s', a')
    DQN:        learn(feat, a, r, feat', done)   -> target network

    Returns dict with full training history.
    """
    label = _algo_label(algo)
    if save_path is None:
        save_path = _algo_model_path(algo)
    if num_episodes is None:
        num_episodes = DQN_DEFAULT_EPISODES if algo == "dqn" else DEFAULT_EPISODES

    if not silent:
        print("\n" + "=" * 70)
        print(f"  TRAINING {label.upper()} CLEANING ROBOT")
        print("=" * 70)
        print(f"  Episodes:    {num_episodes}")
        print(f"  Algorithm:   {label}")
        print(f"  Save path:   {save_path}")
        print(f"  Start time:  {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 70)

    # --- Environment & Agent ------------------------------------------------
    env = CleaningEnv(render_mode=None)
    env_visual = CleaningEnv(render_mode="human") if render_every > 0 else None
    agent = _make_agent(algo, env)

    # --- Tracking lists ------------------------------------------------------
    ep_rewards, ep_tiles, ep_steps = [], [], []
    eps_history, success_history = [], []
    best_reward = float("-inf")
    total_successes = 0
    start = time.time()

    # --- Main training loop --------------------------------------------------
    for ep in range(1, num_episodes + 1):
        should_render = render_every > 0 and ep % render_every == 0
        cur_env = env_visual if should_render else env

        state, info = cur_env.reset()
        ep_reward = 0
        steps = 0
        done = False

        # Algorithm-specific initialisation before inner loop
        if algo == "dqn":
            features = _env_to_features(cur_env)
        elif algo == "sarsa":
            action = agent.choose_action(state, training=True)

        while not done:
            # --- Choose action -----------------------------------------------
            if algo == "dqn":
                action = agent.choose_action(features, training=True)
            elif algo == "qlearning":
                action = agent.choose_action(state, training=True)
            # SARSA: action was already chosen (before loop or at end of prev iter)

            next_state, reward, terminated, truncated, info = cur_env.step(action)
            done = terminated or truncated

            # --- Update agent ------------------------------------------------
            if algo == "dqn":
                next_features = _env_to_features(cur_env)
                agent.learn(features, action, reward, next_features, done)
                features = next_features
            elif algo == "qlearning":
                agent.learn(state, action, reward, next_state, done)
            else:  # sarsa
                next_action = agent.choose_action(next_state, training=True)
                agent.learn(state, action, reward, next_state, next_action, done)
                action = next_action  # carry forward

            ep_reward += reward
            steps += 1
            state = next_state

            if should_render:
                cur_env.render()

        # End-of-episode bookkeeping
        ep_rewards.append(ep_reward)
        ep_tiles.append(info["tiles_cleaned"])
        ep_steps.append(steps)
        eps_history.append(agent.epsilon)
        success = info["tiles_cleaned"] == env.num_cleanable
        success_history.append(success)
        if success:
            total_successes += 1
        if ep_reward > best_reward:
            best_reward = ep_reward
        agent.end_episode()

        # Progress print (DQN every 10 eps since it's slower per episode)
        print_interval = 10 if algo == "dqn" else 100
        if not silent and (ep % print_interval == 0 or ep == 1):
            r100 = ep_rewards[-100:]
            t100 = ep_tiles[-100:]
            s100 = success_history[-100:]
            print(f"  Ep {ep:5d}/{num_episodes} | "
                  f"Reward:{np.mean(r100):7.1f} | "
                  f"Tiles:{np.mean(t100):5.1f}/{env.num_cleanable} | "
                  f"Succ:{sum(s100)/len(s100)*100:5.1f}% | "
                  f"eps:{agent.epsilon:.4f} | "
                  f"T:{format_time(time.time()-start)}")

    elapsed = time.time() - start

    # --- Summary -------------------------------------------------------------
    last_r = ep_rewards[-100:]
    last_t = ep_tiles[-100:]
    last_s = success_history[-100:]

    if not silent:
        print("\n" + "=" * 70)
        print(f"  {label.upper()} TRAINING COMPLETE!")
        print("=" * 70)
        print(f"  Avg Reward (last 100):  {np.mean(last_r):.1f}")
        print(f"  Avg Tiles  (last 100):  {np.mean(last_t):.1f}/{env.num_cleanable}")
        print(f"  Success %  (last 100):  {sum(last_s)/len(last_s)*100:.1f}%")
        print(f"  Best Reward:            {best_reward:.1f}")
        if algo == "dqn":
            print(f"  Network params:         {agent.num_parameters}")
        else:
            print(f"  Q-table entries:        {len(agent.q_table)}")
        print(f"  Total time:             {format_time(elapsed)}")

    # --- Save model ----------------------------------------------------------
    _ensure_dirs()
    agent.save(save_path)

    # --- Save training history -----------------------------------------------
    history = {
        "algo": algo,
        "rewards": ep_rewards,
        "tiles": ep_tiles,
        "steps": ep_steps,
        "epsilon": eps_history,
        "success": success_history,
        "num_episodes": num_episodes,
        "elapsed": elapsed,
    }
    hist_path = _algo_history_path(algo)
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    if not silent:
        print(f"  History saved to: {hist_path}")

    env.close()
    if env_visual:
        env_visual.close()

    return history


# ############################################################################
#                             TESTING  (Pygame UI)
# ############################################################################

def test_agent_ui(algo="qlearning", num_episodes=10, speed="normal"):
    """Run the trained agent with Pygame visualisation + random baseline."""
    label = _algo_label(algo)
    model_path = _algo_model_path(algo)

    print("\n" + "=" * 70)
    print(f"  TESTING {label.upper()} CLEANING ROBOT (Pygame UI)")
    print("=" * 70)

    speed_map = {"slow": 5, "normal": 10, "fast": 20}
    fps = speed_map.get(speed, 10)

    if not os.path.exists(model_path):
        print(f"\n  ERROR: Model not found -> {model_path}")
        print("  Train the agent first.")
        return None

    # --- Load agent ----------------------------------------------------------
    env = CleaningEnv(render_mode="human")
    env.metadata["render_fps"] = fps
    agent = _make_agent(algo, env)
    agent.load(model_path)

    if algo == "dqn":
        print(f"  Model loaded - {agent.num_parameters} params, eps={agent.epsilon:.4f}")
    else:
        print(f"  Model loaded - {len(agent.q_table)} Q-entries, eps={agent.epsilon:.4f}")
    print(f"  Speed: {speed} ({fps} FPS)")
    print("-" * 70)

    # --- Trained agent episodes ----------------------------------------------
    rewards, tiles, steps_all, successes = [], [], [], []

    for ep in range(1, num_episodes + 1):
        state, info = env.reset()
        ep_reward, steps, done = 0, 0, False

        while not done:
            if algo == "dqn":
                features = _env_to_features(env)
                action = agent.choose_action(features, training=False)
            else:
                action = agent.choose_action(state, training=False)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
            state = next_state
            env.render()

        ok = info["tiles_cleaned"] == env.num_cleanable
        rewards.append(ep_reward)
        tiles.append(info["tiles_cleaned"])
        steps_all.append(steps)
        successes.append(ok)
        tag = "SUCCESS" if ok else "INCOMPLETE"
        print(f"  Ep {ep:3d}: Reward={ep_reward:7.1f} | "
              f"Tiles={info['tiles_cleaned']:2d}/{env.num_cleanable} | "
              f"Steps={steps:3d} | {tag}")

    env.close()

    # --- Random baseline (headless) ------------------------------------------
    print("\n  Running random baseline...")
    env2 = CleaningEnv(render_mode=None)
    rand_rew, rand_til, rand_suc = [], [], []
    for _ in range(num_episodes):
        s, _ = env2.reset()
        r_sum, d = 0, False
        while not d:
            s, r, term, trunc, info2 = env2.step(env2.action_space.sample())
            r_sum += r
            d = term or trunc
        rand_rew.append(r_sum)
        rand_til.append(info2["tiles_cleaned"])
        rand_suc.append(info2["tiles_cleaned"] == env2.num_cleanable)
    env2.close()

    # --- Results summary -----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  {label.upper()} TEST RESULTS")
    print("=" * 70)
    print(f"  {'Metric':<22} {'Trained':>12}  {'Random':>12}")
    print("  " + "-" * 50)
    print(f"  {'Avg Reward':<22} {np.mean(rewards):>12.1f}  {np.mean(rand_rew):>12.1f}")
    print(f"  {'Avg Tiles':<22} {np.mean(tiles):>12.1f}  {np.mean(rand_til):>12.1f}")
    print(f"  {'Success %':<22} {sum(successes)/len(successes)*100:>12.1f}  "
          f"{sum(rand_suc)/len(rand_suc)*100:>12.1f}")
    print(f"  {'Avg Steps':<22} {np.mean(steps_all):>12.1f}  {'---':>12}")
    print()

    rate = sum(successes) / len(successes) * 100
    if   rate >= 90: grade = "***** EXCELLENT"
    elif rate >= 70: grade = "****  GOOD"
    elif rate >= 50: grade = "***   MODERATE"
    else:            grade = "**    NEEDS MORE TRAINING"
    print(f"  {grade}")
    print("=" * 70)

    return {
        "trained_rewards": rewards, "trained_tiles": tiles,
        "trained_steps": steps_all, "trained_success": successes,
        "random_rewards": rand_rew, "random_tiles": rand_til,
        "random_success": rand_suc,
    }


# ############################################################################
#                 OPTIMAL PATH EXTRACTION & GRID VISUALISATION
# ############################################################################

def extract_optimal_path(algo="qlearning"):
    """Run one deterministic greedy episode and return the visited path."""
    model_path = _algo_model_path(algo)
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return None

    env = CleaningEnv(render_mode=None)
    agent = _make_agent(algo, env)
    agent.load(model_path)

    state, info = env.reset()
    path = [(env.robot_row, env.robot_col)]
    actions_taken = []
    done = False

    while not done:
        if algo == "dqn":
            features = _env_to_features(env)
            action = agent.choose_action(features, training=False, eval_epsilon=0.0)
        else:
            action = agent.choose_action(state, training=False, eval_epsilon=0.0)
        actions_taken.append(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        path.append((env.robot_row, env.robot_col))
        state = next_state

    env.close()
    return {
        "path": path,
        "actions": actions_taken,
        "tiles_cleaned": info["tiles_cleaned"],
        "total_tiles": env.num_cleanable,
        "steps": len(actions_taken),
    }


def _build_room_grid():
    """Return an RGB grid array coloured by room type and the raw layout."""
    env = CleaningEnv(render_mode=None)
    layout = env.room_layout
    env.close()

    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3))
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            grid[r, c] = _ROOM_COLORS.get(layout[r][c], [0.31, 0.31, 0.31])
    return grid, layout


def visualize_path_on_grid(path_info, title="Optimal Path", ax=None, save=None):
    """
    Draw the agent's cleaning path on a coloured house grid.

    Parameters
    ----------
    path_info : dict   (from extract_optimal_path)
    title     : str
    ax        : matplotlib Axes or None
    save      : file path or None
    """
    if path_info is None:
        print("  No path data to visualise.")
        return

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))

    grid, layout = _build_room_grid()
    ax.imshow(grid, origin="upper",
              extent=(-0.5, GRID_WIDTH - 0.5, GRID_HEIGHT - 0.5, -0.5))

    # Draw grid lines
    for r in range(GRID_HEIGHT + 1):
        ax.axhline(r - 0.5, color="black", linewidth=0.5, alpha=0.4)
    for c in range(GRID_WIDTH + 1):
        ax.axvline(c - 0.5, color="black", linewidth=0.5, alpha=0.4)

    # Room labels
    _room_label(ax, layout)

    path = path_info["path"]

    # Draw arrows between consecutive unique positions
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        if r0 == r1 and c0 == c1:
            continue
        ax.annotate("",
                     xy=(c1, r1), xytext=(c0, r0),
                     arrowprops=dict(arrowstyle="->", color="#E74C3C",
                                     lw=1.8, alpha=0.65))

    # Number the first visit to each cell
    visited = {}
    for i, (r, c) in enumerate(path):
        if (r, c) not in visited:
            visited[(r, c)] = i

    for (r, c), step in visited.items():
        ax.plot(c, r, "o", color="#2ECC71", markersize=14, zorder=5,
                markeredgecolor="black", markeredgewidth=0.8)
        ax.text(c, r, str(step), ha="center", va="center",
                fontsize=7, fontweight="bold", color="black", zorder=6)

    # Start (S) and End (E) markers
    sr, sc = path[0]
    er, ec = path[-1]
    ax.plot(sc, sr, "s", color="#F1C40F", markersize=18, zorder=7,
            markeredgecolor="black", markeredgewidth=1.5)
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=9,
            fontweight="bold", zorder=8)
    ax.plot(ec, er, "D", color="#3498DB", markersize=16, zorder=7,
            markeredgecolor="black", markeredgewidth=1.5)
    ax.text(ec, er, "E", ha="center", va="center", fontsize=8,
            fontweight="bold", color="white", zorder=8)

    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(GRID_HEIGHT - 0.5, -0.5)
    ax.set_xticks(range(GRID_WIDTH))
    ax.set_yticks(range(GRID_HEIGHT))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    tc = path_info["tiles_cleaned"]
    tt = path_info["total_tiles"]
    steps = path_info["steps"]
    ax.set_title(f"{title}\nTiles: {tc}/{tt}  |  Steps: {steps}", fontsize=12)

    # Legend
    patches = [
        mpatches.Patch(color=_ROOM_COLORS[KITCHEN],     label="Kitchen"),
        mpatches.Patch(color=_ROOM_COLORS[LIVING_ROOM], label="Living Room"),
        mpatches.Patch(color=_ROOM_COLORS[HALLWAY],     label="Hallway"),
        mpatches.Patch(color=_ROOM_COLORS[EMPTY],       label="Wall"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    if standalone:
        plt.tight_layout()
        if save:
            _ensure_dirs()
            plt.savefig(save, dpi=150, bbox_inches="tight")
            print(f"  Saved to {save}")
        plt.show()


def _room_label(ax, layout):
    """Put room names at the visual centre of each room."""
    from collections import defaultdict
    cells = defaultdict(list)
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            rt = layout[r][c]
            if rt != EMPTY:
                cells[rt].append((r, c))
    names = {KITCHEN: "Kitchen", LIVING_ROOM: "Living\nRoom", HALLWAY: "Hallway"}
    for rt, pos_list in cells.items():
        rows = [p[0] for p in pos_list]
        cols = [p[1] for p in pos_list]
        cr = (min(rows) + max(rows)) / 2
        cc = (min(cols) + max(cols)) / 2
        ax.text(cc, cr, names.get(rt, ""), ha="center", va="center",
                fontsize=9, fontweight="bold", color="#555555", alpha=0.35)


# ############################################################################
#                    ALGORITHM COMPARISON DASHBOARD
# ############################################################################

def comparison_dashboard(histories=None):
    """
    Smart comparison dashboard for Q-Learning, SARSA & DQN.

    Parameters
    ----------
    histories : dict or None
        Maps algo key -> training history dict.
        If None, loads from saved history files.

    Generates:
      Figure 1: Learning curves (2x2) — reward, tiles, success %, epsilon
                with variance bands + convergence markers
      Figure 2: Side-by-side optimal paths on the house grid
      Figure 3: Smart performance bars (6 metrics) + winner badges
      Figure 4: Convergence & efficiency analysis (convergence speed,
                reward stability, training efficiency, radar chart)
      Console:  Algorithm characteristics, full summary table,
                strengths/weaknesses, overall winner
    """
    # --- Collect histories ---------------------------------------------------
    algos = {}
    for key, info in ALGO_INFO.items():
        if histories and key in histories:
            algos[key] = histories[key]
        elif os.path.exists(info["history"]):
            with open(info["history"], "rb") as f:
                algos[key] = pickle.load(f)

    if len(algos) < 2:
        print("  Need at least 2 trained algorithms to compare.")
        print("  Train more algorithms first.")
        return

    algo_keys = list(algos.keys())
    n_algos = len(algo_keys)

    print("\n" + "=" * 70)
    print("  SMART ALGORITHM COMPARISON DASHBOARD")
    print(f"  Comparing: {', '.join(_algo_label(k) for k in algo_keys)}")
    print("=" * 70)
    _ensure_dirs()

    # ----------------------------------------------------------------
    # DERIVED METRICS (computed once, reused everywhere)
    # ----------------------------------------------------------------
    def smooth(data, w=100):
        out = []
        for i in range(len(data)):
            s = max(0, i - w + 1)
            out.append(np.mean(data[s:i + 1]))
        return out

    def rolling_std(data, w=100):
        out = []
        for i in range(len(data)):
            s = max(0, i - w + 1)
            out.append(np.std(data[s:i + 1]))
        return out

    def _convergence_episode(success_list, threshold=0.90, window=100):
        """Episode number where rolling success rate first exceeds threshold."""
        for i in range(window, len(success_list)):
            rate = sum(success_list[i - window:i]) / window
            if rate >= threshold:
                return i
        return None  # never converged

    # Pre-compute per-algorithm stats
    stats = {}
    for k in algo_keys:
        h = algos[k]
        last = min(100, len(h["rewards"]))
        r_last = h["rewards"][-last:]
        t_last = h["tiles"][-last:]
        s_last = h["success"][-last:]
        st_last = h["steps"][-last:]
        stats[k] = {
            "avg_reward":       np.mean(r_last),
            "avg_tiles":        np.mean(t_last),
            "success_rate":     sum(s_last) / len(s_last) * 100,
            "avg_steps":        np.mean(st_last),
            "reward_std":       np.std(r_last),
            "best_reward":      max(h["rewards"]),
            "worst_reward":     min(h["rewards"]),
            "time":             h["elapsed"],
            "episodes":         h["num_episodes"],
            "conv_50":          _convergence_episode(h["success"], 0.50),
            "conv_75":          _convergence_episode(h["success"], 0.75),
            "conv_90":          _convergence_episode(h["success"], 0.90),
            "conv_100":         _convergence_episode(h["success"], 0.99),
            "reward_per_sec":   np.mean(r_last) / max(h["elapsed"], 0.1),
            "sample_eff":       np.mean(r_last) / max(h["num_episodes"], 1),
        }

    # ====================================================================
    # FIGURE 1 - Learning Curves (2x2) with variance bands + convergence
    # ====================================================================
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig1.suptitle("Learning Curves Comparison — Q-Learning vs SARSA vs DQN",
                  fontsize=15, fontweight="bold")

    for key in algo_keys:
        hist = algos[key]
        c = _algo_color(key)
        lbl = f"{_algo_label(key)} ({hist['num_episodes']} ep)"

        # Rewards with variance band
        sm_r = smooth(hist["rewards"])
        std_r = rolling_std(hist["rewards"])
        x = np.arange(len(sm_r))
        sm_r_arr = np.array(sm_r)
        std_r_arr = np.array(std_r)
        axes[0, 0].plot(x, sm_r, color=c, lw=2, label=lbl)
        axes[0, 0].fill_between(x, sm_r_arr - std_r_arr,
                                sm_r_arr + std_r_arr,
                                color=c, alpha=0.10)

        # Tiles
        sm_t = smooth(hist["tiles"])
        std_t = rolling_std(hist["tiles"])
        sm_t_arr = np.array(sm_t)
        std_t_arr = np.array(std_t)
        axes[0, 1].plot(x, sm_t, color=c, lw=2, label=lbl)
        axes[0, 1].fill_between(x, sm_t_arr - std_t_arr,
                                sm_t_arr + std_t_arr,
                                color=c, alpha=0.10)

        # Success rate
        sm_s = smooth([int(v) * 100 for v in hist["success"]])
        axes[1, 0].plot(sm_s, color=c, lw=2, label=lbl)

        # Add convergence marker (90% success)
        conv_ep = stats[key]["conv_90"]
        if conv_ep is not None:
            axes[1, 0].axvline(conv_ep, color=c, ls="--", alpha=0.5, lw=1)
            axes[1, 0].plot(conv_ep, 90, marker="*", color=c,
                            markersize=14, zorder=5)
            axes[1, 0].annotate(f"  {_algo_label(key)} @{conv_ep}",
                                xy=(conv_ep, 90), fontsize=8, color=c)

        # Epsilon
        axes[1, 1].plot(hist["epsilon"], color=c, lw=2, label=lbl)

    axes[0, 0].set_ylabel("Avg Reward"); axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_title("Episode Reward (smoothed ± σ)")
    axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_ylabel("Tiles Cleaned"); axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_title("Tiles Cleaned per Episode (± σ)")
    axes[0, 1].legend(fontsize=9); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_ylabel("Success %"); axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_title("Success Rate (★ = 90% convergence point)")
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-5, 105)
    axes[1, 0].axhline(90, color="gray", ls=":", alpha=0.4, lw=1)

    axes[1, 1].set_ylabel("Epsilon"); axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_title("Exploration Rate Decay")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.3)

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    save1 = os.path.join(PLOTS_DIR, "comparison_learning_curves.png")
    fig1.savefig(save1, dpi=150, bbox_inches="tight")
    print(f"  [1/4] Saved learning curves   -> {save1}")

    # ====================================================================
    # FIGURE 2 - Side-by-side Optimal Paths
    # ====================================================================
    fig2, axes2 = plt.subplots(1, n_algos, figsize=(8 * n_algos, 7))
    if n_algos == 1:
        axes2 = [axes2]
    fig2.suptitle("Optimal Cleaning Paths — Q-Learning vs SARSA vs DQN",
                  fontsize=15, fontweight="bold")

    path_infos = {}
    for i, key in enumerate(algo_keys):
        path_info = extract_optimal_path(key)
        path_infos[key] = path_info
        tt = f"{_algo_label(key)} Path"
        if path_info:
            tt += f"\n({path_info['tiles_cleaned']}/{path_info['total_tiles']} tiles, {path_info['steps']} steps)"
        visualize_path_on_grid(path_info, title=tt, ax=axes2[i])

    fig2.tight_layout(rect=[0, 0, 1, 0.91])
    save2 = os.path.join(PLOTS_DIR, "comparison_optimal_paths.png")
    fig2.savefig(save2, dpi=150, bbox_inches="tight")
    print(f"  [2/4] Saved optimal paths     -> {save2}")

    # ====================================================================
    # FIGURE 3 - Smart Performance Bars (6 metrics + winner badges)
    # ====================================================================
    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle("Final Performance Comparison (last 100 episodes)",
                  fontsize=15, fontweight="bold")

    labels = [_algo_label(k) for k in algo_keys]
    colors = [_algo_color(k) for k in algo_keys]

    def _barlabel(ax, bars, vals, fmt="{:.1f}", best_idx=None):
        for idx_b, (bar, v) in enumerate(zip(bars, vals)):
            txt = fmt.format(v)
            if best_idx is not None and idx_b == best_idx:
                txt = f"★ {txt}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(abs(max(vals)) * 0.02, 0.5),
                    txt, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    bar_metrics = [
        ("Avg Reward",       [stats[k]["avg_reward"]   for k in algo_keys], "high", "{:.1f}"),
        ("Avg Tiles Cleaned",[stats[k]["avg_tiles"]    for k in algo_keys], "high", "{:.1f}"),
        ("Success Rate %",   [stats[k]["success_rate"] for k in algo_keys], "high", "{:.1f}%"),
        ("Avg Steps",        [stats[k]["avg_steps"]    for k in algo_keys], "low",  "{:.0f}"),
        ("Reward Stability\n(lower σ = better)",
                             [stats[k]["reward_std"]   for k in algo_keys], "low",  "{:.1f}"),
        ("Training Time (s)",[stats[k]["time"]         for k in algo_keys], "low",  "{:.1f}"),
    ]

    for idx, (title, vals, mode, fmt) in enumerate(bar_metrics):
        r, c = divmod(idx, 3)
        ax = axes3[r, c]
        if mode == "high":
            best_i = int(np.argmax(vals))
        else:
            best_i = int(np.argmin(vals))
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
        # Highlight winner bar with a thicker gold edge
        bars[best_i].set_edgecolor("#FFD700")
        bars[best_i].set_linewidth(2.5)
        _barlabel(ax, bars, vals, fmt=fmt, best_idx=best_i)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if "%" in title:
            ax.set_ylim(0, 115)

    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    save3 = os.path.join(PLOTS_DIR, "comparison_bars.png")
    fig3.savefig(save3, dpi=150, bbox_inches="tight")
    print(f"  [3/4] Saved performance bars  -> {save3}")

    # ====================================================================
    # FIGURE 4 - Convergence & Efficiency Deep-Dive (2x2)
    # ====================================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(16, 11))
    fig4.suptitle("Smart Analysis — Convergence, Efficiency & Radar",
                  fontsize=15, fontweight="bold")

    # --- 4a: Convergence Speed (grouped bar) --------------------------------
    ax4a = axes4[0, 0]
    thresholds = [("50%", "conv_50"), ("75%", "conv_75"),
                  ("90%", "conv_90"), ("100%", "conv_100")]
    x_pos = np.arange(len(thresholds))
    bar_w = 0.8 / n_algos
    for i, key in enumerate(algo_keys):
        vals = []
        for _, skey in thresholds:
            ep = stats[key].get(skey)
            vals.append(ep if ep is not None else 0)
        offset = (i - (n_algos - 1) / 2) * bar_w
        bars = ax4a.bar(x_pos + offset, vals, bar_w, color=_algo_color(key),
                        edgecolor="black", linewidth=0.6, label=_algo_label(key))
        for b, v in zip(bars, vals):
            if v > 0:
                ax4a.text(b.get_x() + b.get_width() / 2, b.get_height() + 10,
                          str(v), ha="center", va="bottom", fontsize=8,
                          fontweight="bold")
            else:
                ax4a.text(b.get_x() + b.get_width() / 2, 5,
                          "N/A", ha="center", va="bottom", fontsize=7,
                          color="red", fontstyle="italic")
    ax4a.set_xticks(x_pos)
    ax4a.set_xticklabels([t[0] for t in thresholds])
    ax4a.set_ylabel("Episodes to Converge")
    ax4a.set_title("Convergence Speed (fewer = faster learner)")
    ax4a.legend(fontsize=9); ax4a.grid(axis="y", alpha=0.3)

    # --- 4b: Reward trajectory with rolling std overlay ---------------------
    ax4b = axes4[0, 1]
    for key in algo_keys:
        h = algos[key]
        r_std = rolling_std(h["rewards"], 200)
        ax4b.plot(r_std, color=_algo_color(key), lw=2, label=_algo_label(key))
    ax4b.set_xlabel("Episode"); ax4b.set_ylabel("Reward Std (σ)")
    ax4b.set_title("Reward Stability Over Time (lower = more consistent)")
    ax4b.legend(fontsize=9); ax4b.grid(True, alpha=0.3)

    # --- 4c: Training Efficiency (reward gained per second of training) -----
    ax4c = axes4[1, 0]
    eff_labels = []
    eff_vals = []
    eff_cols = []
    for key in algo_keys:
        eff_labels.append(_algo_label(key))
        eff_vals.append(stats[key]["reward_per_sec"])
        eff_cols.append(_algo_color(key))
    bars = ax4c.barh(eff_labels, eff_vals, color=eff_cols,
                     edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, eff_vals):
        ax4c.text(b.get_width() + max(eff_vals) * 0.02, b.get_y() + b.get_height() / 2,
                  f"{v:.2f}", ha="left", va="center", fontsize=10,
                  fontweight="bold")
    ax4c.set_xlabel("Avg Reward / Training Second")
    ax4c.set_title("Training Efficiency (higher = better return on compute)")
    ax4c.grid(axis="x", alpha=0.3)

    # --- 4d: Radar / Spider Chart -------------------------------------------
    ax4d = axes4[1, 1]
    ax4d.set_visible(False)               # hide cartesian axes
    ax_radar = fig4.add_subplot(2, 2, 4, polar=True)

    radar_dims = ["Avg Reward", "Tiles Cleaned", "Success %",
                  "Stability", "Speed", "Efficiency"]
    n_dims = len(radar_dims)

    # Raw values per algo for each dimension
    raw = {}
    for key in algo_keys:
        s = stats[key]
        raw[key] = [
            s["avg_reward"],
            s["avg_tiles"],
            s["success_rate"],
            100 - min(s["reward_std"], 100),  # invert: lower std = higher score
            max(0, 300 - s["avg_steps"]),     # invert: fewer steps = higher
            s["reward_per_sec"],
        ]

    # Normalise each dimension to 0-1 (across algos)
    norm = {}
    for key in algo_keys:
        norm[key] = []
    for d in range(n_dims):
        col_vals = [raw[k][d] for k in algo_keys]
        lo, hi = min(col_vals), max(col_vals)
        rng = hi - lo if hi != lo else 1.0
        for key in algo_keys:
            norm[key].append((raw[key][d] - lo) / rng * 0.8 + 0.2)  # map to 0.2-1.0

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for key in algo_keys:
        values = norm[key] + norm[key][:1]
        ax_radar.plot(angles, values, "o-", color=_algo_color(key), lw=2,
                      label=_algo_label(key), markersize=5)
        ax_radar.fill(angles, values, color=_algo_color(key), alpha=0.10)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_dims, fontsize=9)
    ax_radar.set_ylim(0, 1.1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
    ax_radar.set_title("Multi-Dimension Radar\n(normalised per metric)",
                       fontsize=11, fontweight="bold", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    save4 = os.path.join(PLOTS_DIR, "comparison_smart_analysis.png")
    fig4.savefig(save4, dpi=150, bbox_inches="tight")
    print(f"  [4/4] Saved smart analysis    -> {save4}")

    # ====================================================================
    # CONSOLE: Algorithm Characteristics
    # ====================================================================
    col_w = 16

    print("\n" + "=" * 70)
    print("  ALGORITHM CHARACTERISTICS")
    print("=" * 70)

    # Type row
    algo_types = {
        "qlearning": "Tabular/Off-policy",
        "sarsa":     "Tabular/On-policy",
        "dqn":       "Deep NN/Off-policy",
    }
    header = f"  {'Property':<24}"
    for k in algo_keys:
        header += f" {_algo_label(k):>{col_w}}"
    print(header)
    print("  " + "-" * (24 + col_w * n_algos))

    char_rows = [
        ("Type",         lambda k: algo_types.get(k, "?")),
        ("Episodes",     lambda k: str(stats[k]["episodes"])),
        ("Training Time",lambda k: f"{stats[k]['time']:.1f}s"),
    ]

    # Model size
    def _model_size(algo_key):
        mp = _algo_model_path(algo_key)
        if not os.path.exists(mp):
            return "N/A"
        try:
            if algo_key == "dqn":
                import torch
                d = torch.load(mp, map_location="cpu", weights_only=False)
                n = sum(v.numel() for v in d["policy_net"].values())
                return f"{n} params"
            else:
                with open(mp, "rb") as f:
                    d = pickle.load(f)
                return f"{len(d.get('q_table', {}))} entries"
        except Exception:
            return "?"

    char_rows.append(("Model Size", lambda k: _model_size(k)))
    char_rows.append(("Update Rule",
                      lambda k: {"qlearning": "max Q(s',·)",
                                 "sarsa": "Q(s',a')",
                                 "dqn": "target net"}.get(k, "?")))

    for name, fn in char_rows:
        row = f"  {name:<24}"
        for k in algo_keys:
            row += f" {fn(k):>{col_w}}"
        print(row)

    # ====================================================================
    # CONSOLE: Full Summary Table with Winners
    # ====================================================================
    print("\n" + "=" * 70)
    print("  PERFORMANCE SUMMARY (last 100 episodes)")
    print("=" * 70)

    header = f"  {'Metric':<28}"
    for k in algo_keys:
        header += f" {_algo_label(k):>{col_w}}"
    header += f" {'Winner':>12}"
    print(header)
    print("  " + "-" * (28 + col_w * n_algos + 14))

    table_rows = [
        ("Avg Reward",         "avg_reward",   "high", "{:.1f}"),
        ("Avg Tiles Cleaned",  "avg_tiles",    "high", "{:.1f}"),
        ("Success Rate %",     "success_rate", "high", "{:.1f}"),
        ("Avg Steps",          "avg_steps",    "low",  "{:.1f}"),
        ("Reward Stability (σ)","reward_std",  "low",  "{:.1f}"),
        ("Best Reward (ever)", "best_reward",  "high", "{:.1f}"),
        ("Training Time (s)",  "time",         "low",  "{:.1f}"),
        ("Reward / Second",    "reward_per_sec","high", "{:.2f}"),
        ("Converge @90%",      "conv_90",      "low",  "{}"),
    ]

    win_count = {k: 0 for k in algo_keys}

    for name, skey, mode, fmt in table_rows:
        vals = {}
        for k in algo_keys:
            v = stats[k][skey]
            vals[k] = v if v is not None else float("inf") if mode == "low" else float("-inf")

        numeric_vals = {k: v for k, v in vals.items()
                        if v is not None and v != float("inf") and v != float("-inf")}
        if not numeric_vals:
            winner_str = "---"
        else:
            if mode == "high":
                best_val = max(numeric_vals.values())
            else:
                best_val = min(numeric_vals.values())
            winners = [k for k, v in numeric_vals.items()
                       if abs(v - best_val) / (abs(best_val) + 1e-9) < 0.02]
            if len(winners) >= n_algos:
                winner_str = "Tie"
            elif len(winners) == 1:
                winner_str = _algo_label(winners[0])
                win_count[winners[0]] += 1
            else:
                winner_str = "Tie"

        row_str = f"  {name:<28}"
        for k in algo_keys:
            v = stats[k][skey]
            if v is None:
                row_str += f" {'N/A':>{col_w}}"
            else:
                row_str += f" {fmt.format(v):>{col_w}}"
        row_str += f" {winner_str:>12}"
        print(row_str)

    # ====================================================================
    # CONSOLE: Strengths / Weaknesses
    # ====================================================================
    print("\n" + "=" * 70)
    print("  STRENGTHS & OBSERVATIONS")
    print("=" * 70)

    algo_traits = {
        "qlearning": {
            "strengths": ["Off-policy: learns optimal policy while exploring",
                          "Simple tabular lookup — fast per-step inference",
                          "Guaranteed convergence with sufficient exploration"],
            "weakness":  "Cannot generalise to unseen states (tabular)",
        },
        "sarsa": {
            "strengths": ["On-policy: learns the policy it actually follows",
                          "Safer behaviour — respects exploration risk",
                          "Good baseline for policy comparison"],
            "weakness":  "Conservative — may converge to suboptimal if exploration is cautious",
        },
        "dqn": {
            "strengths": ["Neural network — can generalise across states",
                          "Experience replay breaks correlation in data",
                          "Target network stabilises training"],
            "weakness":  "More compute, slower training, potential instability",
        },
    }

    for key in algo_keys:
        label = _algo_label(key)
        traits = algo_traits.get(key, {})
        print(f"\n  [{label}]")
        for s in traits.get("strengths", []):
            print(f"    + {s}")
        w = traits.get("weakness", "")
        if w:
            print(f"    - {w}")

        # Data-driven observation
        s = stats[key]
        if s["conv_90"] is not None:
            print(f"    > Reached 90% success at episode {s['conv_90']}")
        else:
            print(f"    > Did NOT reach 90% success rate")
        print(f"    > Final reward: {s['avg_reward']:.1f} ± {s['reward_std']:.1f}")

    # ====================================================================
    # CONSOLE: Overall Winner
    # ====================================================================
    print("\n" + "=" * 70)
    print("  OVERALL WINNER")
    print("=" * 70)

    # Determine winner: most metrics won
    max_wins = max(win_count.values())
    overall_winners = [k for k, v in win_count.items() if v == max_wins]

    header2 = "  Win count: "
    for k in algo_keys:
        header2 += f" {_algo_label(k)}={win_count[k]} "
    print(header2)

    if len(overall_winners) == 1:
        ow = overall_winners[0]
        print(f"\n  >>> {_algo_label(ow)} wins {win_count[ow]}/{len(table_rows)} metrics! <<<")
    elif len(overall_winners) == n_algos:
        print(f"\n  >>> It's a complete TIE — all algorithms perform comparably <<<")
    else:
        tied = " & ".join(_algo_label(k) for k in overall_winners)
        print(f"\n  >>> TIE between {tied} ({max_wins} wins each) <<<")

    # Quick verdict
    fastest_conv = None
    fastest_ep = float("inf")
    for k in algo_keys:
        ep = stats[k]["conv_90"]
        if ep is not None and ep < fastest_ep:
            fastest_ep = ep
            fastest_conv = k
    if fastest_conv:
        print(f"  Fastest to 90% success: {_algo_label(fastest_conv)} "
              f"(ep {fastest_ep})")

    best_reward_key = max(algo_keys, key=lambda k: stats[k]["avg_reward"])
    print(f"  Highest avg reward:     {_algo_label(best_reward_key)} "
          f"({stats[best_reward_key]['avg_reward']:.1f})")

    most_stable = min(algo_keys, key=lambda k: stats[k]["reward_std"])
    print(f"  Most stable:            {_algo_label(most_stable)} "
          f"(σ = {stats[most_stable]['reward_std']:.1f})")

    print("=" * 70)

    # Show all plots
    plt.show()


# ############################################################################
#                          MENU HANDLERS
# ############################################################################

def menu_train(algo):
    label = _algo_label(algo)
    print_header(f"TRAIN {label.upper()} AGENT", width=60)
    default_ep = DQN_DEFAULT_EPISODES if algo == "dqn" else DEFAULT_EPISODES
    eps = get_int("Number of episodes", default_ep, 100, 50000)
    render = get_int("Render every N episodes (0=never)", 0, 0, eps)
    train(algo=algo, num_episodes=eps, render_every=render)
    wait_for_enter()


def menu_test(algo):
    label = _algo_label(algo)
    print_header(f"TEST {label.upper()} AGENT", width=60)
    eps = get_int("Number of test episodes", 10, 1, 200)
    sp = input("  Speed [slow/normal/fast] (normal): ").strip().lower() or "normal"
    test_agent_ui(algo=algo, num_episodes=eps, speed=sp)
    wait_for_enter()


def menu_show_path(algo):
    label = _algo_label(algo)
    print_header(f"OPTIMAL PATH - {label.upper()}", width=60)
    p = extract_optimal_path(algo)
    if p:
        print(f"  Tiles cleaned: {p['tiles_cleaned']}/{p['total_tiles']}")
        print(f"  Steps taken:   {p['steps']}")
        save_file = os.path.join(PLOTS_DIR, f"optimal_path_{algo}.png")
        visualize_path_on_grid(p, title=f"{label} Optimal Path", save=save_file)
    wait_for_enter()


def menu_compare():
    print_header("COMPARE ALL ALGORITHMS", width=60)
    comparison_dashboard()
    wait_for_enter()


def menu_quick():
    print_header("QUICK TRAIN & COMPARE ALL", width=60)
    tab_eps = get_int("Episodes for Q-Learning & SARSA", 3000, 500, 50000)
    dqn_eps = get_int("Episodes for DQN", DQN_DEFAULT_EPISODES, 500, 50000)
    print(f"\n  Q-Learning & SARSA: {tab_eps} episodes each")
    print(f"  DQN:                {dqn_eps} episodes")
    if not yn("Continue?", "y"):
        return
    histories = {}
    histories["qlearning"] = train(algo="qlearning", num_episodes=tab_eps)
    histories["sarsa"] = train(algo="sarsa", num_episodes=tab_eps)
    histories["dqn"] = train(algo="dqn", num_episodes=dqn_eps)
    comparison_dashboard(histories)
    wait_for_enter()


# ############################################################################
#                              MAIN LOOP
# ############################################################################

def main():
    while True:
        clear_screen()
        print_banner()
        print_menu()

        choice = input("\n  Enter your choice [0-11]: ").strip()
        try:
            if   choice == "1":  menu_train("qlearning")
            elif choice == "2":  menu_train("sarsa")
            elif choice == "3":  menu_train("dqn")
            elif choice == "4":  menu_test("qlearning")
            elif choice == "5":  menu_test("sarsa")
            elif choice == "6":  menu_test("dqn")
            elif choice == "7":  menu_show_path("qlearning")
            elif choice == "8":  menu_show_path("sarsa")
            elif choice == "9":  menu_show_path("dqn")
            elif choice == "10": menu_compare()
            elif choice == "11": menu_quick()
            elif choice == "0":
                print("\n  Goodbye!\n")
                break
            else:
                print("  Invalid choice. Enter 0-11.")
                wait_for_enter()
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            break
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()
            wait_for_enter()


# ============================================================================
if __name__ == "__main__":
    main()
