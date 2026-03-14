"""
main.py  —  entry point for the cleaning robot RL project.

Run this file to start:
    python main.py

A small popup window appears where you pick:
  - Mode:     Train / Test / Compare All (trains all three agents)
  - Agent:    Q-Learning / SARSA / DQN
  - Render:   ON draws a live pygame window each episode
  - Episodes: how many episodes to run

After training, PNG charts are saved to the results/ folder.
Trained models are saved to the models/ folder.
"""

import os
import sys
import time

# suppresses a duplicate OpenMP warning that appears on some Windows setups
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from ui.launcher import show_launcher, RunConfig
from environment import Phase2CleaningEnv
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent
from utils.plotting import (
    plot_single_agent,
    plot_comparison_learning_curves,
    plot_comparison_bars,
    plot_agent_optimal_path,
    plot_comparison_optimal_paths,
    plot_comparison_smart_analysis,
    save_results_json,
)

# DQN requires PyTorch — only import if available
try:
    from agents.dqn_agent import DQNAgent
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

MODELS_DIR  = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Early stopping settings (used in train mode and compare mode).
# If rolling average reward stops improving, training ends early.
EARLY_STOP_WINDOW = 50
EARLY_STOP_MIN_EPISODES = 200
EARLY_STOP_PATIENCE = 150
EARLY_STOP_MIN_IMPROVEMENT = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_agent(name: str):
    """Create the right agent from its short name string."""
    if name == "q_learning":
        return QLearningAgent()
    if name == "sarsa":
        return SarsaAgent()
    if name == "dqn":
        if not TORCH_OK:
            print("[ERROR] PyTorch is not installed. Cannot use DQN.")
            sys.exit(1)
        return DQNAgent(
            learning_rate=1e-4,
            discount_factor=0.98,
            epsilon_decay=0.9975,
            memory_size=50_000,
            target_update=400,
            train_every=2,
        )
    raise ValueError(f"Unknown agent name: {name!r}")


def _make_env(use_dqn: bool) -> Phase2CleaningEnv:
    """Create the Phase-2 apartment env in the right observation mode."""
    mode = "dqn" if use_dqn else "tabular"
    return Phase2CleaningEnv(observation_mode=mode)


def _label(name: str) -> str:
    """Human-readable agent label for plot titles and printouts."""
    return {"q_learning": "Q-Learning", "sarsa": "SARSA", "dqn": "DQN"}.get(name, name)


def _model_path(name: str) -> str:
    ext = ".pth" if name == "dqn" else ".pkl"
    return os.path.join(MODELS_DIR, f"{name}_phase2{ext}")


# ─────────────────────────────────────────────────────────────────────────────
# single-episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env,
    agent,
    is_dqn: bool,
    training: bool,
    renderer=None,
    episode_num: int = 0,
    agent_label: str = "agent",
) -> dict:
    """
    Run one full episode and return performance metrics.

    Handles all three agent types:
      Q-Learning — standard off-policy update every step
      SARSA      — on-policy: pre-selects next_action before the update
      DQN        — uses the 8-channel grid tensor as state

    Returns a dict with keys:
      reward, steps, tiles_cleaned, battery_used, battery_eff,
      success (bool), window_open (bool — False if pygame was closed)
    """
    obs, info  = env.reset()
    state      = env.get_dqn_state() if is_dqn else env.get_tabular_state()
    is_sarsa   = isinstance(agent, SarsaAgent)

    total_reward  = 0.0
    steps         = 0
    battery_start = env.battery
    tiles_cleaned = 0
    window_open   = True

    # SARSA needs the first action selected before the loop starts
    action = agent.choose_action(state, training=training) if is_sarsa else None

    while True:
        # pick action — SARSA already chose it above / at end of last step
        if action is None:
            action = agent.choose_action(state, training=training)

        obs, reward, terminated, truncated, info = env.step(action)
        next_state = env.get_dqn_state() if is_dqn else env.get_tabular_state()
        done = terminated or truncated

        total_reward += reward
        steps        += 1
        # count tiles cleaned this step
        if "cleaned" in info.get("event", ""):
            tiles_cleaned += 1

        # ── agent learning update ─────────────────────────────────────────
        if training:
            if is_sarsa:
                # choose the next action first, then pass it to learn()
                next_action = agent.choose_action(next_state, training=True)
                agent.learn(state, action, reward, next_state, next_action, done)
                action = next_action      # carry forward to next iteration
            elif is_dqn:
                agent.learn(state, action, reward, next_state, done)
                action = None
            else:
                # Q-Learning
                agent.learn(state, action, reward, next_state, done)
                action = None
        else:
            action = None                 # during testing: always pick fresh

        state = next_state

        # ── optional rendering ────────────────────────────────────────────
        if renderer is not None:
            window_open = renderer.render(
                env,
                episode=episode_num,
                step=steps,
                ep_reward=total_reward,
                agent_name=agent_label,
            )
            if not window_open:
                break

        if done:
            break

    battery_used = max(1, battery_start - env.battery)

    return {
        "reward":       total_reward,
        "steps":        steps,
        "tiles_cleaned": tiles_cleaned,
        "battery_used": battery_used,
        "battery_eff":  tiles_cleaned / battery_used,
        "success":      bool(info.get("is_apartment_clean", False)),
        "window_open":  window_open,
    }


# ─────────────────────────────────────────────────────────────────────────────
# training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(agent_name: str, episodes: int, render: bool) -> dict:
    """
    Train one agent for the given number of episodes.
    Prints a progress line every 50 episodes.

    Returns a dict of per-episode metric lists:
        {"agent": str, "rewards": [...], "steps": [...], "battery_eff": [...]}
    """
    is_dqn  = (agent_name == "dqn")
    agent   = _make_agent(agent_name)
    env     = _make_env(is_dqn)
    lbl     = _label(agent_name)

    renderer = None
    if render:
        from ui.renderer import ApartmentRenderer
        renderer = ApartmentRenderer(caption=f"Training — {lbl}")

    rewards, steps_list, eff_list, completed_list, tiles_list = [], [], [], [], []
    t0 = time.time()

    # early stopping trackers
    best_rolling_reward = -float("inf")
    no_improve_count = 0
    stopped_early = False
    stop_episode = episodes

    for ep in range(1, episodes + 1):
        m = run_episode(env, agent, is_dqn,
                        training=True,
                        renderer=renderer,
                        episode_num=ep,
                        agent_label=lbl)

        rewards.append(m["reward"])
        steps_list.append(m["steps"])
        eff_list.append(m["battery_eff"])
        completed_list.append(m["success"])
        tiles_list.append(m["tiles_cleaned"])

        agent.decay_epsilon()

        if not m["window_open"]:
            print(f"  [renderer] Window closed at episode {ep}.")
            break

        # using rolling avg reward as a simple convergence signal
        if ep >= EARLY_STOP_MIN_EPISODES and len(rewards) >= EARLY_STOP_WINDOW:
            rolling_reward = float(np.mean(rewards[-EARLY_STOP_WINDOW:]))
            if rolling_reward > (best_rolling_reward + EARLY_STOP_MIN_IMPROVEMENT):
                best_rolling_reward = rolling_reward
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= EARLY_STOP_PATIENCE:
                stopped_early = True
                stop_episode = ep
                print(
                    f"  [early-stop] No reward improvement for "
                    f"{EARLY_STOP_PATIENCE} episodes. Stopping at ep {ep}."
                )
                break

        if ep % 50 == 0 or ep == 1:
            avg_r = float(np.mean(rewards[-50:]))
            eps_v = getattr(agent, "epsilon", 0.0)
            print(f"  ep {ep:>5}/{episodes}  |  "
                  f"avg_reward(50) {avg_r:+7.1f}  |  "
                  f"eps {eps_v:.3f}  |  {time.time()-t0:.0f}s")

    if renderer:
        renderer.close()

    agent.save(_model_path(agent_name))
    print(f"  Model saved → {_model_path(agent_name)}")

    plot_single_agent(
        agent_name=lbl,
        rewards=rewards,
        steps=steps_list,
        battery_eff=eff_list,
        completed=completed_list,
        save_dir=RESULTS_DIR,
    )

    training_time_sec = float(time.time() - t0)

    return {
        "agent":       agent_name,
        "rewards":     rewards,
        "steps":       steps_list,
        "battery_eff": eff_list,
        "completed":   completed_list,
        "tiles_cleaned": tiles_list,
        "training_time_sec": training_time_sec,
        "stopped_early": stopped_early,
        "stop_episode": stop_episode,
    }


def collect_greedy_path(agent_name: str, max_steps: int = 2000) -> dict:
    """
    Load a trained model and roll out one greedy evaluation episode.
    Returns trajectory and summary stats for optimal-path plotting.
    """
    model_file = _model_path(agent_name)
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found for path extraction: {model_file}")

    is_dqn = (agent_name == "dqn")
    agent = _make_agent(agent_name)
    agent.load(model_file)
    env = _make_env(is_dqn)

    _, _info = env.reset()
    state = env.get_dqn_state() if is_dqn else env.get_tabular_state()

    path = [tuple(env.robot_pos)]
    total_reward = 0.0
    steps = 0
    info = {}

    while steps < max_steps:
        action = agent.choose_action(state, training=False)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        path.append(tuple(env.robot_pos))

        state = env.get_dqn_state() if is_dqn else env.get_tabular_state()
        if terminated or truncated:
            break

    return {
        "agent": agent_name,
        "path": path,
        "reward": float(total_reward),
        "steps": int(steps),
        "success": bool(info.get("is_apartment_clean", False)),
        "rows": int(env.rows),
        "cols": int(env.cols),
    }


# ─────────────────────────────────────────────────────────────────────────────
# testing loop
# ─────────────────────────────────────────────────────────────────────────────

def test(agent_name: str, episodes: int, render: bool):
    """
    Load a saved model and run test episodes with no learning.
    Prints per-episode results and a final success-rate summary.
    """
    model_file = _model_path(agent_name)
    if not os.path.exists(model_file):
        print(f"[ERROR] No saved model at {model_file}.")
        print("        Run training first (Mode: Train).")
        return

    is_dqn = (agent_name == "dqn")
    agent  = _make_agent(agent_name)
    agent.load(model_file)
    env    = _make_env(is_dqn)
    lbl    = _label(agent_name)

    renderer = None
    if render:
        from ui.renderer import ApartmentRenderer
        renderer = ApartmentRenderer(caption=f"Testing — {lbl}")

    successes = 0
    for ep in range(1, episodes + 1):
        m = run_episode(env, agent, is_dqn,
                        training=False,
                        renderer=renderer,
                        episode_num=ep,
                        agent_label=lbl)
        tag = "clean!" if m["success"] else "     "
        print(f"  ep {ep:>3}  reward {m['reward']:+7.1f}  "
              f"steps {m['steps']:>4}  {tag}")
        if m["success"]:
            successes += 1
        if not m["window_open"]:
            break

    if renderer:
        renderer.close()

    rate = 100 * successes / max(1, ep)
    print(f"\n  Success rate: {rate:.1f}%  ({successes}/{ep} episodes)")


# ─────────────────────────────────────────────────────────────────────────────
# compare all three agents
# ─────────────────────────────────────────────────────────────────────────────

def compare_all(episodes: int, render: bool):
    """
    Train Q-Learning, SARSA, and DQN (if PyTorch is available) one at a time,
    then produce a side-by-side comparison chart.
    Individual training charts are also saved for each agent.
    """
    names = ["q_learning", "sarsa"]
    if TORCH_OK:
        names.append("dqn")
    else:
        print("  [INFO] PyTorch not found — DQN skipped from comparison.")

    all_results = {}
    for name in names:
        print(f"\n{'─' * 52}")
        print(f"  Training {_label(name)} for {episodes} episodes ...")
        print(f"{'─' * 52}")
        all_results[name] = train(name, episodes, render=render)

    # requested comparison visuals
    plot_comparison_learning_curves(all_results, save_dir=RESULTS_DIR)
    plot_comparison_bars(all_results, save_dir=RESULTS_DIR)
    plot_comparison_smart_analysis(all_results, save_dir=RESULTS_DIR)

    # greedy trajectory per agent + combined path comparison
    path_results = {}
    for name in names:
        path_data = collect_greedy_path(name)
        path_results[name] = path_data
        plot_agent_optimal_path(
            agent_key=name,
            path=path_data["path"],
            reward=path_data["reward"],
            steps=path_data["steps"],
            success=path_data["success"],
            rows=path_data["rows"],
            cols=path_data["cols"],
            save_dir=RESULTS_DIR,
        )

    plot_comparison_optimal_paths(path_results, save_dir=RESULTS_DIR)

    # raw numbers as JSON too
    save_results_json(all_results, save_dir=RESULTS_DIR)
    print(f"\n  All results saved to {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n  Cleaning Robot RL  —  Phase 2")
    print("  Opening settings window ...\n")

    cfg = show_launcher()

    if not cfg.confirmed:
        print("  Cancelled.")
        return

    print(f"  Mode    : {cfg.mode}")
    print(f"  Agent   : {cfg.agent}")
    print(f"  Render  : {cfg.render}")
    print(f"  Episodes: {cfg.episodes}\n")

    if cfg.mode == "train":
        train(cfg.agent, cfg.episodes, cfg.render)

    elif cfg.mode == "test":
        # for testing, cap at 20 episodes by default
        test(cfg.agent, min(cfg.episodes, 20), cfg.render)

    elif cfg.mode == "compare":
        compare_all(cfg.episodes, cfg.render)

    else:
        print(f"  Unknown mode: {cfg.mode!r}")


if __name__ == "__main__":
    main()
