"""
plotting.py  —  saves training charts for the cleaning robot project.

Charts saved:
  1. Rolling average reward per episode
  2. Steps per episode  (how many steps the robot took before done)
  3. Battery efficiency  (tiles cleaned / battery used per episode)

plot_single_agent()   three subplots for one agent
plot_three_agents()   all three agents overlaid for comparison
save_results_json()   raw numbers as JSON for later analysis

All charts are written as PNG files to the results/ folder by default.
"""

import os
import json
import numpy as np
import matplotlib
# non-interactive backend so saving works on machines without a display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

RESULTS_DIR    = "results"
ROLLING_WINDOW = 50     # number of episodes to average for the rolling-avg line


def _rolling(arr, window: int = ROLLING_WINDOW):
    """
    Rolling average using numpy convolution.
    Returns the 'valid' portion — shorter than the input by (window - 1).
    Handles arrays that are shorter than the window gracefully.
    """
    if len(arr) < window:
        return np.array(arr, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def _rolling_nan(arr, window: int = ROLLING_WINDOW):
    """
    Rolling average that ignores NaN values in each window.
    If an entire window is NaN, output NaN for that point.
    """
    a = np.asarray(arr, dtype=float)
    if len(a) < window:
        return a
    out = []
    for i in range(0, len(a) - window + 1):
        w = a[i:i + window]
        if np.isnan(w).all():
            out.append(np.nan)
        else:
            out.append(np.nanmean(w))
    return np.asarray(out, dtype=float)


def _roll_x(data_len: int, window: int = ROLLING_WINDOW):
    """Returns the episode-number x-axis that aligns with _rolling()."""
    return list(range(window, data_len + 1))


def _ensure(d: str):
    os.makedirs(d, exist_ok=True)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _agent_order(results: dict) -> list[str]:
    preferred = ["q_learning", "sarsa", "dqn"]
    present = [k for k in preferred if k in results]
    extras = [k for k in results.keys() if k not in preferred]
    return present + extras


def _agent_label(agent_key: str) -> str:
    return {
        "q_learning": "Q-Learning",
        "sarsa": "SARSA",
        "dqn": "DQN",
    }.get(agent_key, agent_key)


def _agent_color(agent_key: str) -> str:
    return {
        "q_learning": "#89dceb",
        "sarsa": "#a6e3a1",
        "dqn": "#f38ba8",
    }.get(agent_key, "#cdd6f4")


def _agent_slug(agent_key: str) -> str:
    return {
        "q_learning": "qlearning",
        "sarsa": "sarsa",
        "dqn": "dqn",
    }.get(agent_key, agent_key.replace("_", ""))


def _first_reach_episode(values: list[float], threshold_ratio: float = 0.80) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    rolled = _rolling(arr.tolist()) if arr.size >= ROLLING_WINDOW else arr
    if rolled.size == 0:
        return np.nan
    target = threshold_ratio * float(np.nanmax(rolled))
    idx = np.where(rolled >= target)[0]
    if idx.size == 0:
        return np.nan
    return float((idx[0] + ROLLING_WINDOW) if arr.size >= ROLLING_WINDOW else (idx[0] + 1))


# ── single-agent plot ─────────────────────────────────────────────────────────

def plot_single_agent(
    agent_name: str,
    rewards: list,
    steps: list,
    battery_eff: list,
    completed: list | None = None,
    save_dir: str = RESULTS_DIR,
) -> str:
    """
    Save a three-subplot figure for one agent.
    Automatically called at the end of a training run.

    Parameters
    ----------
    agent_name  : human-readable label shown in the chart title
    rewards     : list — total reward collected each episode
    steps       : list — number of steps taken each episode
    battery_eff : list — tiles_cleaned / battery_used per episode
    completed   : optional list[bool]. If provided, steps are plotted only for
                  episodes that reached full apartment clean.
    save_dir    : folder where the PNG is written

    Returns the path of the saved PNG file.
    """
    _ensure(save_dir)

    eps = list(range(1, len(rewards) + 1))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(f"{agent_name} — Training Progress",
                 fontsize=14, fontweight="bold")
    plt.subplots_adjust(hspace=0.45)

    # ── reward ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eps, rewards, color="#94e2d5", alpha=0.30, linewidth=0.8,
            label="Raw reward")
    rolled = _rolling(rewards)
    if len(rolled) > 0:
        ax.plot(_roll_x(len(rewards)), rolled,
                color="#89dceb", linewidth=2,
                label=f"Rolling avg ({ROLLING_WINDOW} eps)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Rolling Average Reward per Episode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # ── steps ─────────────────────────────────────────────────────────────
    ax = axes[1]
    if completed is not None and len(completed) == len(steps):
        steps_to_clean = [float(s) if bool(ok) else np.nan for s, ok in zip(steps, completed)]
    else:
        steps_to_clean = [float(s) for s in steps]
    ax.plot(eps, steps_to_clean, color="#cba6f7", alpha=0.30, linewidth=0.8)
    rolled_s = _rolling_nan(steps_to_clean)
    if len(rolled_s) > 0 and len(steps_to_clean) >= ROLLING_WINDOW:
        ax.plot(_roll_x(len(steps_to_clean)), rolled_s,
                color="#b4befe", linewidth=2,
                label=f"Rolling avg ({ROLLING_WINDOW} eps)")
        ax.legend(fontsize=9)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Steps to Clean 100% Apartment")
    ax.grid(True, alpha=0.2)

    # ── battery efficiency ────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(eps, battery_eff, color="#fab387", alpha=0.30, linewidth=0.8)
    rolled_b = _rolling(battery_eff)
    if len(rolled_b) > 0 and len(battery_eff) >= ROLLING_WINDOW:
        ax.plot(_roll_x(len(battery_eff)), rolled_b,
                color="#f9e2af", linewidth=2,
                label=f"Rolling avg ({ROLLING_WINDOW} eps)")
        ax.legend(fontsize=9)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Tiles / battery unit")
    ax.set_title("Battery Efficiency (Tiles Cleaned per Battery Unit)")
    ax.grid(True, alpha=0.2)

    safe = agent_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(save_dir, f"{safe}_training_{_ts()}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved -> {path}")
    return path


# ── three-agent comparison plot ───────────────────────────────────────────────

def plot_three_agents(results: dict, save_dir: str = RESULTS_DIR) -> str:
    """
    Overlay Q-Learning, SARSA (and optionally DQN) on the same axes
    for easy side-by-side comparison.

    results format:
        {
          "q_learning": {"rewards": [...], "steps": [...], "battery_eff": [...]},
          "sarsa":      { ... },
          "dqn":        { ... },   # optional — only present if torch is installed
        }

    Returns the path of the saved PNG.
    """
    _ensure(save_dir)

    COLORS = {
        "q_learning": "#89dceb",
        "sarsa":      "#a6e3a1",
        "dqn":        "#f38ba8",
    }
    LABELS = {
        "q_learning": "Q-Learning",
        "sarsa":      "SARSA",
        "dqn":        "DQN",
    }

    titles  = [
        "Rolling Average Reward per Episode",
        "Steps to Clean 100% Apartment",
        "Battery Efficiency (Tiles / Battery Unit)",
    ]
    keys    = ["rewards", "steps", "battery_eff"]
    ylabels = ["Total reward", "Steps", "Tiles / battery unit"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle("Algorithm Comparison — Phase 2 Apartment",
                 fontsize=14, fontweight="bold")
    plt.subplots_adjust(hspace=0.45)

    for ax, title, key, ylabel in zip(axes, titles, keys, ylabels):
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)

        for agent_key, data in results.items():
            arr = data.get(key, [])
            if len(arr) == 0:
                continue
            col   = COLORS.get(agent_key, "#cdd6f4")
            label = LABELS.get(agent_key, agent_key)
            eps   = list(range(1, len(arr) + 1))

            # For the steps chart, keep only successful-clean episodes.
            if key == "steps":
                completed = data.get("completed", None)
                if completed is not None and len(completed) == len(arr):
                    arr = [float(s) if bool(ok) else np.nan for s, ok in zip(arr, completed)]

            # faint raw line behind the rolling average
            ax.plot(eps, arr, color=col, alpha=0.20, linewidth=0.7)

            # bolder rolling average on top
            rolled = _rolling_nan(arr) if key == "steps" else _rolling(arr)
            if len(rolled) > 0:
                ax.plot(_roll_x(len(arr)), rolled,
                        color=col, linewidth=2, label=label)

        ax.legend(fontsize=9)

    path = os.path.join(save_dir, f"comparison_{_ts()}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Comparison chart saved -> {path}")
    return path


def plot_comparison_learning_curves(results: dict, save_dir: str = RESULTS_DIR) -> str:
    """
    Save comparison learning curves under the exact required filename:
      comparison_learning_curves.png
    """
    _ensure(save_dir)

    keys = ["rewards", "steps", "battery_eff"]
    titles = [
        "Rolling Average Reward per Episode",
        "Steps to Clean 100% Apartment",
        "Battery Efficiency (Tiles / Battery Unit)",
    ]
    ylabels = ["Total reward", "Steps", "Tiles / battery unit"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle("Comparison — Learning Curves", fontsize=14, fontweight="bold")
    plt.subplots_adjust(hspace=0.45)

    for ax, key, title, ylabel in zip(axes, keys, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)

        for agent_key in _agent_order(results):
            data = results.get(agent_key, {})
            arr = data.get(key, [])
            if not arr:
                continue

            color = _agent_color(agent_key)
            label = _agent_label(agent_key)
            eps = list(range(1, len(arr) + 1))

            if key == "steps":
                completed = data.get("completed", None)
                if completed is not None and len(completed) == len(arr):
                    arr = [float(s) if bool(ok) else np.nan for s, ok in zip(arr, completed)]

            ax.plot(eps, arr, color=color, alpha=0.18, linewidth=0.8)
            rolled = _rolling_nan(arr) if key == "steps" else _rolling(arr)
            if len(rolled) > 0:
                ax.plot(_roll_x(len(arr)), rolled, color=color, linewidth=2, label=label)

        ax.legend(fontsize=9)

    path = os.path.join(save_dir, "comparison_learning_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved -> {path}")
    return path


def plot_comparison_bars(results: dict, save_dir: str = RESULTS_DIR) -> str:
    """
    Save metric bars under exact filename:
      comparison_bars.png

    Metrics:
      - average reward
      - average tiles cleaned
      - success rate
      - average steps to clean
      - reward stability (std; lower is better)
      - training time (sec)
    """
    _ensure(save_dir)
    agents = _agent_order(results)
    labels = [_agent_label(a) for a in agents]
    colors = [_agent_color(a) for a in agents]

    avg_reward = []
    avg_tiles = []
    success_rate = []
    avg_steps_to_clean = []
    reward_std = []
    train_sec = []

    for agent_key in agents:
        d = results.get(agent_key, {})
        rewards = np.asarray(d.get("rewards", []), dtype=float)
        tiles = np.asarray(d.get("tiles_cleaned", []), dtype=float)
        steps = np.asarray(d.get("steps", []), dtype=float)
        completed = np.asarray(d.get("completed", []), dtype=bool)

        avg_reward.append(float(np.nanmean(rewards)) if rewards.size else np.nan)
        avg_tiles.append(float(np.nanmean(tiles)) if tiles.size else np.nan)
        success_rate.append(float(np.mean(completed) * 100.0) if completed.size else np.nan)

        if steps.size and completed.size == steps.size:
            clean_steps = np.where(completed, steps, np.nan)
            avg_steps_to_clean.append(float(np.nanmean(clean_steps)) if not np.isnan(clean_steps).all() else np.nan)
        else:
            avg_steps_to_clean.append(float(np.nanmean(steps)) if steps.size else np.nan)

        reward_std.append(float(np.nanstd(rewards)) if rewards.size else np.nan)
        train_sec.append(float(d.get("training_time_sec", np.nan)))

    metrics = [
        (avg_reward, "Average Reward"),
        (avg_tiles, "Average Tiles Cleaned"),
        (success_rate, "Success Rate (%)"),
        (avg_steps_to_clean, "Average Steps to Clean"),
        (reward_std, "Reward Stability (Std, Lower Better)"),
        (train_sec, "Training Time (seconds)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Comparison — Key Metrics", fontsize=14, fontweight="bold")
    axes = axes.ravel()

    for ax, (vals, title) in zip(axes, metrics):
        x = np.arange(len(agents))
        bars = ax.bar(x, vals, color=colors, alpha=0.90)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

        for b, v in zip(bars, vals):
            if np.isnan(v):
                txt = "n/a"
                y = 0.0
            else:
                txt = f"{v:.1f}"
                y = float(v)
            ax.text(b.get_x() + b.get_width() / 2.0, y, txt,
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(save_dir, "comparison_bars.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved -> {path}")
    return path


def _draw_path_on_axis(
    ax,
    path: list[tuple[int, int]],
    title: str,
    rows: int,
    cols: int,
    success: bool,
    reward: float,
    steps: int,
):
    """Draw a single trajectory over an apartment-sized grid on a matplotlib axis."""
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{title}\nsteps={steps}, reward={reward:.1f}, success={success}")

    if not path:
        return

    pts = np.asarray([(c, r) for r, c in path], dtype=float)
    if pts.shape[0] >= 2:
        ax.plot(pts[:, 0], pts[:, 1], linewidth=2.0, color="#74c7ec", alpha=0.95)

    ax.scatter(pts[0, 0], pts[0, 1], s=70, color="#a6e3a1", label="start", zorder=3)
    ax.scatter(pts[-1, 0], pts[-1, 1], s=70, color="#f38ba8", label="end", zorder=3)
    ax.legend(loc="upper right", fontsize=7)


def plot_agent_optimal_path(
    agent_key: str,
    path: list[tuple[int, int]],
    reward: float,
    steps: int,
    success: bool,
    rows: int = 15,
    cols: int = 15,
    save_dir: str = RESULTS_DIR,
) -> str:
    """Save a single agent path figure with exact naming requirement."""
    _ensure(save_dir)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    _draw_path_on_axis(
        ax=ax,
        path=path,
        title=f"{_agent_label(agent_key)} Optimal Path",
        rows=rows,
        cols=cols,
        success=success,
        reward=reward,
        steps=steps,
    )
    slug = _agent_slug(agent_key)
    path_file = os.path.join(save_dir, f"optimal_path_{slug}.png")
    plt.savefig(path_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved -> {path_file}")
    return path_file


def plot_comparison_optimal_paths(paths_data: dict, save_dir: str = RESULTS_DIR) -> str:
    """
    Save all agent trajectories in one figure under exact filename:
      comparison_optimal_paths.png

    paths_data format:
      {
        "q_learning": {"path": [...], "reward": float, "steps": int, "success": bool},
        ...
      }
    """
    _ensure(save_dir)
    agents = _agent_order(paths_data)
    n = max(1, len(agents))

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, agent_key in zip(axes, agents):
        d = paths_data.get(agent_key, {})
        _draw_path_on_axis(
            ax=ax,
            path=d.get("path", []),
            title=f"{_agent_label(agent_key)}",
            rows=int(d.get("rows", 15)),
            cols=int(d.get("cols", 15)),
            success=bool(d.get("success", False)),
            reward=float(d.get("reward", 0.0)),
            steps=int(d.get("steps", 0)),
        )

    fig.suptitle("Comparison — Optimal Paths", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path_file = os.path.join(save_dir, "comparison_optimal_paths.png")
    plt.savefig(path_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved -> {path_file}")
    return path_file


def plot_comparison_smart_analysis(results: dict, save_dir: str = RESULTS_DIR) -> str:
    """
    Save advanced analysis under exact filename:
      comparison_smart_analysis.png

    Includes:
      - radar chart (normalized, higher is better)
      - convergence episode bars (lower is better)
    """
    _ensure(save_dir)
    agents = _agent_order(results)
    labels = [_agent_label(a) for a in agents]
    colors = [_agent_color(a) for a in agents]

    # raw metrics
    raw = {
        "avg_reward": [],
        "success_rate": [],
        "efficiency": [],
        "speed": [],          # inverse training time
        "stability": [],      # inverse std reward
        "coverage": [],       # average tiles cleaned
        "convergence_ep": [],
    }

    for agent_key in agents:
        d = results.get(agent_key, {})
        rewards = np.asarray(d.get("rewards", []), dtype=float)
        battery_eff = np.asarray(d.get("battery_eff", []), dtype=float)
        completed = np.asarray(d.get("completed", []), dtype=bool)
        tiles = np.asarray(d.get("tiles_cleaned", []), dtype=float)
        train_time = float(d.get("training_time_sec", np.nan))

        avg_reward = float(np.nanmean(rewards)) if rewards.size else np.nan
        success_rate = float(np.mean(completed) * 100.0) if completed.size else np.nan
        efficiency = float(np.nanmean(battery_eff)) if battery_eff.size else np.nan
        reward_std = float(np.nanstd(rewards)) if rewards.size else np.nan
        stability = (1.0 / (1.0 + reward_std)) if not np.isnan(reward_std) else np.nan
        speed = (1.0 / max(train_time, 1e-9)) if not np.isnan(train_time) else np.nan
        coverage = float(np.nanmean(tiles)) if tiles.size else np.nan
        conv_ep = _first_reach_episode(rewards.tolist(), threshold_ratio=0.80)

        raw["avg_reward"].append(avg_reward)
        raw["success_rate"].append(success_rate)
        raw["efficiency"].append(efficiency)
        raw["speed"].append(speed)
        raw["stability"].append(stability)
        raw["coverage"].append(coverage)
        raw["convergence_ep"].append(conv_ep)

    radar_keys = ["avg_reward", "success_rate", "efficiency", "speed", "stability", "coverage"]
    radar_labels = ["Reward", "Success", "Efficiency", "Speed", "Stability", "Coverage"]

    # min-max normalize each radar metric across agents
    norm = {k: [] for k in radar_keys}
    for k in radar_keys:
        vals = np.asarray(raw[k], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            norm[k] = [0.0 for _ in agents]
            continue
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if abs(vmax - vmin) < 1e-9:
            norm[k] = [1.0 if np.isfinite(v) else 0.0 for v in vals]
        else:
            norm[k] = [((float(v) - vmin) / (vmax - vmin)) if np.isfinite(v) else 0.0 for v in vals]

    fig = plt.figure(figsize=(13, 6))
    ax_radar = fig.add_subplot(1, 2, 1, polar=True)
    ax_conv = fig.add_subplot(1, 2, 2)

    # radar plot
    angles = np.linspace(0, 2 * np.pi, len(radar_keys), endpoint=False).tolist()
    angles += angles[:1]
    for i, agent_key in enumerate(agents):
        vals = [norm[k][i] for k in radar_keys]
        vals += vals[:1]
        ax_radar.plot(angles, vals, color=colors[i], linewidth=2, label=labels[i])
        ax_radar.fill(angles, vals, color=colors[i], alpha=0.12)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels)
    ax_radar.set_yticklabels([])
    ax_radar.set_title("Normalized Capability Radar", pad=16)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=8)

    # convergence bars
    conv = np.asarray(raw["convergence_ep"], dtype=float)
    x = np.arange(len(agents))
    bars = ax_conv.bar(x, conv, color=colors, alpha=0.9)
    ax_conv.set_xticks(x)
    ax_conv.set_xticklabels(labels, rotation=15)
    ax_conv.set_ylabel("Episode reaching 80% peak rolling reward")
    ax_conv.set_title("Convergence Speed (Lower is Better)")
    ax_conv.grid(axis="y", alpha=0.2)

    for b, v in zip(bars, conv):
        txt = "n/a" if np.isnan(v) else f"{v:.0f}"
        y = 0.0 if np.isnan(v) else float(v)
        ax_conv.text(b.get_x() + b.get_width() / 2.0, y, txt,
                     ha="center", va="bottom", fontsize=8)

    fig.suptitle("Comparison — Smart Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(save_dir, "comparison_smart_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved -> {path}")
    return path


# ── JSON export ───────────────────────────────────────────────────────────────

def save_results_json(results: dict, save_dir: str = RESULTS_DIR) -> str:
    """
    Save the raw per-episode numbers as a JSON file.
    Useful for post-processing or re-plotting without retraining.

    results format — same as for plot_three_agents().
    """
    _ensure(save_dir)
    path = os.path.join(save_dir, f"results_{_ts()}.json")
    # json doesn't know numpy types so convert everything to plain Python floats
    clean = {}
    for name, data in results.items():
        clean[name] = {
            k: [float(v) for v in lst]
            for k, lst in data.items()
            if isinstance(lst, list)
        }
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  [data] Raw results saved -> {path}")
    return path
