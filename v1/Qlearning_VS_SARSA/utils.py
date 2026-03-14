import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from environment import CleaningRobotEnv, DirtRegenerationEnv
from agents import QLearningAgent, SARSAAgent, BaseAgent
from config import (
    PHASE1_CONFIG, PHASE2_CONFIG, TRAINING_EPISODES, NUM_SEEDS, SEEDS,
    ALPHA, GAMMA,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    UCB_C, OPTIMISTIC_INIT, NUM_ACTIONS,
    DIRT_PATTERN_CONFIG, DIRT_PATTERN_EPISODES, DIRT_PATTERN_MAX_STEPS,
    DIRT_BURST_CONFIG, ROOM_DEFINITIONS,
)


# =============================================================================
# Training Functions
# =============================================================================

def train_qlearning(env, agent, num_episodes=TRAINING_EPISODES):
    """
    Train a Q-Learning agent.

    Parameters
    ----------
    env : CleaningRobotEnv
    agent : QLearningAgent
    num_episodes : int

    Returns
    -------
    metrics : dict
        Episode-level metrics: rewards, coverages, steps, deaths, events.
    """
    metrics = {
        "rewards": [],
        "coverages": [],
        "steps": [],
        "deaths": [],
        "battery_at_end": [],
    }

    for ep in range(num_episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        total_reward = 0.0
        died = False

        while not env.done:
            action = agent.choose_action(state_idx)
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            agent.update(state_idx, action, reward, next_state_idx, done)

            state_idx = next_state_idx
            total_reward += reward

            if info["event"] == "battery_dead":
                died = True

        agent.decay_epsilon()

        metrics["rewards"].append(total_reward)
        metrics["coverages"].append(info["coverage"])
        metrics["steps"].append(info["steps"])
        metrics["deaths"].append(int(died))
        metrics["battery_at_end"].append(info["battery"])

    return metrics


def train_sarsa(env, agent, num_episodes=TRAINING_EPISODES):
    """
    Train a SARSA agent.

    Parameters
    ----------
    env : CleaningRobotEnv
    agent : SARSAAgent
    num_episodes : int

    Returns
    -------
    metrics : dict
        Episode-level metrics.
    """
    metrics = {
        "rewards": [],
        "coverages": [],
        "steps": [],
        "deaths": [],
        "battery_at_end": [],
    }

    for ep in range(num_episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        action = agent.choose_action(state_idx)
        total_reward = 0.0
        died = False

        while not env.done:
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            if done:
                agent.update(state_idx, action, reward, next_state_idx, done)
            else:
                next_action = agent.choose_action(next_state_idx)
                agent.update(
                    state_idx, action, reward, next_state_idx, done,
                    next_action=next_action,
                )
                action = next_action

            state_idx = next_state_idx
            total_reward += reward

            if info["event"] == "battery_dead":
                died = True

        agent.decay_epsilon()

        metrics["rewards"].append(total_reward)
        metrics["coverages"].append(info["coverage"])
        metrics["steps"].append(info["steps"])
        metrics["deaths"].append(int(died))
        metrics["battery_at_end"].append(info["battery"])

    return metrics


def create_agent(algorithm, exploration, env, **kwargs):
    """
    Factory function to create an agent with the specified algorithm and
    exploration strategy.

    Parameters
    ----------
    algorithm : str
        "qlearning" or "sarsa"
    exploration : str
        "epsilon_greedy", "ucb", or "optimistic"
    env : CleaningRobotEnv
    **kwargs : additional overrides for agent parameters

    Returns
    -------
    agent : BaseAgent subclass
    """
    params = {
        "num_states": env.num_states,
        "num_actions": env.num_actions,
        "alpha": kwargs.get("alpha", ALPHA),
        "gamma": kwargs.get("gamma", GAMMA),
        "exploration": exploration,
        "epsilon_start": kwargs.get("epsilon_start", EPSILON_START),
        "epsilon_min": kwargs.get("epsilon_min", EPSILON_MIN),
        "epsilon_decay": kwargs.get("epsilon_decay", EPSILON_DECAY),
        "ucb_c": kwargs.get("ucb_c", UCB_C),
        "optimistic_init": kwargs.get("optimistic_init", 0.0),
    }

    if algorithm == "qlearning":
        return QLearningAgent(**params)
    elif algorithm == "sarsa":
        return SARSAAgent(**params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_agent(env, agent, num_episodes=TRAINING_EPISODES):
    """Route to the correct training function based on agent type."""
    if isinstance(agent, SARSAAgent):
        return train_sarsa(env, agent, num_episodes)
    else:
        return train_qlearning(env, agent, num_episodes)


def run_experiment(algorithm, exploration, config=None, num_episodes=TRAINING_EPISODES,
                   seeds=SEEDS, **agent_kwargs):
    """
    Run a full experiment: train over multiple seeds, collect metrics.

    Parameters
    ----------
    algorithm : str
        "qlearning" or "sarsa"
    exploration : str
        "epsilon_greedy", "ucb", or "optimistic"
    config : dict, optional
        Environment config. Defaults to PHASE1_CONFIG.
    num_episodes : int
    seeds : list of int
    **agent_kwargs : passed to create_agent

    Returns
    -------
    all_metrics : list of dict
        One metrics dict per seed.
    agents : list of BaseAgent
        Trained agents (one per seed).
    """
    cfg = config or PHASE1_CONFIG
    all_metrics = []
    agents = []

    for seed in seeds:
        np.random.seed(seed)
        env = CleaningRobotEnv(cfg)
        agent = create_agent(algorithm, exploration, env, **agent_kwargs)

        env.reset(seed=seed)
        metrics = train_agent(env, agent, num_episodes)

        all_metrics.append(metrics)
        agents.append(agent)

    return all_metrics, agents


def evaluate_agent(env, agent, num_episodes=100, seed=42):
    """
    Evaluate a trained agent greedily (no exploration).

    Returns
    -------
    metrics : dict
        Same structure as training metrics.
    """
    metrics = {
        "rewards": [],
        "coverages": [],
        "steps": [],
        "deaths": [],
        "battery_at_end": [],
    }

    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        state_idx = env.state_to_index(state)
        total_reward = 0.0
        died = False

        while not env.done:
            action = agent.get_greedy_action(state_idx)
            next_state, reward, done, info = env.step(action)
            state_idx = env.state_to_index(next_state)
            total_reward += reward
            if info["event"] == "battery_dead":
                died = True

        metrics["rewards"].append(total_reward)
        metrics["coverages"].append(info["coverage"])
        metrics["steps"].append(info["steps"])
        metrics["deaths"].append(int(died))
        metrics["battery_at_end"].append(info["battery"])

    return metrics


# =============================================================================
# Plotting Functions
# =============================================================================

def smooth(data, window=100):
    """Simple moving average for smoothing curves."""
    if len(data) < window:
        window = max(1, len(data))
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def aggregate_metrics(all_metrics, key, window=100):
    """
    Aggregate a metric across seeds: compute mean and std of smoothed curves.

    Returns
    -------
    mean : np.array
    std : np.array
    """
    smoothed = [smooth(m[key], window) for m in all_metrics]
    min_len = min(len(s) for s in smoothed)
    smoothed = np.array([s[:min_len] for s in smoothed])
    return smoothed.mean(axis=0), smoothed.std(axis=0)


def plot_learning_curves(results_dict, metric="rewards", window=100,
                         title=None, ylabel=None, figsize=(12, 5)):
    """
    Plot learning curves for multiple experiments.

    Parameters
    ----------
    results_dict : dict
        {label: all_metrics} where all_metrics is a list of metric dicts (one per seed).
    metric : str
        Key in metrics dict to plot.
    window : int
        Smoothing window.
    title : str
    ylabel : str
    figsize : tuple
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for label, all_metrics in results_dict.items():
        mean, std = aggregate_metrics(all_metrics, metric, window)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=label, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel or metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"Learning Curves: {metric}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_multi_metric(results_dict, metrics=None, window=100, figsize=(14, 10)):
    """
    Plot multiple metrics in subplots for comparison.

    Parameters
    ----------
    results_dict : dict
        {label: all_metrics}
    metrics : list of str, optional
        Which metrics to plot. Defaults to all main ones.
    figsize : tuple
    """
    if metrics is None:
        metrics = ["rewards", "coverages", "steps", "deaths"]

    ylabels = {
        "rewards": "Total Reward",
        "coverages": "Coverage (%)",
        "steps": "Steps per Episode",
        "deaths": "Battery Death Rate",
        "battery_at_end": "Battery Remaining",
    }

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for label, all_metrics in results_dict.items():
            mean, std = aggregate_metrics(all_metrics, metric, window)
            episodes = np.arange(len(mean))
            ax.plot(episodes, mean, label=label, linewidth=2)
            ax.fill_between(episodes, mean - std, mean + std, alpha=0.15)

        ax.set_ylabel(ylabels.get(metric, metric), fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episode", fontsize=12)
    fig.suptitle("Training Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_coverage_heatmap(env, agent, num_episodes=100, seed=42, figsize=(8, 7)):
    """
    Heatmap of how often the agent visits each tile during evaluation.

    Parameters
    ----------
    env : CleaningRobotEnv
    agent : BaseAgent
    num_episodes : int
    seed : int
    figsize : tuple
    """
    visit_counts = np.zeros((env.rows, env.cols))

    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        state_idx = env.state_to_index(state)
        visit_counts[state[0], state[1]] += 1

        while not env.done:
            action = agent.get_greedy_action(state_idx)
            next_state, reward, done, info = env.step(action)
            state_idx = env.state_to_index(next_state)
            visit_counts[next_state[0], next_state[1]] += 1

    # Normalize
    visit_counts /= num_episodes

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(visit_counts, cmap="YlOrRd", origin="upper")
    ax.set_title("Average Visit Frequency (per episode)", fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, label="Visits per episode")

    # Mark charger
    cr, cc = env.charger_pos
    ax.plot(cc, cr, "s", color="green", markersize=12, label="Charger")
    ax.legend(fontsize=11)

    plt.tight_layout()
    return fig, ax


def plot_battery_analysis(results_dict, window=100, figsize=(12, 5)):
    """
    Plot battery death rate over training for different strategies.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Death rate
    ax = axes[0]
    for label, all_metrics in results_dict.items():
        mean, std = aggregate_metrics(all_metrics, "deaths", window)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean * 100, label=label, linewidth=2)
        ax.fill_between(episodes, (mean - std) * 100, (mean + std) * 100, alpha=0.15)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Death Rate (%)", fontsize=12)
    ax.set_title("Battery Death Rate", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Battery remaining
    ax = axes[1]
    for label, all_metrics in results_dict.items():
        mean, std = aggregate_metrics(all_metrics, "battery_at_end", window)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=label, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Battery Remaining", fontsize=12)
    ax.set_title("Battery at Episode End", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def summary_table(results_dict, last_n=500):
    """
    Print a summary table of final performance (averaged over last_n episodes).

    Parameters
    ----------
    results_dict : dict
        {label: all_metrics}
    last_n : int
        Number of final episodes to average over.
    """
    print(f"{'Agent':<35} {'Reward':>10} {'Coverage':>10} {'Steps':>8} "
          f"{'Death%':>8} {'Battery':>8}")
    print("-" * 85)

    for label, all_metrics in results_dict.items():
        rewards = [np.mean(m["rewards"][-last_n:]) for m in all_metrics]
        coverages = [np.mean(m["coverages"][-last_n:]) for m in all_metrics]
        steps = [np.mean(m["steps"][-last_n:]) for m in all_metrics]
        deaths = [np.mean(m["deaths"][-last_n:]) for m in all_metrics]
        battery = [np.mean(m["battery_at_end"][-last_n:]) for m in all_metrics]

        print(f"{label:<35} "
              f"{np.mean(rewards):>10.1f} "
              f"{np.mean(coverages)*100:>9.1f}% "
              f"{np.mean(steps):>8.1f} "
              f"{np.mean(deaths)*100:>7.1f}% "
              f"{np.mean(battery):>8.1f}")

    print("-" * 85)


def plot_evaluation_comparison(eval_results, figsize=(12, 5)):
    """
    Bar chart comparing evaluation metrics across agents.

    Parameters
    ----------
    eval_results : dict
        {label: eval_metrics} from evaluate_agent().
    """
    labels = list(eval_results.keys())
    n = len(labels)

    avg_reward = [np.mean(eval_results[l]["rewards"]) for l in labels]
    avg_coverage = [np.mean(eval_results[l]["coverages"]) * 100 for l in labels]
    avg_deaths = [np.mean(eval_results[l]["deaths"]) * 100 for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, n))

    axes[0].bar(labels, avg_reward, color=colors)
    axes[0].set_title("Average Reward", fontsize=13)
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].bar(labels, avg_coverage, color=colors)
    axes[1].set_title("Average Coverage (%)", fontsize=13)
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis='x', rotation=30)

    axes[2].bar(labels, avg_deaths, color=colors)
    axes[2].set_title("Battery Death Rate (%)", fontsize=13)
    axes[2].set_ylim(0, 105)
    axes[2].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    return fig, axes


def plot_apartment_layout(env, figsize=(10, 10)):
    """
    Visualize the apartment layout: walls, furniture, walkable tiles, charger,
    and room labels.

    Parameters
    ----------
    env : CleaningRobotEnv
        An environment initialized with the apartment config.
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    # Build a grid where:
    #   0 = walkable (white)
    #   1 = furniture (light gray)
    #   2 = wall (dark gray)
    #   3 = charger (green — plotted as marker)
    grid = np.zeros((env.rows, env.cols))
    for (r, c) in env.furniture:
        grid[r, c] = 1
    for (r, c) in env.walls:
        grid[r, c] = 2

    # Custom colormap: walkable=white, furniture=lightgray, wall=darkgray
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#FFFFFF", "#B0B0B0", "#404040"])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, origin="upper")

    # Draw grid lines
    for i in range(env.rows + 1):
        ax.axhline(i - 0.5, color="black", linewidth=0.5)
    for j in range(env.cols + 1):
        ax.axvline(j - 0.5, color="black", linewidth=0.5)

    # Mark charger
    cr, cc = env.charger_pos
    ax.plot(cc, cr, "s", color="limegreen", markersize=18, label="Charger",
            markeredgecolor="black", markeredgewidth=1.5)

    # Room labels
    room_labels = [
        (2.5, 3.0, "Living Room"),
        (2.5, 11.0, "Kitchen"),
        (7.5, 7.0, "Hallway"),
        (12.0, 3.0, "Bedroom"),
        (12.0, 9.5, "Bath"),
        (12.0, 13.5, "Storage"),
    ]
    for row_pos, col_pos, name in room_labels:
        ax.text(col_pos, row_pos, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                          alpha=0.7, edgecolor="none"))

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FFFFFF", edgecolor="black", label="Walkable"),
        Patch(facecolor="#B0B0B0", edgecolor="black", label="Furniture"),
        Patch(facecolor="#404040", edgecolor="black", label="Wall"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="limegreen",
                   markersize=12, markeredgecolor="black", label="Charger"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9)

    ax.set_title("Apartment Layout (Phase 2)", fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))

    plt.tight_layout()
    return fig, ax


# =============================================================================
# Dirt Regeneration — Training & Evaluation
# =============================================================================

def train_qlearning_dirt(env, agent, num_episodes=DIRT_PATTERN_EPISODES):
    """
    Train a Q-Learning agent in a DirtRegenerationEnv.
    Captures dirt-specific metrics (total_cleans, room_visits, current_dirt).
    """
    metrics = {
        "rewards": [],
        "coverages": [],
        "steps": [],
        "deaths": [],
        "battery_at_end": [],
        "total_cleans": [],
        "room_visits": [],
        "current_dirt": [],
    }

    for ep in range(num_episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        total_reward = 0.0
        died = False

        while not env.done:
            action = agent.choose_action(state_idx)
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)
            agent.update(state_idx, action, reward, next_state_idx, done)
            state_idx = next_state_idx
            total_reward += reward
            if info["event"] == "battery_dead":
                died = True

        agent.decay_epsilon()

        metrics["rewards"].append(total_reward)
        metrics["coverages"].append(info["coverage"])
        metrics["steps"].append(info["steps"])
        metrics["deaths"].append(int(died))
        metrics["battery_at_end"].append(info["battery"])
        metrics["total_cleans"].append(info.get("total_cleans", 0))
        metrics["room_visits"].append(info.get("room_visits", {}))
        metrics["current_dirt"].append(info.get("current_dirt", 0))

    return metrics


def train_sarsa_dirt(env, agent, num_episodes=DIRT_PATTERN_EPISODES):
    """
    Train a SARSA agent in a DirtRegenerationEnv.
    Captures dirt-specific metrics.
    """
    metrics = {
        "rewards": [],
        "coverages": [],
        "steps": [],
        "deaths": [],
        "battery_at_end": [],
        "total_cleans": [],
        "room_visits": [],
        "current_dirt": [],
    }

    for ep in range(num_episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        action = agent.choose_action(state_idx)
        total_reward = 0.0
        died = False

        while not env.done:
            next_state, reward, done, info = env.step(action)
            next_state_idx = env.state_to_index(next_state)
            if done:
                agent.update(state_idx, action, reward, next_state_idx, done)
            else:
                next_action = agent.choose_action(next_state_idx)
                agent.update(
                    state_idx, action, reward, next_state_idx, done,
                    next_action=next_action,
                )
                action = next_action
            state_idx = next_state_idx
            total_reward += reward
            if info["event"] == "battery_dead":
                died = True

        agent.decay_epsilon()

        metrics["rewards"].append(total_reward)
        metrics["coverages"].append(info["coverage"])
        metrics["steps"].append(info["steps"])
        metrics["deaths"].append(int(died))
        metrics["battery_at_end"].append(info["battery"])
        metrics["total_cleans"].append(info.get("total_cleans", 0))
        metrics["room_visits"].append(info.get("room_visits", {}))
        metrics["current_dirt"].append(info.get("current_dirt", 0))

    return metrics


def train_agent_dirt(env, agent, num_episodes=DIRT_PATTERN_EPISODES):
    """Route to the correct dirt-regen training function based on agent type."""
    if isinstance(agent, SARSAAgent):
        return train_sarsa_dirt(env, agent, num_episodes)
    else:
        return train_qlearning_dirt(env, agent, num_episodes)


def run_dirt_experiment(algorithm, exploration, config=None,
                        num_episodes=DIRT_PATTERN_EPISODES,
                        max_steps=DIRT_PATTERN_MAX_STEPS,
                        seeds=SEEDS, **agent_kwargs):
    """
    Run a full dirt-regeneration experiment over multiple seeds.

    Returns
    -------
    all_metrics : list of dict
    agents : list of BaseAgent
    """
    cfg = config or DIRT_PATTERN_CONFIG
    all_metrics = []
    agents = []

    for seed in seeds:
        np.random.seed(seed)
        env = DirtRegenerationEnv(cfg, max_steps=max_steps)
        agent = create_agent(algorithm, exploration, env, **agent_kwargs)
        env.reset(seed=seed)
        metrics = train_agent_dirt(env, agent, num_episodes)
        all_metrics.append(metrics)
        agents.append(agent)

    return all_metrics, agents


def evaluate_dirt_agent(env, agent, num_episodes=100, seed=42):
    """
    Evaluate a trained agent greedily in a DirtRegenerationEnv.

    Returns metrics including room_visits for every evaluation episode.
    """
    metrics = {
        "rewards": [],
        "coverages": [],
        "steps": [],
        "deaths": [],
        "battery_at_end": [],
        "total_cleans": [],
        "room_visits": [],
        "current_dirt": [],
    }

    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        state_idx = env.state_to_index(state)
        total_reward = 0.0
        died = False

        while not env.done:
            action = agent.get_greedy_action(state_idx)
            next_state, reward, done, info = env.step(action)
            state_idx = env.state_to_index(next_state)
            total_reward += reward
            if info["event"] == "battery_dead":
                died = True

        metrics["rewards"].append(total_reward)
        metrics["coverages"].append(info["coverage"])
        metrics["steps"].append(info["steps"])
        metrics["deaths"].append(int(died))
        metrics["battery_at_end"].append(info["battery"])
        metrics["total_cleans"].append(info.get("total_cleans", 0))
        metrics["room_visits"].append(info.get("room_visits", {}))
        metrics["current_dirt"].append(info.get("current_dirt", 0))

    return metrics


# =============================================================================
# Dirt Regeneration — Analysis & Plotting
# =============================================================================

def compute_room_visit_ratios(eval_metrics):
    """
    Compute average visit counts per room from evaluation metrics.

    Parameters
    ----------
    eval_metrics : dict
        Output from evaluate_dirt_agent(), must contain 'room_visits' list.

    Returns
    -------
    avg_visits : dict
        {room_name: average_visits_per_episode}
    visit_ratios : dict
        {room_name: fraction_of_total_visits}
    """
    room_totals = defaultdict(float)
    n = len(eval_metrics["room_visits"])

    for rv in eval_metrics["room_visits"]:
        for room, count in rv.items():
            room_totals[room] += count

    avg_visits = {room: total / n for room, total in room_totals.items()}
    total_all = sum(avg_visits.values())

    if total_all > 0:
        visit_ratios = {room: v / total_all for room, v in avg_visits.items()}
    else:
        visit_ratios = {room: 0.0 for room in avg_visits}

    return avg_visits, visit_ratios


def compute_expected_dirt_ratios(dirt_burst_config=DIRT_BURST_CONFIG):
    """
    Compute expected relative dirt production rate per room.

    The expected dirt rate is proportional to:
        burst_probability * burst_intensity / burst_interval

    Returns
    -------
    dirt_ratios : dict
        {room_name: normalized_dirt_rate} summing to 1.0
    """
    raw_rates = {}
    for room, cfg in dirt_burst_config.items():
        rate = cfg["burst_probability"] * cfg["burst_intensity"] / cfg["burst_interval"]
        raw_rates[room] = rate

    total = sum(raw_rates.values())
    return {room: rate / total for room, rate in raw_rates.items()}


def plot_room_visit_comparison(eval_metrics_dict, dirt_burst_config=DIRT_BURST_CONFIG,
                                figsize=(14, 6)):
    """
    Bar chart comparing room visit ratios across agents, overlaid with
    expected dirt production ratios.

    Parameters
    ----------
    eval_metrics_dict : dict
        {agent_label: eval_metrics} from evaluate_dirt_agent()
    dirt_burst_config : dict
    """
    expected = compute_expected_dirt_ratios(dirt_burst_config)
    room_order = sorted(expected.keys(), key=lambda r: expected[r], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: absolute average visits ---
    ax = axes[0]
    x = np.arange(len(room_order))
    width = 0.8 / (len(eval_metrics_dict) + 1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(eval_metrics_dict) + 1))

    for i, (label, evals) in enumerate(eval_metrics_dict.items()):
        avg_visits, _ = compute_room_visit_ratios(evals)
        vals = [avg_visits.get(room, 0) for room in room_order]
        ax.bar(x + i * width, vals, width, label=label, color=colors[i])

    ax.set_xticks(x + width * len(eval_metrics_dict) / 2)
    ax.set_xticklabels([r.replace("_", " ").title() for r in room_order],
                       rotation=30, ha="right")
    ax.set_ylabel("Average Visits per Episode", fontsize=11)
    ax.set_title("Room Visit Frequency (Absolute)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Right: visit ratios vs expected dirt ratios ---
    ax = axes[1]
    for i, (label, evals) in enumerate(eval_metrics_dict.items()):
        _, ratios = compute_room_visit_ratios(evals)
        vals = [ratios.get(room, 0) for room in room_order]
        ax.bar(x + i * width, vals, width, label=label, color=colors[i])

    # Overlay expected dirt ratios
    expected_vals = [expected[room] for room in room_order]
    ax.bar(x + len(eval_metrics_dict) * width, expected_vals, width,
           label="Expected (dirt rate)", color=colors[len(eval_metrics_dict)],
           edgecolor="black", linewidth=1.5, linestyle="--")

    ax.set_xticks(x + width * (len(eval_metrics_dict) + 1) / 2)
    ax.set_xticklabels([r.replace("_", " ").title() for r in room_order],
                       rotation=30, ha="right")
    ax.set_ylabel("Fraction of Total Visits", fontsize=11)
    ax.set_title("Visit Ratio vs Expected Dirt Production", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig, axes


def plot_dirt_timeline(env, agent, max_steps=500, seed=42, figsize=(14, 5)):
    """
    Run one episode and plot the dirt level over time, marking burst events.

    Parameters
    ----------
    env : DirtRegenerationEnv
    agent : BaseAgent
    max_steps : int
    seed : int
    """
    state = env.reset(seed=seed)
    state_idx = env.state_to_index(state)

    dirt_levels = [int(env.dirt_grid.sum())]
    clean_events = []  # steps where a tile was cleaned

    step = 0
    while not env.done and step < max_steps:
        action = agent.get_greedy_action(state_idx)
        next_state, reward, done, info = env.step(action)
        state_idx = env.state_to_index(next_state)
        dirt_levels.append(info["current_dirt"])
        if info["event"] == "clean_dirty":
            clean_events.append(step)
        step += 1

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(range(len(dirt_levels)), dirt_levels, color="firebrick",
            linewidth=1.5, label="Current dirty tiles")

    # Mark burst events
    for burst_step, bursts in env.burst_log:
        total_re_dirtied = sum(n for _, n in bursts)
        ax.axvline(burst_step, color="orange", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Number of Dirty Tiles", fontsize=12)
    ax.set_title("Dirt Level Over Time (Single Episode)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add orange patch to legend for burst markers
    from matplotlib.patches import Patch
    handles = ax.get_legend_handles_labels()[0]
    handles.append(Patch(facecolor="orange", alpha=0.3, label="Dirt burst"))
    ax.legend(handles=handles, fontsize=11)

    plt.tight_layout()
    return fig, ax


def plot_cleans_over_training(results_dict, window=200, figsize=(12, 5)):
    """
    Plot total cleans per episode over training — shows if the agent learns
    to clean more as it adapts to dirt patterns.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for label, all_metrics in results_dict.items():
        mean, std = aggregate_metrics(all_metrics, "total_cleans", window)
        episodes = np.arange(len(mean))
        ax.plot(episodes, mean, label=label, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Cleans per Episode", fontsize=12)
    ax.set_title("Cleaning Efficiency Over Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_room_visit_evolution(all_metrics, room_order=None, window=500,
                               figsize=(12, 6)):
    """
    Plot how room visit proportions evolve over training (averaged across seeds).

    Parameters
    ----------
    all_metrics : list of dict
        Metrics from one experiment (list of per-seed dicts).
    room_order : list of str, optional
    window : int
    """
    if room_order is None:
        room_order = ["kitchen", "living_room", "bedroom", "bathroom",
                       "hallway", "storage"]

    # Extract room visit ratios per episode per seed
    n_episodes = len(all_metrics[0]["room_visits"])

    # For each seed, compute visit ratio time series per room
    room_ratio_series = {room: [] for room in room_order}
    for seed_metrics in all_metrics:
        for room in room_order:
            series = []
            for rv in seed_metrics["room_visits"]:
                total = sum(rv.values()) if rv else 1
                series.append(rv.get(room, 0) / max(total, 1))
            room_ratio_series[room].append(series)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(room_order)))

    for i, room in enumerate(room_order):
        # Average across seeds, then smooth
        all_seeds = np.array(room_ratio_series[room])
        mean_series = all_seeds.mean(axis=0)
        smoothed = smooth(mean_series, window)
        episodes = np.arange(len(smoothed))
        ax.plot(episodes, smoothed, label=room.replace("_", " ").title(),
                linewidth=2, color=colors[i])

    # Add horizontal lines for expected dirt ratios
    expected = compute_expected_dirt_ratios()
    for i, room in enumerate(room_order):
        ax.axhline(expected[room], color=colors[i], linestyle="--", alpha=0.5)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Fraction of Visits", fontsize=12)
    ax.set_title("Room Visit Proportions Over Training\n(dashed = expected dirt rate)",
                 fontsize=14)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def dirt_summary_table(eval_metrics_dict, dirt_burst_config=DIRT_BURST_CONFIG):
    """
    Print a summary table for dirt-pattern experiments showing:
    - Per-agent: avg reward, total cleans, death rate, battery remaining
    - Per-room visit ratios vs expected dirt ratios
    """
    expected = compute_expected_dirt_ratios(dirt_burst_config)
    room_order = sorted(expected.keys(), key=lambda r: expected[r], reverse=True)

    # Header
    room_headers = "  ".join(f"{r.replace('_', ' ').title():>12}" for r in room_order)
    print(f"{'Agent':<30} {'Reward':>8} {'Cleans':>8} {'Death%':>7} {'Batt':>6}  "
          f"│ Room Visit Ratios (vs Expected)")
    print(f"{'':30} {'':>8} {'':>8} {'':>7} {'':>6}  │ {room_headers}")
    print("-" * (65 + 14 * len(room_order)))

    # Expected row
    exp_str = "  ".join(f"{expected[r]*100:>11.1f}%" for r in room_order)
    print(f"{'Expected (dirt rate)':<30} {'':>8} {'':>8} {'':>7} {'':>6}  │ {exp_str}")
    print("-" * (65 + 14 * len(room_order)))

    for label, evals in eval_metrics_dict.items():
        avg_reward = np.mean(evals["rewards"])
        avg_cleans = np.mean(evals["total_cleans"])
        death_rate = np.mean(evals["deaths"]) * 100
        avg_battery = np.mean(evals["battery_at_end"])

        _, ratios = compute_room_visit_ratios(evals)
        ratio_str = "  ".join(f"{ratios.get(r, 0)*100:>11.1f}%" for r in room_order)

        print(f"{label:<30} {avg_reward:>8.1f} {avg_cleans:>8.1f} "
              f"{death_rate:>6.1f}% {avg_battery:>6.1f}  │ {ratio_str}")

    print("-" * (65 + 14 * len(room_order)))
