"""
================================================================================
PLOTTING UTILITIES - Visualization Functions for Training Results
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning (Q-Learning)
FILE: utils/plotting.py
PURPOSE: Create visualizations of training progress and agent performance

================================================================================
📚 OVERVIEW
================================================================================

This module provides plotting functions for visualizing:
1. Training Progress - Rewards, completion rates over episodes
2. Learning Curves - How the agent improves over time
3. Performance Comparisons - Trained agent vs random baseline

All plots use Matplotlib and are designed to help understand
how the RL agent learns and performs.

================================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_results(
    episode_rewards,
    completion_rates,
    steps_per_episode,
    epsilon_history=None,
    save_path="plots",
    show_plot=True,
    window=100
):
    """
    Create comprehensive training visualization.
    
    Generates a multi-subplot figure showing all key training metrics:
    - Episode rewards with smoothed trend line
    - Completion rates over time
    - Steps per episode
    - Epsilon decay (if provided)
    
    Parameters:
    -----------
    episode_rewards : list
        Reward earned in each episode
    completion_rates : list
        Percentage of cleaning completed each episode (0-100)
    steps_per_episode : list
        Number of steps taken each episode
    epsilon_history : list, optional
        Epsilon value at each episode
    save_path : str
        Directory to save the plot
    show_plot : bool
        Whether to display the plot
    window : int
        Window size for moving average smoothing
    
    Returns:
    --------
    str
        Path to the saved plot image
    """
    
    # ========================================================================
    # SETUP FIGURE
    # ========================================================================
    
    # Determine number of subplots based on available data
    num_plots = 3 if epsilon_history is None else 4
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        num_plots, 1,
        figsize=(12, 3 * num_plots),
        sharex=True
    )
    
    # Main title
    fig.suptitle(
        "Q-Learning Training Progress - Cleaning Robot",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Number of episodes
    episodes = range(1, len(episode_rewards) + 1)
    
    # ========================================================================
    # PLOT 1: EPISODE REWARDS
    # ========================================================================
    
    ax1 = axes[0]
    
    # Plot raw rewards with transparency
    ax1.plot(
        episodes,
        episode_rewards,
        alpha=0.3,
        color='blue',
        linewidth=0.5,
        label='Raw Rewards'
    )
    
    # Plot smoothed rewards (moving average)
    smoothed_rewards = _moving_average(episode_rewards, window)
    ax1.plot(
        episodes,
        smoothed_rewards,
        color='blue',
        linewidth=2,
        label=f'Smoothed ({window}-episode avg)'
    )
    
    # Formatting
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # ========================================================================
    # PLOT 2: COMPLETION RATES
    # ========================================================================
    
    ax2 = axes[1]
    
    # Plot raw completion rates
    ax2.plot(
        episodes,
        completion_rates,
        alpha=0.3,
        color='green',
        linewidth=0.5,
        label='Raw Completion'
    )
    
    # Plot smoothed completion rates
    smoothed_completion = _moving_average(completion_rates, window)
    ax2.plot(
        episodes,
        smoothed_completion,
        color='green',
        linewidth=2,
        label=f'Smoothed ({window}-episode avg)'
    )
    
    # Formatting
    ax2.set_ylabel('Completion %')
    ax2.set_title('Cleaning Completion Rate')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)  # Percentage scale
    ax2.axhline(y=100, color='gold', linestyle='--', alpha=0.7, linewidth=2, label='100% Target')
    
    # ========================================================================
    # PLOT 3: STEPS PER EPISODE
    # ========================================================================
    
    ax3 = axes[2]
    
    # Plot raw steps
    ax3.plot(
        episodes,
        steps_per_episode,
        alpha=0.3,
        color='purple',
        linewidth=0.5,
        label='Raw Steps'
    )
    
    # Plot smoothed steps
    smoothed_steps = _moving_average(steps_per_episode, window)
    ax3.plot(
        episodes,
        smoothed_steps,
        color='purple',
        linewidth=2,
        label=f'Smoothed ({window}-episode avg)'
    )
    
    # Formatting
    ax3.set_ylabel('Steps')
    ax3.set_title('Steps Per Episode')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # ========================================================================
    # PLOT 4: EPSILON DECAY (if available)
    # ========================================================================
    
    if epsilon_history is not None and len(axes) > 3:
        ax4 = axes[3]
        
        ax4.plot(
            episodes,
            epsilon_history,
            color='orange',
            linewidth=2,
            label='Epsilon'
        )
        
        # Formatting
        ax4.set_ylabel('Epsilon (ε)')
        ax4.set_title('Exploration Rate (Epsilon Decay)')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-0.05, 1.05)
        ax4.set_xlabel('Episode')
    else:
        axes[-1].set_xlabel('Episode')
    
    # ========================================================================
    # SAVE AND DISPLAY
    # ========================================================================
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Create save directory if needed
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "training_progress.png")
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {save_file}")
    else:
        save_file = None
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return save_file


def plot_comparison(
    trained_rewards,
    random_rewards,
    trained_completions,
    random_completions,
    save_path="plots",
    show_plot=True
):
    """
    Create comparison visualization between trained and random agents.
    
    Shows side-by-side comparison to demonstrate learning effectiveness.
    
    Parameters:
    -----------
    trained_rewards : list
        Rewards from trained agent episodes
    random_rewards : list
        Rewards from random agent episodes
    trained_completions : list
        Completion rates from trained agent
    random_completions : list
        Completion rates from random agent
    save_path : str
        Directory to save the plot
    show_plot : bool
        Whether to display the plot
    
    Returns:
    --------
    str
        Path to saved plot
    """
    
    # ========================================================================
    # CALCULATE STATISTICS
    # ========================================================================
    
    trained_reward_avg = np.mean(trained_rewards)
    trained_reward_std = np.std(trained_rewards)
    random_reward_avg = np.mean(random_rewards)
    random_reward_std = np.std(random_rewards)
    
    trained_comp_avg = np.mean(trained_completions)
    random_comp_avg = np.mean(random_completions)
    
    # ========================================================================
    # CREATE FIGURE
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    fig.suptitle(
        "Q-Learning Agent vs Random Agent Comparison",
        fontsize=14,
        fontweight='bold'
    )
    
    # ========================================================================
    # PLOT 1: REWARD COMPARISON (BAR CHART)
    # ========================================================================
    
    ax1 = axes[0]
    
    # Bar positions and data
    labels = ['Trained Agent', 'Random Agent']
    rewards = [trained_reward_avg, random_reward_avg]
    stds = [trained_reward_std, random_reward_std]
    colors = ['#2ecc71', '#e74c3c']  # Green and Red
    
    # Create bars
    bars = ax1.bar(labels, rewards, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax1.errorbar(
        labels, rewards,
        yerr=stds,
        fmt='none',
        color='black',
        capsize=10,
        capthick=2
    )
    
    # Add value labels on bars
    for bar, reward, std in zip(bars, rewards, stds):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + std + 5,
            f'{reward:.1f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Average Reward Comparison', fontsize=12)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # ========================================================================
    # PLOT 2: COMPLETION RATE COMPARISON (BAR CHART)
    # ========================================================================
    
    ax2 = axes[1]
    
    completions = [trained_comp_avg, random_comp_avg]
    
    # Create bars
    bars = ax2.bar(labels, completions, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, comp in zip(bars, completions):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f'{comp:.1f}%',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    ax2.set_ylabel('Completion Rate (%)', fontsize=12)
    ax2.set_title('Average Completion Rate Comparison', fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color='gold', linestyle='--', alpha=0.7, linewidth=2)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # ========================================================================
    # SAVE AND DISPLAY
    # ========================================================================
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "agent_comparison.png")
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"  Comparison plot saved to: {save_file}")
    else:
        save_file = None
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return save_file


def plot_q_values_heatmap(
    q_table,
    num_positions,
    action_names=['Up', 'Down', 'Left', 'Right', 'Clean'],
    save_path="plots",
    show_plot=True
):
    """
    Visualize Q-values as a heatmap.
    
    Shows the learned Q-values for each state-action pair,
    helping understand what the agent has learned.
    
    Parameters:
    -----------
    q_table : dict
        Q-table dictionary {state: [q_values per action]}
    num_positions : int
        Number of position states (typically 23)
    action_names : list
        Names for each action
    save_path : str
        Directory to save plot
    show_plot : bool
        Whether to display
    
    Returns:
    --------
    str
        Path to saved plot
    """
    
    # ========================================================================
    # EXTRACT Q-VALUES
    # ========================================================================
    
    num_states = num_positions * 2  # positions × dirty status
    num_actions = len(action_names)
    
    # Create matrix of Q-values
    q_matrix = np.zeros((num_states, num_actions))
    
    for state in range(num_states):
        if state in q_table:
            q_matrix[state] = q_table[state]
    
    # ========================================================================
    # CREATE HEATMAP
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Create heatmap
    im = ax.imshow(q_matrix, aspect='auto', cmap='RdYlGn')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-Value', fontsize=12)
    
    # Set labels
    ax.set_xticks(range(num_actions))
    ax.set_xticklabels(action_names, fontsize=10)
    ax.set_xlabel('Action', fontsize=12)
    
    ax.set_yticks(range(num_states))
    state_labels = [f"Pos{s//2}-{'D' if s%2 else 'C'}" for s in range(num_states)]
    ax.set_yticklabels(state_labels, fontsize=8)
    ax.set_ylabel('State (Position-DirtyStatus)', fontsize=12)
    
    ax.set_title('Q-Values Heatmap\n(Green=High, Red=Low)', fontsize=14)
    
    # ========================================================================
    # SAVE AND DISPLAY
    # ========================================================================
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "q_values_heatmap.png")
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"  Q-values heatmap saved to: {save_file}")
    else:
        save_file = None
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return save_file


def _moving_average(data, window):
    """
    Calculate moving average for smoothing.
    
    Internal helper function for plot smoothing.
    
    Parameters:
    -----------
    data : list
        Data to smooth
    window : int
        Window size
    
    Returns:
    --------
    list
        Smoothed data
    """
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = data[start:i + 1]
        avg = sum(window_data) / len(window_data)
        result.append(avg)
    return result


# ================================================================================
# MODULE TEST
# ================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TESTING PLOTTING FUNCTIONS")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    num_episodes = 500
    
    # Simulate training progress (improving over time)
    base_reward = -50 + np.arange(num_episodes) * 0.5
    noise = np.random.randn(num_episodes) * 30
    episode_rewards = (base_reward + noise).tolist()
    
    base_completion = 10 + np.arange(num_episodes) * 0.18
    noise = np.random.randn(num_episodes) * 10
    completion_rates = np.clip(base_completion + noise, 0, 100).tolist()
    
    base_steps = 200 - np.arange(num_episodes) * 0.2
    noise = np.random.randn(num_episodes) * 20
    steps_per_episode = np.clip(base_steps + noise, 23, 200).tolist()
    
    epsilon_history = [max(0.01, 1.0 * (0.995 ** i)) for i in range(num_episodes)]
    
    print("\n1. Testing plot_training_results():")
    print(f"   Generated {num_episodes} episodes of sample data")
    
    # Create plot
    plot_training_results(
        episode_rewards=episode_rewards,
        completion_rates=completion_rates,
        steps_per_episode=steps_per_episode,
        epsilon_history=epsilon_history,
        save_path="plots",
        show_plot=True
    )
    
    print("\n2. Testing plot_comparison():")
    
    # Generate comparison data
    trained_rewards = [150 + np.random.randn() * 20 for _ in range(20)]
    random_rewards = [-30 + np.random.randn() * 15 for _ in range(20)]
    trained_completions = [95 + np.random.randn() * 5 for _ in range(20)]
    random_completions = [25 + np.random.randn() * 10 for _ in range(20)]
    
    plot_comparison(
        trained_rewards=trained_rewards,
        random_rewards=random_rewards,
        trained_completions=trained_completions,
        random_completions=random_completions,
        save_path="plots",
        show_plot=True
    )
    
    print("\nAll plotting tests completed!")
