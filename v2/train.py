"""
================================================================================
TRAINING SCRIPT - Train the Q-Learning Cleaning Robot
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning (Q-Learning)
FILE: train.py
PURPOSE: Training loop for the Q-Learning agent

================================================================================
📚 TRAINING PROCESS OVERVIEW
================================================================================

The training loop follows this structure for each episode:

1. RESET: Start a new episode with all tiles dirty
2. LOOP until episode ends:
   a. Agent observes current state
   b. Agent chooses action (epsilon-greedy)
   c. Environment executes action
   d. Environment returns reward and next state
   e. Agent updates Q-table using Q-learning update rule
   f. Move to next state
3. END EPISODE: Decay epsilon, record statistics
4. REPEAT for many episodes

Each episode ends when:
- All tiles are cleaned (success - terminated)
- Maximum steps reached (truncated)

================================================================================
🎯 TRAINING PARAMETERS
================================================================================

NUMBER OF EPISODES:
- More episodes = better learning (but takes longer)
- For pure RL (no hints): 3000-5000 episodes recommended
- For complex tasks: May need 10,000+ episodes

EPSILON SCHEDULE:
- Start: 1.0 (100% random exploration)
- Decay: 0.998 per episode
- End: 0.01 (1% random after many episodes)

After ~1000 episodes with 0.998 decay: epsilon ≈ 0.13 (13% exploration)
After ~2000 episodes: epsilon ≈ 0.018 (mostly exploitation)

================================================================================
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from env.cleaning_env import CleaningEnv
from agent.q_learning_agent import QLearningAgent
from utils.helpers import print_progress_bar, format_time
from utils.plotting import plot_training_results


def train(num_episodes=5000, render_every=1000, save_path="models/q_table.pkl"):
    """
    Train the Q-Learning agent to clean the house.
    
    This function implements the main training loop for Q-Learning.
    The agent learns through trial and error over many episodes,
    gradually discovering the optimal cleaning policy.
    
    Parameters:
    -----------
    num_episodes : int
        Number of episodes to train for.
        - 2000-3000: Quick training, decent results
        - 5000+: Better learning, recommended for pure RL
        - 10000+: Very thorough training
    
    render_every : int
        How often to show visual rendering (0 = never render).
        - 500: Show every 500 episodes (good for monitoring)
        - 0: No rendering (fastest training)
    
    save_path : str
        Path to save the trained Q-table.
        The directory will be created if it doesn't exist.
    
    Returns:
    --------
    dict
        Training results containing:
        - rewards: List of episode rewards
        - tiles_cleaned: List of tiles cleaned per episode
        - epsilon_history: Epsilon values over training
        - steps_per_episode: Steps taken per episode
    """
    print("\n" + "=" * 70)
    print("  TRAINING Q-LEARNING CLEANING ROBOT (Pure RL)")
    print("=" * 70)
    print(f"\n  Training Parameters:")
    print(f"    Episodes:         {num_episodes}")
    print(f"    Render every:     {render_every if render_every > 0 else 'Never'}")
    print(f"    Save path:        {save_path}")
    print(f"    Start time:       {datetime.now().strftime('%H:%M:%S')}")
    print("\n" + "-" * 70)
    
    # ==========================================================================
    # STEP 1: CREATE ENVIRONMENT AND AGENT
    # ==========================================================================
    
    # Create environment (no rendering for training speed)
    env = CleaningEnv(render_mode=None)
    
    # Also create a rendered environment for occasional visualization
    env_visual = CleaningEnv(render_mode="human") if render_every > 0 else None
    
    # Create Q-Learning agent with optimized hyperparameters for 2300-state space
    agent = QLearningAgent(
        state_size=env.observation_space.n,    # 2300 states (pos × dirt × dir × DNUT)
        action_size=env.action_space.n,        # 6 actions
        learning_rate=0.15,                    # α = 0.15 (slightly faster for larger space)
        discount_factor=0.99,                  # γ = 0.99 (strongly value future rewards)
        epsilon_start=1.0,                     # Start with 100% exploration
        epsilon_end=0.02,                      # End with 2% exploration
        epsilon_decay=0.998                    # Faster decay → ε ≈ 0.002 at ep 3000
    )
    
    # ==========================================================================
    # STEP 2: INITIALIZE TRACKING VARIABLES
    # ==========================================================================
    
    # Lists to store training metrics
    episode_rewards = []           # Total reward per episode
    episode_tiles_cleaned = []     # Tiles cleaned per episode
    episode_steps = []             # Steps taken per episode
    epsilon_history = []           # Epsilon value per episode
    success_history = []           # Whether episode was successful
    
    # Variables for progress tracking
    best_reward = float('-inf')    # Best episode reward seen
    best_tiles = 0                 # Best tiles cleaned seen
    total_successes = 0            # Total successful episodes
    start_time = time.time()       # Training start time
    
    # ==========================================================================
    # STEP 3: MAIN TRAINING LOOP
    # ==========================================================================
    
    print(f"\n  Starting training at epsilon = {agent.epsilon:.2f}")
    print("  " + "-" * 68)
    
    for episode in range(1, num_episodes + 1):
        # ======================================================================
        # RESET EPISODE
        # ======================================================================
        # Choose which environment to use (visual or fast)
        should_render = render_every > 0 and episode % render_every == 0
        current_env = env_visual if should_render else env
        
        # Reset environment to get initial state
        state, info = current_env.reset()
        
        # Episode tracking
        episode_reward = 0     # Total reward this episode
        steps = 0              # Steps this episode
        done = False           # Episode finished flag
        
        # ======================================================================
        # EPISODE LOOP - Run until episode ends
        # ======================================================================
        while not done:
            # ------------------------------------------------------------------
            # 1. Choose action using epsilon-greedy policy
            # ------------------------------------------------------------------
            action = agent.choose_action(state, training=True)
            
            # ------------------------------------------------------------------
            # 2. Execute action in environment
            # ------------------------------------------------------------------
            next_state, reward, terminated, truncated, info = current_env.step(action)
            done = terminated or truncated
            
            # ------------------------------------------------------------------
            # 3. Update Q-table using Q-learning rule
            # ------------------------------------------------------------------
            agent.learn(state, action, reward, next_state, done)
            
            # ------------------------------------------------------------------
            # 4. Update tracking and move to next state
            # ------------------------------------------------------------------
            episode_reward += reward
            steps += 1
            state = next_state
            
            # ------------------------------------------------------------------
            # 5. Render if this is a visual episode
            # ------------------------------------------------------------------
            if should_render:
                current_env.render()
        
        # ======================================================================
        # END OF EPISODE - Update statistics and decay epsilon
        # ======================================================================
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_tiles_cleaned.append(info['tiles_cleaned'])
        episode_steps.append(steps)
        epsilon_history.append(agent.epsilon)
        
        # Track success (all tiles cleaned)
        success = info['tiles_cleaned'] == env.num_cleanable
        success_history.append(success)
        if success:
            total_successes += 1
        
        # Update best performance
        if episode_reward > best_reward:
            best_reward = episode_reward
        if info['tiles_cleaned'] > best_tiles:
            best_tiles = info['tiles_cleaned']
        
        # Decay epsilon for next episode
        agent.end_episode()
        
        # ======================================================================
        # PRINT PROGRESS (every 100 episodes)
        # ======================================================================
        if episode % 100 == 0 or episode == 1:
            # Calculate recent averages (last 100 episodes)
            recent_rewards = episode_rewards[-100:]
            recent_tiles = episode_tiles_cleaned[-100:]
            recent_success = success_history[-100:]
            
            avg_reward = np.mean(recent_rewards)
            avg_tiles = np.mean(recent_tiles)
            success_rate = sum(recent_success) / len(recent_success) * 100
            
            # Calculate elapsed time
            elapsed = time.time() - start_time
            
            # Print progress line
            print(f"  Episode {episode:5d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:7.1f} | "
                  f"Avg Tiles: {avg_tiles:5.1f}/{env.num_cleanable} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Time: {format_time(elapsed)}")
    
    # ==========================================================================
    # STEP 4: TRAINING COMPLETE - Save and report results
    # ==========================================================================
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    
    # Calculate final statistics
    final_100_rewards = episode_rewards[-100:]
    final_100_tiles = episode_tiles_cleaned[-100:]
    final_100_success = success_history[-100:]
    
    print(f"\n  Final Statistics (last 100 episodes):")
    print(f"    Average Reward:    {np.mean(final_100_rewards):.1f}")
    print(f"    Average Tiles:     {np.mean(final_100_tiles):.1f}/{env.num_cleanable}")
    print(f"    Success Rate:      {sum(final_100_success)}%")
    print(f"    Best Reward:       {best_reward:.1f}")
    print(f"    Best Tiles:        {best_tiles}/{env.num_cleanable}")
    
    print(f"\n  Training Summary:")
    print(f"    Total Episodes:    {num_episodes}")
    print(f"    Total Successes:   {total_successes} ({total_successes/num_episodes*100:.1f}%)")
    print(f"    Final Epsilon:     {agent.epsilon:.4f}")
    print(f"    Q-Table Entries:   {len(agent.q_table)}")
    print(f"    Total Time:        {format_time(total_time)}")
    print(f"    Time/Episode:      {total_time/num_episodes*1000:.1f}ms")
    
    # ==========================================================================
    # STEP 5: SAVE THE TRAINED MODEL
    # ==========================================================================
    
    print(f"\n  Saving trained model...")
    agent.save(save_path)
    
    # ==========================================================================
    # STEP 6: GENERATE TRAINING PLOTS
    # ==========================================================================
    
    print(f"\n  Generating training plots...")
    
    try:
        # Calculate completion rates from tiles cleaned
        completion_rates = [(tc / env.num_cleanable * 100) for tc in episode_tiles_cleaned]
        
        plot_training_results(
            episode_rewards=episode_rewards,
            completion_rates=completion_rates,
            steps_per_episode=episode_steps,
            epsilon_history=epsilon_history,
            save_path="plots",
            show_plot=False
        )
        print(f"    Plots saved to: plots/")
    except Exception as e:
        print(f"    Warning: Could not generate plots: {e}")
    
    # ==========================================================================
    # STEP 7: CLEANUP
    # ==========================================================================
    
    env.close()
    if env_visual:
        env_visual.close()
    
    print("\n" + "=" * 70)
    
    training_results = {
        "rewards": episode_rewards,
        "tiles_cleaned": episode_tiles_cleaned,
        "steps_per_episode": episode_steps,
        "epsilon_history": epsilon_history,
        "success_history": success_history,
    }
    
    return training_results


def train_with_baseline(num_episodes=3000, save_path="models/q_table.pkl"):
    """
    Train the agent and compare with random baseline.
    
    This function trains the Q-Learning agent and also runs a random
    agent for comparison, showing how much the trained agent improves.
    
    Parameters:
    -----------
    num_episodes : int
        Number of episodes for training
    save_path : str
        Path to save trained model
    
    Returns:
    --------
    dict
        Training results plus baseline comparison
    """
    print("\n" + "=" * 70)
    print("  TRAINING WITH BASELINE COMPARISON")
    print("=" * 70)
    
    # ======================================================================
    # Run baseline (random agent) first
    # ======================================================================
    print("\n  Running random baseline (100 episodes)...")
    
    env = CleaningEnv(render_mode=None)
    baseline_rewards = []
    baseline_tiles = []
    
    for _ in range(100):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()  # Random action
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        baseline_rewards.append(episode_reward)
        baseline_tiles.append(info['tiles_cleaned'])
    
    baseline_avg_reward = np.mean(baseline_rewards)
    baseline_avg_tiles = np.mean(baseline_tiles)
    
    print(f"    Random Baseline Results:")
    print(f"      Avg Reward: {baseline_avg_reward:.1f}")
    print(f"      Avg Tiles:  {baseline_avg_tiles:.1f}/{env.num_cleanable}")
    
    env.close()
    
    # ======================================================================
    # Train the Q-Learning agent
    # ======================================================================
    results = train(
        num_episodes=num_episodes,
        render_every=500,
        save_path=save_path
    )
    
    # ======================================================================
    # Compare with baseline
    # ======================================================================
    trained_avg = np.mean(results["rewards"][-100:])
    improvement = trained_avg - baseline_avg_reward
    
    print("\n" + "=" * 70)
    print("  IMPROVEMENT OVER RANDOM BASELINE")
    print("=" * 70)
    print(f"\n    Random Baseline:  {baseline_avg_reward:.1f} avg reward")
    print(f"    Trained Agent:    {trained_avg:.1f} avg reward")
    print(f"    Improvement:      {improvement:+.1f} ({improvement/abs(baseline_avg_reward)*100:+.1f}%)")
    print("\n" + "=" * 70)
    
    results["baseline_reward"] = baseline_avg_reward
    results["baseline_tiles"] = baseline_avg_tiles
    
    return results


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    """
    Run training when this script is executed directly.
    
    Usage:
        python train.py
        python train.py --episodes 5000
        python train.py --baseline
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Q-Learning Cleaning Robot")
    parser.add_argument("--episodes", type=int, default=3000,
                       help="Number of training episodes (default: 3000)")
    parser.add_argument("--render", type=int, default=500,
                       help="Render every N episodes (0 = never, default: 500)")
    parser.add_argument("--save", type=str, default="models/q_table.pkl",
                       help="Path to save trained model")
    parser.add_argument("--baseline", action="store_true",
                       help="Also run baseline comparison")
    
    args = parser.parse_args()
    
    # Run training
    print("\n" + "=" * 70)
    print("  CLEANING ROBOT Q-LEARNING TRAINER")
    print("=" * 70)
    
    if args.baseline:
        results = train_with_baseline(
            num_episodes=args.episodes,
            save_path=args.save
        )
    else:
        results = train(
            num_episodes=args.episodes,
            render_every=args.render,
            save_path=args.save
        )
    
    print("\n  Training script finished!")
    print("  Run 'python test.py' to test the trained agent.")
    print("  Run 'python main.py' for interactive menu.\n")
