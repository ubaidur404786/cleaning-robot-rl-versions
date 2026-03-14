"""
================================================================================
TESTING SCRIPT - Evaluate the Trained Q-Learning Agent
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning (Q-Learning)
FILE: test.py
PURPOSE: Test and evaluate the trained cleaning robot agent

================================================================================
📚 TESTING OVERVIEW
================================================================================

Testing differs from training in several key ways:

TRAINING:
- Epsilon-greedy exploration (random actions with probability ε)
- Q-table updates after each action
- Focus on learning, not performance

TESTING:
- Pure exploitation (ε = 0, always use best action)
- No Q-table updates (frozen policy)
- Focus on evaluating learned performance

================================================================================
🎯 TEST METRICS
================================================================================

The test script evaluates the agent on these metrics:

1. AVERAGE REWARD:
   - Higher is better
   - Includes cleaning rewards (+20 to +50) and penalties

2. TILES CLEANED:
   - Out of 23 total tiles
   - 100% completion = 23/23 tiles

3. SUCCESS RATE:
   - Percentage of episodes where ALL tiles are cleaned
   - Target: 90%+ for well-trained agent

4. STEPS TAKEN:
   - Fewer steps = more efficient
   - Shows if agent learns efficient paths

5. COMPARISON TO RANDOM:
   - How much better is trained agent vs random actions?
   - Shows that learning actually occurred

================================================================================
"""

import os
import sys
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from env.cleaning_env import CleaningEnv
from agent.q_learning_agent import QLearningAgent
from utils.helpers import format_time


def test(num_episodes=10, render=True, speed="normal", model_path="models/q_table.pkl"):
    """
    Test the trained Q-Learning agent.
    
    This function loads a trained agent and runs it through multiple
    test episodes to evaluate its performance. Unlike training, no
    learning occurs and the agent always uses its best learned action.
    
    Parameters:
    -----------
    num_episodes : int
        Number of test episodes to run.
        - 1-5: Quick test to see behavior
        - 10-20: Good statistical evaluation
        - 100: Thorough performance assessment
    
    render : bool
        Whether to show visual rendering.
        - True: Watch the robot clean (slower but informative)
        - False: Run without graphics (faster evaluation)
    
    speed : str
        Rendering speed: "slow", "normal", or "fast"
        - "slow": 5 FPS - Easy to follow each action
        - "normal": 10 FPS - Default speed
        - "fast": 20 FPS - Quick visualization
    
    model_path : str
        Path to the saved Q-table file.
    
    Returns:
    --------
    dict
        Test results containing performance metrics
    """
    print("\n" + "=" * 70)
    print("  TESTING Q-LEARNING CLEANING ROBOT")
    print("=" * 70)
    
    # Set rendering FPS based on speed
    speed_map = {"slow": 5, "normal": 10, "fast": 20}
    fps = speed_map.get(speed, 10)
    
    print(f"\n  Test Parameters:")
    print(f"    Episodes:     {num_episodes}")
    print(f"    Render:       {'Yes' if render else 'No'}")
    print(f"    Speed:        {speed} ({fps} FPS)")
    print(f"    Model:        {model_path}")
    
    # ==========================================================================
    # STEP 1: CREATE ENVIRONMENT AND AGENT
    # ==========================================================================
    
    print("\n  Loading environment and agent...")
    
    # Create environment (with or without rendering)
    render_mode = "human" if render else None
    env = CleaningEnv(render_mode=render_mode)
    
    # Set custom FPS if rendering
    if render:
        env.metadata["render_fps"] = fps
    
    # Create agent and load trained Q-table
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n
    )
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"\n  ERROR: Model file not found: {model_path}")
        print("  Please train the agent first using: python train.py")
        return None
    
    agent.load(model_path)
    
    # ==========================================================================
    # STEP 2: RUN TEST EPISODES (TRAINED AGENT)
    # ==========================================================================
    
    print("\n" + "-" * 70)
    print("  Running Test Episodes (Trained Agent)")
    print("-" * 70)
    
    # Tracking variables
    test_rewards = []
    test_tiles = []
    test_steps = []
    test_successes = []
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        state, info = env.reset()
        
        episode_reward = 0
        steps = 0
        done = False
        
        # Run episode
        while not done:
            # Choose best action (no exploration during testing)
            action = agent.choose_action(state, training=False)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Render if enabled
            if render:
                env.render()
        
        # Record episode results
        test_rewards.append(episode_reward)
        test_tiles.append(info['tiles_cleaned'])
        test_steps.append(steps)
        success = info['tiles_cleaned'] == env.num_cleanable
        test_successes.append(success)
        
        # Print episode summary
        status = "SUCCESS" if success else "INCOMPLETE"
        print(f"  Episode {episode:3d}: Reward={episode_reward:7.1f} | "
              f"Tiles={info['tiles_cleaned']:2d}/{env.num_cleanable} | "
              f"Steps={steps:3d} | {status}")
    
    # ==========================================================================
    # STEP 3: RUN RANDOM BASELINE FOR COMPARISON
    # ==========================================================================
    
    print("\n" + "-" * 70)
    print("  Running Random Baseline (for comparison)")
    print("-" * 70)
    
    # Run same number of episodes with random actions
    env_fast = CleaningEnv(render_mode=None)  # No rendering for speed
    
    random_rewards = []
    random_tiles = []
    random_successes = []
    
    for episode in range(1, num_episodes + 1):
        state, info = env_fast.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env_fast.action_space.sample()  # Random action
            state, reward, terminated, truncated, info = env_fast.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        random_rewards.append(episode_reward)
        random_tiles.append(info['tiles_cleaned'])
        random_successes.append(info['tiles_cleaned'] == env.num_cleanable)
    
    env_fast.close()
    
    # ==========================================================================
    # STEP 4: CALCULATE AND DISPLAY RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("  TEST RESULTS SUMMARY")
    print("=" * 70)
    
    # Trained agent statistics
    trained_avg_reward = np.mean(test_rewards)
    trained_avg_tiles = np.mean(test_tiles)
    trained_avg_steps = np.mean(test_steps)
    trained_success_rate = sum(test_successes) / num_episodes * 100
    
    # Random baseline statistics
    random_avg_reward = np.mean(random_rewards)
    random_avg_tiles = np.mean(random_tiles)
    random_success_rate = sum(random_successes) / num_episodes * 100
    
    # Calculate improvement
    reward_improvement = trained_avg_reward - random_avg_reward
    tiles_improvement = trained_avg_tiles - random_avg_tiles
    
    print(f"\n  Trained Agent Performance:")
    print(f"    Average Reward:    {trained_avg_reward:8.1f}")
    print(f"    Average Tiles:     {trained_avg_tiles:8.1f}/{env.num_cleanable}")
    print(f"    Average Steps:     {trained_avg_steps:8.1f}")
    print(f"    Success Rate:      {trained_success_rate:8.1f}%")
    print(f"    Best Episode:      {max(test_rewards):8.1f} reward")
    
    print(f"\n  Random Baseline:")
    print(f"    Average Reward:    {random_avg_reward:8.1f}")
    print(f"    Average Tiles:     {random_avg_tiles:8.1f}/{env.num_cleanable}")
    print(f"    Success Rate:      {random_success_rate:8.1f}%")
    
    print(f"\n  Improvement over Random:")
    print(f"    Reward:            {reward_improvement:+8.1f}")
    print(f"    Tiles Cleaned:     {tiles_improvement:+8.1f}")
    print(f"    Success Rate:      {trained_success_rate - random_success_rate:+8.1f}%")
    
    # Performance grade
    print("\n  Performance Grade:")
    if trained_success_rate >= 90:
        print("    ★★★★★ EXCELLENT - Agent has mastered the task!")
    elif trained_success_rate >= 70:
        print("    ★★★★☆ GOOD - Agent performs well, could improve")
    elif trained_success_rate >= 50:
        print("    ★★★☆☆ MODERATE - Agent shows learning but needs more training")
    elif trained_avg_tiles > random_avg_tiles:
        print("    ★★☆☆☆ BASIC - Some learning detected, needs more episodes")
    else:
        print("    ★☆☆☆☆ POOR - Agent may not have learned effectively")
    
    print("\n" + "=" * 70)
    
    # ==========================================================================
    # STEP 5: CLEANUP
    # ==========================================================================
    
    env.close()
    
    # Return results
    return {
        "trained_rewards": test_rewards,
        "trained_tiles": test_tiles,
        "trained_steps": test_steps,
        "trained_success_rate": trained_success_rate,
        "random_rewards": random_rewards,
        "random_tiles": random_tiles,
        "random_success_rate": random_success_rate,
        "improvement": reward_improvement
    }


def test_single_episode(model_path="models/q_table.pkl", speed="slow"):
    """
    Run a single test episode with detailed action logging.
    
    This function is useful for debugging and understanding
    exactly what the agent is doing step by step.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved Q-table
    speed : str
        Rendering speed: "slow", "normal", or "fast"
    """
    print("\n" + "=" * 70)
    print("  SINGLE EPISODE DETAILED TEST")
    print("=" * 70)
    
    # Set up environment and agent
    speed_map = {"slow": 5, "normal": 10, "fast": 20}
    fps = speed_map.get(speed, 5)
    
    env = CleaningEnv(render_mode="human")
    env.metadata["render_fps"] = fps
    
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n
    )
    
    if not agent.load(model_path):
        print("  Cannot load model. Please train first.")
        return
    
    # Run episode
    state, info = env.reset()
    env.render()
    time.sleep(1)  # Pause to see initial state
    
    print(f"\n  Starting cleaning run...")
    print(f"  Initial dirty tiles: {info['dirty_tiles']}")
    print("\n  Step | Action       | Reward   | Tiles Left | Room")
    print("  " + "-" * 60)
    
    total_reward = 0
    step_count = 0
    done = False
    
    while not done:
        # Get action
        action = agent.choose_action(state, training=False)
        action_name = env.get_action_name(action)
        
        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        step_count += 1
        
        # Print step details
        print(f"  {step_count:4d} | {action_name:12} | {reward:+7.1f} | "
              f"{info['dirty_tiles']:10d} | {info['room']}")
        
        # Render
        env.render()
        
        state = next_state
    
    # Print summary
    print("\n  " + "-" * 60)
    print(f"\n  Episode Complete!")
    print(f"    Total Reward:  {total_reward:.1f}")
    print(f"    Tiles Cleaned: {info['tiles_cleaned']}/{env.num_cleanable}")
    print(f"    Steps Taken:   {step_count}")
    print(f"    Success:       {'YES' if info['tiles_cleaned'] == env.num_cleanable else 'NO'}")
    
    # Wait before closing
    print("\n  Press Ctrl+C or wait 5 seconds to close...")
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    
    env.close()


def compare_random_vs_trained(num_episodes=20, model_path="models/q_table.pkl"):
    """
    Side-by-side comparison of random vs trained agent.
    
    This function runs both agents and provides a clear comparison
    of their performance, demonstrating the value of training.
    
    Parameters:
    -----------
    num_episodes : int
        Number of episodes to compare
    model_path : str
        Path to the trained model
    """
    print("\n" + "=" * 70)
    print("  RANDOM vs TRAINED AGENT COMPARISON")
    print("=" * 70)
    
    env = CleaningEnv(render_mode=None)
    
    # Test random agent
    print("\n  Testing Random Agent...")
    random_results = {"rewards": [], "tiles": [], "successes": 0}
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        random_results["rewards"].append(episode_reward)
        random_results["tiles"].append(info['tiles_cleaned'])
        if info['tiles_cleaned'] == env.num_cleanable:
            random_results["successes"] += 1
    
    # Test trained agent
    print("  Testing Trained Agent...")
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n
    )
    
    if not agent.load(model_path):
        print("  Cannot load model. Please train first.")
        return
    
    trained_results = {"rewards": [], "tiles": [], "successes": 0}
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        trained_results["rewards"].append(episode_reward)
        trained_results["tiles"].append(info['tiles_cleaned'])
        if info['tiles_cleaned'] == env.num_cleanable:
            trained_results["successes"] += 1
    
    env.close()
    
    # Display comparison
    print("\n" + "-" * 70)
    print(f"  {'Metric':<25} | {'Random':>15} | {'Trained':>15} | {'Diff':>10}")
    print("  " + "-" * 70)
    
    # Average reward
    r_avg = np.mean(random_results["rewards"])
    t_avg = np.mean(trained_results["rewards"])
    print(f"  {'Avg Reward':<25} | {r_avg:>15.1f} | {t_avg:>15.1f} | {t_avg-r_avg:>+10.1f}")
    
    # Average tiles
    r_tiles = np.mean(random_results["tiles"])
    t_tiles = np.mean(trained_results["tiles"])
    print(f"  {'Avg Tiles Cleaned':<25} | {r_tiles:>15.1f} | {t_tiles:>15.1f} | {t_tiles-r_tiles:>+10.1f}")
    
    # Success rate
    r_success = random_results["successes"] / num_episodes * 100
    t_success = trained_results["successes"] / num_episodes * 100
    print(f"  {'Success Rate (%)':<25} | {r_success:>15.1f} | {t_success:>15.1f} | {t_success-r_success:>+10.1f}")
    
    # Best reward
    print(f"  {'Best Episode Reward':<25} | {max(random_results['rewards']):>15.1f} | "
          f"{max(trained_results['rewards']):>15.1f} | "
          f"{max(trained_results['rewards'])-max(random_results['rewards']):>+10.1f}")
    
    print("\n" + "=" * 70)
    
    # Conclusion
    if t_avg > r_avg and t_tiles > r_tiles:
        print("\n  CONCLUSION: Trained agent significantly outperforms random!")
        print("  This demonstrates successful Q-Learning.")
    elif t_avg > r_avg:
        print("\n  CONCLUSION: Trained agent shows improvement in rewards.")
        print("  More training may improve tile completion rate.")
    else:
        print("\n  CONCLUSION: Training may not be sufficient yet.")
        print("  Try training for more episodes.")
    
    print("\n" + "=" * 70)


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    """
    Run testing when this script is executed directly.
    
    Usage:
        python test.py
        python test.py --episodes 20
        python test.py --single
        python test.py --compare
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Q-Learning Cleaning Robot")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of test episodes (default: 10)")
    parser.add_argument("--no-render", action="store_true",
                       help="Run without visualization")
    parser.add_argument("--speed", choices=["slow", "normal", "fast"], default="normal",
                       help="Rendering speed (default: normal)")
    parser.add_argument("--model", type=str, default="models/q_table.pkl",
                       help="Path to trained model")
    parser.add_argument("--single", action="store_true",
                       help="Run single detailed episode")
    parser.add_argument("--compare", action="store_true",
                       help="Compare random vs trained agent")
    
    args = parser.parse_args()
    
    if args.single:
        test_single_episode(model_path=args.model, speed=args.speed)
    elif args.compare:
        compare_random_vs_trained(num_episodes=args.episodes, model_path=args.model)
    else:
        test(
            num_episodes=args.episodes,
            render=not args.no_render,
            speed=args.speed,
            model_path=args.model
        )
    
    print("\n  Testing script finished!")
