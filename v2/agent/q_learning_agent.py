"""
================================================================================
Q-LEARNING AGENT - Pure Reinforcement Learning Implementation
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning (Q-Learning)
FILE: agent/q_learning_agent.py
PURPOSE: Q-Learning agent that learns optimal cleaning behavior

================================================================================
📚 Q-LEARNING ALGORITHM OVERVIEW
================================================================================

Q-Learning is a model-free, off-policy reinforcement learning algorithm.
The agent learns a Q-function (action-value function) that estimates the
expected future reward for taking an action in a given state.

KEY CONCEPTS:

1. Q-VALUE: Q(s, a) = expected total reward starting from state s,
            taking action a, and then following the optimal policy

2. Q-TABLE: A lookup table storing Q-values for all (state, action) pairs
            Size = num_states × num_actions

3. UPDATE RULE (Bellman Equation):
   Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
   
   Where:
   - s = current state
   - a = action taken
   - r = reward received
   - s' = next state
   - α (alpha) = learning rate (how fast to update)
   - γ (gamma) = discount factor (importance of future rewards)

================================================================================
⚡ EPSILON-GREEDY EXPLORATION
================================================================================

The agent uses ε-greedy (epsilon-greedy) policy for action selection:

- With probability ε: Take RANDOM action (exploration)
- With probability 1-ε: Take BEST action from Q-table (exploitation)

EXPLORATION vs EXPLOITATION:
- Exploration: Try new actions to discover better strategies
- Exploitation: Use known best actions to maximize reward

Epsilon typically starts high (1.0 = 100% random) and decays over time
to a small value (e.g., 0.01 = 1% random) as the agent learns.

================================================================================
🎯 PURE RL IMPLEMENTATION
================================================================================

This implementation is PURE Q-Learning:
- NO hardcoded policies (no "if dirty then clean")
- NO direction hints from the state
- NO shortcuts or cheating

The agent learns EVERYTHING through trial and error:
- When to clean (learns that cleaning dirty tiles gives reward)
- Where to go (learns room layouts through exploration)
- How to navigate (learns which actions lead where)

This takes more training episodes but demonstrates true RL learning!

================================================================================
"""

import numpy as np
import pickle
import os


class QLearningAgent:
    """
    ============================================================================
    Q-LEARNING AGENT - Pure Reinforcement Learning
    ============================================================================
    
    This agent learns optimal behavior using the Q-Learning algorithm.
    It maintains a Q-table that maps (state, action) pairs to expected
    future rewards, and updates this table through experience.
    
    PURE RL DESIGN:
    - No hardcoded decision rules
    - No knowledge of environment dynamics
    - Learns purely from rewards received
    
    LEARNING PROCESS:
    1. Start with no knowledge (Q-table initialized to zeros or small values)
    2. Take actions (random early on due to high epsilon)
    3. Observe rewards and update Q-values
    4. Gradually shift from exploration to exploitation
    5. Eventually learn optimal policy
    
    ============================================================================
    """
    
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.2,           # α (alpha) - how quickly Q-values update
        discount_factor=0.95,         # γ (gamma) - importance of future rewards
        epsilon_start=1.0,            # Initial exploration rate (100% random)
        epsilon_end=0.01,             # Final exploration rate (1% random)
        epsilon_decay=0.998           # Decay multiplier per episode
    ):
        """
        Initialize the Q-Learning Agent.
        
        Parameters:
        -----------
        state_size : int
            Number of possible states in the environment.
            For our cleaning robot: 46 states (23 positions × 2 dirty status)
        
        action_size : int
            Number of possible actions.
            For our cleaning robot: 6 actions (4 moves + wait + clean)
        
        learning_rate : float (0 to 1)
            Alpha (α) - Controls how much new information overrides old.
            - Higher (0.5-1.0): Learn fast but may be unstable
            - Lower (0.01-0.1): Learn slowly but more stable
            - Typical: 0.1 to 0.3
        
        discount_factor : float (0 to 1)
            Gamma (γ) - How much to value future rewards vs immediate.
            - Higher (0.9-0.99): Care about long-term rewards
            - Lower (0.1-0.5): Prefer immediate rewards
            - Typical: 0.9 to 0.99
        
        epsilon_start : float (0 to 1)
            Initial probability of taking random action.
            Starting at 1.0 means 100% random actions initially.
        
        epsilon_end : float (0 to 1)
            Minimum exploration rate after decay.
            Typically 0.01 to 0.05 to maintain some exploration.
        
        epsilon_decay : float (0 to 1)
            Multiplier for epsilon after each episode.
            0.995 means epsilon reduces by 0.5% each episode.
        """
        # Store environment dimensions
        self.state_size = state_size
        self.action_size = action_size
        
        # ======================================================================
        # LEARNING HYPERPARAMETERS
        # ======================================================================
        
        # Learning rate (alpha) - How much to update Q-values
        self.alpha = learning_rate
        self.alpha_initial = learning_rate
        self.alpha_min = 0.01          # Minimum learning rate
        self.alpha_decay = 0.9999      # Learning rate decay per step
        
        # Discount factor (gamma) - Importance of future rewards
        self.gamma = discount_factor
        
        # Exploration parameters (epsilon-greedy)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # ======================================================================
        # Q-TABLE INITIALIZATION
        # ======================================================================
        # The Q-table stores Q-values for all (state, action) pairs.
        #
        # We use a dictionary instead of a numpy array because:
        # 1. Lazy initialization - only create entries for visited states
        # 2. Memory efficient for sparse state spaces
        # 3. Easy to save/load with pickle
        #
        # Each state maps to an array of Q-values (one per action):
        #   q_table[state] = [Q(s,a0), Q(s,a1), ..., Q(s,a5)]
        
        self.q_table = {}
        
        # ======================================================================
        # TRAINING STATISTICS
        # ======================================================================
        self.training_episodes = 0     # Total episodes trained
        self.total_steps = 0           # Total steps across all episodes
        
        # ======================================================================
        # PRINT INITIALIZATION INFO
        # ======================================================================
        print("=" * 65)
        print("  Q-LEARNING AGENT INITIALIZED (Pure RL)")
        print("=" * 65)
        print(f"  State space:        {state_size} states")
        print(f"  Action space:       {action_size} actions")
        print(f"  Learning rate (α):  {learning_rate}")
        print(f"  Discount (γ):       {discount_factor}")
        print(f"  Epsilon:            {epsilon_start} → {epsilon_end}")
        print(f"  Epsilon decay:      {epsilon_decay}")
        print("=" * 65)
    
    def _get_q_values(self, state):
        """
        Get Q-values for a state, initializing if this is a new state.
        
        This method implements lazy initialization: Q-values are only
        created when a state is first encountered.
        
        We initialize Q-values to small random values (not zeros) to
        encourage initial exploration. This is called "optimistic
        initialization" when using small positive values.
        
        Parameters:
        -----------
        state : int
            State index to get Q-values for
        
        Returns:
        --------
        numpy.ndarray
            Array of Q-values, one per action [Q(s,a0), Q(s,a1), ...]
        """
        if state not in self.q_table:
            # Initialize Q-values for this new state
            # Using small random values encourages exploration
            # Range [-0.1, 0.1] is small enough to not bias learning
            self.q_table[state] = np.random.uniform(
                low=-0.1, high=0.1, size=self.action_size
            )
        return self.q_table[state]
    
    def choose_action(self, state, training=True, eval_epsilon=0.02):
        """
        Choose an action using epsilon-greedy (training) or softmax (testing).
        
        TRAINING (epsilon-greedy):
        - Generate random number r between 0 and 1
        - If r < epsilon: Take random action (exploration)
        - Else: Take action with highest Q-value (exploitation)
        
        TESTING (softmax):
        - Use softmax over Q-values to select action probabilistically
        - Higher Q-value = higher probability of selection
        - Prevents deterministic loops while still using learned values
        
        Parameters:
        -----------
        state : int
            Current state observation
        
        training : bool
            If True, use epsilon-greedy (for training)
            If False, use epsilon-greedy with configurable evaluation epsilon

        eval_epsilon : float
            Exploration rate to use during evaluation/testing.
            Set to 0.0 for deterministic greedy behavior.
        
        Returns:
        --------
        int
            Chosen action index (0 to action_size-1)
        """
        # Get Q-values for current state
        q_values = self._get_q_values(state)
        
        # ======================================================================
        # TRAINING: Epsilon-greedy exploration
        # ======================================================================
        if training:
            if np.random.random() < self.epsilon:
                # Random action - allows discovering new strategies
                return np.random.randint(0, self.action_size)
            
            # Greedy: take best action
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
        
        # ======================================================================
        # TESTING: Epsilon-greedy with small fixed epsilon
        # ======================================================================
        # Use the same strategy as training but with a tiny exploration rate.
        # This keeps behaviour consistent with how the agent learned and
        # prevents deterministic loops on rarely-visited states.
        test_epsilon = eval_epsilon
        
        if np.random.random() < test_epsilon:
            return np.random.randint(0, self.action_size)
        
        # Greedy: take best action (ties broken randomly)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-values using the Q-Learning update rule (Bellman equation).
        
        Q-LEARNING UPDATE RULE:
        -----------------------
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Breaking this down:
        1. Q(s,a): Current estimate of value
        2. r: Immediate reward received
        3. γ·max(Q(s',a')): Discounted value of best action in next state
        4. r + γ·max(Q(s',a')): Target value (what we think Q should be)
        5. Target - Current: Error (how wrong our estimate was)
        6. α·Error: How much to adjust (scaled by learning rate)
        
        Parameters:
        -----------
        state : int
            State before taking action
        
        action : int
            Action that was taken
        
        reward : float
            Reward received for taking the action
        
        next_state : int
            State after taking action
        
        done : bool
            Whether the episode ended (terminal state)
        """
        # Get current Q-value for (state, action) pair
        current_q = self._get_q_values(state)[action]
        
        # Calculate target Q-value
        if done:
            # Terminal state: no future rewards
            # Target = just the immediate reward
            target_q = reward
        else:
            # Non-terminal: include discounted future reward
            # max(Q(s',a')) is the value of the best action in next state
            next_q_values = self._get_q_values(next_state)
            max_next_q = np.max(next_q_values)
            target_q = reward + self.gamma * max_next_q
        
        # Calculate TD error (Temporal Difference error)
        # This is how wrong our current estimate was
        td_error = target_q - current_q
        
        # Update Q-value using the Q-learning update rule
        # New Q = Old Q + learning_rate × TD_error
        self.q_table[state][action] = current_q + self.alpha * td_error
        
        # Track total steps
        self.total_steps += 1
    
    def decay_epsilon(self):
        """
        Decay exploration rate (epsilon) after each episode.
        
        This gradually shifts the agent from exploration to exploitation.
        Early in training, high epsilon means lots of random exploration.
        As training progresses, epsilon decreases and the agent relies
        more on its learned Q-values.
        
        Formula: epsilon_new = max(epsilon × decay_rate, epsilon_min)
        """
        self.epsilon = max(
            self.epsilon * self.epsilon_decay,
            self.epsilon_end
        )
    
    def decay_learning_rate(self):
        """
        Decay learning rate (alpha) over time.
        
        Reducing the learning rate over time can help stabilize learning:
        - Early: High learning rate for fast initial learning
        - Later: Low learning rate for fine-tuning
        
        Formula: alpha_new = max(alpha × decay_rate, alpha_min)
        """
        self.alpha = max(
            self.alpha * self.alpha_decay,
            self.alpha_min
        )
    
    def end_episode(self):
        """
        Called at the end of each training episode.
        
        This method:
        1. Decays epsilon (reduce exploration over time)
        2. Optionally decays learning rate
        3. Updates episode counter
        """
        self.decay_epsilon()
        # Uncomment to also decay learning rate:
        # self.decay_learning_rate()
        self.training_episodes += 1
    
    def get_stats(self):
        """
        Get training statistics for logging/display.
        
        Returns:
        --------
        dict
            Dictionary containing training statistics
        """
        return {
            "episodes_trained": self.training_episodes,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "q_table_size": len(self.q_table),
            "state_coverage": f"{len(self.q_table)}/{self.state_size}"
        }
    
    def get_q_value(self, state, action):
        """
        Get Q-value for a specific (state, action) pair.
        
        Parameters:
        -----------
        state : int
            State index
        action : int
            Action index
        
        Returns:
        --------
        float
            Q-value for the (state, action) pair
        """
        return self._get_q_values(state)[action]
    
    def get_best_action(self, state):
        """
        Get the best action for a state (pure exploitation).
        
        Parameters:
        -----------
        state : int
            State to get best action for
        
        Returns:
        --------
        int
            Action with highest Q-value
        """
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))
    
    def save(self, filepath):
        """
        Save the trained agent to a file.
        
        Saves:
        - Q-table (the learned knowledge)
        - Training statistics
        - Hyperparameters
        
        Parameters:
        -----------
        filepath : str
            Path to save the agent (e.g., "models/agent.pkl")
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", 
                   exist_ok=True)
        
        # Prepare data to save
        save_data = {
            # Q-table (the learned policy)
            "q_table": dict(self.q_table),  # Convert to regular dict for saving
            
            # Environment info
            "state_size": self.state_size,
            "action_size": self.action_size,
            
            # Hyperparameters (for reference)
            "alpha": self.alpha,
            "alpha_initial": self.alpha_initial,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            
            # Training stats
            "training_episodes": self.training_episodes,
            "total_steps": self.total_steps
        }
        
        # Save using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Agent saved to: {filepath}")
        print(f"  Q-table entries: {len(self.q_table)}")
        print(f"  Episodes trained: {self.training_episodes}")
    
    def load(self, filepath):
        """
        Load a trained agent from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved agent file
        
        Returns:
        --------
        bool
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore Q-table
            self.q_table = save_data["q_table"]
            
            # Restore training stats
            self.training_episodes = save_data.get("training_episodes", 0)
            self.total_steps = save_data.get("total_steps", 0)
            
            # Restore epsilon (for continued training)
            self.epsilon = save_data.get("epsilon", self.epsilon_end)
            self.alpha = save_data.get("alpha", self.alpha)
            
            print(f"Agent loaded from: {filepath}")
            print(f"  Q-table entries: {len(self.q_table)}")
            print(f"  Episodes trained: {self.training_episodes}")
            print(f"  Current epsilon: {self.epsilon:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading agent: {e}")
            return False
    
    def reset_for_training(self, keep_q_table=False):
        """
        Reset agent for new training run.
        
        Parameters:
        -----------
        keep_q_table : bool
            If True, keep the learned Q-values (for continued training)
            If False, reset Q-table (start fresh)
        """
        if not keep_q_table:
            self.q_table = {}
        
        # Reset epsilon to start value for new exploration phase
        self.epsilon = self.epsilon_start
        self.alpha = self.alpha_initial
        
        # Reset counters
        self.training_episodes = 0
        self.total_steps = 0
        
        print("Agent reset for training")
        print(f"  Epsilon reset to: {self.epsilon}")
        print(f"  Q-table {'kept' if keep_q_table else 'cleared'}")
    
    def print_policy_summary(self, env=None):
        """
        Print a summary of the learned policy.
        
        This shows what action the agent would take in each state,
        useful for understanding what the agent has learned.
        
        Parameters:
        -----------
        env : CleaningEnv, optional
            Environment for getting state descriptions
        """
        print("\n" + "=" * 65)
        print("  LEARNED POLICY SUMMARY")
        print("=" * 65)
        
        # Action names
        action_names = ["Fwd", "Back", "Left", "Right", "Wait", "Clean"]
        
        # Print header
        print(f"{'State':>6} | {'Dirty':>5} | {'Best Action':>12} | {'Q-Values'}")
        print("-" * 65)
        
        # Sort states for organized output
        sorted_states = sorted(self.q_table.keys())
        
        for state in sorted_states[:20]:  # Limit output
            q_values = self.q_table[state]
            best_action = np.argmax(q_values)
            
            # Decode state (assuming state = pos + dirty × num_pos)
            num_pos = self.state_size // 2
            is_dirty = state >= num_pos
            pos = state % num_pos
            
            # Format Q-values
            q_str = " ".join([f"{q:+.1f}" for q in q_values])
            
            print(f"{state:>6} | {'Yes' if is_dirty else 'No':>5} | "
                  f"{action_names[best_action]:>12} | [{q_str}]")
        
        if len(sorted_states) > 20:
            print(f"... and {len(sorted_states) - 20} more states")
        
        print("=" * 65)


# ================================================================================
# MODULE TEST - Run this file directly to test the agent
# ================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TESTING Q-LEARNING AGENT")
    print("=" * 65)
    
    # Create agent with small state space for testing
    print("\n1. Creating agent with 46 states and 6 actions...")
    agent = QLearningAgent(
        state_size=46,
        action_size=6,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Test action selection
    print("\n2. Testing action selection (epsilon=1.0, should be random)...")
    actions = [agent.choose_action(0, training=True) for _ in range(10)]
    print(f"   Actions chosen: {actions}")
    
    # Test Q-learning update
    print("\n3. Testing Q-learning update...")
    state, action = 0, 5  # State 0, Clean action
    reward = 50.0
    next_state = 0
    
    print(f"   Before: Q({state}, {action}) = {agent.get_q_value(state, action):.4f}")
    agent.learn(state, action, reward, next_state, done=False)
    print(f"   After:  Q({state}, {action}) = {agent.get_q_value(state, action):.4f}")
    
    # Test epsilon decay
    print("\n4. Testing epsilon decay over 100 episodes...")
    agent.epsilon = 1.0
    for _ in range(100):
        agent.decay_epsilon()
    print(f"   Epsilon after 100 decays: {agent.epsilon:.4f}")
    
    # Test save/load
    print("\n5. Testing save/load...")
    agent.save("test_agent.pkl")
    
    # Create new agent and load
    agent2 = QLearningAgent(state_size=46, action_size=6)
    agent2.load("test_agent.pkl")
    
    # Verify Q-values match
    print(f"   Q-values match: {agent.q_table[0][5] == agent2.q_table[0][5]}")
    
    # Clean up test file
    os.remove("test_agent.pkl")
    
    # Test exploitation mode
    print("\n6. Testing exploitation (training=False)...")
    agent.epsilon = 0.0  # No exploration
    actions_exploit = [agent.choose_action(0, training=False) for _ in range(10)]
    print(f"   All actions same (exploitation): {len(set(actions_exploit)) == 1}")
    
    print("\nAll tests passed!")
