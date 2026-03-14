"""
================================================================================
SARSA AGENT - On-Policy Reinforcement Learning Implementation
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning (SARSA)
FILE: agent/sarsa_agent.py
PURPOSE: SARSA agent that learns cleaning behavior on-policy

================================================================================
📚 SARSA ALGORITHM OVERVIEW
================================================================================

SARSA is a model-free, ON-POLICY reinforcement learning algorithm.
Unlike Q-Learning which uses the *best* next action to update Q-values,
SARSA uses the *actual* next action chosen by the current policy.

The name SARSA comes from the quintuple used in each update:
  (S, A, R, S', A') → (State, Action, Reward, next State, next Action)

KEY CONCEPTS:

1. Q-VALUE: Q(s, a) = expected total reward starting from state s,
            taking action a, and following the CURRENT policy

2. Q-TABLE: A lookup table storing Q-values for all (state, action) pairs
            Size = num_states × num_actions

3. UPDATE RULE:
   Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
   
   Where:
   - s  = current state
   - a  = action taken
   - r  = reward received
   - s' = next state
   - a' = next action ACTUALLY CHOSEN by the policy (not max!)
   - α (alpha) = learning rate
   - γ (gamma) = discount factor

================================================================================
⚡ ON-POLICY vs OFF-POLICY
================================================================================

Q-LEARNING (off-policy):
  - Updates use max(Q(s',a')) regardless of what action is actually taken next
  - Learns the OPTIMAL policy while following an exploratory policy
  - More aggressive: assumes optimal future behaviour even while exploring

SARSA (on-policy):
  - Updates use Q(s', a') where a' is the action actually chosen next
  - Learns the value of the CURRENT policy (including exploration)
  - More conservative: accounts for the fact that exploration may happen
  - Tends to learn safer policies (avoids risky states near penalties)

PRACTICAL DIFFERENCE:
  Because SARSA incorporates the exploration noise into its value estimates,
  it tends to be more cautious. If a state is near a wall (penalty), SARSA
  penalises it more because the epsilon-random actions might hit the wall,
  whereas Q-Learning ignores that risk (it assumes optimal future moves).

================================================================================
🎯 TRAINING LOOP DIFFERENCE
================================================================================

Q-Learning loop:
  1. Observe state s
  2. Choose action a (ε-greedy)
  3. Take action a → get reward r, next state s'
  4. Update: Q(s,a) ← ... + γ·max Q(s',·) ...
  5. s ← s'

SARSA loop:
  1. Observe state s, choose action a (ε-greedy)
  2. Take action a → get reward r, next state s'
  3. Choose NEXT action a' (ε-greedy)           ← key difference
  4. Update: Q(s,a) ← ... + γ·Q(s',a') ...     ← uses a', not max
  5. s ← s', a ← a'

Notice that SARSA selects the next action BEFORE updating, so the update
reflects what the agent will actually do, not what it *could* do.

================================================================================
"""

import numpy as np
import pickle
import os


class SarsaAgent:
    """
    ============================================================================
    SARSA AGENT - On-Policy Reinforcement Learning
    ============================================================================
    
    This agent learns behavior using the SARSA algorithm.
    It maintains a Q-table identical in structure to Q-Learning, but
    updates values using the actual next action chosen by the policy
    rather than the greedy maximum.
    
    ON-POLICY DESIGN:
    - The update rule reflects the agent's own exploratory behavior
    - Values learned account for the randomness of epsilon-greedy
    - Tends to produce more conservative / safer policies
    
    LEARNING PROCESS:
    1. Choose first action for the episode
    2. Take action, observe reward and next state
    3. Choose next action (BEFORE updating)
    4. Update Q(s,a) using actual next action's Q-value
    5. Shift: s ← s', a ← a'
    6. Repeat until episode ends
    
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
        Initialize the SARSA Agent.
        
        Parameters are identical to QLearningAgent so that results
        are directly comparable between the two algorithms.
        
        Parameters:
        -----------
        state_size : int
            Number of possible states in the environment.
        
        action_size : int
            Number of possible actions.
        
        learning_rate : float (0 to 1)
            Alpha (α) - Controls how much new information overrides old.
        
        discount_factor : float (0 to 1)
            Gamma (γ) - How much to value future rewards vs immediate.
        
        epsilon_start : float (0 to 1)
            Initial probability of taking random action.
        
        epsilon_end : float (0 to 1)
            Minimum exploration rate after decay.
        
        epsilon_decay : float (0 to 1)
            Multiplier for epsilon after each episode.
        """
        # Store environment dimensions
        self.state_size = state_size
        self.action_size = action_size
        
        # ======================================================================
        # LEARNING HYPERPARAMETERS
        # ======================================================================
        
        self.alpha = learning_rate
        self.alpha_initial = learning_rate
        self.alpha_min = 0.01
        self.alpha_decay = 0.9999
        
        self.gamma = discount_factor
        
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # ======================================================================
        # Q-TABLE INITIALIZATION
        # ======================================================================
        # Same lazy-init dictionary approach as Q-Learning for fair comparison.
        
        self.q_table = {}
        
        # ======================================================================
        # TRAINING STATISTICS
        # ======================================================================
        self.training_episodes = 0
        self.total_steps = 0
        
        # ======================================================================
        # PRINT INITIALIZATION INFO
        # ======================================================================
        print("=" * 65)
        print("  SARSA AGENT INITIALIZED (On-Policy RL)")
        print("=" * 65)
        print(f"  State space:        {state_size} states")
        print(f"  Action space:       {action_size} actions")
        print(f"  Learning rate (α):  {learning_rate}")
        print(f"  Discount (γ):       {discount_factor}")
        print(f"  Epsilon:            {epsilon_start} → {epsilon_end}")
        print(f"  Epsilon decay:      {epsilon_decay}")
        print("=" * 65)
    
    # ==========================================================================
    # Q-TABLE ACCESS
    # ==========================================================================
    
    def _get_q_values(self, state):
        """
        Get Q-values for a state, initializing if this is a new state.
        
        Lazy initialization with small random values (identical to
        QLearningAgent so the comparison is fair).
        
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
            self.q_table[state] = np.random.uniform(
                low=-0.1, high=0.1, size=self.action_size
            )
        return self.q_table[state]
    
    # ==========================================================================
    # ACTION SELECTION (identical to Q-Learning for fair comparison)
    # ==========================================================================
    
    def choose_action(self, state, training=True, eval_epsilon=0.02):
        """
        Choose an action using epsilon-greedy.
        
        This is intentionally identical to QLearningAgent.choose_action()
        so that the only difference between the two algorithms is the
        update rule.
        
        Parameters:
        -----------
        state : int
            Current state observation
        
        training : bool
            If True, use epsilon-greedy with current epsilon.
            If False, use epsilon-greedy with configurable evaluation epsilon.

        eval_epsilon : float
            Exploration rate to use during evaluation/testing.
            Set to 0.0 for deterministic greedy behavior.
        
        Returns:
        --------
        int
            Chosen action index (0 to action_size-1)
        """
        q_values = self._get_q_values(state)
        
        # ==================================================================
        # TRAINING: Epsilon-greedy exploration
        # ==================================================================
        if training:
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.action_size)
            
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
        
        # ==================================================================
        # TESTING: Epsilon-greedy with small fixed epsilon
        # ==================================================================
        test_epsilon = eval_epsilon
        
        if np.random.random() < test_epsilon:
            return np.random.randint(0, self.action_size)
        
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
    
    # ==========================================================================
    # SARSA UPDATE RULE (the core difference from Q-Learning)
    # ==========================================================================
    
    def learn(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-values using the SARSA update rule.
        
        SARSA UPDATE RULE:
        ------------------
        Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
        
        The critical difference from Q-Learning:
        - Q-Learning uses max(Q(s',·)) — the best possible future value
        - SARSA uses Q(s',a') — the value of the action actually chosen
        
        This means SARSA's updates reflect the current policy (including
        exploration), making it on-policy. The agent learns the value of
        what it actually does, not what it could optimally do.
        
        Parameters:
        -----------
        state : int
            State before taking action (s)
        
        action : int
            Action that was taken (a)
        
        reward : float
            Reward received for taking the action (r)
        
        next_state : int
            State after taking action (s')
        
        next_action : int
            Action chosen for the next step (a') — already selected
            by choose_action() BEFORE this update is called
        
        done : bool
            Whether the episode ended (terminal state)
        """
        # Current Q-value: Q(s, a)
        current_q = self._get_q_values(state)[action]
        
        # Calculate target Q-value
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: use the ACTUAL next action's Q-value
            # This is Q(s', a') — NOT max Q(s', ·)
            next_q = self._get_q_values(next_state)[next_action]
            target_q = reward + self.gamma * next_q
        
        # TD error
        td_error = target_q - current_q
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.alpha * td_error
        
        # Track total steps
        self.total_steps += 1
    
    # ==========================================================================
    # EPSILON / LEARNING-RATE DECAY
    # ==========================================================================
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(
            self.epsilon * self.epsilon_decay,
            self.epsilon_end
        )
    
    def decay_learning_rate(self):
        """Decay learning rate over time."""
        self.alpha = max(
            self.alpha * self.alpha_decay,
            self.alpha_min
        )
    
    def end_episode(self):
        """
        Called at the end of each training episode.
        
        Decays epsilon and increments the episode counter.
        """
        self.decay_epsilon()
        self.training_episodes += 1
    
    # ==========================================================================
    # STATISTICS & INSPECTION
    # ==========================================================================
    
    def get_stats(self):
        """Get training statistics for logging/display."""
        return {
            "episodes_trained": self.training_episodes,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "q_table_size": len(self.q_table),
            "state_coverage": f"{len(self.q_table)}/{self.state_size}"
        }
    
    def get_q_value(self, state, action):
        """Get Q-value for a specific (state, action) pair."""
        return self._get_q_values(state)[action]
    
    def get_best_action(self, state):
        """Get the best action for a state (pure exploitation)."""
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))
    
    # ==========================================================================
    # SAVE / LOAD
    # ==========================================================================
    
    def save(self, filepath):
        """
        Save the trained agent to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the agent (e.g., "models/sarsa_agent.pkl")
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
                     exist_ok=True)
        
        save_data = {
            "q_table": dict(self.q_table),
            "state_size": self.state_size,
            "action_size": self.action_size,
            "alpha": self.alpha,
            "alpha_initial": self.alpha_initial,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "training_episodes": self.training_episodes,
            "total_steps": self.total_steps
        }
        
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
            
            self.q_table = save_data["q_table"]
            self.training_episodes = save_data.get("training_episodes", 0)
            self.total_steps = save_data.get("total_steps", 0)
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
        
        self.epsilon = self.epsilon_start
        self.alpha = self.alpha_initial
        self.training_episodes = 0
        self.total_steps = 0
        
        print("Agent reset for training")
        print(f"  Epsilon reset to: {self.epsilon}")
        print(f"  Q-table {'kept' if keep_q_table else 'cleared'}")
    
    def print_policy_summary(self, env=None):
        """
        Print a summary of the learned policy.
        
        Parameters:
        -----------
        env : CleaningEnv, optional
            Environment for getting state descriptions
        """
        print("\n" + "=" * 65)
        print("  LEARNED POLICY SUMMARY (SARSA)")
        print("=" * 65)
        
        action_names = ["Fwd", "Back", "Left", "Right", "Wait", "Clean"]
        
        print(f"{'State':>6} | {'Dirty':>5} | {'Best Action':>12} | {'Q-Values'}")
        print("-" * 65)
        
        sorted_states = sorted(self.q_table.keys())
        
        for state in sorted_states[:20]:
            q_values = self.q_table[state]
            best_action = np.argmax(q_values)
            
            num_pos = self.state_size // 2
            is_dirty = state >= num_pos
            pos = state % num_pos
            
            q_str = " ".join([f"{q:+.1f}" for q in q_values])
            
            print(f"{state:>6} | {'Yes' if is_dirty else 'No':>5} | "
                  f"{action_names[best_action]:>12} | [{q_str}]")
        
        if len(sorted_states) > 20:
            print(f"... and {len(sorted_states) - 20} more states")
        
        print("=" * 65)


# ================================================================================
# MODULE TEST - Run this file directly to test the SARSA agent
# ================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TESTING SARSA AGENT")
    print("=" * 65)
    
    # Create agent with small state space for testing
    print("\n1. Creating SARSA agent with 46 states and 6 actions...")
    agent = SarsaAgent(
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
    
    # Test SARSA update (note: needs next_action parameter)
    print("\n3. Testing SARSA update...")
    state, action = 0, 5          # State 0, Clean action
    reward = 50.0
    next_state = 1
    next_action = 3               # Actual next action chosen by policy
    
    print(f"   Before: Q({state}, {action}) = {agent.get_q_value(state, action):.4f}")
    agent.learn(state, action, reward, next_state, next_action, done=False)
    print(f"   After:  Q({state}, {action}) = {agent.get_q_value(state, action):.4f}")
    
    # Verify that SARSA uses Q(s',a') not max Q(s',·)
    print("\n4. Verifying on-policy update...")
    agent2 = SarsaAgent(state_size=46, action_size=6)
    agent2.q_table[1] = np.array([10.0, 5.0, 3.0, 1.0, 0.0, -1.0])
    agent2.q_table[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # next_action=3 → Q(1,3)=1.0, not max Q(1,·)=10.0
    agent2.learn(0, 0, 10.0, 1, 3, done=False)
    expected = 0.0 + 0.2 * (10.0 + 0.95 * 1.0 - 0.0)  # = 2.19
    actual = agent2.q_table[0][0]
    print(f"   Q(0,0) after update: {actual:.4f}  (expected ≈ {expected:.4f})")
    print(f"   Uses Q(s',a')=1.0 not max=10.0: {abs(actual - expected) < 0.001}")
    
    # Test epsilon decay
    print("\n5. Testing epsilon decay over 100 episodes...")
    agent.epsilon = 1.0
    for _ in range(100):
        agent.decay_epsilon()
    print(f"   Epsilon after 100 decays: {agent.epsilon:.4f}")
    
    # Test save/load
    print("\n6. Testing save/load...")
    agent.save("test_sarsa.pkl")
    
    agent3 = SarsaAgent(state_size=46, action_size=6)
    agent3.load("test_sarsa.pkl")
    print(f"   Q-values match: {agent.q_table[0][5] == agent3.q_table[0][5]}")
    
    os.remove("test_sarsa.pkl")
    
    print("\nAll SARSA tests passed!")
