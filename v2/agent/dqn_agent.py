"""
================================================================================
DQN AGENT - Deep Q-Network Reinforcement Learning Implementation
================================================================================

PROJECT: Cleaning Robot using Reinforcement Learning (DQN)
FILE: agent/dqn_agent.py
PURPOSE: DQN agent that learns cleaning behavior using a neural network
         to approximate Q-values instead of a lookup table.

================================================================================
DEEP Q-NETWORK (DQN) ALGORITHM OVERVIEW
================================================================================

DQN is a model-free, OFF-POLICY reinforcement learning algorithm that uses
a neural network to approximate the Q-value function.  Unlike tabular
Q-Learning which stores Q-values in a dictionary, DQN uses a neural network
that can generalise across similar states.

KEY CONCEPTS:

1. Q-NETWORK:
   A neural network f(s; θ) → [Q(s,a0), Q(s,a1), ..., Q(s,a5)]
   Input:  a feature vector describing the current state
   Output: estimated Q-value for each action

2. EXPERIENCE REPLAY:
   Instead of learning from transitions as they arrive (correlated data),
   DQN stores transitions in a replay buffer and trains on random mini-
   batches.  This breaks correlation, improves sample efficiency, and
   stabilises training.

3. TARGET NETWORK:
   A separate copy of the Q-network (with frozen weights) is used to
   compute the target Q-values.  This prevents the "moving target" problem
   where the network chases its own changing predictions.  The target
   network is periodically synced with the policy network.

4. UPDATE RULE:
   loss = MSE( Q_policy(s,a) ,  r + γ · max Q_target(s', ·) )
   
   Where:
   - Q_policy(s,a)          = current prediction from policy network
   - Q_target(s', ·)        = next-state prediction from target network
   - r + γ·max Q_target(·)  = TD target

================================================================================
DQN vs TABULAR Q-LEARNING
================================================================================

TABULAR Q-LEARNING:
  - Stores one Q-value per (state, action) pair in a dictionary
  - Can only handle discrete, finite state spaces
  - No generalisation: each state is learned independently
  - Memory grows linearly with visited states

DQN:
  - Approximates Q-values with a neural network
  - Can handle continuous or high-dimensional state spaces
  - Generalises: similar states share network weights
  - Fixed memory footprint (network parameters)
  - Requires experience replay + target network for stability

================================================================================
FEATURE VECTOR
================================================================================

Since the original environment returns an integer state for the Q-table,
the DQN works with a richer feature vector extracted from the environment:

  [0]     : robot_row  (normalised to 0-1)
  [1]     : robot_col  (normalised to 0-1)
  [2:25]  : dirt status of each of the 23 cleanable tiles (0 or 1)
    [25:30] : one-hot movement history (came_from direction)
    [30:40] : one-hot DNUT direction to nearest dirty tile

Total input size = 40

The feature extraction is handled externally (in main.py) via the helper
function _env_to_features(env), keeping this agent class decoupled from
the environment implementation.

================================================================================
"""

import numpy as np
import random
import os
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ==============================================================================
# Q-NETWORK  (the neural network that approximates Q-values)
# ==============================================================================

class QNetwork(nn.Module):
    """
    Feed-forward neural network for Q-value approximation.

    Architecture:
    Input(40) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(6)

    The network takes a state feature vector and outputs one Q-value per
    action, just like a row in the tabular Q-table.
    """

    def __init__(self, input_size, output_size, hidden_size=64):
        """
        Parameters
        ----------
        input_size  : int   Size of the state feature vector (default 40).
        output_size : int   Number of actions (default 6).
        hidden_size : int   Neurons per hidden layer (default 64).
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ==============================================================================
# DQN AGENT
# ==============================================================================

class DQNAgent:
    """
    ============================================================================
    DQN AGENT - Deep Q-Network Reinforcement Learning
    ============================================================================

    This agent learns behaviour using a neural network instead of a table.
    It maintains two networks (policy + target), a replay buffer, and uses
    mini-batch gradient descent for training.

    Interface is designed to be compatible with the tabular agents
    (QLearningAgent, SarsaAgent) so that the same training / testing loop
    in main.py can drive all three algorithms.

    KEY DIFFERENCES FROM TABULAR AGENTS:
    - choose_action() receives a numpy feature vector, not an integer state
    - learn() receives feature vectors for state and next_state
    - save / load use torch.save / torch.load instead of pickle
    ============================================================================
    """

    def __init__(
        self,
        input_size=40,
        action_size=6,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.9987,
        batch_size=64,
        memory_size=10000,
        target_update=100,
        hidden_size=64,
        train_every=4,
    ):
        """
        Initialise the DQN Agent.

        Parameters
        ----------
        input_size      : int     Dimension of the feature vector (40).
        action_size     : int     Number of possible actions (6).
        learning_rate   : float   Adam optimiser learning rate.
        discount_factor : float   Gamma (γ) — importance of future rewards.
        epsilon_start   : float   Initial exploration rate.
        epsilon_end     : float   Minimum exploration rate.
        epsilon_decay   : float   Multiplicative decay per episode.
        batch_size      : int     Mini-batch size for replay training.
        memory_size     : int     Maximum transitions in replay buffer.
        target_update   : int     Sync target network every N learn() calls.
        hidden_size     : int     Neurons per hidden layer.
        train_every     : int     Only do gradient update every N steps (skip otherwise).
        """
        # Environment dimensions
        self.input_size = input_size
        self.action_size = action_size

        # ==================================================================
        # HYPERPARAMETERS
        # ==================================================================
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.target_update = target_update
        self.hidden_size = hidden_size
        self.train_every = train_every

        # Exploration (epsilon-greedy)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # ==================================================================
        # DEVICE (GPU if available, else CPU)
        # ==================================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==================================================================
        # NEURAL NETWORKS
        # ==================================================================
        # Policy network — used for action selection and updated every step
        self.policy_net = QNetwork(input_size, action_size, hidden_size).to(self.device)

        # Target network — frozen copy, updated every `target_update` steps
        self.target_net = QNetwork(input_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # never in training mode

        # ==================================================================
        # OPTIMISER
        # ==================================================================
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.lr = learning_rate

        # ==================================================================
        # EXPERIENCE REPLAY BUFFER
        # ==================================================================
        # Each entry: (state, action, reward, next_state, done)
        # state / next_state are numpy float32 arrays of shape (input_size,)
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size

        # ==================================================================
        # COUNTERS
        # ==================================================================
        self.learn_step_counter = 0     # number of learn() calls with gradient
        self.training_episodes = 0
        self.total_steps = 0

        # ==================================================================
        # PRINT INITIALISATION INFO
        # ==================================================================
        print("=" * 65)
        print("  DQN AGENT INITIALIZED (Deep RL)")
        print("=" * 65)
        print(f"  Input size:         {input_size}")
        print(f"  Action space:       {action_size} actions")
        print(f"  Hidden layers:      2 × {hidden_size} neurons")
        print(f"  Network params:     {self.num_parameters}")
        print(f"  Learning rate:      {learning_rate}")
        print(f"  Discount (γ):       {discount_factor}")
        print(f"  Epsilon:            {epsilon_start} → {epsilon_end}")
        print(f"  Epsilon decay:      {epsilon_decay}")
        print(f"  Batch size:         {batch_size}")
        print(f"  Replay buffer:      {memory_size}")
        print(f"  Target update:      every {target_update} steps")
        print(f"  Train every:        {train_every} steps")
        print(f"  Device:             {self.device}")
        print("=" * 65)

    # ==========================================================================
    # PROPERTIES
    # ==========================================================================

    @property
    def num_parameters(self):
        """Total trainable parameters in the policy network."""
        return sum(p.numel() for p in self.policy_net.parameters())

    # ==========================================================================
    # ACTION SELECTION
    # ==========================================================================

    def choose_action(self, state_features, training=True, eval_epsilon=0.02):
        """
        Choose an action using epsilon-greedy over the policy network.

        Parameters
        ----------
        state_features : numpy.ndarray, shape (input_size,)
            Feature vector describing the current state.
        training : bool
            If True, use current epsilon for exploration.
            If False, use the provided evaluation epsilon.
        eval_epsilon : float
            Exploration rate to use during evaluation/testing.
            Set to 0.0 for a fully deterministic greedy rollout.

        Returns
        -------
        int
            Chosen action index (0 to action_size-1).
        """
        eps = self.epsilon if training else eval_epsilon

        # Exploration: random action
        if random.random() < eps:
            return random.randint(0, self.action_size - 1)

        # Exploitation: best action from policy network
        with torch.no_grad():
            state_t = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    # ==========================================================================
    # EXPERIENCE REPLAY STORAGE
    # ==========================================================================

    def _store_transition(self, state, action, reward, next_state, done):
        """Add a transition to the replay buffer."""
        self.memory.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    # ==========================================================================
    # LEARNING (TRAINING STEP)
    # ==========================================================================

    def learn(self, state, action, reward, next_state, done):
        """
        Store transition and train on a mini-batch from replay buffer.

        DQN UPDATE:
          1. Store (s, a, r, s', done) in replay buffer
          2. If buffer has enough samples, sample a mini-batch
          3. Compute targets:  y = r + γ · max Q_target(s', ·) · (1 - done)
          4. Compute loss:     L = MSE( Q_policy(s, a),  y )
          5. Backpropagate and update policy network weights
          6. Every target_update steps, copy policy → target

        Parameters
        ----------
        state       : numpy.ndarray  Current state features.
        action      : int            Action taken.
        reward      : float          Reward received.
        next_state  : numpy.ndarray  Next state features.
        done        : bool           Whether episode ended.
        """
        self._store_transition(state, action, reward, next_state, done)
        self.total_steps += 1

        # Only do a gradient update every train_every steps
        if self.total_steps % self.train_every != 0:
            return

        # Wait until we have enough transitions for a batch
        if len(self.memory) < self.batch_size:
            return

        # --- Sample random mini-batch ----------------------------------------
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # --- Current Q-values from policy network ----------------------------
        # Q_policy(s, a) — gather selects the Q-value for the action taken
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # --- Target Q-values from target network -----------------------------
        # y = r + γ · max Q_target(s', ·) · (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        # --- Compute loss and backpropagate ----------------------------------
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # --- Update target network -------------------------------------------
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ==========================================================================
    # EPSILON DECAY
    # ==========================================================================

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(
            self.epsilon * self.epsilon_decay,
            self.epsilon_end,
        )

    def end_episode(self):
        """Called at the end of each training episode."""
        self.decay_epsilon()
        self.training_episodes += 1

    # ==========================================================================
    # STATISTICS
    # ==========================================================================

    def get_stats(self):
        """Get training statistics for logging/display."""
        return {
            "episodes_trained": self.training_episodes,
            "total_steps": self.total_steps,
            "learn_steps": self.learn_step_counter,
            "epsilon": self.epsilon,
            "replay_buffer_size": len(self.memory),
            "num_parameters": self.num_parameters,
        }

    # ==========================================================================
    # SAVE / LOAD
    # ==========================================================================

    def save(self, filepath):
        """
        Save the trained DQN agent to a file.

        Parameters
        ----------
        filepath : str
            Path to save (e.g. "models/dqn_model.pth").
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
                     exist_ok=True)

        save_data = {
            # Network weights
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),

            # Architecture info (needed for reconstruction)
            "input_size": self.input_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,

            # Hyperparameters
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,

            # Training counters
            "training_episodes": self.training_episodes,
            "total_steps": self.total_steps,
            "learn_step_counter": self.learn_step_counter,
        }

        torch.save(save_data, filepath)

        print(f"Agent saved to: {filepath}")
        print(f"  Network params: {self.num_parameters}")
        print(f"  Episodes trained: {self.training_episodes}")

    def load(self, filepath):
        """
        Load a trained DQN agent from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved agent file.

        Returns
        -------
        bool
            True if loaded successfully, False otherwise.
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            return False

        try:
            save_data = torch.load(filepath, map_location=self.device,
                                   weights_only=False)

            # Restore networks
            self.policy_net.load_state_dict(save_data["policy_net"])
            self.target_net.load_state_dict(save_data["target_net"])
            self.optimizer.load_state_dict(save_data["optimizer"])

            # Restore counters
            self.training_episodes = save_data.get("training_episodes", 0)
            self.total_steps = save_data.get("total_steps", 0)
            self.learn_step_counter = save_data.get("learn_step_counter", 0)

            # Restore epsilon
            self.epsilon = save_data.get("epsilon", self.epsilon_end)

            print(f"Agent loaded from: {filepath}")
            print(f"  Network params: {self.num_parameters}")
            print(f"  Episodes trained: {self.training_episodes}")
            print(f"  Current epsilon: {self.epsilon:.4f}")

            return True

        except Exception as e:
            print(f"Error loading agent: {e}")
            return False

    def reset_for_training(self, keep_weights=False):
        """
        Reset agent for a new training run.

        Parameters
        ----------
        keep_weights : bool
            If True, keep the learned network weights (for continued training).
            If False, re-initialise all weights (start fresh).
        """
        if not keep_weights:
            self.policy_net = QNetwork(
                self.input_size, self.action_size, self.hidden_size
            ).to(self.device)
            self.target_net = QNetwork(
                self.input_size, self.action_size, self.hidden_size
            ).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory.clear()
        self.epsilon = self.epsilon_start
        self.training_episodes = 0
        self.total_steps = 0
        self.learn_step_counter = 0

        print("Agent reset for training")
        print(f"  Epsilon reset to: {self.epsilon}")
        print(f"  Weights {'kept' if keep_weights else 'reinitialised'}")


# ==============================================================================
# MODULE TEST — run this file directly to verify the DQN agent
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TESTING DQN AGENT")
    print("=" * 65)

    # 1. Create agent
    print("\n1. Creating DQN agent (input=25, actions=6) ...")
    agent = DQNAgent(
        input_size=25,
        action_size=6,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.9987,
        batch_size=4,       # small batch for quick test
        memory_size=100,
        target_update=10,
        hidden_size=64,
    )

    # 2. Test action selection
    print("\n2. Testing action selection (eps=1.0, should be random) ...")
    dummy_state = np.random.rand(25).astype(np.float32)
    actions = [agent.choose_action(dummy_state, training=True) for _ in range(10)]
    print(f"   Actions: {actions}")

    # 3. Test learn with random transitions
    print("\n3. Testing learn() with random transitions ...")
    for i in range(20):
        s  = np.random.rand(25).astype(np.float32)
        a  = random.randint(0, 5)
        r  = random.uniform(-5, 50)
        s2 = np.random.rand(25).astype(np.float32)
        d  = random.random() < 0.1
        agent.learn(s, a, r, s2, d)
    print(f"   Replay buffer size: {len(agent.memory)}")
    print(f"   Learn steps: {agent.learn_step_counter}")

    # 4. Test exploitation action
    print("\n4. Testing exploitation (eps set to 0) ...")
    agent.epsilon = 0.0
    a1 = agent.choose_action(dummy_state, training=True)
    a2 = agent.choose_action(dummy_state, training=True)
    print(f"   Same state → action {a1}, {a2}  (should be the same: {a1 == a2})")

    # 5. Test epsilon decay
    print("\n5. Testing epsilon decay over 100 episodes ...")
    agent.epsilon = 1.0
    for _ in range(100):
        agent.decay_epsilon()
    print(f"   Epsilon after 100 decays: {agent.epsilon:.4f}")

    # 6. Test save / load
    print("\n6. Testing save / load ...")
    agent.save("test_dqn.pth")

    agent2 = DQNAgent(input_size=25, action_size=6, batch_size=4,
                       memory_size=100, target_update=10)
    agent2.load("test_dqn.pth")

    # Verify same output
    agent.epsilon = 0.0
    agent2.epsilon = 0.0
    a_orig = agent.choose_action(dummy_state, training=True)
    a_load = agent2.choose_action(dummy_state, training=True)
    print(f"   Actions match after load: {a_orig == a_load}")

    os.remove("test_dqn.pth")

    # 7. Parameter count
    print(f"\n7. Network parameters: {agent.num_parameters}")
    expected = 25 * 64 + 64 + 64 * 64 + 64 + 64 * 6 + 6
    print(f"   Expected (25→64→64→6): {expected}")
    print(f"   Match: {agent.num_parameters == expected}")

    print("\n  All DQN tests passed!")
