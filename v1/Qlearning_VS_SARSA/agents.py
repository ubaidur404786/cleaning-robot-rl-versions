import numpy as np
from config import (
    ALPHA, GAMMA,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    UCB_C, OPTIMISTIC_INIT,
    NUM_ACTIONS,
)


class BaseAgent:
    """
    Base class for tabular RL agents.

    Handles Q-table management, state indexing, and exploration strategies.
    Subclasses implement the specific update rule (Q-Learning vs SARSA).
    """

    def __init__(
        self,
        num_states,
        num_actions=NUM_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        exploration="epsilon_greedy",
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        ucb_c=UCB_C,
        optimistic_init=0.0,
    ):
        """
        Parameters
        ----------
        num_states : int
            Total number of discrete states.
        num_actions : int
            Number of possible actions.
        alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        exploration : str
            One of "epsilon_greedy", "ucb", "optimistic".
        epsilon_start : float
            Starting epsilon for epsilon-greedy.
        epsilon_min : float
            Minimum epsilon.
        epsilon_decay : float
            Multiplicative decay per episode.
        ucb_c : float
            Exploration constant for UCB.
        optimistic_init : float
            Initial Q-values (set high for optimistic initialization).
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration

        # Epsilon-greedy params
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # UCB params
        self.ucb_c = ucb_c

        # Initialize Q-table
        init_val = OPTIMISTIC_INIT if exploration == "optimistic" else optimistic_init
        self.q_table = np.full((num_states, num_actions), init_val, dtype=np.float64)

        # Visit counts for UCB
        self.action_counts = np.zeros((num_states, num_actions), dtype=np.int64)
        self.total_steps = 0

        # Episode tracking
        self.episode_count = 0

    def choose_action(self, state_idx):
        """
        Select an action based on the current exploration strategy.

        Parameters
        ----------
        state_idx : int
            Flat state index.

        Returns
        -------
        action : int
            Selected action.
        """
        self.total_steps += 1

        if self.exploration == "epsilon_greedy" or self.exploration == "optimistic":
            return self._epsilon_greedy(state_idx)
        elif self.exploration == "ucb":
            return self._ucb(state_idx)
        else:
            raise ValueError(f"Unknown exploration strategy: {self.exploration}")

    def _epsilon_greedy(self, state_idx):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            # Break ties randomly
            q_vals = self.q_table[state_idx]
            max_q = np.max(q_vals)
            best_actions = np.where(q_vals == max_q)[0]
            return np.random.choice(best_actions)

    def _ucb(self, state_idx):
        """
        UCB action selection.

        For unvisited actions, return them first (infinite UCB bonus).
        Otherwise: Q(s,a) + c * sqrt(ln(total_steps) / N(s,a))
        """
        counts = self.action_counts[state_idx]

        # Return first unvisited action
        unvisited = np.where(counts == 0)[0]
        if len(unvisited) > 0:
            return np.random.choice(unvisited)

        q_vals = self.q_table[state_idx]
        ucb_values = q_vals + self.ucb_c * np.sqrt(
            np.log(self.total_steps) / counts
        )
        max_ucb = np.max(ucb_values)
        best_actions = np.where(ucb_values == max_ucb)[0]
        return np.random.choice(best_actions)

    def update(self, state_idx, action, reward, next_state_idx, done, next_action=None):
        """
        Update Q-values. Must be implemented by subclasses.

        Parameters
        ----------
        state_idx : int
            Current state index.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state_idx : int
            Next state index.
        done : bool
            Whether episode is over.
        next_action : int, optional
            Next action (used by SARSA only).
        """
        raise NotImplementedError

    def decay_epsilon(self):
        """Decay epsilon at the end of an episode."""
        self.episode_count += 1
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def get_greedy_action(self, state_idx):
        """Return the greedy action (for evaluation, no exploration)."""
        q_vals = self.q_table[state_idx]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return np.random.choice(best_actions)

    def reset_exploration(self):
        """Reset exploration parameters (for new training run)."""
        self.epsilon = self.epsilon_start
        self.episode_count = 0
        self.total_steps = 0
        self.action_counts[:] = 0


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent (off-policy).

    Update rule:
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

    Uses the maximum Q-value over all next actions (optimistic about future).
    """

    def update(self, state_idx, action, reward, next_state_idx, done, next_action=None):
        self.action_counts[state_idx, action] += 1

        current_q = self.q_table[state_idx, action]

        if done:
            target = reward
        else:
            # Off-policy: use max over next state's actions
            target = reward + self.gamma * np.max(self.q_table[next_state_idx])

        self.q_table[state_idx, action] += self.alpha * (target - current_q)


class SARSAAgent(BaseAgent):
    """
    SARSA agent (on-policy).

    Update rule:
        Q(s,a) ← Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]

    Uses the Q-value of the action actually chosen by the policy (cautious).
    This makes SARSA account for its own exploration noise, leading to
    more conservative behavior — potentially better for battery management.
    """

    def update(self, state_idx, action, reward, next_state_idx, done, next_action=None):
        self.action_counts[state_idx, action] += 1

        current_q = self.q_table[state_idx, action]

        if done:
            target = reward
        else:
            if next_action is None:
                raise ValueError("SARSA requires next_action for update.")
            # On-policy: use the actual next action
            target = reward + self.gamma * self.q_table[next_state_idx, next_action]

        self.q_table[state_idx, action] += self.alpha * (target - current_q)
