"""
================================================================================
Q-LEARNING AGENT  —  Phase-2 Directional Cleaning Robot
================================================================================

Q-Learning is a model-free, OFF-POLICY Temporal Difference (TD) control
algorithm.  The agent learns a Q-function mapping every (state, action) pair
to an estimate of the total future discounted reward, independently of the
exploratory policy used during training.

ALGORITHM  (Watkins, 1989)
--------------------------
For each step inside an episode:
  1. Observe tabular state  s = (row, col, orientation, battery_bin, is_clean)
  2. Select action a via ε-greedy
  3. Execute a  →  observe reward r, next state s'
  4. Apply off-policy Bellman backup:

        Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a')  −  Q(s,a) ]

  5. Advance: s ← s'

WHY THIS STATE REPRESENTATION
------------------------------
The Phase-2 environment exposes the compact tabular state:
    (row, col, orientation, battery_bin, is_apartment_clean)

  • row ∈ [0,14], col ∈ [0,14]       → 15 × 15 = 225 positions
  • orientation ∈ {N,E,S,W}          →   4 orientations
  • battery_bin ∈ {0..4}             →   5 battery levels  (BATTERY_BINS=5)
  • is_apartment_clean ∈ {0, 1}      →   2 modes  (seek-dirt / go-home)

  Total: 225 × 4 × 5 × 2 = 9,000 possible states  — perfectly tractable
  for a look-up table.  Encoding the full 15×15 dirt grid would yield 2^225
  states, making tabular RL completely infeasible.  The is_apartment_clean
  flag gives the agent an implicit mode switch: seek dirt while 0, navigate
  home while 1.

OFF-POLICY ADVANTAGE
--------------------
Because the Bellman target uses max Q(s',·), Q-Learning converges to the
OPTIMAL policy regardless of the exploratory behaviour that generates data.
This makes it aggressive and sample-efficient — at the cost of being slightly
less stable than on-policy SARSA in stochastic/noisy environments.
================================================================================
"""

from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import numpy as np


# Tabular state type alias used throughout the module
TabularState = Tuple[int, int, int, int, int]
# (row, col, orientation, battery_bin, is_apartment_clean)

NUM_ACTIONS = 4  # Phase-2: move_forward / rotate_left / rotate_right / charge


class QLearningAgent:
    """
    Tabular Q-Learning agent for the Phase-2 directional vacuum robot.

    The Q-table is implemented as a lazy dictionary:
        q_table[state_tuple] → np.ndarray  shape (NUM_ACTIONS,)

    Lazy initialisation means memory is only allocated for states the agent
    actually visits — efficient for large but sparsely-explored state spaces.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
    ):
        """
        Parameters
        ----------
        learning_rate   : α  — step size for Q-value updates (0 < α ≤ 1).
        discount_factor : γ  — weight of future rewards  (0 < γ ≤ 1).
        epsilon_start   : Initial exploration probability (1.0 = fully random).
        epsilon_end     : Minimum exploration probability after decay.
        epsilon_decay   : Multiplicative decay applied once per episode.
        """
        self.alpha         = learning_rate
        self.gamma         = discount_factor
        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: dict { TabularState tuple → np.ndarray(NUM_ACTIONS,) }
        self.q_table: dict = {}

        self.training_episodes: int = 0
        self.total_steps: int       = 0

        print("=" * 60)
        print("  Q-LEARNING AGENT  (Phase-2, Off-Policy TD)")
        print("=" * 60)
        print(f"  State  : (row, col, orient, batt_bin, is_clean)")
        print(f"  Max states: 9,000  (15×15×4×5×2)")
        print(f"  Actions   : {NUM_ACTIONS}  (fwd / L / R / charge)")
        print(f"  α         : {learning_rate}")
        print(f"  γ         : {discount_factor}")
        print(f"  ε         : {epsilon_start} → {epsilon_end}  (decay {epsilon_decay})")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _q(self, state: TabularState) -> np.ndarray:
        """
        Return Q-values for *state*, lazily creating zero-initialised
        entry on first visit.
        Zero initialisation is neutral — no optimistic / pessimistic bias.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(NUM_ACTIONS, dtype=np.float64)
        return self.q_table[state]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(
        self,
        state: TabularState,
        training: bool = True,
        eval_epsilon: float = 0.0,
    ) -> int:
        """
        ε-greedy action selection.

        Training    : explore with probability self.epsilon.
        Evaluation  : explore with probability eval_epsilon (default 0 = greedy).

        Ties in Q-values are broken randomly to avoid systematic bias.

        Parameters
        ----------
        state        : Tabular state tuple from env.get_tabular_state().
        training     : Whether to use the decaying epsilon.
        eval_epsilon : Fixed exploration rate during evaluation.

        Returns
        -------
        int  — action index in {0, 1, 2, 3}
        """
        eps = self.epsilon if training else eval_epsilon

        if np.random.random() < eps:
            return int(np.random.randint(NUM_ACTIONS))

        q = self._q(state)
        # Break ties randomly among actions with equal max Q-value
        return int(np.random.choice(np.flatnonzero(q == q.max())))

    # ------------------------------------------------------------------
    # Learning update  (off-policy Bellman backup)
    # ------------------------------------------------------------------

    def learn(
        self,
        state: TabularState,
        action: int,
        reward: float,
        next_state: TabularState,
        done: bool,
    ) -> float:
        """
        Perform one Q-Learning off-policy update step.

        Update rule:
            Q(s,a) ← Q(s,a) + α · [ target  −  Q(s,a) ]

        where   target = r                          if  done
                       = r + γ · max_a' Q(s',a')  otherwise

        Returns
        -------
        float — |TD error|  (useful for training diagnostics / logging).
        """
        q_sa = self._q(state)[action]

        if done:
            td_target = reward
        else:
            # Off-policy: use the greedy maximum over all next actions
            td_target = reward + self.gamma * self._q(next_state).max()

        td_error = td_target - q_sa
        self._q(state)[action] += self.alpha * td_error
        self.total_steps += 1

        return abs(td_error)

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Decay exploration rate by one step.  Call once per episode end."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_episodes += 1

    # end_episode is an alias so the training loop can call either name
    def end_episode(self) -> None:
        self.decay_epsilon()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise Q-table and agent state to a pickle file."""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "total_steps": self.total_steps,
            "hyperparams": {
                "alpha": self.alpha, "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
            },
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(
            f"[QLearningAgent] Saved → {path}"
            f"  (states={len(self.q_table)}, ε={self.epsilon:.4f})"
        )

    def load(self, path: str) -> None:
        """Restore Q-table and agent state from a pickle file."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.q_table            = data["q_table"]
        self.epsilon            = data["epsilon"]
        self.training_episodes  = data["training_episodes"]
        self.total_steps        = data["total_steps"]
        hp = data.get("hyperparams", {})
        for attr in ("alpha", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay"):
            if attr in hp:
                setattr(self, attr, hp[attr])
        print(
            f"[QLearningAgent] Loaded ← {path}"
            f"  (states={len(self.q_table)}, ε={self.epsilon:.4f},"
            f"  episodes={self.training_episodes})"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a snapshot of training metrics."""
        return {
            "algorithm":        "Q-Learning (off-policy)",
            "states_visited":   len(self.q_table),
            "epsilon":          round(self.epsilon, 5),
            "training_episodes": self.training_episodes,
            "total_steps":      self.total_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  QUICK SMOKE TEST — QLearningAgent (Phase-2)")
    print("=" * 60)

    agent = QLearningAgent(learning_rate=0.1, epsilon_start=1.0)

    # Phase-2 tabular state: (row, col, orientation, battery_bin, is_clean)
    s  = (0, 0, 0, 4, 0)
    s2 = (1, 0, 0, 3, 0)

    print("\n1. Q-values before learning:", agent._q(s))
    td = agent.learn(s, 0, 10.0, s2, False)
    print("   Q-values after  learning:", agent._q(s))
    print(f"   |TD error| = {td:.4f}")

    print("\n2. choose_action (ε=1.0, random):",
          [agent.choose_action(s) for _ in range(5)])
    agent.epsilon = 0.0
    print("   choose_action (ε=0.0, greedy):",
          [agent.choose_action(s, training=False) for _ in range(5)])

    agent.epsilon = 1.0
    for _ in range(200):
        agent.decay_epsilon()
    print(f"\n3. ε after 200 decays: {agent.epsilon:.5f}")

    agent.save("/tmp/ql_test.pkl")
    agent2 = QLearningAgent()
    agent2.load("/tmp/ql_test.pkl")
    print(f"\n4. Save/load OK: Q-values match = {list(agent._q(s)) == list(agent2._q(s))}")
    os.remove("/tmp/ql_test.pkl")

    print("\n5. Stats:", agent.stats())
    print("\nAll Q-Learning smoke tests passed!")
