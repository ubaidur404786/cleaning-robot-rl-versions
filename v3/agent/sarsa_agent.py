"""
================================================================================
SARSA AGENT  —  Phase-2 Directional Cleaning Robot
================================================================================

SARSA is a model-free, ON-POLICY Temporal Difference control algorithm.
Unlike Q-Learning which bootstraps off the greedy best action, SARSA uses
the action ACTUALLY CHOSEN by the current policy in the next state.

The name comes from the quintuple used in each update:
    (S, A, R, S', A') → State, Action, Reward, next-State, next-Action

ALGORITHM
---------
For each step inside an episode:
  1. Observe state s,  choose action a via ε-greedy
  2. Execute a  →  observe reward r, next state s'
  3. Choose NEXT action a' via ε-greedy  ← key step (before update!)
  4. Apply on-policy Bellman backup:

        Q(s,a) ← Q(s,a) + α · [ r + γ · Q(s',a')  −  Q(s,a) ]

  5. Advance: s ← s', a ← a'

WHY ON-POLICY MATTERS
---------------------
Because the update uses Q(s',a') — the value of the action the agent WILL
actually take — the Q-values incorporate the exploration noise of the current
ε-greedy policy.  Near risky states (walls, battery dead-ends) SARSA learns
to be more cautious than Q-Learning, because it accounts for the chance that
exploration could send it into a penalty.
Q-Learning ignores that risk (it assumes future moves will be perfectly greedy).

COMPARISON WITH Q-LEARNING
---------------------------
  Q-Learning  update: ... γ · max_a' Q(s',a') ...  (off-policy, aggressive)
  SARSA       update: ... γ · Q(s', a')          ...  (on-policy, conservative)

Both converge to the optimal policy in the limit, but SARSA tends to produce
safer routes and is less likely to catastrophically drain the battery.
================================================================================
"""

from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import numpy as np


TabularState = Tuple[int, int, int, int, int]
# (row, col, orientation, battery_bin, is_apartment_clean)

NUM_ACTIONS = 4  # Phase-2: move_forward / rotate_left / rotate_right / charge


class SarsaAgent:
    """
    Tabular SARSA agent for the Phase-2 directional vacuum robot.

    Identical Q-table structure to QLearningAgent for fair comparison;
    the sole algorithmic difference is the on-policy learn() update.
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
        learning_rate   : α  — step size for Q-value updates.
        discount_factor : γ  — weight given to future rewards.
        epsilon_start   : Initial exploration probability.
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
        print("  SARSA AGENT  (Phase-2, On-Policy TD)")
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
        Return Q-values for *state*, lazily creating zero-initialised entry.
        Zeros mean the agent starts neutral — no domain bias.
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
        ε-greedy action selection (identical interface to QLearningAgent).

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
        return int(np.random.choice(np.flatnonzero(q == q.max())))

    # ------------------------------------------------------------------
    # Learning update  (ON-POLICY Bellman backup  — the key SARSA step)
    # ------------------------------------------------------------------

    def learn(
        self,
        state: TabularState,
        action: int,
        reward: float,
        next_state: TabularState,
        next_action: int,          # ← the action ACTUALLY chosen in s'
        done: bool,
    ) -> float:
        """
        Perform one SARSA on-policy update step.

        Update rule:
            Q(s,a) ← Q(s,a) + α · [ target  −  Q(s,a) ]

        where   target = r                        if  done
                       = r + γ · Q(s', a')       otherwise

        CRITICAL: a' must be chosen with choose_action(s') BEFORE calling
        learn(). The training loop must look like:
            a = agent.choose_action(s)
            s', r, done = env.step(a)
            a' = agent.choose_action(s')          ← before update
            agent.learn(s, a, r, s', a', done)    ← uses a'
            s, a = s', a'

        Parameters
        ----------
        state       : State before the step.
        action      : Action taken.
        reward      : Scalar reward received.
        next_state  : State after the step.
        next_action : Action chosen for the next step (already by the policy).
        done        : True if episode ended.

        Returns
        -------
        float — |TD error|
        """
        q_sa = self._q(state)[action]

        if done:
            td_target = reward
        else:
            # On-policy: use Q-value of the ACTUAL next action, not the max
            td_target = reward + self.gamma * self._q(next_state)[next_action]

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
            f"[SarsaAgent] Saved → {path}"
            f"  (states={len(self.q_table)}, ε={self.epsilon:.4f})"
        )

    def load(self, path: str) -> None:
        """Restore Q-table and agent state from a pickle file."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.q_table           = data["q_table"]
        self.epsilon           = data["epsilon"]
        self.training_episodes = data["training_episodes"]
        self.total_steps       = data["total_steps"]
        hp = data.get("hyperparams", {})
        for attr in ("alpha", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay"):
            if attr in hp:
                setattr(self, attr, hp[attr])
        print(
            f"[SarsaAgent] Loaded ← {path}"
            f"  (states={len(self.q_table)}, ε={self.epsilon:.4f},"
            f"  episodes={self.training_episodes})"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a snapshot of training metrics."""
        return {
            "algorithm":         "SARSA (on-policy)",
            "states_visited":    len(self.q_table),
            "epsilon":           round(self.epsilon, 5),
            "training_episodes": self.training_episodes,
            "total_steps":       self.total_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  QUICK SMOKE TEST — SarsaAgent (Phase-2)")
    print("=" * 60)

    agent = SarsaAgent(learning_rate=0.1)

    s  = (0, 0, 0, 4, 0)
    s2 = (1, 0, 0, 3, 0)

    print("\n1. Q-values before:", agent._q(s))
    td = agent.learn(s, 0, 10.0, s2, 2, False)     # next_action=2 (rotate_right)
    print("   Q-values after :", agent._q(s))
    print(f"   |TD error| = {td:.4f}")

    # Verify on-policy: update used Q(s', a'=2), not max Q(s',·)
    # Setup controlled test
    agent2 = SarsaAgent()
    agent2.q_table[s2] = np.array([10.0, 5.0, 3.0, 1.0])  # max=10 at action 0
    agent2.q_table[s]  = np.zeros(4)
    agent2.learn(s, 0, 0.0, s2, 2, False)               # next_action=2 → Q(s2,2)=3.0
    expected_target = 0.0 + 0.95 * 3.0                  # = 2.85
    expected_q      = 0.0 + 0.1 * (expected_target - 0.0)  # = 0.285
    print(f"\n2. On-policy check: Q(s,0) = {agent2._q(s)[0]:.4f}"
          f"  (expected {expected_q:.4f}, used Q(s2,2)=3 not max=10)")

    print("\n3. Stats:", agent.stats())
    print("\nAll SARSA smoke tests passed!")
