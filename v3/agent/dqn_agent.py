"""
================================================================================
DQN AGENT  —  Phase-2 Directional Cleaning Robot
================================================================================

Deep Q-Network (DQN) extends tabular Q-Learning to high-dimensional state
spaces by approximating the Q-function with a neural network.

INPUT  →  9-channel 15×15 grid tensor  (from Phase2CleaningEnv.get_dqn_state())
   ch 0:  walls        (static binary map)
   ch 1:  furniture    (static binary map)
   ch 2:  current dirt (dynamic binary map, updated each step)
   ch 3:  charger      (single-cell indicator)
   ch 4:  robot facing NORTH  (one-hot orientation channels 4-7)
   ch 5:  robot facing EAST
   ch 6:  robot facing SOUTH
    ch 7:  robot facing WEST
    ch 8:  battery level (normalized map-wide channel)

OUTPUT →  4 Q-values   (move_forward / rotate_left / rotate_right / charge)

WHY CNN?
--------
The environment is a 2-D spatial grid.  A Convolutional Neural Network
naturally captures local spatial relationships (wall adjacency, dirt clusters,
robot orientation) without requiring flattening, which would discard the 2-D
structure entirely.  Convolutions also reuse weights across positions, which
significantly reduces parameter count and improves generalisation.

KEY DQN STABILISATION TECHNIQUES
---------------------------------
1. Experience Replay:   Transitions (s, a, r, s', done) are stored in a
   circular replay buffer.  Training samples random mini-batches instead of
   consecutive steps, breaking temporal correlation in the data.

2. Target Network:  A frozen copy of the Q-network computes the TD target
   y = r + γ · max Q_target(s',·) · (1-done).
   Weights are periodically copied from the policy network to prevent the
   "moving target" problem that destabilises pure DQN training.

3. Gradient Clipping:  Gradients are clipped to norm ≤ 10 to prevent
   exploding updates in early training when Q-values are far from optimal.
================================================================================
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# CNN Q-NETWORK
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Convolutional Q-network for Phase-2 grid input.

    Architecture
    ------------
    Input : (batch, 9, 15, 15)  — 9-channel grid
    Conv1 : Conv2d(9,  32, 3, padding=1) → (batch, 32, 15, 15)  + ReLU
    Conv2 : Conv2d(32, 64, 3, padding=1) → (batch, 64, 15, 15)  + ReLU
    Conv3 : Conv2d(64, 64, 3, stride=2, padding=1) → (batch, 64, 8, 8)  + ReLU
    Flatten: 64 × 8 × 8 = 4096
    FC1   : Linear(4096, 256)  + ReLU
    FC2   : Linear(256,    4)   → 4 Q-values

    Why three conv layers?
      • Layers 1-2 encode local features (nearby walls, dirt tiles).
      • Layer 3 (stride=2) reduces resolution, building a wider receptive field
        so the network can reason about room-scale structure.
    """

    # Precomputed conv output spatial size:
    #   floor((15 + 2*1 - 3) / 2) + 1 = 8  →  64 × 8 × 8 = 4096 feature units

    CONV_FLAT = 64 * 8 * 8   # = 4096

    def __init__(self, in_channels: int = 9, num_actions: int = 4, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.CONV_FLAT, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (batch, 9, 15, 15),  dtype float32, values in [0, 1]

        Returns
        -------
        Tensor shape (batch, 4)  — Q-value estimate for each action
        """
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)   # flatten spatial dims
        return self.fc(x)


# ---------------------------------------------------------------------------
# DQN AGENT
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    DQN agent for the Phase-2 directional vacuum robot.

    Training interface
    ------------------
    The agent consumes the 9×15×15 tensor returned by
    ``Phase2CleaningEnv.get_dqn_state()``.

    Typical training loop::

        obs, _ = env.reset()
        state  = env.get_dqn_state()          # (9, 15, 15) np.ndarray
        while not done:
            action  = agent.choose_action(state)
            obs, r, terminated, truncated, _ = env.step(action)
            next_st = env.get_dqn_state()
            done    = terminated or truncated
            agent.learn(state, action, r, next_st, done)
            state   = next_st
        agent.decay_epsilon()
    """

    IN_CHANNELS = 9
    GRID_H      = 15
    GRID_W      = 15
    NUM_ACTIONS = 4

    def __init__(
        self,
        learning_rate:    float = 3e-4,
        discount_factor:  float = 0.99,
        epsilon_start:    float = 1.0,
        epsilon_end:      float = 0.05,
        epsilon_decay:    float = 0.9990,
        batch_size:       int   = 64,
        memory_size:      int   = 20_000,
        target_update:    int   = 200,    # copy policy → target every N learn() calls
        hidden:           int   = 256,
        train_every:      int   = 4,      # gradient step every N environment steps
    ):
        """
        Parameters
        ----------
        learning_rate   : Adam optimiser LR.
        discount_factor : γ — importance of future rewards.
        epsilon_start   : Initial exploration rate.
        epsilon_end     : Minimum exploration rate.
        epsilon_decay   : Multiplicative decay per episode.
        batch_size      : Mini-batch size drawn from replay buffer.
        memory_size     : Replay buffer capacity (FIFO).
        target_update   : Sync target network every N gradient steps.
        hidden          : Hidden layer width in the Q-network FC section.
        train_every     : Only run a gradient step every N environment steps.
        """
        self.gamma         = discount_factor
        self.batch_size    = batch_size
        self.target_update = target_update
        self.train_every   = train_every

        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay

        # GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network (updated every train_every steps via backprop)
        self.policy_net = QNetwork(
            self.IN_CHANNELS, self.NUM_ACTIONS, hidden
        ).to(self.device)

        # Target network (frozen copy, synced every target_update steps)
        self.target_net = QNetwork(
            self.IN_CHANNELS, self.NUM_ACTIONS, hidden
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.lr = learning_rate

        # Replay buffer: stores (state, action, reward, next_state, done)
        # state / next_state are stored as float32 arrays of shape (9, 15, 15)
        self.memory: deque = deque(maxlen=memory_size)

        self.learn_step_counter: int = 0
        self.training_episodes: int  = 0
        self.total_steps: int        = 0

        print("=" * 62)
        print("  DQN AGENT  (Phase-2, CNN, Off-Policy)")
        print("=" * 62)
        print(f"  Input  : ({self.IN_CHANNELS}, {self.GRID_H}, {self.GRID_W}) grid tensor")
        print(f"  Network: Conv×3 → Flatten → FC×2 → {self.NUM_ACTIONS} actions")
        print(f"  Params : {self.num_parameters:,}")
        print(f"  Device : {self.device}")
        print(f"  α (LR) : {learning_rate}")
        print(f"  γ      : {discount_factor}")
        print(f"  ε      : {epsilon_start} → {epsilon_end}  (decay {epsilon_decay})")
        print(f"  Batch  : {batch_size}  |  Buffer: {memory_size}")
        print(f"  Target update every {target_update} gradient steps")
        print("=" * 62)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(
        self,
        state: np.ndarray,
        training: bool = True,
        eval_epsilon: float = 0.0,
    ) -> int:
        """
        ε-greedy action selection over the CNN policy network.

        Parameters
        ----------
        state        : np.ndarray  shape (8, 15, 15), from env.get_dqn_state().
        training     : Whether to use the decaying training epsilon.
        eval_epsilon : Fixed exploration rate for evaluation.

        Returns
        -------
        int  — action index in {0, 1, 2, 3}
        """
        eps = self.epsilon if training else eval_epsilon

        if random.random() < eps:
            return random.randint(0, self.NUM_ACTIONS - 1)

        self.policy_net.eval()
        with torch.no_grad():
            state_t = (
                torch.FloatTensor(state)       # (8, 15, 15)
                .unsqueeze(0)                  # (1, 8, 15, 15)
                .to(self.device)
            )
            q_values = self.policy_net(state_t)          # (1, 4)
        self.policy_net.train()
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def _store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition into the circular replay buffer."""
        self.memory.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    # ------------------------------------------------------------------
    # Learning update  (experience replay + target network)
    # ------------------------------------------------------------------

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Store transition and optionally train on a mini-batch.

        DQN training steps
        ------------------
        1. Push (s, a, r, s', done) into replay buffer.
        2. Every train_every steps, if buffer is large enough:
           a. Sample random mini-batch.
           b. Compute targets:  y = r + γ · max Q_target(s',·) · (1-done)
           c. Loss = MSE( Q_policy(s, a),  y )
           d. Backprop + gradient clip + Adam step.
           e. Every target_update steps, copy policy → target.

        Parameters
        ----------
        state      : np.ndarray (8,15,15)  current DQN state.
        action     : Action taken.
        reward     : Reward received.
        next_state : np.ndarray (8,15,15)  next DQN state.
        done       : Episode ended flag.

        Returns
        -------
        float or None — loss value if a gradient step was taken, else None.
        """
        self._store(state, action, reward, next_state, done)
        self.total_steps += 1

        # Only train every train_every environment steps
        if self.total_steps % self.train_every != 0:
            return None

        # Wait until replay buffer has enough transitions
        if len(self.memory) < self.batch_size:
            return None

        # ── Sample random mini-batch ──────────────────────────────────
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # ── Current Q-values from policy network ───────────────────────
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # ── Target Q-values from frozen target network ─────────────────
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q   = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        # ── Loss + backprop ────────────────────────────────────────────
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Periodic target network sync ──────────────────────────────
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Decay exploration rate by one step.  Call once per episode end."""
        self.epsilon = max(
            self.epsilon_end, self.epsilon * self.epsilon_decay
        )
        self.training_episodes += 1

    def end_episode(self) -> None:
        self.decay_epsilon()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy network weights and agent state."""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state":   self.optimizer.state_dict(),
                "epsilon":           self.epsilon,
                "training_episodes": self.training_episodes,
                "total_steps":       self.total_steps,
                "hyperparams": {
                    "gamma": self.gamma, "lr": self.lr,
                    "epsilon_start": self.epsilon_start,
                    "epsilon_end": self.epsilon_end,
                    "epsilon_decay": self.epsilon_decay,
                    "batch_size": self.batch_size,
                    "target_update": self.target_update,
                },
            },
            path,
        )
        print(
            f"[DQNAgent] Saved → {path}"
            f"  (ε={self.epsilon:.4f}, episodes={self.training_episodes})"
        )

    def load(self, path: str) -> None:
        """Load policy network weights and agent state."""
        data = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(data["policy_state_dict"])
        self.target_net.load_state_dict(data["target_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state"])
        self.epsilon           = data["epsilon"]
        self.training_episodes = data["training_episodes"]
        self.total_steps       = data["total_steps"]
        print(
            f"[DQNAgent] Loaded ← {path}"
            f"  (ε={self.epsilon:.4f}, episodes={self.training_episodes})"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a snapshot of training metrics."""
        return {
            "algorithm":         "DQN (CNN, off-policy)",
            "device":            str(self.device),
            "epsilon":           round(self.epsilon, 5),
            "training_episodes": self.training_episodes,
            "total_steps":       self.total_steps,
            "replay_size":       len(self.memory),
            "network_params":    self.num_parameters,
            "grad_steps":        self.learn_step_counter,
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  QUICK SMOKE TEST — DQNAgent (Phase-2)")
    print("=" * 62)

    agent = DQNAgent(batch_size=8, memory_size=100, target_update=10, train_every=1)

    # Simulate (8, 15, 15) tensors from Phase2CleaningEnv.get_dqn_state()
    rng = np.random.default_rng(42)
    state      = rng.random((8, 15, 15)).astype(np.float32)
    next_state = rng.random((8, 15, 15)).astype(np.float32)

    print("\n1. choose_action (ε=1.0, random):",
          [agent.choose_action(state, training=True) for _ in range(5)])
    agent.epsilon = 0.0
    print("   choose_action (ε=0.0, greedy):",
          [agent.choose_action(state, training=False) for _ in range(5)])
    agent.epsilon = 1.0

    print("\n2. Filling replay buffer and triggering a learn step...")
    losses = []
    for i in range(20):
        s  = rng.random((8, 15, 15)).astype(np.float32)
        ns = rng.random((8, 15, 15)).astype(np.float32)
        loss = agent.learn(s, random.randint(0, 3), float(rng.random()), ns, False)
        if loss is not None:
            losses.append(loss)
    print(f"   Losses recorded: {len(losses)},  last: {losses[-1]:.6f}" if losses else "   No loss yet (buffer too small)")

    print(f"\n3. Replay buffer size: {len(agent.memory)}")

    agent.save("/tmp/dqn_test.pth")
    agent2 = DQNAgent(batch_size=8, memory_size=100, target_update=10, train_every=1)
    agent2.load("/tmp/dqn_test.pth")
    print(f"\n4. Save/load OK: epsilon match = {agent.epsilon == agent2.epsilon}")
    os.remove("/tmp/dqn_test.pth")

    print("\n5. Stats:", agent.stats())
    print("\nAll DQN smoke tests passed!")
