import numpy as np
from config import (
    PHASE1_CONFIG, REWARDS, BATTERY_BINS, NUM_ACTIONS, MAX_STEPS_PER_EPISODE,
    ROOM_DEFINITIONS, DIRT_BURST_CONFIG,
)


class CleaningRobotEnv:
    """Grid-world environment for a vacuum cleaning robot with battery."""

    def __init__(self, config=None):
        """
        Initialize the environment.

        Parameters
        ----------
        config : dict, optional
            Environment configuration. Defaults to PHASE1_CONFIG.
        """
        cfg = config or PHASE1_CONFIG

        self.rows = cfg["rows"]
        self.cols = cfg["cols"]
        self.charger_pos = cfg["charger_pos"]
        self.start_pos = cfg["start_pos"]
        self.battery_capacity = cfg["battery_capacity"]
        self.walls = set(map(tuple, cfg.get("walls", [])))
        self.furniture = set(map(tuple, cfg.get("furniture", [])))
        self.dirt_ratio = cfg.get("dirt_ratio", 1.0)

        # Blocked cells = walls + furniture 
        self.blocked = self.walls | self.furniture

        # All walkable tiles 
        self.walkable = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.blocked:
                    self.walkable.add((r, c))

        # Tiles that can be dirty
        self.cleanable = self.walkable - {self.charger_pos}

        # Movement deltas: Up, Down, Left, Right
        self._deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }

        self.num_actions = NUM_ACTIONS
        self.battery_bins = BATTERY_BINS
        self.max_steps = MAX_STEPS_PER_EPISODE

        # State space size
        self.num_states = self.rows * self.cols * self.battery_bins

        # Will be set by reset()
        self.agent_pos = None
        self.battery = None
        self.dirt_grid = None
        self.steps = 0
        self.done = False
        self.total_dirty = 0
        self.cleaned_count = 0

    def reset(self, seed=None):
        """
        Reset the environment to the initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        state : tuple
            Initial state (row, col, battery_bin).
        """
        if seed is not None:
            np.random.seed(seed)

        self.agent_pos = self.start_pos
        self.battery = self.battery_capacity
        self.steps = 0
        self.done = False
        self.cleaned_count = 0

        # Initialize dirt grid: 1 = dirty, 0 = clean
        self.dirt_grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        if self.dirt_ratio >= 1.0:
            for (r, c) in self.cleanable:
                self.dirt_grid[r, c] = 1
        else:
            # Random subset of cleanable tiles are dirty
            cleanable_list = list(self.cleanable)
            n_dirty = int(len(cleanable_list) * self.dirt_ratio)
            dirty_tiles = np.random.choice(
                len(cleanable_list), size=n_dirty, replace=False
            )
            for idx in dirty_tiles:
                r, c = cleanable_list[idx]
                self.dirt_grid[r, c] = 1

        self.total_dirty = int(self.dirt_grid.sum())

        return self._get_state()

    def step(self, action):
        """
        Execute an action in the environment.

        Parameters
        ----------
        action : int
            0=Up, 1=Down, 2=Left, 3=Right, 4=Charge

        Returns
        -------
        state : tuple
            New state (row, col, battery_bin).
        reward : float
            Reward received.
        done : bool
            Whether the episode is over.
        info : dict
            Additional information.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        self.steps += 1
        reward = 0.0
        info = {"event": "none"}

        if action == 4:
            # ---- CHARGE action ----
            if self.agent_pos == self.charger_pos:
                if self.battery < self.battery_capacity:
                    self.battery = self.battery_capacity
                    reward = REWARDS["charge_success"]
                    info["event"] = "charge_success"
                else:
                    reward = REWARDS["charge_full"]
                    info["event"] = "charge_full"
            else:
                reward = REWARDS["charge_away"]
                info["event"] = "charge_away"
                # Charging away still costs 1 battery (wasted action)
                self.battery -= 1
        else:
            # ---- MOVE action ----
            dr, dc = self._deltas[action]
            new_r = self.agent_pos[0] + dr
            new_c = self.agent_pos[1] + dc

            if (
                new_r < 0 or new_r >= self.rows
                or new_c < 0 or new_c >= self.cols
                or (new_r, new_c) in self.blocked
            ):
                # Hit wall / out of bounds / blocked
                reward = REWARDS["wall_hit"]
                info["event"] = "wall_hit"
                # Battery still drains for the attempt
                self.battery -= 1
            else:
                # Successful move
                self.agent_pos = (new_r, new_c)
                self.battery -= 1

                if self.dirt_grid[new_r, new_c] == 1:
                    # Auto-clean: stepped on dirty tile
                    self.dirt_grid[new_r, new_c] = 0
                    self.cleaned_count += 1
                    reward = REWARDS["clean_dirty"]
                    info["event"] = "clean_dirty"
                else:
                    # Stepped on clean tile
                    reward = REWARDS["step_cost"]
                    info["event"] = "step_clean"

        # ---- Check termination conditions ----

        # Battery death
        if self.battery <= 0:
            reward = REWARDS["battery_dead"]
            info["event"] = "battery_dead"
            self.done = True

        # All tiles cleaned
        elif self.dirt_grid.sum() == 0:
            info["event"] = "all_clean"
            self.done = True

        # Max steps reached
        elif self.steps >= self.max_steps:
            info["event"] = "max_steps"
            self.done = True

        # Build info
        info["battery"] = self.battery
        info["cleaned"] = self.cleaned_count
        info["total_dirty"] = self.total_dirty
        info["coverage"] = (
            self.cleaned_count / self.total_dirty if self.total_dirty > 0 else 1.0
        )
        info["steps"] = self.steps

        return self._get_state(), reward, self.done, info

    def _get_state(self):
        """
        Get the current state as (row, col, battery_bin).

        Battery is discretized into bins:
        - bin 0: battery in [0, capacity/bins)
        - bin 1: battery in [capacity/bins, 2*capacity/bins)
        - ...
        - bin (bins-1): battery in [(bins-1)*capacity/bins, capacity]
        """
        bin_size = self.battery_capacity / self.battery_bins
        battery_bin = min(
            int(self.battery / bin_size),
            self.battery_bins - 1,
        )
        return (self.agent_pos[0], self.agent_pos[1], battery_bin)

    def state_to_index(self, state):
        """Convert (row, col, battery_bin) to a flat index for Q-table."""
        r, c, b = state
        return r * self.cols * self.battery_bins + c * self.battery_bins + b

    def get_coverage(self):
        """Return the fraction of dirty tiles that have been cleaned."""
        if self.total_dirty == 0:
            return 1.0
        return self.cleaned_count / self.total_dirty

    def get_dirt_map(self):
        """Return a copy of the current dirt grid."""
        return self.dirt_grid.copy()

    def __repr__(self):
        return (
            f"CleaningRobotEnv({self.rows}x{self.cols}, "
            f"battery={self.battery}/{self.battery_capacity}, "
            f"cleaned={self.cleaned_count}/{self.total_dirty})"
        )


class DirtRegenerationEnv(CleaningRobotEnv):
    """
    Extended apartment environment with periodic dirt bursts.

    Each room regenerates dirt at a different rate, simulating real-world
    usage patterns (kitchen dirtier than storage). The agent must learn —
    purely from reward signals — to prioritize high-traffic rooms.

    Key differences from CleaningRobotEnv:
    - Dirt bursts: every N steps per room, a fraction of its tiles re-dirty.
    - "all_clean" termination is disabled (dirt always comes back).
    - Coverage tracks cumulative cleans (including re-cleans), not just
      initial dirty tiles.
    - Room-level visit tracking for analysis.
    """

    def __init__(self, config, max_steps=None):
        """
        Parameters
        ----------
        config : dict
            Must include 'dirt_burst_config' and 'room_definitions'.
        max_steps : int, optional
            Override MAX_STEPS_PER_EPISODE for longer episodes.
        """
        super().__init__(config)

        if max_steps is not None:
            self.max_steps = max_steps

        self.dirt_burst_config = config["dirt_burst_config"]
        raw_room_defs = config["room_definitions"]

        # Filter room tile lists to only include walkable tiles (no walls/furniture).
        # Also exclude charger tile  it's never dirty.
        self.room_tiles = {}
        for room_name, tile_list in raw_room_defs.items():
            self.room_tiles[room_name] = [
                (r, c) for (r, c) in tile_list
                if (r, c) in self.cleanable
            ]

        # Tracking: per-room visit counts (reset each episode)
        self.room_visits = None
        # Cumulative cleans including re-cleans
        self.total_cleans = 0
        # Track how many dirt bursts fired (for analysis)
        self.burst_log = []

    def reset(self, seed=None):
        """Reset environment and room-visit tracking."""
        state = super().reset(seed=seed)

        # Initialize room visit tracking
        self.room_visits = {room: 0 for room in self.room_tiles}
        self.total_cleans = 0
        self.burst_log = []

        return state

    def step(self, action):
        """
        Execute action, then check for dirt bursts.

        Returns same (state, reward, done, info) tuple as parent, but info
        includes additional keys:
        - 'room_visits': dict of room visit counts this episode
        - 'total_cleans': cumulative tiles cleaned (including re-cleans)
        - 'bursts_this_step': list of (room_name, n_re_dirtied)
        """
        # --- 1. Execute the normal step ---
        state, reward, done, info = super().step(action)

        # Track room visits based on current position
        for room_name, tiles in self.room_tiles.items():
            if self.agent_pos in tiles:
                self.room_visits[room_name] += 1
                break

        # Update cumulative clean count (parent tracks cleaned_count for
        # initial dirty tiles, but we need to count re-cleans too)
        if info["event"] == "clean_dirty":
            self.total_cleans += 1

        # --- 2. Override "all_clean" termination ---
        # With dirt regen, we never end early because dirt comes back.
        if info["event"] == "all_clean":
            self.done = False
            done = False

        # --- 3. Dirt burst check (only if episode is still running) ---
        bursts_this_step = []
        if not done:
            for room_name, burst_cfg in self.dirt_burst_config.items():
                interval = burst_cfg["burst_interval"]
                probability = burst_cfg["burst_probability"]
                intensity = burst_cfg["burst_intensity"]

                if (self.steps % interval == 0
                        and np.random.random() < probability):
                    # Fire a burst: re-dirty a fraction of the room's tiles
                    room_cleanable = self.room_tiles.get(room_name, [])
                    if not room_cleanable:
                        continue

                    # Only re-dirty tiles that are currently clean
                    clean_tiles = [
                        (r, c) for (r, c) in room_cleanable
                        if self.dirt_grid[r, c] == 0
                    ]
                    if not clean_tiles:
                        continue

                    n_to_dirty = max(1, int(len(room_cleanable) * intensity))
                    n_to_dirty = min(n_to_dirty, len(clean_tiles))

                    chosen = np.random.choice(
                        len(clean_tiles), size=n_to_dirty, replace=False
                    )
                    for idx in chosen:
                        r, c = clean_tiles[idx]
                        self.dirt_grid[r, c] = 1

                    bursts_this_step.append((room_name, n_to_dirty))

            if bursts_this_step:
                self.burst_log.append((self.steps, bursts_this_step))

        # --- 4. Enrich info dict ---
        info["room_visits"] = dict(self.room_visits)
        info["total_cleans"] = self.total_cleans
        info["bursts_this_step"] = bursts_this_step
        info["current_dirt"] = int(self.dirt_grid.sum())

        return state, reward, done, info
