"""
Phase 2 environment: realistic 15x15 apartment with directional robot physics.

This module is intentionally independent from the legacy `CleaningEnv` so the
existing project workflow keeps working while we build the modular Phase-2 stack.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# -----------------------------------------------------------------------------
# Apartment configuration (provided by the user)
# -----------------------------------------------------------------------------

PHASE2_CONFIG = {
    "rows": 15,
    "cols": 15,
    "charger_pos": (0, 0),
    "start_pos": (0, 0),
    # fixed battery budget: 2000 actions/steps before empty
    "battery_capacity": 2000,
    # keep battery fixed instead of scaling it by complexity
    "auto_battery": False,
    "walls": [
        (0, 7), (0, 8), (1, 7), (1, 8), (3, 7), (4, 7), (4, 8), (5, 7), (5, 8),
        (6, 0), (6, 1), (6, 2), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 11), (6, 12), (6, 13), (6, 14),
        (9, 0), (9, 1), (9, 2), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 11), (9, 12), (9, 13), (9, 14),
        (10, 7), (11, 7), (13, 7), (14, 7),
        (10, 12), (11, 12), (13, 12), (14, 12),
    ],
    "furniture": [
        (0, 5), (1, 1), (1, 2), (2, 2),
        (0, 9), (0, 10), (0, 11), (0, 13), (3, 10), (3, 11),
        (11, 1), (11, 2), (11, 3), (11, 4), (12, 1), (12, 2), (12, 3), (10, 5), (10, 6),
        (10, 9), (10, 10), (14, 9), (14, 10),
        (10, 13), (10, 14), (11, 13),
    ],
    "dirt_ratio": 1.0,
}

BATTERY_BINS = 5


# -----------------------------------------------------------------------------
# Reward configuration (dynamic dictionary)
# -----------------------------------------------------------------------------

DEFAULT_REWARD_CONFIG: Dict[str, Any] = {
    # Base penalties & mechanics
    "step_cost": -0.1,
    "wall_hit": -2.0,
    "charge_success": 5.0,  # charging when battery is low
    "charge_full": -1.0,  # charging while already full
    "charge_away": -5.0,  # charge action away from charger
    "battery_dead": -50.0,
    # Region-specific cleaning priorities
    "clean_kitchen": 20.0,
    "clean_bathroom": 20.0,
    "clean_living_room": 10.0,
    "clean_hallway": 10.0,
    "clean_bedroom": 5.0,
    "clean_storage": 5.0,
    # Small exploration bonuses help the agent leave the start room.
    "new_tile_bonus": 0.6,
    "new_room_bonus": 3.0,
    # End-game objective
    "mission_complete": 100.0,
    # Tunable threshold for "battery is low"
    "battery_low_threshold": 0.30,
}


# -----------------------------------------------------------------------------
# Orientation and action constants
# -----------------------------------------------------------------------------

ORIENT_NORTH = 0
ORIENT_EAST = 1
ORIENT_SOUTH = 2
ORIENT_WEST = 3

ORIENTATION_NAMES = {
    ORIENT_NORTH: "N",
    ORIENT_EAST: "E",
    ORIENT_SOUTH: "S",
    ORIENT_WEST: "W",
}

ACTION_MOVE_FORWARD = 0
ACTION_ROTATE_LEFT = 1
ACTION_ROTATE_RIGHT = 2
ACTION_CHARGE = 3

ACTION_NAMES = {
    ACTION_MOVE_FORWARD: "move_forward",
    ACTION_ROTATE_LEFT: "rotate_left",
    ACTION_ROTATE_RIGHT: "rotate_right",
    ACTION_CHARGE: "charge",
}


class Phase2CleaningEnv(gym.Env):
    """
    Smart vacuum apartment environment with directional robot mechanics.

    Why this state representation (explicitly):
    ------------------------------------------
    1) Tabular state: (x, y, orientation, battery_bin, is_apartment_clean)
       - This keeps the tabular state space tractable.
       - Encoding full dirt layout in a tabular key would explode the number of
         states combinatorially and make learning impractical.
       - `is_apartment_clean` gives a mode switch: before completion the policy
         should seek dirt; after completion it should go home to the charger.

    2) DQN state: multi-channel grid tensor
       - Deep methods can handle high-dimensional structured input.
       - We provide spatial channels for walls, furniture, dirt, charger, and
         orientation-aware robot position (4 one-hot orientation channels),
         preserving geometry and obstacle layout for convolution-style learning.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[Dict] = None,
        observation_mode: str = "tabular",
        battery_bins: int = BATTERY_BINS,
        max_steps: int = 600,
        reward_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.config = dict(PHASE2_CONFIG if config is None else config)
        self.rows = int(self.config["rows"])
        self.cols = int(self.config["cols"])
        self.charger_pos = tuple(self.config["charger_pos"])
        self.start_pos = tuple(self.config["start_pos"])
        self.base_battery_capacity = int(self.config["battery_capacity"])
        self.auto_battery = bool(self.config.get("auto_battery", True))
        self.dirt_ratio = float(self.config.get("dirt_ratio", 1.0))

        self.walls: Set[Tuple[int, int]] = set(map(tuple, self.config.get("walls", [])))
        self.furniture: Set[Tuple[int, int]] = set(map(tuple, self.config.get("furniture", [])))
        self.blocked: Set[Tuple[int, int]] = self.walls | self.furniture

        self.battery_bins = int(battery_bins)
        self.max_steps = int(max_steps)
        self.reward_cfg: Dict[str, Any] = dict(DEFAULT_REWARD_CONFIG)
        if reward_config:
            self.reward_cfg.update(reward_config)

        if observation_mode not in {"tabular", "dqn", "both"}:
            raise ValueError("observation_mode must be one of: 'tabular', 'dqn', 'both'")
        self.observation_mode = observation_mode

        self.action_space = spaces.Discrete(4)

        self.tabular_observation_space = spaces.MultiDiscrete(
            [self.rows, self.cols, 4, self.battery_bins, 2]
        )
        # Channels: walls, furniture, dirt, charger, robot_N/E/S/W, battery_level
        self.dqn_channels = 9
        self.dqn_observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.dqn_channels, self.rows, self.cols),
            dtype=np.float32,
        )

        if self.observation_mode == "tabular":
            self.observation_space = self.tabular_observation_space
        elif self.observation_mode == "dqn":
            self.observation_space = self.dqn_observation_space
        else:
            self.observation_space = spaces.Dict(
                {
                    "tabular": self.tabular_observation_space,
                    "dqn": self.dqn_observation_space,
                }
            )

        self.walkable_tiles: List[Tuple[int, int]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                if cell in self.blocked:
                    continue
                self.walkable_tiles.append(cell)

        if self.charger_pos in self.blocked:
            raise ValueError("charger_pos cannot be a wall/furniture tile")
        if self.start_pos in self.blocked:
            raise ValueError("start_pos cannot be a wall/furniture tile")

        # adapt battery capacity to map complexity (bigger/harder map => more battery)
        self.layout_complexity = self._compute_layout_complexity()
        if self.auto_battery:
            self.battery_capacity = self._derive_battery_capacity(
                self.base_battery_capacity,
                self.layout_complexity,
            )
        else:
            self.battery_capacity = self.base_battery_capacity

        self.rng = np.random.default_rng()
        self.robot_pos: Tuple[int, int] = self.start_pos
        self.orientation: int = ORIENT_NORTH
        self.battery: int = self.battery_capacity
        self.dirt: np.ndarray = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.steps_taken: int = 0
        self.last_action: Optional[int] = None
        self.visited_tiles: Set[Tuple[int, int]] = set()
        self.visited_rooms: Set[str] = set()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_blocked(self, pos: Tuple[int, int]) -> bool:
        return pos in self.blocked

    def _is_walkable(self, pos: Tuple[int, int]) -> bool:
        return self._in_bounds(pos) and not self._is_blocked(pos)

    def _forward_delta(self) -> Tuple[int, int]:
        if self.orientation == ORIENT_NORTH:
            return (-1, 0)
        if self.orientation == ORIENT_EAST:
            return (0, 1)
        if self.orientation == ORIENT_SOUTH:
            return (1, 0)
        return (0, -1)

    def _battery_bin(self) -> int:
        if self.battery_capacity <= 0:
            return 0
        ratio = np.clip(self.battery / self.battery_capacity, 0.0, 1.0)
        # ratio == 1.0 should map to the top bin
        idx = int(np.floor(ratio * self.battery_bins))
        return min(self.battery_bins - 1, idx)

    def _compute_layout_complexity(self) -> float:
        """
        Estimate how hard this grid is to navigate.

        We combine a few simple signals:
        - obstacle ratio (more blocked cells => harder)
        - branching penalty (narrower walkable graph => harder)
        - size factor (bigger grid => harder)
        - farthest walk distance from charger (long trips => harder)

        Returns a multiplier in a safe range [1.0, 2.2].
        """
        total_cells = max(1, self.rows * self.cols)
        obstacle_ratio = len(self.blocked) / total_cells

        walkable_set = set(self.walkable_tiles)
        if not walkable_set:
            return 1.0

        # using grid-neighbour degree as a cheap graph complexity proxy
        degrees = []
        for r, c in walkable_set:
            deg = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in walkable_set:
                    deg += 1
            degrees.append(deg)
        avg_degree = float(np.mean(degrees)) if degrees else 2.0
        branching_penalty = (4.0 - avg_degree) / 4.0

        size_factor = total_cells / 225.0  # 15x15 is baseline 1.0

        charger_r, charger_c = self.charger_pos
        farthest_dist = max(abs(r - charger_r) + abs(c - charger_c) for r, c in walkable_set)
        distance_penalty = farthest_dist / max(1.0, (self.rows + self.cols))

        complexity = (
            1.0
            + 0.60 * obstacle_ratio
            + 0.40 * branching_penalty
            + 0.20 * max(0.0, size_factor - 1.0)
            + 0.25 * distance_penalty
        )
        return float(np.clip(complexity, 1.0, 2.2))

    def _derive_battery_capacity(self, base_capacity: int, complexity: float) -> int:
        """Scale battery from a base value using the complexity multiplier."""
        # keep enough battery so random exploration can still traverse most of the map
        walkable_floor = int(np.ceil(len(self.walkable_tiles) * 1.35))
        min_cap = max(80, int(np.ceil(base_capacity * 1.10)), walkable_floor)
        max_cap = max(min_cap + 1, int(np.ceil(base_capacity * 2.00)))
        scaled = int(round(base_capacity * complexity))
        return int(np.clip(scaled, min_cap, max_cap))

    def is_apartment_clean(self) -> bool:
        return int(self.dirt.sum()) == 0

    def _build_tabular_state(self) -> np.ndarray:
        r, c = self.robot_pos
        state = np.array(
            [
                r,
                c,
                self.orientation,
                self._battery_bin(),
                int(self.is_apartment_clean()),
            ],
            dtype=np.int32,
        )
        return state

    def _build_dqn_state(self) -> np.ndarray:
        obs = np.zeros((self.dqn_channels, self.rows, self.cols), dtype=np.float32)

        # Static channels
        for r, c in self.walls:
            obs[0, r, c] = 1.0
        for r, c in self.furniture:
            obs[1, r, c] = 1.0

        # Dynamic channels
        obs[2] = self.dirt.astype(np.float32)
        cr, cc = self.charger_pos
        obs[3, cr, cc] = 1.0

        rr, rc = self.robot_pos
        # Orientation-aware robot channels
        obs[4 + self.orientation, rr, rc] = 1.0

        # Battery channel keeps DQN state informative when energy is running out.
        battery_ratio = self.battery / max(1, self.battery_capacity)
        obs[8, :, :] = float(np.clip(battery_ratio, 0.0, 1.0))

        return obs

    def get_tabular_state(self) -> Tuple[int, int, int, int, int]:
        """Convenience API: return a Python tuple for dict/Q-table indexing."""
        arr = self._build_tabular_state()
        return (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]), int(arr[4]))

    def get_dqn_state(self) -> np.ndarray:
        """Convenience API: return the multi-channel tensor."""
        return self._build_dqn_state()

    def _build_observation(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if self.observation_mode == "tabular":
            return self._build_tabular_state()
        if self.observation_mode == "dqn":
            return self._build_dqn_state()
        return {
            "tabular": self._build_tabular_state(),
            "dqn": self._build_dqn_state(),
        }

    def _seed_dirt(self) -> None:
        self.dirt.fill(0)
        candidates = [p for p in self.walkable_tiles if p != self.charger_pos]
        if self.dirt_ratio >= 1.0:
            selected = candidates
        elif self.dirt_ratio <= 0.0:
            selected = []
        else:
            n = int(np.ceil(len(candidates) * self.dirt_ratio))
            idx = self.rng.choice(len(candidates), size=n, replace=False)
            selected = [candidates[i] for i in idx]

        for r, c in selected:
            self.dirt[r, c] = 1

    def _consume_battery(self, amount: int = 1) -> bool:
        if self.battery <= 0:
            return False
        self.battery = max(0, self.battery - amount)
        return True

    def _room_for_position(self, pos: Tuple[int, int]) -> str:
        """Return semantic apartment region used by reward shaping."""
        r, c = pos

        if 0 <= r <= 5 and 0 <= c <= 6:
            return "living_room"
        if 0 <= r <= 5 and 8 <= c <= 14:
            return "kitchen"
        if 7 <= r <= 8:
            return "hallway"
        if 10 <= r <= 14 and 0 <= c <= 6:
            return "bedroom"
        if 10 <= r <= 14 and 8 <= c <= 11:
            return "bathroom"
        if 10 <= r <= 14 and 13 <= c <= 14:
            return "storage"

        # Doorway/connector cells not explicitly inside a room are treated as hallway.
        return "hallway"

    def _clean_reward_for_position(self, pos: Tuple[int, int]) -> float:
        room = self._room_for_position(pos)
        return float(self.reward_cfg.get(f"clean_{room}", 0.0))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.robot_pos = self.start_pos
        self.orientation = ORIENT_NORTH
        self.battery = self.battery_capacity
        self.steps_taken = 0
        self.last_action = None
        self._seed_dirt()
        self.visited_tiles = {self.robot_pos}
        self.visited_rooms = {self._room_for_position(self.robot_pos)}

        obs = self._build_observation()
        info = {
            "battery": self.battery,
            "battery_capacity": self.battery_capacity,
            "battery_bin": self._battery_bin(),
            "layout_complexity": round(self.layout_complexity, 3),
            "is_apartment_clean": self.is_apartment_clean(),
            "robot_pos": self.robot_pos,
            "orientation": self.orientation,
            "orientation_name": ORIENTATION_NAMES[self.orientation],
            "dirt_remaining": int(self.dirt.sum()),
        }
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps_taken += 1
        self.last_action = int(action)
        reward = float(self.reward_cfg["step_cost"])
        terminated = False
        truncated = False
        event = "none"

        # Battery dead away from charger immediately ends episode.
        if self.battery <= 0 and self.robot_pos != self.charger_pos:
            reward += float(self.reward_cfg["battery_dead"])
            event = "battery_dead"
            terminated = True
        else:
            if action == ACTION_MOVE_FORWARD:
                if self._consume_battery(1):
                    dr, dc = self._forward_delta()
                    nr = self.robot_pos[0] + dr
                    nc = self.robot_pos[1] + dc
                    next_pos = (nr, nc)
                    if self._is_walkable(next_pos):
                        self.robot_pos = next_pos
                        event = "moved"

                        # Give a small bonus when the robot explores unseen tiles.
                        if next_pos not in self.visited_tiles:
                            reward += float(self.reward_cfg["new_tile_bonus"])
                            self.visited_tiles.add(next_pos)

                        # Give a room-entry bonus once per room each episode.
                        next_room = self._room_for_position(next_pos)
                        if next_room not in self.visited_rooms:
                            reward += float(self.reward_cfg["new_room_bonus"])
                            self.visited_rooms.add(next_room)

                        rr, rc = self.robot_pos
                        if self.dirt[rr, rc] == 1:
                            self.dirt[rr, rc] = 0
                            reward += self._clean_reward_for_position(self.robot_pos)
                            event = "moved_and_cleaned"
                    else:
                        reward += float(self.reward_cfg["wall_hit"])
                        event = "wall_hit"
                else:
                    reward += float(self.reward_cfg["battery_dead"])
                    event = "battery_dead"
                    terminated = True

            elif action == ACTION_ROTATE_LEFT:
                if self._consume_battery(1):
                    self.orientation = (self.orientation - 1) % 4
                    event = "rotated_left"
                else:
                    reward += float(self.reward_cfg["battery_dead"])
                    event = "battery_dead"
                    terminated = True

            elif action == ACTION_ROTATE_RIGHT:
                if self._consume_battery(1):
                    self.orientation = (self.orientation + 1) % 4
                    event = "rotated_right"
                else:
                    reward += float(self.reward_cfg["battery_dead"])
                    event = "battery_dead"
                    terminated = True

            else:  # ACTION_CHARGE
                if self.robot_pos == self.charger_pos:
                    if self.battery >= self.battery_capacity:
                        reward += float(self.reward_cfg["charge_full"])
                        event = "charge_full"
                    else:
                        was_low = (
                            self.battery
                            <= int(
                                np.floor(
                                    self.battery_capacity
                                    * float(self.reward_cfg["battery_low_threshold"])
                                )
                            )
                        )
                        self.battery = self.battery_capacity
                        if was_low:
                            reward += float(self.reward_cfg["charge_success"])
                            event = "charge_success_low_battery"
                        else:
                            event = "charged_not_low"
                else:
                    reward += float(self.reward_cfg["charge_away"])
                    event = "charge_away"

        apartment_clean = self.is_apartment_clean()

        # Mission complete is granted ONLY when all dirt is cleaned AND robot is home.
        if apartment_clean and self.robot_pos == self.charger_pos and not terminated:
            reward += float(self.reward_cfg["mission_complete"])
            terminated = True

        if self.steps_taken >= self.max_steps:
            truncated = True

        obs = self._build_observation()
        info = {
            "event": event,
            "battery": self.battery,
            "battery_capacity": self.battery_capacity,
            "battery_bin": self._battery_bin(),
            "layout_complexity": round(self.layout_complexity, 3),
            "is_apartment_clean": apartment_clean,
            "robot_pos": self.robot_pos,
            "orientation": self.orientation,
            "orientation_name": ORIENTATION_NAMES[self.orientation],
            "dirt_remaining": int(self.dirt.sum()),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Simple ANSI rendering for debugging (kept lightweight and dependency-free)."""
        chars = np.full((self.rows, self.cols), ".", dtype="<U1")

        for r, c in self.walls:
            chars[r, c] = "#"
        for r, c in self.furniture:
            chars[r, c] = "F"
        dirty_positions = np.argwhere(self.dirt == 1)
        for r, c in dirty_positions:
            chars[r, c] = "*"

        cr, cc = self.charger_pos
        chars[cr, cc] = "C"

        rr, rc = self.robot_pos
        chars[rr, rc] = ORIENTATION_NAMES[self.orientation]

        lines = [" ".join(row.tolist()) for row in chars]
        out = "\n".join(lines)
        print(out)
        print(
            f"battery={self.battery}/{self.battery_capacity} | "
            f"dirt_remaining={int(self.dirt.sum())} | steps={self.steps_taken}"
        )
        return out


__all__ = [
    "Phase2CleaningEnv",
    "PHASE2_CONFIG",
    "BATTERY_BINS",
    "ACTION_MOVE_FORWARD",
    "ACTION_ROTATE_LEFT",
    "ACTION_ROTATE_RIGHT",
    "ACTION_CHARGE",
    "ORIENT_NORTH",
    "ORIENT_EAST",
    "ORIENT_SOUTH",
    "ORIENT_WEST",
]
