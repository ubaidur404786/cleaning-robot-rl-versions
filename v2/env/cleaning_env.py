"""Gymnasium environment for the cleaning robot task (v2)."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# Constants

# Grid dimensions
GRID_WIDTH = 8    # Number of columns in the grid
GRID_HEIGHT = 6   # Number of rows in the grid

# Room type identifiers (used in room_layout array)
EMPTY = 0         # Wall/Outside area - robot cannot enter
KITCHEN = 1       # Kitchen room (highest cleaning priority)
LIVING_ROOM = 2   # Living room (medium cleaning priority)
HALLWAY = 3       # Hallway (lowest cleaning priority)

# Tile cleanliness states
CLEAN = 0         # Tile has been cleaned
DIRTY = 1         # Tile needs cleaning

# Actions

ACTION_FORWARD = 0    # Move up (decrease row)
ACTION_BACKWARD = 1   # Move down (increase row)
ACTION_LEFT = 2       # Move left (decrease column)
ACTION_RIGHT = 3      # Move right (increase column)
ACTION_WAIT = 4       # Stay in place (do nothing)
ACTION_CLEAN = 5      # Clean the current tile

# Action names
ACTION_NAMES = {
    0: "Forward",
    1: "Backward",
    2: "Left",
    3: "Right",
    4: "Wait",
    5: "Clean"
}

# Number of available actions
NUM_ACTIONS = 6

# Rewards (positive rewards only if tile is dirty)

# Cleaning rewards (positive) - Only awarded when tile is DIRTY
REWARD_CLEAN_KITCHEN = 50      # Kitchen has highest priority
REWARD_CLEAN_LIVING = 35       # Living room has medium priority
REWARD_CLEAN_HALLWAY = 20      # Hallway has lowest priority

# Completion bonus - Big reward for cleaning everything
REWARD_ALL_CLEAN_BONUS = 200   # Encourages completing the task

# Penalty values (negative) - Discourage wasteful actions
REWARD_STEP_ON_CLEAN = -5         # Stepping on an already-clean tile
REWARD_CLEAN_ALREADY_CLEAN = -10  # Using Clean action on clean tile
REWARD_HIT_WALL = -5              # Penalize trying to move into walls
REWARD_WAIT = -3                  # Penalize waiting (wastes time)
REWARD_STEP_PENALTY = -0.1        # Small penalty per step (encourages efficiency)

# Visualization colors

COLOR_KITCHEN = (255, 255, 180)       # Light yellow for kitchen
COLOR_LIVING_ROOM = (180, 200, 255)   # Light blue for living room
COLOR_HALLWAY = (200, 200, 200)       # Light gray for hallway
COLOR_WALL = (80, 80, 80)             # Dark gray for walls
COLOR_ROBOT = (50, 180, 50)           # Green for robot
COLOR_DIRTY = (139, 90, 43)           # Brown for dirty tiles
COLOR_CLEAN_MARKER = (100, 220, 100)  # Bright green checkmark for cleaned
COLOR_GRID_LINE = (50, 50, 50)        # Dark lines for grid
COLOR_TEXT = (255, 255, 255)          # White text
COLOR_BLACK = (0, 0, 0)               # Black for outlines


class CleaningEnv(gym.Env):
    """Custom Gymnasium environment for the cleaning robot task."""
    
    # Gymnasium metadata - tells Gymnasium what render modes we support
    metadata = {
        "render_modes": ["human", "rgb_array"],  # Supported render modes
        "render_fps": 10                          # Frames per second for rendering
    }
    
    def __init__(self, render_mode=None):
        """
        Initialize the Cleaning Robot Environment.
        
        This constructor sets up:
        1. The room layout (which cells belong to which room type)
        2. The action and observation spaces (Gymnasium requirements)
        3. Pygame visualization components (for rendering)
        4. Position indexing for state encoding
        
        Parameters:
        -----------
        render_mode : str or None
            How to render the environment:
            - None: No rendering (fastest for training)
            - "human": Display Pygame window (for watching)
            - "rgb_array": Return RGB array (for recording)
        """
        # Call parent class constructor (required by Gymnasium)
        super().__init__()
        
        # Store render mode for later use
        self.render_mode = render_mode
        
        # ======================================================================
        # PYGAME VISUALIZATION SETUP
        # ======================================================================
        # These variables will be initialized when render() is first called
        self.window = None           # Pygame window object
        self.clock = None            # Pygame clock for frame rate control
        self.cell_size = 100         # Pixel size of each grid cell
        
        # ======================================================================
        # ACTION SPACE DEFINITION
        # ======================================================================
        # Gymnasium Discrete space: Actions are integers from 0 to 5
        # This tells Gymnasium what actions are valid in this environment
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        # ======================================================================
        # CREATE ROOM LAYOUT
        # ======================================================================
        # Build the 2D array that defines which room each cell belongs to
        self.room_layout = self._create_room_layout()
        
        # ======================================================================
        # IDENTIFY CLEANABLE TILES
        # ======================================================================
        # Create a list of all tiles that can be cleaned (i.e., not walls)
        # We iterate through the grid and collect positions where room != EMPTY
        self.cleanable_tiles = []
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                if self.room_layout[row][col] != EMPTY:
                    self.cleanable_tiles.append((row, col))
        
        # Store the total number of cleanable tiles
        self.num_cleanable = len(self.cleanable_tiles)
        
        # ======================================================================
        # POSITION INDEX MAPPING
        # ======================================================================
        # Map (row, col) coordinates to integer indices for state encoding
        # This allows us to convert 2D positions to a single number for Q-table
        #
        # Example:
        #   pos_to_index[(1,1)] = 0   (first kitchen tile)
        #   pos_to_index[(1,2)] = 1   (second kitchen tile)
        #   ... and so on
        self.pos_to_index = {}     # (row, col) -> integer index
        self.index_to_pos = {}     # integer index -> (row, col)
        for idx, (row, col) in enumerate(self.cleanable_tiles):
            self.pos_to_index[(row, col)] = idx
            self.index_to_pos[idx] = (row, col)
        
        # ======================================================================
        # OBSERVATION SPACE DEFINITION (Position + Dirt + History + DNUT)
        # ======================================================================
        # State = position + is_dirty + came_from_direction + dnut_direction
        #
        # Components:
        # - position: Which tile robot is on (0-22 = 23 positions)
        # - is_dirty: Current tile dirt status (0 or 1)
        # - came_from: Direction robot just came from (0-4: N/S/E/W/none)
        # - dnut: Relative direction to nearest dirty tile (0-9)
        #         0-8 = (dx+1)*3 + (dy+1) where dx,dy ∈ {-1,0,+1}
        #         9   = no dirty tiles remain
        #
        # Total: 23 positions × 2 dirt × 5 directions × 10 DNUT = 2300 states
        
        self.num_directions = 5   # N, S, E, W, none
        self.num_dnut = 10        # 3×3 direction grid + none
        self.state_space_size = self.num_cleanable * 2 * self.num_directions * self.num_dnut  # 2300 states
        self.observation_space = spaces.Discrete(self.state_space_size)
        
        # ======================================================================
        # ROOM REWARD MAPPING
        # ======================================================================
        # Different rewards for cleaning different room types
        # Kitchen pays the most, hallway the least (priority-based)
        self.room_rewards = {
            KITCHEN: REWARD_CLEAN_KITCHEN,
            LIVING_ROOM: REWARD_CLEAN_LIVING,
            HALLWAY: REWARD_CLEAN_HALLWAY
        }
        
        # ======================================================================
        # ROOM NAME MAPPING (for visualization and debugging)
        # ======================================================================
        self.room_names = {
            EMPTY: "Wall",
            KITCHEN: "Kitchen",
            LIVING_ROOM: "Living Room",
            HALLWAY: "Hallway"
        }
        
        # ======================================================================
        # STATE VARIABLES (will be initialized in reset())
        # ======================================================================
        self.robot_row = 0        # Robot's current row position
        self.robot_col = 0        # Robot's current column position
        self.dirt_map = None      # 2D array tracking dirty status of each tile
        self.steps_taken = 0      # Number of steps in current episode
        self.tiles_cleaned = 0    # Number of tiles cleaned this episode
        self.max_steps = 300      # Maximum steps per episode (prevents infinite)
        self.last_direction = 4   # Direction we came from (0=N, 1=S, 2=E, 3=W, 4=none)
        
        # ======================================================================
        # PRINT INITIALIZATION INFO
        # ======================================================================
        print("=" * 65)
        print("  CLEANING ROBOT ENVIRONMENT INITIALIZED (Pure Q-Learning)")
        print("=" * 65)
        print(f"  Grid size:          {GRID_WIDTH} × {GRID_HEIGHT}")
        print(f"  Cleanable tiles:    {self.num_cleanable}")
        print(f"  State space size:   {self.state_space_size} states")
        print(f"  Action space size:  {NUM_ACTIONS} actions")
        print(f"  Max steps/episode:  {self.max_steps}")
        print("-" * 65)
        print("  Room Cleaning Rewards (dirty tiles only):")
        print(f"    Kitchen:     +{REWARD_CLEAN_KITCHEN} points")
        print(f"    Living Room: +{REWARD_CLEAN_LIVING} points")
        print(f"    Hallway:     +{REWARD_CLEAN_HALLWAY} points")
        print(f"    Clean tile:  {REWARD_STEP_ON_CLEAN} penalty (step)")
        print(f"    All Clean:   +{REWARD_ALL_CLEAN_BONUS} bonus")
        print(f"  DNUT feature:   enabled (10 direction bins)")
        print("=" * 65)
    
    def _create_room_layout(self):
        """
        Create the house layout as a 2D numpy array.
        
        This method defines which cells belong to which room type.
        The layout is hardcoded for this simulation but could be
        modified for different house configurations.
        
        Returns:
        --------
        numpy.ndarray
            2D array of shape (GRID_HEIGHT, GRID_WIDTH) containing
            room type identifiers (EMPTY, KITCHEN, LIVING_ROOM, HALLWAY)
        
        Layout Visualization:
        --------------------
        Row 0: All walls (boundary)
        Row 1: Wall | Kitchen (3 tiles) | Wall | Living (2 tiles) | Wall
        Row 2: Wall | Kitchen (3 tiles) | Hall | Living (2 tiles) | Wall
        Row 3: Wall | Kitchen (3 tiles) | Hall | Living (2 tiles) | Wall
        Row 4: Wall | Hallway (6 tiles connecting both rooms)      | Wall
        Row 5: All walls (boundary)
        
        This layout creates an interesting navigation challenge:
        - Kitchen and Living Room are separated by a wall
        - Hallway connects both rooms
        - Robot must learn to navigate through hallway
        """
        # Initialize grid with all walls (EMPTY)
        layout = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
        
        # ======================================================================
        # KITCHEN: 3×3 area in top-left corner (9 tiles)
        # ======================================================================
        # Kitchen is the highest priority room for cleaning (+50 per tile)
        # Located at rows 1-3, columns 1-3
        for row in range(1, 4):      # Rows 1, 2, 3
            for col in range(1, 4):  # Columns 1, 2, 3
                layout[row][col] = KITCHEN
        
        # ======================================================================
        # LIVING ROOM: 3×2 area in top-right corner (6 tiles)
        # ======================================================================
        # Living room has medium priority for cleaning (+35 per tile)
        # Located at rows 1-3, columns 5-6
        for row in range(1, 4):      # Rows 1, 2, 3
            for col in range(5, 7):  # Columns 5, 6
                layout[row][col] = LIVING_ROOM
        
        # ======================================================================
        # HALLWAY: Connecting corridor (8 tiles total)
        # ======================================================================
        # Hallway connects the kitchen and living room (+20 per tile)
        # 
        # Vertical connection piece (allows access from rooms to main hallway):
        layout[2][4] = HALLWAY  # Row 2, Col 4 - connects to both rooms
        layout[3][4] = HALLWAY  # Row 3, Col 4 - connects to both rooms
        
        # Horizontal main hallway (bottom strip connecting entire house):
        for col in range(1, 7):      # Columns 1 through 6
            layout[4][col] = HALLWAY
        
        return layout
    
    def _get_nearest_dirty_direction(self):
        """
        DNUT (Detection of Nearest Uncleaned Tile).
        
        Find the nearest dirty tile by Manhattan distance and return the
        relative direction as an encoded integer.
        
        Returns:
        --------
        int
            0-8: encoded direction (dx+1)*3 + (dy+1) where dx, dy ∈ {-1,0,+1}
            9:   no dirty tiles remain
        """
        best_dist = float('inf')
        best_dr, best_dc = 0, 0
        
        for row, col in self.cleanable_tiles:
            if self.dirt_map[row][col] == DIRTY:
                dist = abs(row - self.robot_row) + abs(col - self.robot_col)
                if dist < best_dist:
                    best_dist = dist
                    best_dr = row - self.robot_row
                    best_dc = col - self.robot_col
        
        if best_dist == float('inf'):
            # No dirty tiles left
            return 9
        
        # Convert to sign: -1, 0, +1
        dx = (best_dr > 0) - (best_dr < 0)
        dy = (best_dc > 0) - (best_dc < 0)
        
        return (dx + 1) * 3 + (dy + 1)
    
    def _get_state(self):
        """
        Convert current environment state to a single integer for Q-table lookup.
        
        STATE REPRESENTATION (Position + Dirt + Movement History + DNUT):
        
        Components combined:
        1. position: Which tile (0-22)
        2. is_dirty: Current tile status (0 or 1)
        3. came_from: Direction robot just came from (0-4)
           - 0: Came from North (moved South to get here)
           - 1: Came from South (moved North to get here)
           - 2: Came from East (moved West to get here)
           - 3: Came from West (moved East to get here)
           - 4: No movement yet (start of episode)
        4. dnut: Relative direction to nearest dirty tile (0-9)
           - 0-8: (dx+1)*3 + (dy+1) directional encoding
           - 9: no dirty tiles remain
        
        Formula: state = pos + is_dirty*23 + came_from*46 + dnut*230
        Total: 2300 states (23 × 2 × 5 × 10)
        
        Returns:
        --------
        int
            State index for the Q-table (0 to 2299)
        """
        # Get position index (0 to 22)
        pos_index = self.pos_to_index.get((self.robot_row, self.robot_col), 0)
        
        # Get dirt status of current tile (0 = clean, 1 = dirty)
        is_dirty = 1 if self.dirt_map[self.robot_row][self.robot_col] == DIRTY else 0
        
        # Get the direction we came from (set by step() after each move)
        came_from = self.last_direction
        
        # Get DNUT direction to nearest dirty tile
        dnut = self._get_nearest_dirty_direction()
        
        # Encode as single integer:
        # Each component contributes to non-overlapping ranges
        state = (pos_index
                 + is_dirty * self.num_cleanable
                 + came_from * self.num_cleanable * 2
                 + dnut * self.num_cleanable * 2 * self.num_directions)
        
        return int(state)
    
    def _get_room_dirty_combo(self):
        """
        Calculate a 3-bit encoding of which rooms have dirty tiles.
        
        Returns:
        --------
        int
            Value 0-7 representing room dirty status
            - 2^0=Bit 0 (value 1): Kitchen has dirty tiles
            - 2^1=Bit 1 (value 2): Living room has dirty tiles
            - 2^2= Bit 2 (value 4): Hallway has dirty tiles
            
        Examples:
        ---------
        - 0 = No rooms have dirty tiles (all clean!)
        - 1 = Only kitchen has dirty tiles
        - 3 = Kitchen and living room have dirty tiles
        - 7 = All rooms have dirty tiles
        """
        combo = 0
        
        # Check each room for dirty tiles
        kitchen_dirty = False
        living_dirty = False
        hallway_dirty = False
        
        for row, col in self.cleanable_tiles:
            if self.dirt_map[row][col] == DIRTY:
                room_type = self.room_layout[row][col]
                if room_type == KITCHEN:
                    kitchen_dirty = True
                elif room_type == LIVING_ROOM:
                    living_dirty = True
                elif room_type == HALLWAY:
                    hallway_dirty = True
        
        # Encode as 3-bit value
        if kitchen_dirty:
            combo += 1  # Bit 0
        if living_dirty:
            combo += 2  # Bit 1
        if hallway_dirty:
            combo += 4  # Bit 2
        
        return combo
    
    def _count_dirty_tiles(self):
        """
        Count how many tiles are still dirty.
        
        This method iterates through all cleanable tiles and counts
        how many still have DIRTY status.
        
        Returns:
        --------
        int
            Number of dirty tiles remaining (0 when all clean)
        """
        dirty_count = 0
        for row, col in self.cleanable_tiles:
            if self.dirt_map[row][col] == DIRTY:
                dirty_count += 1
        return dirty_count
    
    def _is_valid_position(self, row, col):
        """
        Check if a position is valid for the robot to occupy.
        
        A position is valid if:
        1. It's within the grid boundaries
        2. It's not a wall (EMPTY)
        
        Parameters:
        -----------
        row : int
            Row coordinate to check
        col : int
            Column coordinate to check
        
        Returns:
        --------
        bool
            True if robot can move to this position, False otherwise
        """
        # Check grid boundaries
        if row < 0 or row >= GRID_HEIGHT:
            return False
        if col < 0 or col >= GRID_WIDTH:
            return False
        
        # Check if position is a wall (EMPTY means wall/outside)
        if self.room_layout[row][col] == EMPTY:
            return False
        
        return True
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        This method is called at the start of each training/testing episode.
        It:
        1. Resets step counter and tiles cleaned counter
        2. Places robot at starting position (center of hallway)
        3. Makes all tiles dirty (fresh start)
        
        Parameters:
        -----------
        seed : int or None
            Random seed for reproducibility (passed to parent class)
        options : dict or None
            Additional reset options (not used in this implementation)
        
        Returns:
        --------
        tuple (observation, info)
            observation : int
                Initial state observation
            info : dict
                Additional information dictionary
        """
        # Call parent reset to handle seeding properly
        super().reset(seed=seed)
        
        # Reset episode tracking variables
        self.steps_taken = 0
        self.tiles_cleaned = 0
        
        # ======================================================================
        # PLACE ROBOT AT STARTING POSITION
        # ======================================================================
        # Start in the center of the bottom hallway strip
        # This gives the robot roughly equal access to all rooms
        # Position (row=4, col=3) is in the middle of the hallway
        self.robot_row = 4
        self.robot_col = 3
        
        # ======================================================================
        # INITIALIZE DIRT MAP - ALL TILES START DIRTY
        # ======================================================================
        # Create a fresh dirt map where all cleanable tiles are dirty
        # This represents a house that needs complete cleaning
        self.dirt_map = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
        for row, col in self.cleanable_tiles:
            self.dirt_map[row][col] = DIRTY
        
        # Reset movement history (4 = no previous direction)
        self.last_direction = 4
        
        # Get initial observation (state)
        observation = self._get_state()
        
        # Build info dictionary with useful debugging information
        info = {
            "robot_position": (self.robot_row, self.robot_col),
            "dirty_tiles": self._count_dirty_tiles(),
            "tiles_cleaned": 0,
            "room": self.room_names.get(
                self.room_layout[self.robot_row][self.robot_col], "Unknown"
            )
        }
        
        return observation, info
    
    def step(self, action):
        """
        Execute one action in the environment.
        
        This is the core method that processes robot actions and returns
        the resulting state, reward, and termination status. This is called
        once per timestep during training/testing.
        
        The Q-Learning update equation uses the reward and next state from here:
            Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Parameters:
        -----------
        action : int
            Action to take (0-5, see ACTION_* constants)
            0: Forward, 1: Backward, 2: Left, 3: Right, 4: Wait, 5: Clean
        
        Returns:
        --------
        tuple (observation, reward, terminated, truncated, info)
            observation : int
                New state after action
            reward : float
                Reward received for this action
            terminated : bool
                True if episode ended naturally (all tiles clean)
            truncated : bool
                True if episode ended by step limit
            info : dict
                Additional information dictionary
        """
        # Increment step counter
        self.steps_taken += 1
        
        # Initialize reward with small step penalty
        # This encourages the robot to be efficient (fewer steps = better)
        reward = REWARD_STEP_PENALTY
        
        # Track action outcome for info dictionary
        action_result = "unknown"
        
        # ======================================================================
        # PROCESS ACTION - Execute the robot's chosen action
        # ======================================================================
        
        if action == ACTION_FORWARD:
            # Move up (decrease row)
            new_row = self.robot_row - 1
            new_col = self.robot_col
            if self._is_valid_position(new_row, new_col):
                self.robot_row = new_row
                action_result = "moved_forward"
                self.last_direction = 1
                # Dirt-conditional reward for stepping on the tile
                if self.dirt_map[new_row][new_col] == DIRTY:
                    room_type = self.room_layout[new_row][new_col]
                    reward += self.room_rewards.get(room_type, 20)
                    self.dirt_map[new_row][new_col] = CLEAN
                    self.tiles_cleaned += 1
                else:
                    reward += REWARD_STEP_ON_CLEAN
            else:
                reward += REWARD_HIT_WALL
                action_result = "hit_wall"
        
        elif action == ACTION_BACKWARD:
            # Move down (increase row)
            new_row = self.robot_row + 1
            new_col = self.robot_col
            if self._is_valid_position(new_row, new_col):
                self.robot_row = new_row
                action_result = "moved_backward"
                self.last_direction = 0
                if self.dirt_map[new_row][new_col] == DIRTY:
                    room_type = self.room_layout[new_row][new_col]
                    reward += self.room_rewards.get(room_type, 20)
                    self.dirt_map[new_row][new_col] = CLEAN
                    self.tiles_cleaned += 1
                else:
                    reward += REWARD_STEP_ON_CLEAN
            else:
                reward += REWARD_HIT_WALL
                action_result = "hit_wall"
        
        elif action == ACTION_LEFT:
            # Move left (decrease column)
            new_row = self.robot_row
            new_col = self.robot_col - 1
            if self._is_valid_position(new_row, new_col):
                self.robot_col = new_col
                action_result = "moved_left"
                self.last_direction = 2
                if self.dirt_map[new_row][new_col] == DIRTY:
                    room_type = self.room_layout[new_row][new_col]
                    reward += self.room_rewards.get(room_type, 20)
                    self.dirt_map[new_row][new_col] = CLEAN
                    self.tiles_cleaned += 1
                else:
                    reward += REWARD_STEP_ON_CLEAN
            else:
                reward += REWARD_HIT_WALL
                action_result = "hit_wall"
        
        elif action == ACTION_RIGHT:
            # Move right (increase column)
            new_row = self.robot_row
            new_col = self.robot_col + 1
            if self._is_valid_position(new_row, new_col):
                self.robot_col = new_col
                action_result = "moved_right"
                self.last_direction = 3
                if self.dirt_map[new_row][new_col] == DIRTY:
                    room_type = self.room_layout[new_row][new_col]
                    reward += self.room_rewards.get(room_type, 20)
                    self.dirt_map[new_row][new_col] = CLEAN
                    self.tiles_cleaned += 1
                else:
                    reward += REWARD_STEP_ON_CLEAN
            else:
                reward += REWARD_HIT_WALL
                action_result = "hit_wall"
        
        elif action == ACTION_WAIT:
            # Stay in place - penalize because it wastes time
            reward += REWARD_WAIT
            action_result = "waited"
        
        elif action == ACTION_CLEAN:
            # Try to clean the current tile
            if self.dirt_map[self.robot_row][self.robot_col] == DIRTY:
                # Tile is dirty - successful clean!
                self.dirt_map[self.robot_row][self.robot_col] = CLEAN
                self.tiles_cleaned += 1
                
                # Get room-specific reward
                room_type = self.room_layout[self.robot_row][self.robot_col]
                cleaning_reward = self.room_rewards.get(room_type, 20)
                reward += cleaning_reward
                action_result = f"cleaned_{self.room_names.get(room_type, 'tile')}"
            else:
                # Tile already clean - penalize wasted action
                reward += REWARD_CLEAN_ALREADY_CLEAN
                action_result = "clean_failed_already_clean"
        
        # ======================================================================
        # CHECK TERMINATION CONDITIONS
        # ======================================================================
        
        # Count remaining dirty tiles
        dirty_remaining = self._count_dirty_tiles()
        
        # Episode terminates successfully if all tiles are clean
        terminated = False
        if dirty_remaining == 0:
            # Big bonus for completing the task!
            reward += REWARD_ALL_CLEAN_BONUS
            terminated = True
        
        # Episode is truncated if max steps reached (time limit)
        truncated = (self.steps_taken >= self.max_steps)
        
        # ======================================================================
        # GET NEW OBSERVATION
        # ======================================================================
        observation = self._get_state()
        
        # ======================================================================
        # BUILD INFO DICTIONARY
        # ======================================================================
        # This provides useful information for debugging and analysis
        
        # Calculate completion rate as percentage
        completion_rate = (self.tiles_cleaned / self.num_cleanable * 100) if self.num_cleanable > 0 else 0.0
        
        info = {
            "robot_position": (self.robot_row, self.robot_col),
            "action": ACTION_NAMES.get(action, "Unknown"),
            "action_result": action_result,
            "dirty_tiles": dirty_remaining,
            "tiles_cleaned": self.tiles_cleaned,
            "total_dirty": self.num_cleanable,
            "completion_rate": completion_rate,
            "steps": self.steps_taken,
            "room": self.room_names.get(
                self.room_layout[self.robot_row][self.robot_col], "Unknown"
            )
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the current state of the environment using Pygame.
        
        This method creates a visual representation of the house showing:
        - Room layout with different colors per room type
        - Dirty tiles (brown dirt marks)
        - Cleaned tiles (green checkmark)
        - Robot position (green circle with eyes)
        - Statistics overlay (steps, cleaned count, etc.)
        
        Returns:
        --------
        numpy.ndarray or None
            RGB array if render_mode is "rgb_array", None otherwise
        """
        # If render_mode is None, skip rendering
        if self.render_mode is None:
            return None
        
        # ======================================================================
        # INITIALIZE PYGAME IF NEEDED
        # ======================================================================
        # First time render() is called, we need to set up Pygame
        if self.window is None:
            pygame.init()
            pygame.display.init()
            
            # Calculate window size (grid + space for stats panel)
            window_width = GRID_WIDTH * self.cell_size
            window_height = GRID_HEIGHT * self.cell_size + 80  # Extra for stats
            
            if self.render_mode == "human":
                # Create visible window for human viewing
                pygame.display.set_caption("Cleaning Robot - Pure Q-Learning")
                self.window = pygame.display.set_mode((window_width, window_height))
            else:
                # Create off-screen surface for rgb_array mode
                self.window = pygame.Surface((window_width, window_height))
            
            # Create clock for frame rate control
            self.clock = pygame.time.Clock()
            
            # Initialize fonts for text rendering
            self.font = pygame.font.Font(None, 28)
            self.small_font = pygame.font.Font(None, 22)
        
        # ======================================================================
        # DRAW BACKGROUND AND GRID
        # ======================================================================
        self.window.fill(COLOR_WALL)  # Fill with wall color as background
        
        # Map room types to colors
        room_colors = {
            EMPTY: COLOR_WALL,
            KITCHEN: COLOR_KITCHEN,
            LIVING_ROOM: COLOR_LIVING_ROOM,
            HALLWAY: COLOR_HALLWAY
        }
        
        # Draw each cell in the grid
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                # Calculate cell rectangle position
                x = col * self.cell_size
                y = row * self.cell_size
                cell_rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # Get room type and corresponding color
                room_type = self.room_layout[row][col]
                base_color = room_colors.get(room_type, COLOR_WALL)
                
                # Draw cell background
                pygame.draw.rect(self.window, base_color, cell_rect)
                
                # Draw dirt or clean markers for room tiles (not walls)
                if room_type != EMPTY and self.dirt_map is not None:
                    if self.dirt_map[row][col] == DIRTY:
                        # Draw brown dirt spots to indicate dirty tile
                        center_x = x + self.cell_size // 2
                        center_y = y + self.cell_size // 2
                        # Draw multiple dirt spots in a pattern
                        dirt_positions = [(-15, -10), (10, -5), (-5, 15), (15, 10), (0, 0)]
                        for dx, dy in dirt_positions:
                            pygame.draw.circle(
                                self.window, COLOR_DIRTY,
                                (center_x + dx, center_y + dy), 8
                            )
                    else:
                        # Draw green checkmark to indicate cleaned tile
                        center_x = x + self.cell_size // 2
                        center_y = y + self.cell_size // 2
                        # Draw checkmark circle
                        pygame.draw.circle(
                            self.window, COLOR_CLEAN_MARKER,
                            (center_x, center_y), 15
                        )
                        # Draw check symbol inside
                        pygame.draw.lines(
                            self.window, (255, 255, 255), False,
                            [
                                (center_x - 8, center_y),
                                (center_x - 2, center_y + 8),
                                (center_x + 10, center_y - 8)
                            ], 3
                        )
                
                # Draw grid lines
                pygame.draw.rect(self.window, COLOR_GRID_LINE, cell_rect, 2)
        
        # ======================================================================
        # DRAW ROBOT
        # ======================================================================
        # Calculate robot center position
        robot_x = self.robot_col * self.cell_size + self.cell_size // 2
        robot_y = self.robot_row * self.cell_size + self.cell_size // 2
        robot_radius = self.cell_size // 3
        
        # Draw robot body (green circle)
        pygame.draw.circle(self.window, COLOR_ROBOT, (robot_x, robot_y), robot_radius)
        # Draw robot outline
        pygame.draw.circle(self.window, (30, 100, 30), (robot_x, robot_y), robot_radius, 3)
        
        # Draw robot eyes (to give it character and show direction)
        eye_size = 8
        eye_offset_x = robot_radius // 3
        eye_offset_y = robot_radius // 4
        # White part of eyes
        pygame.draw.circle(self.window, (255, 255, 255),
                          (robot_x - eye_offset_x, robot_y - eye_offset_y), eye_size)
        pygame.draw.circle(self.window, (255, 255, 255),
                          (robot_x + eye_offset_x, robot_y - eye_offset_y), eye_size)
        # Black pupils
        pygame.draw.circle(self.window, COLOR_BLACK,
                          (robot_x - eye_offset_x, robot_y - eye_offset_y), 4)
        pygame.draw.circle(self.window, COLOR_BLACK,
                          (robot_x + eye_offset_x, robot_y - eye_offset_y), 4)
        
        # ======================================================================
        # DRAW STATISTICS PANEL
        # ======================================================================
        stats_y = GRID_HEIGHT * self.cell_size + 10
        
        # Calculate current statistics
        dirty_count = self._count_dirty_tiles() if self.dirt_map is not None else 0
        completion_pct = ((self.num_cleanable - dirty_count) / self.num_cleanable) * 100
        current_room = self.room_names.get(
            self.room_layout[self.robot_row][self.robot_col], "Unknown"
        )
        
        # Draw main stats line
        stats_text = (f"Room: {current_room} | "
                     f"Steps: {self.steps_taken}/{self.max_steps} | "
                     f"Cleaned: {self.tiles_cleaned}/{self.num_cleanable} | "
                     f"Progress: {completion_pct:.0f}%")
        text_surface = self.font.render(stats_text, True, COLOR_TEXT)
        self.window.blit(text_surface, (10, stats_y))
        
        # Draw legend line
        legend_y = stats_y + 30
        legend_text = "Kitchen (+50) | Living (+35) | Hallway (+20)"
        legend_surface = self.small_font.render(legend_text, True, COLOR_TEXT)
        self.window.blit(legend_surface, (10, legend_y))
        
        # Draw color legend boxes
        legend_items = [
            (450, "Kitchen", COLOR_KITCHEN),
            (530, "Living", COLOR_LIVING_ROOM),
            (600, "Hall", COLOR_HALLWAY),
        ]
        for x_pos, name, color in legend_items:
            pygame.draw.rect(self.window, color, (x_pos, legend_y, 20, 20))
            pygame.draw.rect(self.window, COLOR_BLACK, (x_pos, legend_y, 20, 20), 1)
        
        # ======================================================================
        # UPDATE DISPLAY
        # ======================================================================
        if self.render_mode == "human":
            # Process Pygame events (required to keep window responsive)
            pygame.event.pump()
            # Update the display
            pygame.display.flip()
            # Control frame rate
            self.clock.tick(self.metadata["render_fps"])
        
        # Return RGB array if requested (for video recording)
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)),
                axes=(1, 0, 2)
            )
        
        return None
    
    def close(self):
        """
        Clean up Pygame resources when environment is closed.
        
        This method should be called when you're done using the environment
        to properly release Pygame resources and close the window.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def get_action_name(self, action):
        """
        Get human-readable name for an action.
        
        Parameters:
        -----------
        action : int
            Action number (0-5)
        
        Returns:
        --------
        str
            Human-readable action name (e.g., "Forward", "Clean")
        """
        return ACTION_NAMES.get(action, "Unknown")
    
    def get_room_name(self, row=None, col=None):
        """
        Get the room name at a specific position or current position.
        
        Parameters:
        -----------
        row : int, optional
            Row coordinate (uses robot position if None)
        col : int, optional
            Column coordinate (uses robot position if None)
        
        Returns:
        --------
        str
            Room name at the given position
        """
        if row is None:
            row = self.robot_row
        if col is None:
            col = self.robot_col
        room_type = self.room_layout[row][col]
        return self.room_names.get(room_type, "Unknown")


# Module test

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  TESTING CLEANING ROBOT ENVIRONMENT")
    print("=" * 65)
    
    # Create environment with human rendering
    print("\n1. Creating environment with render_mode='human'...")
    env = CleaningEnv(render_mode="human")
    
    # Test reset
    print("\n2. Testing reset()...")
    obs, info = env.reset(seed=42)
    print(f"   Initial observation: {obs}")
    print(f"   Initial info: {info}")
    
    # Test render
    print("\n3. Rendering initial state...")
    env.render()
    
    # Test each action
    print("\n4. Testing each action:")
    for action in range(6):
        obs, reward, term, trunc, info = env.step(action)
        print(f"   Action {action} ({env.get_action_name(action)}): "
              f"reward={reward:+.1f}, state={obs}, result={info['action_result']}")
        env.render()
    
    # Test random episode
    print("\n5. Running 50 random steps...")
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        
        if terminated or truncated:
            print(f"   Episode ended at step {step + 1}")
            break
    
    print(f"   Total reward: {total_reward:.1f}")
    print(f"   Tiles cleaned: {info['tiles_cleaned']}")
    
    # Wait before closing
    import time
    print("\n6. Waiting 2 seconds before closing...")
    time.sleep(2)
    
    # Clean up
    env.close()
    print("\nTest complete! Environment working correctly.")
