"""Compatibility entrypoint for the project environment.

Requested structure includes environment.py, so this file re-exports the
Phase-2 environment implementation.
"""

from env.phase2_cleaning_env import (
    Phase2CleaningEnv,
    PHASE2_CONFIG,
    BATTERY_BINS,
    ACTION_MOVE_FORWARD,
    ACTION_ROTATE_LEFT,
    ACTION_ROTATE_RIGHT,
    ACTION_CHARGE,
    ORIENT_NORTH,
    ORIENT_EAST,
    ORIENT_SOUTH,
    ORIENT_WEST,
)

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
