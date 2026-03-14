"""Compatibility agent package requested by the project structure spec."""

from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent

try:
    from agents.dqn_agent import DQNAgent, QNetwork
except ImportError:
    pass

__all__ = ["QLearningAgent", "SarsaAgent", "DQNAgent", "QNetwork"]
