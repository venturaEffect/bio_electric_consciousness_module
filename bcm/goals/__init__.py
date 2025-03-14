"""
Components for primitive goal-directed behavior in bioelectric systems.
"""
from bcm.goals.homeostasis import HomeostasisRegulator
from bcm.goals.goal_states import GoalState

__all__ = ['HomeostasisRegulator', 'GoalState']