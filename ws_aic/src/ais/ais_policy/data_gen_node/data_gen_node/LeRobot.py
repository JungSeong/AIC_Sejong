"""LeRobot episode collection policy.

This module exposes the basic episode collector through the
``data_gen_node.LeRobot`` import path.
"""

from .DataCollect import DataCollect


class LeRobot(DataCollect):
    """Collect basic LeRobot-format insertion episodes."""
