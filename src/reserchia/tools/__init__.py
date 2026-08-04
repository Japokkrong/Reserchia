"""Tool registry.

Add new tools here -- `TOOLS` is the single list the agent binds against.
"""

from .datetime_tools import get_current_datetime

TOOLS = [get_current_datetime]

__all__ = ["TOOLS", "get_current_datetime"]
