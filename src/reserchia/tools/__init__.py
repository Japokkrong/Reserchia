"""Tool registry.

Add new tools here -- `TOOLS` is the single list the agent binds against.
"""

from .arxiv_tools import (
    browse_arxiv,
    get_arxiv_fulltext,
    get_arxiv_paper,
    search_arxiv,
)
from .datetime_tools import get_current_datetime

TOOLS = [
    get_current_datetime,
    search_arxiv,
    browse_arxiv,
    get_arxiv_paper,
    get_arxiv_fulltext,
]

__all__ = [
    "TOOLS",
    "browse_arxiv",
    "get_arxiv_fulltext",
    "get_arxiv_paper",
    "get_current_datetime",
    "search_arxiv",
]
