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
from .rag_tools import list_paper_library, search_paper_library
from .visual_tools import render_diagram, render_equation

TOOLS = [
    get_current_datetime,
    search_arxiv,
    browse_arxiv,
    get_arxiv_paper,
    # Library first, API second: search_paper_library answers from papers
    # already read, and get_arxiv_fulltext both answers and fills the library.
    search_paper_library,
    list_paper_library,
    get_arxiv_fulltext,
    # Presentation, not retrieval: these show what the others found.
    render_diagram,
    render_equation,
]

__all__ = [
    "TOOLS",
    "browse_arxiv",
    "get_arxiv_fulltext",
    "get_arxiv_paper",
    "get_current_datetime",
    "list_paper_library",
    "render_diagram",
    "render_equation",
    "search_arxiv",
    "search_paper_library",
]
