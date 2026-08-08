"""Diagrams and equations the agent asks to show.

Same shape as `rag/citations.py`: a tool registers something, and whichever
front end is attached decides how to display it. The terminal prints source and
a file path; Chainlit shows an image and typeset maths.

**Rendering is the validation.** Mermaid's grammar is large and its parser is
JavaScript, so a hand-written Python check would be a heuristic that still
waves through diagrams mermaid rejects. Kroki runs the real parser and returns
its error text, which goes back to the model in time for it to fix and retry:

    HTTP 400  SyntaxError: No diagram type detected matching given
              configuration for text: grph TD

The diagram source is sent to kroki.io. It is model-authored text about public
papers rather than anything of the user's, but it is an outbound call and is
documented as one in the README.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import get_settings
from .observability import track

KROKI_URL = "https://kroki.io/mermaid/png"
TIMEOUT = 45.0

#: Every diagram type mermaid currently accepts as an opening keyword. Used only
#: for a pre-check, to fail obvious garbage without a 2-second round trip --
#: anything that passes here is still validated properly by Kroki.
DIAGRAM_TYPES = (
    "graph", "flowchart", "sequencediagram", "classdiagram", "statediagram",
    "erdiagram", "journey", "gantt", "pie", "quadrantchart", "requirementdiagram",
    "gitgraph", "mindmap", "timeline", "zenuml", "sankey-beta", "xychart-beta",
    "block-beta", "packet-beta", "kanban", "architecture-beta", "radar-beta",
    "treemap-beta", "c4context", "c4container", "c4component", "c4dynamic",
)

_BEGIN = re.compile(r"\\begin\{([^}]*)\}")
_END = re.compile(r"\\end\{([^}]*)\}")


@dataclass
class Diagram:
    source: str
    caption: str
    png: bytes = b""
    path: Path | None = None


@dataclass
class Equation:
    latex: str
    caption: str
    number: int = 0


@dataclass
class Turn:
    diagrams: list[Diagram] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)

    def empty(self) -> bool:
        return not self.diagrams and not self.equations


_lock = threading.Lock()
_turn = Turn()


def start_turn() -> None:
    """Drop anything registered by the previous turn."""
    global _turn
    with _lock:
        _turn = Turn()


def take() -> Turn:
    """Hand the current turn's visuals to the UI and reset."""
    global _turn
    with _lock:
        current, _turn = _turn, Turn()
        return current


def add_diagram(diagram: Diagram) -> None:
    with _lock:
        _turn.diagrams.append(diagram)


def add_equation(equation: Equation) -> int:
    with _lock:
        equation.number = len(_turn.equations) + 1
        _turn.equations.append(equation)
        return equation.number


# -- mermaid ------------------------------------------------------------------


def _cache_dir() -> Path:
    path = get_settings().store_dir / "diagrams"
    path.mkdir(parents=True, exist_ok=True)
    return path


def precheck(source: str) -> str | None:
    """Catch obvious nonsense locally. Returns an error, or None to proceed."""
    text = (source or "").strip()
    if not text:
        return "the diagram is empty"

    # Strip a markdown fence if the model wrapped the source in one.
    first = text.splitlines()[0].strip().lower()
    if first.startswith("```"):
        return "send the mermaid source only, without the ``` fence"

    # Prefix match, not equality: real types carry suffixes the list cannot
    # enumerate -- stateDiagram-v2, gitGraph:, and whatever mermaid adds next.
    # This check exists only to skip a pointless 2-second round trip, so it must
    # err towards letting things through; Kroki does the real validation.
    opener = re.split(r"[\s:;{(]", first, maxsplit=1)[0]
    if not any(opener.startswith(kind) for kind in DIAGRAM_TYPES):
        return (
            f"{first[:40]!r} does not start a mermaid diagram. The first line must "
            f"name the type, e.g. 'flowchart TD', 'sequenceDiagram', 'classDiagram'."
        )
    return None


def render_mermaid(source: str) -> tuple[bytes, Path] | str:
    """Render to PNG via Kroki. Returns (png, cache path) or a readable error."""
    problem = precheck(source)
    if problem:
        return f"Diagram not rendered: {problem}."

    text = source.strip()
    digest = hashlib.sha256(text.encode()).hexdigest()
    cached = _cache_dir() / f"{digest}.png"
    if cached.exists():
        with track("kroki.render", "tool", cache="hit"):
            return cached.read_bytes(), cached

    try:
        with track("kroki.render", "tool", cache="miss") as span:
            response = httpx.post(KROKI_URL, content=text.encode(), timeout=TIMEOUT)
            span.set(status=response.status_code, bytes=len(response.content))
    except httpx.RequestError as exc:
        return (
            f"Diagram not rendered: could not reach the renderer "
            f"({type(exc).__name__}). Describe the structure in words instead."
        )

    if response.status_code != 200:
        # Kroki hands back mermaid's own parser message; passing it through
        # verbatim is what lets the model correct the diagram.
        detail = response.text.strip().splitlines()
        message = " ".join(line for line in detail[:3] if line.strip())
        return f"Diagram not rendered, mermaid rejected it: {message[:400]}"

    cached.write_bytes(response.content)
    return response.content, cached


# -- latex --------------------------------------------------------------------


def check_latex(source: str) -> str | None:
    """Shallow structural check on an equation. Returns an error, or None.

    Deliberately shallow: it catches truncation and unbalanced groups, which is
    what a model actually gets wrong, and cannot catch an undefined macro.
    KaTeX shows those in red at render time, which is a good enough backstop.
    """
    text = (source or "").strip()
    if not text:
        return "the equation is empty"
    if text.startswith("$") or text.endswith("$"):
        return "send the LaTeX only, without the surrounding $ or $$ delimiters"

    depth = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return "a closing brace '}' has no matching '{'"
    if depth:
        return f"{depth} brace(s) left unclosed"

    starts = _BEGIN.findall(text)
    ends = _END.findall(text)
    if sorted(starts) != sorted(ends):
        missing = set(starts) ^ set(ends)
        return f"\\begin and \\end do not match for: {', '.join(sorted(missing))}"
    return None
