"""Showing a diagram or an equation rather than describing it.

Both tools validate, register the visual for the front end to display, and
return a confirmation. The confirmation matters as much as the rendering: it
tells the model the visual is *already shown*, because otherwise it narrates
the diagram it just drew and the answer doubles in length for no gain.

On failure they return the renderer's own error text and the docstrings tell
the model to fix and retry. That only works because the diagram is rendered
eagerly -- a lazy render would surface the syntax error after the answer was
already written, too late to act on.
"""

from __future__ import annotations

from langchain_core.tools import tool

from ..visuals import Diagram, Equation, add_diagram, add_equation, check_latex
from ..visuals import render_mermaid


@tool
def render_diagram(mermaid: str, caption: str = "") -> str:
    """Draw a diagram to show structure the reader would otherwise have to assemble.

    Reach for this when the answer has shape worth seeing: a pipeline, a model
    architecture, a multi-stage training recipe, how components connect, a
    decision flow. A drawn structure is grasped far faster than the same thing
    unrolled into a paragraph.

    Do not draw one for a fact, a definition, or two related items -- an
    unnecessary diagram is noise. Do not describe the diagram in prose
    afterwards; the reader can see it.

    Args:
        mermaid: The diagram source, mermaid syntax only, with no ``` fence.
            The first line must name the type, for example 'flowchart TD',
            'sequenceDiagram', 'classDiagram', 'stateDiagram-v2', 'timeline'.
            Keep node labels short. If this returns a syntax error, fix the
            source and call again once.
        caption: A short line naming what is drawn, e.g. 'GLM-OCR inference
            pipeline'. Shown beneath the diagram.
    """
    result = render_mermaid(mermaid)
    if isinstance(result, str):
        return result  # already a readable error, from mermaid's own parser

    png, path = result
    add_diagram(Diagram(source=mermaid.strip(), caption=caption.strip(), png=png, path=path))
    label = f" ({caption.strip()})" if caption.strip() else ""
    return (
        f"Diagram rendered and shown to the user{label}. "
        "Do not repeat its contents in prose -- refer to it instead, e.g. "
        "'as the diagram shows'."
    )


@tool
def render_equation(latex: str, caption: str = "") -> str:
    """Set a display equation when the formula itself is the point of the answer.

    Use this for an equation a paper defines and the answer turns on -- an
    attention formula, a loss, a scoring rule. It is shown as its own numbered,
    captioned block rather than buried mid-sentence.

    For ordinary maths inside a sentence, do not call this: just write it inline
    as $x_i$ and it will be typeset.

    Args:
        latex: The LaTeX body only, with no surrounding $ or $$ delimiters and
            no ``` fence. For example:
            '\\text{Attention}(Q,K,V) = \\text{softmax}(QK^T/\\sqrt{d_k})V'.
            If this returns an error, fix it and call again once.
        caption: A short line naming the equation, e.g. 'Scaled dot-product
            attention'. Shown with the equation number.
    """
    problem = check_latex(latex)
    if problem:
        return f"Equation not shown: {problem}. Fix the LaTeX and call again."

    number = add_equation(Equation(latex=latex.strip(), caption=caption.strip()))
    return (
        f"Equation ({number}) rendered and shown to the user. "
        "Refer to it as equation "
        f"({number}) rather than writing the formula out again."
    )
