"""`reserchia-ui` -- launch the Chainlit front end.

Chainlit owns its own server and CLI, so this only locates the app file and
hands over. Its real job is to fail with a useful message when the optional
`ui` dependency group is not installed, rather than a bare ImportError.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent.parent / "ui" / "app.py"


def main() -> int:
    try:
        from chainlit.cli import cli  # noqa: F401
    except ImportError:
        print(
            "chainlit is not installed. It is an optional dependency:\n"
            "    uv sync --group ui",
            file=sys.stderr,
        )
        return 1

    if not APP.exists():
        print(f"cannot find the Chainlit app at {APP}", file=sys.stderr)
        return 1

    # Chainlit's CLI is a click group; invoking `run` through argv keeps every
    # flag it supports (--port, --host, -w) working without re-declaring them.
    sys.argv = ["chainlit", "run", str(APP), *sys.argv[1:]]
    from chainlit.cli import cli as chainlit_cli

    chainlit_cli()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
