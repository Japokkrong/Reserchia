"""Current date/time lookup."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

_EXAMPLE_ZONES = ("UTC", "Asia/Tokyo", "America/New_York", "Europe/London")


@tool
def get_current_datetime(timezone: str | None = None) -> str:
    """Get the current date, day of week, month, year, and time of day.

    Use this for any question about the present moment -- today's date, the
    current month or year, what day of the week it is, or the current time.

    Args:
        timezone: Optional IANA timezone name, e.g. 'Asia/Tokyo', 'UTC',
            'America/New_York'. Omit it to use the system's local timezone.
    """
    if timezone:
        try:
            now = datetime.now(ZoneInfo(timezone))
        except (ZoneInfoNotFoundError, ValueError):
            examples = ", ".join(_EXAMPLE_ZONES)
            return (
                f"Error: {timezone!r} is not a valid IANA timezone name. "
                f"Try one of: {examples}. "
                "Retry this tool with a corrected timezone, or omit the "
                "argument to use local time."
            )
    else:
        now = datetime.now().astimezone()

    offset = now.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if offset else "unknown"

    return "\n".join(
        [
            f"ISO 8601: {now.isoformat()}",
            f"Date: {now.strftime('%Y-%m-%d')}",
            f"Day of week: {now.strftime('%A')}",
            f"Month: {now.strftime('%B')} ({now.month})",
            f"Year: {now.year}",
            f"Time (24h): {now.strftime('%H:%M:%S')}",
            f"Time (12h): {now.strftime('%I:%M:%S %p').lstrip('0')}",
            f"Timezone: {now.tzname()} (UTC{offset})",
            f"Unix timestamp: {int(now.timestamp())}",
        ]
    )
