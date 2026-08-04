"""Environment-backed settings for the Reserchia agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

VALID_REASONING = ("enabled", "disabled")
# OpenRouter's effort ladder, roughly % of the token budget spent reasoning.
VALID_EFFORT = ("minimal", "low", "medium", "high", "xhigh", "max")


class ConfigError(RuntimeError):
    """Raised when the environment is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    reasoning: str
    reasoning_effort: str
    temperature: float
    site_url: str = ""
    app_name: str = ""

    @property
    def reasoning_enabled(self) -> bool:
        return self.reasoning == "enabled"

    @property
    def reasoning_body(self) -> dict[str, object]:
        """The `reasoning` block OpenRouter expects in the request body.

        Sent explicitly in both directions: DeepSeek V4 reasons by default, so
        omitting this would silently leave thinking on.
        """
        if not self.reasoning_enabled:
            return {"reasoning": {"enabled": False}}
        return {"reasoning": {"enabled": True, "effort": self.reasoning_effort}}

    @property
    def headers(self) -> dict[str, str]:
        """Optional attribution headers for OpenRouter's leaderboards."""
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers


def _choice(name: str, default: str, valid: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in valid:
        raise ConfigError(
            f"{name} must be one of {', '.join(valid)} (got {value!r})."
        )
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "OPENROUTER_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key from "
            "https://openrouter.ai/keys"
        )

    raw_temperature = os.getenv("OPENROUTER_TEMPERATURE", "0").strip()
    try:
        temperature = float(raw_temperature)
    except ValueError as exc:
        raise ConfigError(
            f"OPENROUTER_TEMPERATURE must be a number (got {raw_temperature!r})."
        ) from exc

    return Settings(
        api_key=api_key,
        base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).strip(),
        model=os.getenv(
            "OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731"
        ).strip(),
        reasoning=_choice("OPENROUTER_REASONING", "disabled", VALID_REASONING),
        reasoning_effort=_choice(
            "OPENROUTER_REASONING_EFFORT", "low", VALID_EFFORT
        ),
        temperature=temperature,
        site_url=os.getenv("OPENROUTER_SITE_URL", "").strip(),
        app_name=os.getenv("OPENROUTER_APP_NAME", "").strip(),
    )
