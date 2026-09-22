"""Configuration management for WisprHelp.

Reads environment variables from .env and exposes strongly-typed configuration
with safe fallbacks and validation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Automatically locate and load .env from project root or workspace
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
else:
    load_dotenv(override=False)


@dataclass(frozen=True)
class Config:
    """Immutable application configuration."""

    # TypeSafe AI Jev (System One)
    jev_api_key: str = field(
        default_factory=lambda: (
            os.getenv("JEV_API_KEY")
            or os.getenv("TYPESAFE_API_KEY")
            or ""
        ).strip()
    )
    jev_api_url: str = field(
        default_factory=lambda: os.getenv(
            "JEV_API_URL", "https://api.typesafe.ai/v1/systemone"
        ).strip()
    )
    jev_model: str = field(
        default_factory=lambda: os.getenv("JEV_MODEL", "jev-latest").strip()
    )

    # Confidence and Safety Thresholds
    intent_confidence_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.60")
        )
    )
    destructive_risk_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("DESTRUCTIVE_RISK_THRESHOLD", "0.65")
        )
    )

    # Frontier Vision Agent (System Two)
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    gemini_api_key: str = field(
        default_factory=lambda: (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        ).strip()
    )

    # Speech-to-Text (Ear)
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip()
    )

    # Execution limits & Guardrails
    max_agent_steps: int = field(
        default_factory=lambda: int(os.getenv("MAX_AGENT_STEPS", "15"))
    )
    kill_switch_enabled: bool = field(
        default_factory=lambda: os.getenv("KILL_SWITCH_ENABLED", "true").lower()
        in ("1", "true", "yes")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    def validate_jev_key(self) -> None:
        """Validates that a Jev API key is configured."""
        if not self.jev_api_key:
            raise ValueError(
                "Missing TypeSafe JEV API key. Please configure JEV_API_KEY or "
                "TYPESAFE_API_KEY in your .env file."
            )


# Default singleton instance
config = Config()
