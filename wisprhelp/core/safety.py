"""Safety interlocks and emergency kill-switch supervision.

Enforces calibrated destructive risk confirmation gates and physical kill-switch hooks.
"""

from __future__ import annotations

import logging
import threading
import re
from typing import Callable, Optional

from wisprhelp.classification.schemas import JevDecision
from wisprhelp.config import config

logger = logging.getLogger(__name__)


def is_explicit_confirmation(response: str) -> bool:
    """Accept a complete consent phrase; text containing 'yes' is not consent."""
    clean = re.sub(r"[.!?,]+$", "", response.casefold().strip()).strip()
    return clean in {
        "yes", "yes please", "yes i confirm", "yes, i confirm", "confirm", "i confirm",
        "proceed", "go ahead", "do it", "sure", "ok", "okay", "haan", "haan ji", "ha",
        "yes, proceed", "yes proceed", "yes, we do", "yes we do",
    }


class EmergencyKillSwitchTriggered(Exception):
    """Raised when user hits Esc or flings mouse to abort agent actions."""
    pass


class SafetyGate:
    """Evaluates destructive risk and manages confirmation gates."""

    def __init__(self, risk_threshold: Optional[float] = None) -> None:
        self.risk_threshold = config.destructive_risk_threshold if risk_threshold is None else risk_threshold
        self._pending_action: Optional[JevDecision] = None
        self._lock = threading.Lock()

    def check_risk(self, decision: JevDecision) -> bool:
        """Returns True if the decision requires explicit user confirmation before execution."""
        return decision.is_high_risk(self.risk_threshold)

    def hold_for_confirmation(self, decision: JevDecision) -> str:
        """Puts a risky action on hold and returns a confirmation prompt for the user."""
        with self._lock:
            self._pending_action = decision

        prompt = (
            f"High-risk action detected (risk probability: {decision.destructive_probability:.2f}). "
            f"This command may delete files, terminate processes, or modify critical data. "
            f"Are you sure you want to proceed?"
        )
        logger.warning("Safety gate engaged for command. Prompt: %s", prompt)
        return prompt

    def resolve_confirmation(self, user_confirmation: str) -> Optional[JevDecision]:
        """Resolves a pending high-risk action based on user's verbal/text confirmation."""
        with self._lock:
            if not self._pending_action:
                return None

            if is_explicit_confirmation(user_confirmation):
                action = self._pending_action
                self._pending_action = None
                logger.info("User confirmed high-risk action: %s", action.intent)
                return action

            # User cancelled or denied
            logger.info("User declined high-risk action with response: '%s'", user_confirmation)
            self._pending_action = None
            return None

    def cancel_pending(self) -> None:
        """Cancels any pending confirmation."""
        with self._lock:
            self._pending_action = None
