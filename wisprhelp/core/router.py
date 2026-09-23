"""Confidence-gated command router.

Evaluates TypeSafe Jev decision matrix against calibrated confidence and risk
thresholds to select the optimal execution path.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Tuple

from wisprhelp.classification.schemas import IntentType, JevDecision
from wisprhelp.config import config

logger = logging.getLogger(__name__)


class RouteTarget(str, Enum):
    """The execution pathway selected by the router."""
    CLARIFY = "clarify"
    SAFETY_INTERLOCK = "safety_interlock"
    PRIORITY_1_LAUNCH = "priority_1_launch"
    PRIORITY_1_MEDIA = "priority_1_media"
    TEXT_INJECTION = "text_injection"
    GUI_WORKFLOW_AUTOMATION = "gui_workflow_automation"
    CONVERSATIONAL_QUERY = "conversational_query"


class CommandRouter:
    """Routes calibrated Jev decisions to appropriate execution handlers."""

    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        destructive_threshold: Optional[float] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold or config.intent_confidence_threshold
        self.destructive_threshold = destructive_threshold or config.destructive_risk_threshold

    def route(self, decision: JevDecision) -> Tuple[RouteTarget, str]:
        """Evaluates Jev decision and returns the optimal execution route and rationale.

        Args:
            decision: JevDecision returned by the TypeSafe Jev classifier.

        Returns:
            Tuple of (RouteTarget, explanation_string).
        """
        # 1. Destructive Risk Interlock (Safety takes absolute precedence)
        if decision.is_high_risk(self.destructive_threshold):
            reason = (
                f"Destructive risk score ({decision.destructive_probability:.2f} >= {self.destructive_threshold:.2f}). "
                f"Safety interlock engaged."
            )
            logger.warning("Routing decision to SAFETY_INTERLOCK: %s", reason)
            return RouteTarget.SAFETY_INTERLOCK, reason

        # 2. Ambiguity / Low Confidence Check (conversational queries answer directly)
        if decision.needs_clarification(self.confidence_threshold) and decision.intent != IntentType.CONVERSATIONAL_QUERY.value:
            reason = (
                f"Low intent confidence ({decision.intent_confidence:.2f} < {self.confidence_threshold:.2f}). "
                f"Needs verbal clarification."
            )
            logger.info("Routing decision to CLARIFY: %s", reason)
            return RouteTarget.CLARIFY, reason

        # 3. Deterministic Priority 1: Direct App Launch
        if (
            decision.complexity_score <= 0.5
            and decision.intent == IntentType.DIRECT_APP_LAUNCH.value
        ):
            reason = f"Fast-path launch for application '{decision.target_app}'."
            return RouteTarget.PRIORITY_1_LAUNCH, reason

        # 4. Deterministic Priority 1: System Media Control
        if (
            decision.complexity_score <= 0.5
            and decision.intent == IntentType.SYSTEM_MEDIA_CONTROL.value
        ):
            reason = "Fast-path system media / volume adjustment."
            return RouteTarget.PRIORITY_1_MEDIA, reason

        # 5. Conversational Query or Knowledge Question
        if decision.intent == IntentType.CONVERSATIONAL_QUERY.value:
            reason = "Conversational query or knowledge question."
            return RouteTarget.CONVERSATIONAL_QUERY, reason

        # 6. Form Fill & Multilingual Unicode Typing
        if decision.is_text_injection():
            reason = (
                f"Text injection into active focus (Multilingual: {decision.has_multilingual_content})."
            )
            return RouteTarget.TEXT_INJECTION, reason

        # 7. Multi-Step GUI Workflow Automation (Priority 2 -> Priority 3)
        if decision.is_gui_agent_required():
            reason = (
                f"Multi-step GUI workflow targeting '{decision.target_app}' "
                f"(Complexity: {decision.complexity_score:.2f})."
            )
            return RouteTarget.GUI_WORKFLOW_AUTOMATION, reason

        # Default fallback to GUI workflow
        return RouteTarget.GUI_WORKFLOW_AUTOMATION, "Default workflow execution."
