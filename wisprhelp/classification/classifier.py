"""High-level TypeSafe JEV Classifier for WisprHelp.

Takes a natural language user query and OS context, runs the 5-question
parallel evaluation, and returns a calibrated JevDecision.
"""

from __future__ import annotations

import logging
import platform
import re
from typing import Any, Dict, Optional

from wisprhelp.classification.client import JevClient
from wisprhelp.classification.schemas import (
    JEV_QUESTIONS_SPEC,
    IntentType,
    JevDecision,
    TargetApp,
)

logger = logging.getLogger(__name__)


class JevClassifier:
    """Classifies user goals into calibrated OS interaction decisions."""

    def __init__(self, client: Optional[JevClient] = None) -> None:
        self.client = client or JevClient()

    def classify(
        self,
        user_query: str,
        os_context: Optional[Dict[str, Any]] = None,
    ) -> JevDecision:
        """Evaluates a spoken or typed user command against desktop context.

        Args:
            user_query: The natural language command spoken/typed by the user.
            os_context: Optional dictionary containing active window title, process, etc.

        Returns:
            JevDecision containing parsed choices, scores, probabilities, and confidence.
        """
        cleaned_query = user_query.strip()
        if not cleaned_query:
            # Degenerate empty input
            return JevDecision(
                intent=IntentType.CONVERSATIONAL_QUERY.value,
                intent_confidence=1.0,
                target_app=TargetApp.GENERIC_OR_ACTIVE.value,
                target_app_confidence=1.0,
                complexity_score=0.0,
                complexity_confidence=1.0,
                is_destructive=False,
                destructive_probability=0.0,
                has_multilingual_content=False,
                multilingual_probability=0.0,
                raw_response={},
            )

        context = {
            "user_query": cleaned_query,
            "os_platform": platform.system().lower(),
            **(os_context or {}),
        }

        try:
            raw = self.client.evaluate(state=context, questions=JEV_QUESTIONS_SPEC)
            return self._parse_response(raw)
        except Exception as exc:
            logger.error("TypeSafe JEV API evaluation failed: %s. Using local fallback.", exc)
            return self._offline_fallback_classify(cleaned_query, context)

    def _parse_response(self, raw: Dict[str, Any]) -> JevDecision:
        """Parses raw JEV response into typed JevDecision dataclass."""
        answers = raw.get("answers", {})

        # 1. Intent (Choice)
        intent_data = answers.get("intent", {})
        intent = intent_data.get("choice", IntentType.GUI_WORKFLOW_AUTOMATION.value)
        intent_conf = float(intent_data.get("confidence", 0.8))

        # 2. Target App (Choice)
        app_data = answers.get("target_app", {})
        target_app = app_data.get("choice", TargetApp.GENERIC_OR_ACTIVE.value)
        app_conf = float(app_data.get("confidence", 0.8))

        # 3. Complexity (Score)
        comp_data = answers.get("complexity", {})
        comp_score = float(comp_data.get("score", 1.0))
        comp_conf = float(comp_data.get("confidence", 0.8))

        # 4. Destructive (Noul)
        dest_data = answers.get("is_destructive", {})
        dest_prob = float(dest_data.get("noul", 0.0))
        is_destructive = dest_prob >= 0.50

        # 5. Multilingual Content (Noul)
        multi_data = answers.get("has_multilingual_content", {})
        multi_prob = float(multi_data.get("noul", 0.0))
        has_multilingual = multi_prob >= 0.50

        return JevDecision(
            intent=intent,
            intent_confidence=intent_conf,
            target_app=target_app,
            target_app_confidence=app_conf,
            complexity_score=comp_score,
            complexity_confidence=comp_conf,
            is_destructive=is_destructive,
            destructive_probability=dest_prob,
            has_multilingual_content=has_multilingual,
            multilingual_probability=multi_prob,
            raw_response=raw,
        )

    def _offline_fallback_classify(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> JevDecision:
        """Rule-based offline heuristic classifier for offline graceful degradation."""
        lower_query = query.lower()

        # Check for non-ASCII / Devanagari characters
        has_non_ascii = any(ord(char) > 127 for char in query)

        # Destructive patterns
        destructive_keywords = [
            "delete", "remove", "erase", "format", "wipe", "drop",
            "kill", "terminate", "uninstall", "destroy", "shutdown",
        ]
        is_destructive = any(kw in lower_query for kw in destructive_keywords)
        dest_prob = 0.95 if is_destructive else 0.05

        # Volume / System media patterns
        media_keywords = ["volume", "mute", "unmute", "brightness", "pause", "play next", "lock screen"]
        if any(kw in lower_query for kw in media_keywords):
            return JevDecision(
                intent=IntentType.SYSTEM_MEDIA_CONTROL.value,
                intent_confidence=0.85,
                target_app=TargetApp.MEDIA_PLAYER.value,
                target_app_confidence=0.85,
                complexity_score=0.0,
                complexity_confidence=0.90,
                is_destructive=False,
                destructive_probability=0.05,
                has_multilingual_content=has_non_ascii,
                multilingual_probability=0.95 if has_non_ascii else 0.05,
                raw_response={"fallback": True},
            )

        # Direct app launch patterns
        launch_match = re.match(r"^(open|launch|start|run)\s+([a-zA-Z0-9\s]+)$", lower_query)
        if launch_match and len(lower_query.split()) <= 3:
            app_name = launch_match.group(2).strip()
            target_app = TargetApp.GENERIC_OR_ACTIVE.value
            if "whatsapp" in app_name:
                target_app = TargetApp.WHATSAPP.value
            elif any(b in app_name for b in ("chrome", "browser", "edge", "firefox", "safari")):
                target_app = TargetApp.BROWSER.value
            elif any(s in app_name for s in ("store", "market")):
                target_app = TargetApp.APP_STORE.value
            elif any(t in app_name for t in ("code", "terminal", "powershell", "cmd")):
                target_app = TargetApp.TERMINAL_CODE.value
            elif any(f in app_name for f in ("files", "explorer", "finder")):
                target_app = TargetApp.FILE_MANAGER.value

            return JevDecision(
                intent=IntentType.DIRECT_APP_LAUNCH.value,
                intent_confidence=0.90,
                target_app=target_app,
                target_app_confidence=0.85,
                complexity_score=0.0,
                complexity_confidence=0.95,
                is_destructive=is_destructive,
                destructive_probability=dest_prob,
                has_multilingual_content=has_non_ascii,
                multilingual_probability=0.95 if has_non_ascii else 0.05,
                raw_response={"fallback": True},
            )

        # Typing patterns
        if lower_query.startswith(("type ", "write ", "लिखो ", "टाइप करो ")):
            return JevDecision(
                intent=IntentType.TEXT_INPUT_INJECTION.value,
                intent_confidence=0.88,
                target_app=TargetApp.GENERIC_OR_ACTIVE.value,
                target_app_confidence=0.75,
                complexity_score=1.0,
                complexity_confidence=0.85,
                is_destructive=is_destructive,
                destructive_probability=dest_prob,
                has_multilingual_content=has_non_ascii,
                multilingual_probability=0.95 if has_non_ascii else 0.05,
                raw_response={"fallback": True},
            )

        # Default complex multi-step GUI workflow
        target_app = TargetApp.GENERIC_OR_ACTIVE.value
        if "whatsapp" in lower_query:
            target_app = TargetApp.WHATSAPP.value
        elif any(b in lower_query for b in ("chrome", "google", "website", "youtube", "browser")):
            target_app = TargetApp.BROWSER.value
        elif "store" in lower_query:
            target_app = TargetApp.APP_STORE.value

        return JevDecision(
            intent=IntentType.GUI_WORKFLOW_AUTOMATION.value,
            intent_confidence=0.75,
            target_app=target_app,
            target_app_confidence=0.75,
            complexity_score=2.0,
            complexity_confidence=0.80,
            is_destructive=is_destructive,
            destructive_probability=dest_prob,
            has_multilingual_content=has_non_ascii,
            multilingual_probability=0.95 if has_non_ascii else 0.05,
            raw_response={"fallback": True},
        )
