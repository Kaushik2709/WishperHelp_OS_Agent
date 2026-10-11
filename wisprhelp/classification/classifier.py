"""High-level TypeSafe JEV Classifier for WisprHelp.

Takes a natural language user query and OS context, runs the 5-question
parallel evaluation, and returns a calibrated JevDecision.
"""

from __future__ import annotations

import logging
import platform
import re
import math
from typing import Any, Dict, Optional

from wisprhelp.classification.client import JevClient
from wisprhelp.classification.decomposer import SemanticDecomposer
from wisprhelp.classification.schemas import (
    JEV_QUESTIONS_SPEC,
    DecomposedQuery,
    IntentType,
    JevDecision,
    TargetApp,
)

logger = logging.getLogger(__name__)


class JevClassifier:
    """Classifies user goals into calibrated OS interaction decisions."""

    def __init__(
        self,
        client: Optional[JevClient] = None,
        decomposer: Optional[SemanticDecomposer] = None,
    ) -> None:
        self.client = client or JevClient()
        self.decomposer = decomposer or SemanticDecomposer()

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

        # 0. Pre-Jev Semantic Decomposition (Groq LPU / Deterministic Fallback)
        decomposed: Optional[DecomposedQuery] = None
        try:
            decomposed = self.decomposer.decompose(cleaned_query, os_context)
        except Exception as exc:
            logger.warning("Semantic decomposition exception: %s", exc)

        context = {
            "user_query": cleaned_query,
            "os_platform": platform.system().lower(),
            **(os_context or {}),
        }
        if decomposed:
            context["is_compound"] = decomposed.is_compound
            if decomposed.target_app:
                context["target_app_hint"] = decomposed.target_app
            context["normalized_intent"] = decomposed.normalized_intent

        try:
            raw = self.client.evaluate(state=context, questions=JEV_QUESTIONS_SPEC)
            decision = self._parse_response(raw, decomposed)
        except Exception as exc:
            logger.error("TypeSafe JEV API evaluation failed: %s. Using local fallback.", exc)
            decision = self._offline_fallback_classify(cleaned_query, context, decomposed)
            decision.raw_response["degraded_reason"] = "api_unavailable"

        return decision

    def _parse_response(
        self,
        raw: Dict[str, Any],
        decomposed: Optional[DecomposedQuery] = None,
    ) -> JevDecision:
        """Parses raw JEV response into typed JevDecision dataclass with calibrated confidence."""
        answers = raw.get("answers", {}) if isinstance(raw, dict) else {}
        def score(section, key, upper=1.0):
            value = answers.get(section, {}).get(key)
            if isinstance(value, bool):
                raise ValueError("Boolean classification score")
            result = float(value)
            if not math.isfinite(result) or not 0 <= result <= upper:
                raise ValueError("Invalid classification score")
            return result

        # Incomplete/malformed API responses must not acquire confident defaults.
        try:
            score("intent", "confidence")
            score("target_app", "confidence")
            score("complexity", "score", 2)
            score("complexity", "confidence")
            score("is_destructive", "noul")
            score("has_multilingual_content", "noul")
            if answers["intent"].get("choice") not in {i.value for i in IntentType}:
                raise ValueError("Unknown intent")
        except (ValueError, TypeError, AttributeError, KeyError):
            return JevDecision(IntentType.GUI_WORKFLOW_AUTOMATION.value, 0.0,
                               TargetApp.GENERIC_OR_ACTIVE.value, 0.0, 2.0, 0.0,
                               False, 0.0, False, 0.0,
                               raw_response={"invalid_classification": True}, decomposed_plan=decomposed)

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

        # Boost confidence & resolve target app if decomposed plan provides high certainty
        if decomposed:
            if decomposed.is_compound:
                intent = IntentType.GUI_WORKFLOW_AUTOMATION.value
                # Decomposition can change the route, never increase calibration.
                comp_score = max(comp_score, 2.0)
            if decomposed.target_app and target_app == TargetApp.GENERIC_OR_ACTIVE.value:
                mapped = self._map_app_to_target_enum(decomposed.target_app)
                if mapped != TargetApp.GENERIC_OR_ACTIVE.value:
                    target_app = mapped

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
            decomposed_plan=decomposed,
        )

    @staticmethod
    def _map_app_to_target_enum(app_name: Optional[str]) -> str:
        """Maps canonical application name string to TargetApp enum value."""
        if not app_name:
            return TargetApp.GENERIC_OR_ACTIVE.value
        low = app_name.lower()
        if "whatsapp" in low:
            return TargetApp.WHATSAPP.value
        if any(b in low for b in ("chrome", "browser", "edge", "firefox", "brave", "opera", "safari")):
            return TargetApp.BROWSER.value
        if any(s in low for s in ("store", "market")):
            return TargetApp.APP_STORE.value
        if any(t in low for t in ("code", "terminal", "powershell", "cmd", "bash", "ide")):
            return TargetApp.TERMINAL_CODE.value
        if any(f in low for f in ("file", "explorer", "finder", "directory")):
            return TargetApp.FILE_MANAGER.value
        if any(m in low for m in ("media", "spotify", "music", "player", "youtube")):
            return TargetApp.MEDIA_PLAYER.value
        if any(s in low for s in ("setting", "control panel", "config")):
            return TargetApp.SYSTEM_SETTINGS.value
        return TargetApp.GENERIC_OR_ACTIVE.value

    def _offline_fallback_classify(
        self,
        query: str,
        context: Dict[str, Any],
        decomposed: Optional[DecomposedQuery] = None,
    ) -> JevDecision:
        """Rule-based offline heuristic classifier for offline graceful degradation."""
        lower_query = query.lower()

        # Check for non-ASCII / Devanagari characters
        has_non_ascii = any(ord(char) > 127 for char in query)

        # Destructive patterns
        destructive_keywords = [
            "delete", "remove", "erase", "format", "wipe", "drop",
            "kill", "terminate", "uninstall", "destroy", "shutdown",
            "send", "submit", "pay", "purchase", "buy", "overwrite",
        ]
        is_destructive = any(kw in lower_query for kw in destructive_keywords)
        dest_prob = 0.95 if is_destructive else 0.05

        # If semantic decomposer identified compound workflow, prioritize high confidence
        if decomposed and decomposed.is_compound:
            tgt_app = self._map_app_to_target_enum(decomposed.target_app)
            return JevDecision(
                intent=IntentType.GUI_WORKFLOW_AUTOMATION.value,
                intent_confidence=0.95,
                target_app=tgt_app,
                target_app_confidence=0.92,
                complexity_score=2.0,
                complexity_confidence=0.90,
                is_destructive=is_destructive,
                destructive_probability=dest_prob,
                has_multilingual_content=has_non_ascii,
                multilingual_probability=0.95 if has_non_ascii else 0.05,
                raw_response={"fallback": True},
                decomposed_plan=decomposed,
            )

        # Conversational / knowledge questions (who are you, what can you do, jokes, greetings)
        conversational_triggers = (
            "who are you", "what is your name", "what can you do", "what are your capabilities",
            "tell me a joke", "tell me about yourself", "how are you",
            "what is wisprhelp", "what can i ask", "koun ho", "kya kar sakte ho",
            "namaste", "good morning", "good afternoon", "good evening", "how's it going",
        )
        single_words = {"hi", "hello", "hey", "help", "who"}
        q_words = set(re.findall(r"\b\w+\b", lower_query))
        if (
            (decomposed and decomposed.normalized_intent == IntentType.CONVERSATIONAL_QUERY.value)
            or any(w in lower_query for w in conversational_triggers)
            or bool(re.match(r"tell me (?:a |an )?.*joke", lower_query))
            or (bool(q_words & single_words) and len(q_words) <= 2)
        ):
            return JevDecision(
                intent=IntentType.CONVERSATIONAL_QUERY.value,
                intent_confidence=0.98,
                target_app=TargetApp.GENERIC_OR_ACTIVE.value,
                target_app_confidence=0.95,
                complexity_score=0.0,
                complexity_confidence=0.95,
                is_destructive=False,
                destructive_probability=0.01,
                has_multilingual_content=has_non_ascii,
                multilingual_probability=0.95 if has_non_ascii else 0.05,
                raw_response={"fallback": True},
            )

        # Volume / System media patterns
        media_keywords = [
            "volume", "vol", "sound", "awaaz", "audio", "percent", "%",
            "mute", "unmute", "brightness", "pause", "play next", "lock screen",
            "louder", "quieter",
        ]
        if any(kw in lower_query for kw in media_keywords):
            return JevDecision(
                intent=IntentType.SYSTEM_MEDIA_CONTROL.value,
                intent_confidence=0.85,
                target_app=TargetApp.MEDIA_PLAYER.value,
                target_app_confidence=0.85,
                complexity_score=0.0,
                complexity_confidence=0.90,
                is_destructive=is_destructive,
                destructive_probability=dest_prob,
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

        # Typing patterns (strictly excluded if click/button terms present)
        is_click = bool(re.match(r"(?:please\s+)?(?:click|press|tap|select)\b", lower_query))
        is_composition = bool(re.match(r"write\s+(?:something|an?\s+(?:email|essay|story|poem|paragraph)|about)\b", lower_query))
        if not is_click and not is_composition and lower_query.startswith(("type ", "write ", "लिखो ", "टाइप करो ")):
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
            decomposed_plan=decomposed,
        )
