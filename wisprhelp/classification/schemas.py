"""TypeSafe Jev classification schemas and decision models.

Defines the 5-question multi-evaluation matrix and strongly-typed dataclasses
for calibrated decision routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IntentType(str, Enum):
    """Primary intent categories for OS interaction."""
    DIRECT_APP_LAUNCH = "direct_app_launch"
    TEXT_INPUT_INJECTION = "text_input_injection"
    SYSTEM_MEDIA_CONTROL = "system_media_control"
    GUI_WORKFLOW_AUTOMATION = "gui_workflow_automation"
    CONVERSATIONAL_QUERY = "conversational_query"


class TargetApp(str, Enum):
    """Target application domain for the command."""
    WHATSAPP = "whatsapp"
    BROWSER = "browser"
    APP_STORE = "app_store"
    FILE_MANAGER = "file_manager"
    TERMINAL_CODE = "terminal_code"
    SYSTEM_SETTINGS = "system_settings"
    MEDIA_PLAYER = "media_player"
    GENERIC_OR_ACTIVE = "generic_or_active"


# Standard 5-Question Specification for TypeSafe Jev (System One)
JEV_QUESTIONS_SPEC: Dict[str, Dict[str, Any]] = {
    "intent": {
        "type": "choice",
        "instructions": "What is the primary action requested by the user?",
        "criteria": {
            "direct_app_launch": "Opening, launching, starting, or switching to an application or system tool.",
            "text_input_injection": "Typing, dictating, or entering text/message directly into the active field or window.",
            "system_media_control": "Controlling system-level settings like volume, mute, brightness, playback, or locking the screen.",
            "gui_workflow_automation": "Executing a multi-step task inside an application requiring searching, clicking buttons, selecting items, or navigating.",
            "conversational_query": "Asking a general knowledge question, calculation, or conversational query without expecting OS manipulation.",
        },
    },
    "target_app": {
        "type": "choice",
        "instructions": "Which application or service is the target of this command?",
        "criteria": {
            "whatsapp": "WhatsApp desktop, web, or messaging app.",
            "browser": "Chrome, Edge, Safari, Firefox, or web browsing.",
            "app_store": "Microsoft Store, Mac App Store, package manager, or software center.",
            "file_manager": "Windows Explorer, macOS Finder, or Linux file manager.",
            "terminal_code": "Terminal, Command Prompt, PowerShell, VS Code, or IDE.",
            "system_settings": "OS Settings, Control Panel, Network, or display panels.",
            "media_player": "Spotify, YouTube, VLC, or music/video playback.",
            "generic_or_active": "Currently active window, desktop, or generic interface.",
        },
    },
    "complexity": {
        "type": "score",
        "instructions": "Rate the operational complexity required to execute this command",
        "criteria": [
            "Level 0: Deterministic operation executable via single system call, URI protocol, or media hotkey.",
            "Level 1: Single targeted interaction in the active window (e.g. paste text into active focus).",
            "Level 2: Multi-step workflow requiring visual perception, element grounding, and iterative screen verification.",
        ],
    },
    "is_destructive": {
        "type": "noul",
        "instructions": "Does this action delete files, make purchases, send unverified messages, terminate processes, or modify critical system settings?",
    },
    "has_multilingual_content": {
        "type": "noul",
        "instructions": "Does the input contain text meant to be typed in Hindi, Hinglish, Spanish, Arabic, or another non-Latin/complex script?",
    },
}


@dataclass(frozen=True)
class DecomposedStep:
    """An atomic sub-action extracted from a compound user command."""

    step_number: int
    query_text: str
    action_type: str  # "launch", "in_app_search", "click_element", "type_text", "media_control", "keyboard_shortcut"
    target: Optional[str] = None
    payload: Optional[str] = None


@dataclass(frozen=True)
class DecomposedQuery:
    """Structured semantic decomposition of a user query produced by Groq or fallback heuristic."""

    original_query: str
    target_app: Optional[str] = None
    is_compound: bool = False
    normalized_intent: str = IntentType.GUI_WORKFLOW_AUTOMATION.value
    sub_queries: List[DecomposedStep] = field(default_factory=list)


@dataclass(frozen=True)
class JevDecision:
    """Calibrated decision object returned by TypeSafe Jev classification."""

    intent: str
    intent_confidence: float
    target_app: str
    target_app_confidence: float
    complexity_score: float
    complexity_confidence: float
    is_destructive: bool
    destructive_probability: float
    has_multilingual_content: bool
    multilingual_probability: float
    raw_response: Dict[str, Any] = field(default_factory=dict)
    decomposed_plan: Optional[DecomposedQuery] = None

    def is_high_risk(self, threshold: float = 0.65) -> bool:
        """Checks if the action has an unacceptably high destructive probability."""
        return self.is_destructive or self.destructive_probability >= threshold

    def needs_clarification(self, threshold: float = 0.60) -> bool:
        """Checks if the intent confidence is too low to proceed safely."""
        return self.intent_confidence < threshold

    def is_fast_path(self) -> bool:
        """Checks if the command is a Level 0 deterministic system or launch command."""
        return (
            self.complexity_score <= 0.5
            and self.intent in (
                IntentType.DIRECT_APP_LAUNCH.value,
                IntentType.SYSTEM_MEDIA_CONTROL.value,
            )
        )

    def is_text_injection(self) -> bool:
        """Checks if the command is a text typing/dictation task."""
        return (
            self.intent == IntentType.TEXT_INPUT_INJECTION.value
            or (self.complexity_score <= 1.0 and self.has_multilingual_content)
        )

    def is_gui_agent_required(self) -> bool:
        """Checks if a full multi-step visual UI agent loop is needed."""
        return (
            self.complexity_score >= 1.5
            or self.intent == IntentType.GUI_WORKFLOW_AUTOMATION.value
        )
