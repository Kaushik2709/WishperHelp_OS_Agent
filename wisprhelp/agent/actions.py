"""Platform-neutral Action Schema for WisprHelp Semantic Brain.

Defines the abstract, general action model for arbitrary computer use
across applications, filesystems, browsers, media, and OS controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    """General platform-neutral action types."""
    LAUNCH_APP = "launch_app"
    FOCUS_WINDOW = "focus_window"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    KEY_COMBINATION = "key_combination"
    SCROLL = "scroll"
    SELECT = "select"
    OPEN_FILE = "open_file"
    FIND_FILE = "find_file"
    SEARCH = "search"
    NAVIGATE = "navigate"
    CLOSE = "close"
    WAIT = "wait"
    ASK_CLARIFICATION = "ask_clarification"
    RESPOND = "respond"
    MEDIA_CONTROL = "media_control"


@dataclass
class Action:
    """A general, typed desktop interaction primitive."""

    type: ActionType
    target: Optional[str] = None
    payload: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    is_destructive: bool = False
    grounding_tier: Optional[int] = None  # 1 (Shell/P1), 2 (UIA/P2), 3 (Vision/P3)
    rationale: str = ""
    resolved_target: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes action to a clean dictionary."""
        return {
            "type": self.type.value if isinstance(self.type, ActionType) else str(self.type),
            "target": self.target,
            "payload": self.payload,
            "params": self.params,
            "confidence": self.confidence,
            "is_destructive": self.is_destructive,
            "grounding_tier": self.grounding_tier,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Action:
        """Constructs an Action from a dictionary."""
        act_type_str = data.get("type", "respond")
        try:
            act_type = ActionType(act_type_str)
        except ValueError:
            act_type = ActionType.RESPOND

        return cls(
            type=act_type,
            target=data.get("target"),
            payload=data.get("payload"),
            params=data.get("params", {}),
            confidence=float(data.get("confidence", 1.0)),
            is_destructive=bool(data.get("is_destructive", False)),
            grounding_tier=data.get("grounding_tier"),
            rationale=data.get("rationale", ""),
        )

    # --- Convenience Factory Constructors ---

    @classmethod
    def launch_app(
        cls, target: str, params: Optional[str] = None, rationale: str = ""
    ) -> Action:
        return cls(
            type=ActionType.LAUNCH_APP,
            target=target,
            payload=params,
            grounding_tier=1,
            rationale=rationale or f"Launch application '{target}'",
        )

    @classmethod
    def focus_window(cls, target: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.FOCUS_WINDOW,
            target=target,
            grounding_tier=1,
            rationale=rationale or f"Focus window matching '{target}'",
        )

    @classmethod
    def click(
        cls,
        target: str,
        button: str = "left",
        clicks: int = 1,
        rationale: str = "",
    ) -> Action:
        act_type = ActionType.DOUBLE_CLICK if clicks == 2 else (ActionType.RIGHT_CLICK if button == "right" else ActionType.CLICK)
        return cls(
            type=act_type,
            target=target,
            params={"button": button, "clicks": clicks},
            rationale=rationale or f"Click target '{target}'",
        )

    @classmethod
    def type_text(
        cls, text: str, press_enter: bool = False, rationale: str = ""
    ) -> Action:
        return cls(
            type=ActionType.TYPE_TEXT,
            payload=text,
            params={"press_enter": press_enter},
            grounding_tier=1,
            rationale=rationale or f"Type text '{text[:30]}'",
        )

    @classmethod
    def press_key(cls, key: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.PRESS_KEY,
            target=key,
            grounding_tier=1,
            rationale=rationale or f"Press key '{key}'",
        )

    @classmethod
    def key_combination(cls, keys: List[str], rationale: str = "") -> Action:
        return cls(
            type=ActionType.KEY_COMBINATION,
            params={"keys": keys},
            grounding_tier=1,
            rationale=rationale or f"Press hotkey combination '{'+'.join(keys)}'",
        )

    @classmethod
    def scroll(
        cls, direction: str = "down", amount: int = 3, rationale: str = ""
    ) -> Action:
        return cls(
            type=ActionType.SCROLL,
            params={"direction": direction, "amount": amount},
            grounding_tier=1,
            rationale=rationale or f"Scroll {direction} by {amount} units",
        )

    @classmethod
    def select(cls, target: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.SELECT,
            target=target,
            rationale=rationale or f"Select '{target}'",
        )

    @classmethod
    def open_file(cls, target: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.OPEN_FILE,
            target=target,
            grounding_tier=1,
            rationale=rationale or f"Open file '{target}'",
        )

    @classmethod
    def find_file(
        cls, query: str, directory: Optional[str] = None, rationale: str = ""
    ) -> Action:
        return cls(
            type=ActionType.FIND_FILE,
            target=query,
            params={"directory": directory},
            grounding_tier=1,
            rationale=rationale or f"Find file matching '{query}'",
        )

    @classmethod
    def search(
        cls, query: str, engine_or_app: Optional[str] = None, rationale: str = ""
    ) -> Action:
        return cls(
            type=ActionType.SEARCH,
            payload=query,
            target=engine_or_app,
            rationale=rationale or f"Search for '{query}' in {engine_or_app or 'browser'}",
        )

    @classmethod
    def navigate(cls, destination: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.NAVIGATE,
            payload=destination,
            rationale=rationale or f"Navigate to '{destination}'",
        )

    @classmethod
    def close(cls, target: Optional[str] = None, rationale: str = "") -> Action:
        return cls(
            type=ActionType.CLOSE,
            target=target or "active_window",
            rationale=rationale or f"Close target '{target or 'active window'}'",
        )

    @classmethod
    def wait(
        cls, condition: str, timeout: float = 3.0, rationale: str = ""
    ) -> Action:
        return cls(
            type=ActionType.WAIT,
            target=condition,
            params={"timeout": timeout},
            grounding_tier=1,
            rationale=rationale or f"Wait for '{condition}'",
        )

    @classmethod
    def ask_clarification(
        cls, question: str, choices: Optional[List[str]] = None, rationale: str = ""
    ) -> Action:
        return cls(
            type=ActionType.ASK_CLARIFICATION,
            payload=question,
            params={"choices": choices or []},
            rationale=rationale or "Ask user for clarification",
        )

    @classmethod
    def respond(cls, text: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.RESPOND,
            payload=text,
            rationale=rationale or "Conversational response",
        )

    @classmethod
    def media_control(cls, command: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.MEDIA_CONTROL,
            payload=command,
            grounding_tier=1,
            rationale=rationale or f"Media control '{command}'",
        )
