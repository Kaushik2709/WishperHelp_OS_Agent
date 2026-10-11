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
    SELECT_ALL = "select_all"
    CLEAR_TEXT = "clear_text"
    SCROLL = "scroll"
    DRAG = "drag"
    SELECT = "select"
    OPEN_FILE = "open_file"
    CREATE_FILE = "create_file"
    WRITE_FILE = "write_file"
    APPEND_FILE = "append_file"
    RENAME_FILE = "rename_file"
    MOVE_FILE = "move_file"
    DELETE_FILE = "delete_file"
    READ_FILE = "read_file"
    OPEN_FOLDER = "open_folder"
    FIND_FILE = "find_file"
    SEARCH = "search"
    NAVIGATE = "navigate"
    CLOSE = "close"
    WAIT = "wait"
    ASK_CLARIFICATION = "ask_clarification"
    RESPOND = "respond"
    MEDIA_CONTROL = "media_control"


# Conditions that can be evaluated at runtime before executing a step.
SUPPORTED_CONDITIONS = (
    "document_has_content",
    "document_is_empty",
    "clipboard_non_empty",
)


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
    condition: Optional[str] = None  # Runtime predicate; step is skipped when false

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
            "condition": self.condition,
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
            condition=data.get("condition"),
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
        condition: Optional[str] = None,
    ) -> Action:
        act_type = ActionType.DOUBLE_CLICK if clicks == 2 else (ActionType.RIGHT_CLICK if button == "right" else ActionType.CLICK)
        return cls(
            type=act_type,
            target=target,
            params={"button": button, "clicks": clicks},
            rationale=rationale or f"Click target '{target}'",
            condition=condition,
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
    def select_all(cls, target: str = "active_document", rationale: str = "") -> Action:
        return cls(
            type=ActionType.SELECT_ALL,
            target=target,
            grounding_tier=1,
            rationale=rationale or "Select all content in the active document",
        )

    @classmethod
    def clear_text(cls, target: str = "active_document", rationale: str = "", condition: Optional[str] = None) -> Action:
        return cls(
            type=ActionType.CLEAR_TEXT,
            is_destructive=True,
            target=target,
            grounding_tier=1,
            rationale=rationale or "Clear all content from the active document",
            condition=condition,
        )

    @classmethod
    def drag(
        cls,
        source: str,
        destination: str,
        rationale: str = "",
        condition: Optional[str] = None,
    ) -> Action:
        return cls(
            type=ActionType.DRAG,
            target=source,
            payload=destination,
            rationale=rationale or f"Drag '{source}' to '{destination}'",
            condition=condition,
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

    @classmethod
    def create_file(cls, path: str, content: str = "", rationale: str = "") -> Action:
        return cls(
            type=ActionType.CREATE_FILE,
            target=path,
            payload=content,
            grounding_tier=1,
            rationale=rationale or f"Create file '{path}' natively",
        )

    @classmethod
    def write_file(cls, path: str, content: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.WRITE_FILE,
            target=path,
            payload=content,
            grounding_tier=1,
            rationale=rationale or f"Write to file '{path}' natively",
        )

    @classmethod
    def append_file(cls, path: str, content: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.APPEND_FILE,
            target=path,
            payload=content,
            grounding_tier=1,
            rationale=rationale or f"Append to file '{path}' natively",
        )

    @classmethod
    def rename_file(cls, old_path: str, new_name: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.RENAME_FILE,
            target=old_path,
            payload=new_name,
            grounding_tier=1,
            rationale=rationale or f"Rename '{old_path}' to '{new_name}'",
        )

    @classmethod
    def move_file(cls, source_path: str, destination_dir: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.MOVE_FILE,
            target=source_path,
            payload=destination_dir,
            grounding_tier=1,
            rationale=rationale or f"Move '{source_path}' to '{destination_dir}'",
        )

    @classmethod
    def delete_file(cls, path: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.DELETE_FILE,
            target=path,
            grounding_tier=1,
            is_destructive=True,  # Safety interlock gate required
            rationale=rationale or f"Delete file '{path}' natively",
        )

    @classmethod
    def read_file(cls, path: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.READ_FILE,
            target=path,
            grounding_tier=1,
            rationale=rationale or f"Read file '{path}' natively",
        )

    @classmethod
    def open_folder(cls, path: str, rationale: str = "") -> Action:
        return cls(
            type=ActionType.OPEN_FOLDER,
            target=path,
            grounding_tier=1,
            rationale=rationale or f"Open folder '{path}'",
        )
