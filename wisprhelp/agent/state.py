"""Generalized Computer State and Adaptive Observation Engine.

Maintains a unified model of active application, open windows, focused controls,
clipboard data, accessibility subtrees, and recent OS mutations without
incurring excessive latency or vision token overhead.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from wisprhelp.agent.actions import Action
from wisprhelp.hal.base import BaseOSAdapter, WindowInfo
from wisprhelp.hal.factory import get_os_adapter

logger = logging.getLogger(__name__)


@dataclass
class UIElementInfo:
    """Lightweight representation of an interactive UI element."""
    name: str
    control_type: str
    auto_id: str = ""
    rect: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[int, int]] = None
    is_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "control_type": self.control_type,
            "auto_id": self.auto_id,
            "center": self.center,
        }


@dataclass
class ComputerState:
    """Unified snapshot of the workstation state."""

    os_platform: str = field(default_factory=lambda: platform.system().lower())
    active_window: Optional[WindowInfo] = None
    open_windows: List[WindowInfo] = field(default_factory=list)
    focused_element: Optional[UIElementInfo] = None
    clipboard_text: Optional[str] = None
    visible_elements: List[UIElementInfo] = field(default_factory=list)
    recent_actions: List[Action] = field(default_factory=list)
    recent_observations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def active_app_name(self) -> str:
        """Heuristic for the primary active application name."""
        if not self.active_window:
            return "desktop"
        proc = (self.active_window.process_name or "").lower().replace(".exe", "")
        title = self.active_window.title or ""
        if "chrome" in proc or "chrome" in title.lower():
            return "Chrome"
        if "code" in proc or "visual studio code" in title.lower():
            return "VS Code"
        if "spotify" in proc or "spotify" in title.lower():
            return "Spotify"
        if "whatsapp" in proc or "whatsapp" in title.lower():
            return "WhatsApp"
        if "notepad" in proc:
            return "Notepad"
        if "calc" in proc:
            return "Calculator"
        if "explorer" in proc:
            return "File Explorer"
        return proc.title() if proc else title[:30]

    def format_summary(self, max_elements: int = 12) -> str:
        """Formats a compact, token-efficient state summary for LLM context."""
        win_title = self.active_window.title if self.active_window else "None"
        proc = self.active_window.process_name if self.active_window else "None"
        clip_preview = f"'{self.clipboard_text[:50]}...'" if self.clipboard_text else "None"

        lines = [
            f"OS: {self.os_platform}",
            f"Active Window: '{win_title}' (Process: {proc}, App: {self.active_app_name})",
            f"Clipboard: {clip_preview}",
        ]

        if self.open_windows:
            open_titles = [f"'{w.title}'" for w in self.open_windows[:5] if w.title and w.title != win_title]
            if open_titles:
                lines.append(f"Other Windows: {', '.join(open_titles)}")

        if self.visible_elements:
            elem_strs = [
                f"[{e.control_type}] '{e.name}'"
                for e in self.visible_elements[:max_elements]
                if e.name
            ]
            if elem_strs:
                lines.append(f"Interactive UI Elements ({len(elem_strs)}): {', '.join(elem_strs)}")

        if self.recent_actions:
            act_strs = [
                f"{a.type.value if hasattr(a.type, 'value') else a.type}({a.target or a.payload or ''})"
                for a in self.recent_actions[-3:]
            ]
            lines.append(f"Recent Actions: {' -> '.join(act_strs)}")

        return "\n".join(lines)


class ComputerStateManager:
    """Gathers and caches computer state using adaptive observation."""

    def __init__(self, os_adapter: Optional[BaseOSAdapter] = None) -> None:
        self.os_adapter = os_adapter or get_os_adapter()
        self._last_state: Optional[ComputerState] = None

    def capture_state(
        self,
        adaptive: bool = True,
        include_elements: bool = True,
        include_clipboard: bool = False,
    ) -> ComputerState:
        """Captures a snapshot of current computer state.

        Args:
            adaptive: If True, uses fast deterministic OS inspection without heavy tree traversal.
            include_elements: If True, extracts visible accessibility elements of the active window.
            include_clipboard: If True, inspects clipboard text.

        Returns:
            ComputerState object.
        """
        active_win = self.os_adapter.get_active_window()

        # Open windows inspection (if supported)
        open_windows: List[WindowInfo] = []
        if hasattr(self.os_adapter, "get_open_windows"):
            try:
                open_windows = self.os_adapter.get_open_windows()
            except Exception as exc:
                logger.debug("Failed to get open windows: %s", exc)

        # Clipboard inspection
        clipboard_text = None
        if include_clipboard and hasattr(self.os_adapter, "get_clipboard_text"):
            try:
                clipboard_text = self.os_adapter.get_clipboard_text()
            except Exception:
                pass

        # Interactive UI elements via accessibility
        visible_elements: List[UIElementInfo] = []
        if include_elements and hasattr(self.os_adapter, "get_visible_elements"):
            try:
                raw_elems = self.os_adapter.get_visible_elements()
                for item in raw_elems:
                    visible_elements.append(
                        UIElementInfo(
                            name=item.get("name", ""),
                            control_type=item.get("control_type", "control"),
                            auto_id=item.get("auto_id", ""),
                            center=item.get("center"),
                        )
                    )
            except Exception as exc:
                logger.debug("Failed to get visible UI elements: %s", exc)

        state = ComputerState(
            os_platform=platform.system().lower(),
            active_window=active_win,
            open_windows=open_windows,
            clipboard_text=clipboard_text,
            visible_elements=visible_elements,
            recent_actions=self._last_state.recent_actions if self._last_state else [],
            recent_observations=self._last_state.recent_observations if self._last_state else [],
            timestamp=time.time(),
        )
        self._last_state = state
        return state

    def update_with_action(self, action: Action, observation: str = "") -> None:
        """Updates internal state with an executed action and result observation."""
        if not self._last_state:
            self._last_state = ComputerState()

        # Maintain bounded recent actions (max 10)
        self._last_state.recent_actions.append(action)
        if len(self._last_state.recent_actions) > 10:
            self._last_state.recent_actions.pop(0)

        if observation:
            self._last_state.recent_observations.append(observation)
            if len(self._last_state.recent_observations) > 10:
                self._last_state.recent_observations.pop(0)
