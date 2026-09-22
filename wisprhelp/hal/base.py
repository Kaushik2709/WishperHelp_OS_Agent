"""Base Hardware & OS Abstraction Layer (HAL) interface.

Defines the contract for cross-platform OS automation (Windows, macOS, Linux),
window inspection, universal Unicode text injection, accessibility element discovery,
and physical mouse/keyboard simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WindowInfo:
    """Information about an OS window."""
    title: str
    process_name: str
    pid: int
    rect: Tuple[int, int, int, int]  # (x, y, width, height)
    is_minimized: bool = False

    @property
    def center(self) -> Tuple[int, int]:
        """Returns the center coordinate (x, y) of the window."""
        x, y, w, h = self.rect
        return (x + w // 2, y + h // 2)


class BaseOSAdapter(ABC):
    """Abstract interface for native operating system interactions."""

    @abstractmethod
    def launch_app(self, app_identifier: str, parameter: Optional[str] = None) -> bool:
        """Priority 1: Launches application by bundle ID, protocol URI, or executable name.

        Args:
            app_identifier: Application name, URI (e.g. 'whatsapp:'), or executable path.
            parameter: Optional command-line parameter or URL.

        Returns:
            True if launch command was dispatched successfully.
        """
        pass

    @abstractmethod
    def get_active_window(self) -> WindowInfo:
        """Returns information about the currently focused desktop window."""
        pass

    @abstractmethod
    def inject_unicode_text(self, text: str) -> bool:
        """Injects multilingual Unicode text (Hindi, Hinglish, emojis) into active focus.

        Must avoid character corruption by using OS clipboard buffer swapping or
        native Unicode event injection.
        """
        pass

    @abstractmethod
    def media_control(self, command: str) -> bool:
        """Controls system volume or media playback.

        Args:
            command: 'vol_up', 'vol_down', 'mute', 'play_pause', 'next', 'prev'.
        """
        pass

    @abstractmethod
    def find_accessibility_element(
        self, query: str, control_type: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """Priority 2: Searches OS native accessibility tree (UIA/AX/AT-SPI) for a control.

        Args:
            query: Button label, input placeholder, or name to search for.
            control_type: Optional filter (e.g., 'Button', 'Edit', 'TabItem').

        Returns:
            Center coordinates (x, y) if found, else None.
        """
        pass

    @abstractmethod
    def move_mouse(self, x: int, y: int, smooth: bool = True) -> None:
        """Moves cursor to pixel coordinates (x, y)."""
        pass

    @abstractmethod
    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        """Performs mouse click at coordinates (x, y) or current position."""
        pass

    @abstractmethod
    def key_combination(self, keys: List[str]) -> None:
        """Presses and releases a combination of keys (e.g. ['ctrl', 'v'], ['alt', 'tab'])."""
        pass

    @abstractmethod
    def press_key(self, key: str) -> None:
        """Presses and releases a single key (e.g. 'enter', 'tab', 'esc')."""
        pass

    def wait_for_active_window(self, target_keywords: List[str], timeout: float = 2.0) -> bool:
        """Polls until the foreground window title or process contains one of target_keywords."""
        import time
        start = time.perf_counter()
        targets = [k.lower().strip() for k in target_keywords if k and k.strip()]
        while time.perf_counter() - start < timeout:
            win = self.get_active_window()
            title_lower = win.title.lower()
            proc_lower = win.process_name.lower()
            for k in targets:
                if k in title_lower or k in proc_lower:
                    return True
            time.sleep(0.1)
        return False
