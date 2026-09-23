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

    def find_ordinal_element(
        self, ordinal_index: int, item_type: str = "item", target_app: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """Searches OS accessibility tree for an item at an ordinal position (0=first, 1=second, -1=last, -2=second-last).

        Args:
            ordinal_index: 0-based or negative index from the end.
            item_type: 'chat', 'item', 'row', 'tab', 'button', etc.
            target_app: Optional target application name.

        Returns:
            Center coordinates (x, y) if located, else None.
        """
        return None


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

    def scroll(self, direction: str = "down", amount: int = 3) -> None:
        """Scrolls the current window or focused viewport up or down."""
        try:
            import pyautogui
            clicks = -amount if direction == "down" else amount
            pyautogui.scroll(clicks * 100)
        except Exception:
            self.press_key("pagedown" if direction == "down" else "pageup")

    def get_open_windows(self) -> List[WindowInfo]:
        """Returns a list of visible open desktop windows."""
        active = self.get_active_window()
        return [active] if active else []

    def focus_window(self, title_or_keyword: str) -> bool:
        """Switches focus to the window matching the given title or keyword."""
        return self.launch_app(title_or_keyword)

    def get_clipboard_text(self) -> str:
        """Returns the current string contents of the OS clipboard."""
        try:
            import pyperclip
            return pyperclip.paste() or ""
        except Exception:
            return ""

    def open_file(self, file_path: str) -> bool:
        """Opens a local file using the default OS application."""
        import os
        import platform
        import subprocess
        expanded = os.path.expanduser(file_path)
        try:
            if platform.system() == "Windows":
                os.startfile(expanded)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", expanded])
            else:
                subprocess.Popen(["xdg-open", expanded])
            return True
        except Exception:
            return False

    def get_visible_elements(self, max_depth: int = 4) -> List[Dict[str, Any]]:
        """Returns interactive UI controls (buttons, tabs, inputs) visible in the foreground."""
        return []


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

