"""macOS adapter implementation.

Implements Priority 1 (open -a, custom URL schemes),
Priority 2 (AppleScript UI elements / Accessibility AX),
Universal Unicode text injection via Cmd+V, and macOS system events.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import List, Optional, Tuple

import pyautogui
import pyperclip

from wisprhelp.config import config
from wisprhelp.hal.base import BaseOSAdapter, WindowInfo

logger = logging.getLogger(__name__)


class MacOSAdapter(BaseOSAdapter):
    """Native macOS implementation."""

    def __init__(self) -> None:
        pyautogui.FAILSAFE = config.kill_switch_enabled
        pyautogui.PAUSE = 0.05

    def launch_app(self, app_identifier: str, parameter: Optional[str] = None) -> bool:
        """Priority 1: Launches app via macOS `open -a` or URL scheme."""
        target = app_identifier.strip()
        lower_target = target.lower()

        # Custom URL schemes
        if "whatsapp" in lower_target:
            cmd = ["open", "whatsapp://"]
        elif "spotify" in lower_target:
            cmd = ["open", "spotify://"]
        else:
            cmd = ["open", "-a", target]

        if parameter:
            cmd.extend(["--args", parameter])

        try:
            subprocess.Popen(cmd)
            logger.info("Launched macOS app: %s", cmd)
            return True
        except Exception as exc:
            logger.error("Failed to launch macOS app '%s': %s", target, exc)
            return False

    def get_active_window(self) -> WindowInfo:
        """Queries active application title via AppleScript."""
        script = """
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set appName to name of frontApp
            set appPID to unix id of frontApp
            return appName & "||" & (appPID as string)
        end tell
        """
        try:
            output = subprocess.check_output(["osascript", "-e", script], text=True).strip()
            parts = output.split("||")
            app_name = parts[0] if len(parts) > 0 else "Finder"
            pid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            return WindowInfo(
                title=app_name,
                process_name=app_name,
                pid=pid,
                rect=(0, 0, 1920, 1080),
                is_minimized=False,
            )
        except Exception as exc:
            logger.debug("Failed to get macOS active window via AppleScript: %s", exc)
            return WindowInfo(
                title="macOS Desktop",
                process_name="Finder",
                pid=0,
                rect=(0, 0, 1920, 1080),
                is_minimized=False,
            )

    def inject_unicode_text(self, text: str) -> bool:
        """Universal Unicode injection on macOS using clipboard and Command+V."""
        if not text:
            return True

        old_clipboard = ""
        try:
            old_clipboard = pyperclip.paste()
        except Exception:
            pass

        try:
            pyperclip.copy(text)
            time.sleep(0.04)
            pyautogui.hotkey("command", "v")
            time.sleep(0.04)
            return True
        except Exception as exc:
            logger.error("Failed to inject unicode text on macOS: %s", exc)
            return False
        finally:
            try:
                time.sleep(0.1)
                pyperclip.copy(old_clipboard)
            except Exception:
                pass

    def media_control(self, command: str) -> bool:
        """Controls macOS system audio via AppleScript."""
        cmd = command.lower().strip()
        script = ""
        if cmd in ("mute", "unmute"):
            script = "set volume output muted not (output muted of (get volume settings))"
        elif cmd in ("vol_up", "volume_up"):
            script = "set volume output volume ((output volume of (get volume settings)) + 10)"
        elif cmd in ("vol_down", "volume_down"):
            script = "set volume output volume ((output volume of (get volume settings)) - 10)"
        elif cmd in ("play", "pause", "play_pause"):
            script = """
            tell application "System Events"
                key code 16 using {command down, option down}
            end tell
            """

        if not script:
            return False

        try:
            subprocess.run(["osascript", "-e", script], check=True)
            return True
        except Exception as exc:
            logger.error("Error executing macOS media control: %s", exc)
            return False

    def find_accessibility_element(
        self, query: str, control_type: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """Priority 2: Queries macOS Accessibility via AppleScript UI elements."""
        clean_query = query.replace('"', '\\"').strip()
        script = f"""
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            tell frontApp
                try
                    set targetBtn to first UI element whose name contains "{clean_query}"
                    set btnPos to position of targetBtn
                    set btnSize to size of targetBtn
                    return (item 1 of btnPos) & "," & (item 2 of btnPos) & "," & (item 1 of btnSize) & "," & (item 2 of btnSize)
                on error
                    return "NOT_FOUND"
                end try
            end tell
        end tell
        """
        try:
            out = subprocess.check_output(["osascript", "-e", script], text=True).strip()
            if out and out != "NOT_FOUND" and "," in out:
                x, y, w, h = map(int, out.split(","))
                center_x = x + w // 2
                center_y = y + h // 2
                logger.info("macOS Priority 2 AX found element '%s' at (%d, %d)", query, center_x, center_y)
                return (center_x, center_y)
        except Exception as exc:
            logger.debug("macOS accessibility query error for '%s': %s", query, exc)

        return None

    def move_mouse(self, x: int, y: int, smooth: bool = True) -> None:
        duration = 0.15 if smooth else 0.0
        pyautogui.moveTo(x, y, duration=duration)

    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)

    def key_combination(self, keys: List[str]) -> None:
        pyautogui.hotkey(*keys)

    def press_key(self, key: str) -> None:
        pyautogui.press(key)
