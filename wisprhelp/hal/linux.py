"""Linux OS adapter implementation (X11 & Wayland).

Implements Priority 1 (xdg-open, .desktop launchers),
Priority 2 (xdotool / AT-SPI),
Universal Unicode text injection via clipboard swap, and Linux system controls.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pyautogui
import pyperclip

from wisprhelp.config import config
from wisprhelp.hal.base import BaseOSAdapter, WindowInfo

logger = logging.getLogger(__name__)


class LinuxOSAdapter(BaseOSAdapter):
    """Native Linux OS implementation (supporting X11 and Wayland)."""

    def __init__(self) -> None:
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05

    def launch_app(self, app_identifier: str, parameter: Optional[str] = None) -> bool:
        """Priority 1: Launches app via xdg-open, gio launch, or binary."""
        target = app_identifier.strip()
        cmd = ["xdg-open", target] if "://" in target else [target]

        if parameter:
            cmd.append(parameter)

        try:
            subprocess.Popen(cmd)
            logger.info("Launched Linux app: %s", cmd)
            return True
        except Exception:
            try:
                # Fallback to gio launch
                desktop = Path("/usr/share/applications") / f"{target}.desktop"
                if Path(target).name != target or not desktop.is_file():
                    return False
                subprocess.run(["gio", "launch", str(desktop)], check=True, timeout=3.0)
                return True
            except Exception as exc:
                logger.error("Failed to launch Linux app '%s': %s", target, exc)
                return False

    def get_active_window(self) -> WindowInfo:
        """Queries active window title via xdotool or wmctrl."""
        if shutil.which("xdotool"):
            try:
                win_id = subprocess.check_output(["xdotool", "getactivewindow"], text=True, timeout=3.0).strip()
                title = subprocess.check_output(["xdotool", "getwindowname", win_id], text=True, timeout=3.0).strip()
                pid_out = subprocess.check_output(["xdotool", "getwindowpid", win_id], text=True, timeout=3.0).strip()
                pid = int(pid_out) if pid_out.isdigit() else 0
                return WindowInfo(
                    title=title or "Linux Window",
                    process_name="unknown",
                    pid=pid,
                    rect=(0, 0, 1920, 1080),
                    is_minimized=False,
                )
            except Exception as exc:
                logger.debug("xdotool active window query failed: %s", exc)

        return WindowInfo(
            title="Linux Desktop",
            process_name="desktop",
            pid=0,
            rect=(0, 0, 1920, 1080),
            is_minimized=False,
        )

    def inject_unicode_text(self, text: str) -> bool:
        """Universal Unicode injection on Linux using clipboard and Ctrl+V."""
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
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.04)
            return True
        except Exception as exc:
            if type(exc).__name__ in ("FailSafeException", "EmergencyKillSwitchTriggered"):
                raise
            logger.error("Failed to inject unicode text on Linux: %s", exc)
            return False
        finally:
            try:
                time.sleep(0.1)
                pyperclip.copy(old_clipboard)
            except Exception:
                pass

    def media_control(self, command: str) -> bool:
        """Controls system audio using pactl or playerctl."""
        cmd = command.lower().strip()
        try:
            if cmd in ("mute", "unmute"):
                subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if cmd == "mute" else "0"], check=True, timeout=3.0)
            elif cmd in ("vol_up", "volume_up"):
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"], check=True)
            elif cmd in ("vol_down", "volume_down"):
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"], check=True)
            elif cmd in ("play", "pause", "play_pause"):
                subprocess.run(["playerctl", "play-pause"], check=True)
            elif cmd == "next":
                subprocess.run(["playerctl", "next"], check=True)
            elif cmd == "prev":
                subprocess.run(["playerctl", "previous"], check=True)
            else:
                return False
            return True
        except Exception as exc:
            logger.error("Linux media control failed for '%s': %s", command, exc)
            return False

    def find_accessibility_element(
        self, query: str, control_type: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """AT-SPI lookup is not implemented; never invent button coordinates.

        An xdotool window position does not locate a control inside it.
        """
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
