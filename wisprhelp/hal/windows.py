"""Windows OS adapter implementation.

Implements Priority 1 (Shell, Get-StartApps, URI protocols),
Priority 2 (Microsoft UI Automation via uiautomation),
Universal Unicode text injection via clipboard swap, and Win32 controls.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import pyautogui
import pyperclip

try:
    import win32api
    import win32con
    import win32gui
    import win32process
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

try:
    import uiautomation as auto
    HAS_UIA = True
except ImportError:
    HAS_UIA = False

from wisprhelp.config import config
from wisprhelp.hal.base import BaseOSAdapter, WindowInfo

logger = logging.getLogger(__name__)

# Virtual Key Codes for Media
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

WIN32_VK_MAP: Dict[str, int] = {
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "shift": 0x10,
    "win": 0x5B,
    "windows": 0x5B,
    "enter": 0x0D,
    "return": 0x0D,
    "tab": 0x09,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "backspace": 0x08,
    "delete": 0x2E,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
}


class WindowsOSAdapter(BaseOSAdapter):
    """Native Windows OS implementation."""

    def __init__(self) -> None:
        # Enable PyAutoGUI fail-safe (fling mouse to screen corner to abort)
        pyautogui.FAILSAFE = config.kill_switch_enabled
        pyautogui.PAUSE = 0.05
        self._start_apps_cache: Dict[str, str] = {}
        self._app_paths_cache: Dict[str, str] = {}
        self._cache_initialized = False
        self._app_paths_initialized = False

    def _ensure_app_paths_cache(self) -> None:
        """Scans Windows Registry for registered application executables (HKLM & HKCU)."""
        if self._app_paths_initialized:
            return
        try:
            import winreg

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    key = winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths")
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        sub = winreg.EnumKey(key, i)
                        try:
                            sub_key = winreg.OpenKey(key, sub)
                            exe_path = winreg.QueryValue(sub_key, "")
                            if exe_path and os.path.exists(exe_path):
                                clean_name = sub.lower().replace(".exe", "")
                                self._app_paths_cache[clean_name] = exe_path
                        except Exception:
                            pass
                except Exception:
                    pass
            self._app_paths_initialized = True
            logger.info("Loaded %d Windows Registry App Paths into cache.", len(self._app_paths_cache))
        except Exception as exc:
            logger.debug("Failed to scan App Paths registry: %s", exc)
            self._app_paths_initialized = True

    def _ensure_start_apps_cache(self) -> None:
        """Caches Windows Start Menu apps and AppIDs via PowerShell Get-StartApps."""
        if self._cache_initialized:
            return
        try:
            cmd = "powershell -NoProfile -Command \"Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress\""
            output = subprocess.check_output(cmd, shell=True, text=True, timeout=4.0)
            import json
            data = json.loads(output)
            if isinstance(data, list):
                for item in data:
                    name = item.get("Name", "").lower().strip()
                    app_id = item.get("AppID", "")
                    if name and app_id:
                        self._start_apps_cache[name] = app_id
            elif isinstance(data, dict):
                name = data.get("Name", "").lower().strip()
                app_id = data.get("AppID", "")
                if name and app_id:
                    self._start_apps_cache[name] = app_id
            self._cache_initialized = True
            logger.info("Loaded %d Windows StartApps into cache.", len(self._start_apps_cache))
        except Exception as exc:
            logger.debug("Failed to populate Get-StartApps cache: %s", exc)
            self._cache_initialized = True

    def _find_start_app(self, target: str) -> Optional[str]:
        """Performs multi-heuristic fuzzy, token, and acronym matching against Windows StartApps."""
        self._ensure_start_apps_cache()
        t_clean = target.lower().strip()
        t_norm = re.sub(r"[^a-z0-9]", "", t_clean)

        # 1. Exact name match
        if t_clean in self._start_apps_cache:
            return self._start_apps_cache[t_clean]

        # 2. Normalized alphanumeric exact match
        for name, app_id in self._start_apps_cache.items():
            name_norm = re.sub(r"[^a-z0-9]", "", name)
            if t_norm and t_norm == name_norm:
                return app_id

        # 3. Substring match (whole word or boundary containment)
        for name, app_id in self._start_apps_cache.items():
            if t_clean in name or (len(name) > 3 and name in t_clean):
                return app_id

        # 4. Token overlap match (all target tokens must be in name, or high Jaccard overlap >= 0.67)
        t_tokens = set(re.split(r"[\s\-_]+", t_clean))
        t_tokens.discard("")
        if t_tokens:
            for name, app_id in self._start_apps_cache.items():
                name_tokens = set(re.split(r"[\s\-_]+", name))
                if t_tokens.issubset(name_tokens):
                    return app_id
                overlap = len(t_tokens & name_tokens)
                if overlap > 0 and (overlap / len(t_tokens)) >= 0.67:
                    return app_id

        return None

    def _launch_via_start_search(self, target: str) -> bool:
        """Universal Human Fallback: Opens Start Menu, types app name, and presses Enter."""
        try:
            import ctypes
            import time

            VK_LWIN = 0x5B
            VK_RETURN = 0x0D
            KEYEVENTF_KEYUP = 0x0002

            logger.info("Invoking universal human-like Windows Start Search for '%s'...", target)
            # 1. Press Windows Key
            ctypes.windll.user32.keybd_event(VK_LWIN, 0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
            time.sleep(0.20)  # Windows 11 Start animation

            # 2. Type target app name cleanly
            self.inject_unicode_text(target)
            time.sleep(0.35)  # Windows Search query indexing

            # 3. Press Enter to launch best match
            ctypes.windll.user32.keybd_event(VK_RETURN, 0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0)
            time.sleep(0.15)
            logger.info("Dispatched Windows Start Search Enter for '%s'.", target)
            return True
        except Exception as exc:
            logger.error("Start search fallback failed for '%s': %s", target, exc)
            return False

    def launch_app(self, app_identifier: str, parameter: Optional[str] = None) -> bool:
        """Universal 5-layer Application Launcher:
        Layer 1: Known URI protocols & direct mappings (< 10ms)
        Layer 2: Windows Registry App Paths for exact .exe paths (< 15ms)
        Layer 3: Windows StartApps fuzzy & token matcher (< 25ms)
        Layer 4: Direct PATH / shutil.which / startfile (< 30ms)
        Layer 5: Universal human Start Search (Win key + type + Enter) (< 450ms)
        """
        target = app_identifier.strip()
        lower_target = target.lower()

        # --- Layer 1: Protocol schemes & common aliases (< 10ms) ---
        protocol_map = {
            "whatsapp": "whatsapp:",
            "store": "ms-windows-store:",
            "ms-windows-store": "ms-windows-store:",
            "app_store": "ms-windows-store:",
            "settings": "ms-settings:",
            "calculator": "calculator:",
            "calc": "calculator:",
            "spotify": "spotify:",
            "vscode": "vscode:",
            "vs code": "vscode:",
            "visual studio code": "vscode:",
            "code": "vscode:",
            "terminal_code": "vscode:",
            "edge": "microsoft-edge:",
            "microsoft edge": "microsoft-edge:",
            "msedge": "microsoft-edge:",
            "browser": "https://www.google.com",
            "chrome": "chrome",
            "google chrome": "chrome",
            "brave": "brave",
            "brave browser": "brave",
            "firefox": "firefox",
            "notepad": "notepad",
            "terminal": "wt",
            "windows terminal": "wt",
            "wt": "wt",
            "cmd": "cmd",
            "command prompt": "cmd",
            "powershell": "powershell",
            "task manager": "taskmgr",
            "taskmgr": "taskmgr",
            "file explorer": "explorer",
            "explorer": "explorer",
            "files": "explorer",
            "control panel": "control",
            "control": "control",
            "paint": "mspaint",
            "mspaint": "mspaint",
            "snipping tool": "snippingtool",
            "snippingtool": "snippingtool",
            "word": "winword",
            "winword": "winword",
            "excel": "excel",
            "powerpoint": "powerpnt",
            "powerpnt": "powerpnt",
            "steam": "steam:",
            "discord": "discord:",
            "telegram": "telegram:",
            "slack": "slack:",
        }
        for key, uri in protocol_map.items():
            if key == lower_target or (len(key) > 3 and key in lower_target):
                try:
                    if ":" in uri and not uri.endswith(".exe"):
                        os.startfile(uri)
                        logger.info("Layer 1: Launched app via protocol: %s -> %s", key, uri)
                        return True
                    else:
                        import shutil
                        which_exe = shutil.which(uri) or shutil.which(f"{uri}.exe")
                        if which_exe:
                            subprocess.Popen([which_exe])
                            logger.info("Layer 1: Launched app via alias in PATH: %s -> %s", key, which_exe)
                            return True
                except Exception as exc:
                    logger.debug("Layer 1 failed for %s (%s): %s", key, uri, exc)

        # --- Layer 2: Windows Registry App Paths (Exact Match < 15ms) ---
        self._ensure_app_paths_cache()
        if lower_target in self._app_paths_cache:
            exe_path = self._app_paths_cache[lower_target]
            try:
                cmd_list = [exe_path] if not parameter else [exe_path, parameter]
                subprocess.Popen(cmd_list)
                logger.info("Layer 2: Launched via exact Registry App Path: %s (%s)", lower_target, exe_path)
                return True
            except Exception as exc:
                logger.debug("Layer 2 exact launch failed for %s: %s", exe_path, exc)

        # --- Layer 3: Windows StartApps Fuzzy & Token Matcher (< 25ms) ---
        app_id = self._find_start_app(target)
        if app_id:
            try:
                subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
                logger.info("Layer 3: Launched via StartApp AppID: %s (%s)", target, app_id)
                return True
            except Exception as exc:
                logger.debug("Layer 3 launch failed for AppID %s: %s", app_id, exc)

        # --- Layer 4: Windows Registry App Paths (Substring Match < 30ms) ---
        for name, exe_path in self._app_paths_cache.items():
            if lower_target in name or name in lower_target:
                try:
                    cmd_list = [exe_path] if not parameter else [exe_path, parameter]
                    subprocess.Popen(cmd_list)
                    logger.info("Layer 4: Launched via partial Registry App Path: %s (%s)", name, exe_path)
                    return True
                except Exception as exc:
                    logger.debug("Layer 4 launch failed for %s: %s", exe_path, exc)

        # --- Layer 4: Direct executable / PATH / startfile (< 30ms) ---
        try:
            import shutil
            found_which = shutil.which(target) or shutil.which(f"{target}.exe")
            if found_which:
                subprocess.Popen([found_which])
                logger.info("Layer 4: Launched executable from PATH: %s", found_which)
                return True
            os.startfile(target)
            logger.info("Layer 4: Launched via os.startfile: %s", target)
            return True
        except Exception:
            pass

        # --- Layer 5: Universal Human-Like Windows Start Search Fallback (< 450ms) ---
        return self._launch_via_start_search(target)

    def get_active_window(self) -> WindowInfo:
        """Returns information about the foreground window."""
        if not HAS_WIN32:
            return WindowInfo(
                title="Unknown",
                process_name="unknown.exe",
                pid=0,
                rect=(0, 0, 1920, 1080),
                is_minimized=False,
            )

        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) or "Desktop"
            _, pid = win32process.GetWindowThreadProcessId(hwnd)

            rect = (0, 0, 1920, 1080)
            try:
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                rect = (left, top, max(0, right - left), max(0, bottom - top))
            except Exception:
                pass

            is_minimized = bool(win32gui.IsIconic(hwnd))

            # Retrieve process executable name
            process_name = "unknown.exe"
            try:
                import psutil
                process_name = psutil.Process(pid).name()
            except Exception:
                pass

            return WindowInfo(
                title=title,
                process_name=process_name,
                pid=pid,
                rect=rect,
                is_minimized=is_minimized,
            )
        except Exception as exc:
            logger.debug("Error getting active window: %s", exc)
            return WindowInfo(
                title="Unknown Window",
                process_name="unknown.exe",
                pid=0,
                rect=(0, 0, 1920, 1080),
                is_minimized=False,
            )

    def inject_unicode_text(self, text: str) -> bool:
        """Universal Unicode injection for Hindi Devanagari, emojis, and Hinglish."""
        if not text:
            return True

        old_clipboard = ""
        try:
            old_clipboard = pyperclip.paste()
        except Exception:
            pass

        try:
            pyperclip.copy(text)
            time.sleep(0.04)  # Ensure clipboard buffer update
            if HAS_WIN32:
                # Use native Win32 keyboard events for instantaneous paste
                VK_CONTROL = 0x11
                VK_V = 0x56
                win32api.keybd_event(VK_CONTROL, 0, 0, 0)
                win32api.keybd_event(VK_V, 0, 0, 0)
                time.sleep(0.02)
                win32api.keybd_event(VK_V, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32api.keybd_event(VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
            else:
                pyautogui.hotkey("ctrl", "v")
            time.sleep(0.04)
            return True
        except Exception as exc:
            logger.error("Failed to inject unicode text via clipboard: %s", exc)
            return False
        finally:
            # Restore original clipboard
            try:
                time.sleep(0.1)
                pyperclip.copy(old_clipboard)
            except Exception:
                pass

    def media_control(self, command: str) -> bool:
        """Sends native Windows multimedia events or sets master volume scalar via pycaw."""
        if not HAS_WIN32:
            return False

        cmd = command.lower().strip()

        # Handle exact volume percentage scaling via pycaw: set_volume:<pct>
        if cmd.startswith("set_volume:"):
            try:
                pct_str = cmd.split(":", 1)[1]
                pct = max(0, min(100, int(pct_str)))
                try:
                    from pycaw.pycaw import AudioUtilities
                    speakers = AudioUtilities.GetSpeakers()
                    endpoint = speakers.EndpointVolume
                    endpoint.SetMasterVolumeLevelScalar(pct / 100.0, None)
                    logger.info("Set Windows master volume to %d%% via pycaw.", pct)
                    return True
                except Exception as pycaw_exc:
                    logger.warning("pycaw volume adjustment failed: %s; trying key fallback.", pycaw_exc)
            except Exception as exc:
                logger.error("Error parsing volume target: %s", exc)
                return False

        # Handle relative volume adjustment via pycaw: adjust_volume:<delta>
        if cmd.startswith("adjust_volume:"):
            try:
                delta_str = cmd.split(":", 1)[1]
                delta = int(delta_str)
                try:
                    from pycaw.pycaw import AudioUtilities
                    speakers = AudioUtilities.GetSpeakers()
                    endpoint = speakers.EndpointVolume
                    current = endpoint.GetMasterVolumeLevelScalar()
                    new_val = max(0.0, min(1.0, current + (delta / 100.0)))
                    endpoint.SetMasterVolumeLevelScalar(new_val, None)
                    logger.info("Adjusted Windows master volume from %d%% to %d%% via pycaw.", int(current * 100), int(new_val * 100))
                    return True
                except Exception as pycaw_exc:
                    logger.warning("pycaw relative volume failed: %s", pycaw_exc)
            except Exception as exc:
                logger.error("Error parsing volume delta: %s", exc)
                return False

        vk_map = {
            "mute": VK_VOLUME_MUTE,
            "unmute": VK_VOLUME_MUTE,
            "vol_up": VK_VOLUME_UP,
            "volume_up": VK_VOLUME_UP,
            "vol_down": VK_VOLUME_DOWN,
            "volume_down": VK_VOLUME_DOWN,
            "play": VK_MEDIA_PLAY_PAUSE,
            "pause": VK_MEDIA_PLAY_PAUSE,
            "play_pause": VK_MEDIA_PLAY_PAUSE,
            "next": VK_MEDIA_NEXT_TRACK,
            "prev": VK_MEDIA_PREV_TRACK,
        }

        vk = vk_map.get(cmd)
        if vk is None:
            logger.warning("Unrecognized media control command: %s", command)
            return False

        try:
            win32api.keybd_event(vk, 0, 0, 0)
            time.sleep(0.02)
            win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
            logger.info("Executed native Windows media key: %s (0x%X)", cmd, vk)
            return True
        except Exception as exc:
            logger.error("Error dispatching media key %s: %s", cmd, exc)
            return False

    def find_ordinal_element(
        self, ordinal_index: int, item_type: str = "item", target_app: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """Finds the coordinates of a list/row/tab item at the specified ordinal position.

        Args:
            ordinal_index: 0=first, 1=second, 2=third, -1=last, -2=second-last, etc.
            item_type: 'chat', 'item', 'row', 'tab', 'button', etc.
            target_app: Target application name if available.

        Returns:
            Center coordinates (x, y) if found, else None.
        """
        if not HAS_UIA:
            return None

        try:
            root = None
            if HAS_WIN32:
                try:
                    hwnd = win32gui.GetForegroundWindow()
                    if hwnd:
                        root = auto.ControlFromHandle(hwnd)
                except Exception:
                    pass
            if not root:
                root = auto.GetForegroundControl()
            if not root:
                root = auto.GetRootControl()

            candidates: List[auto.Control] = []
            item_type_lower = (item_type or "item").lower()

            def _collect_items(ctrl: auto.Control, depth: int) -> bool:
                ctrl_type = (ctrl.ControlTypeName or "").lower()
                rect = ctrl.BoundingRectangle
                w, h = rect.width(), rect.height()

                # Basic sanity check: visible on screen
                if w <= 10 or h <= 10 or rect.top < 0 or rect.left < 0:
                    return False

                # Chat / List item matching
                if item_type_lower in ("chat", "conversation", "contact", "item", "row", "message"):
                    if any(t in ctrl_type for t in ("listitem", "custom", "dataitem", "pane")):
                        if w >= 80 and 20 <= h <= 200:
                            candidates.append(ctrl)
                elif item_type_lower in ("tab", "tabitem"):
                    if "tabitem" in ctrl_type or ("button" in ctrl_type and h <= 50):
                        candidates.append(ctrl)
                elif item_type_lower in ("button", "btn"):
                    if "button" in ctrl_type:
                        candidates.append(ctrl)
                else:
                    if any(t in ctrl_type for t in ("listitem", "custom", "dataitem", "button")):
                        if w >= 60 and 20 <= h <= 200:
                            candidates.append(ctrl)
                return False

            auto.WalkControl(root, _collect_items, maxDepth=12)

            if not candidates:
                logger.info("No candidates found for ordinal element type '%s'", item_type)
                return None

            # Deduplicate items with identical or nearly identical bounding boxes
            unique_candidates: List[auto.Control] = []
            seen_boxes = set()
            for c in candidates:
                r = c.BoundingRectangle
                box_key = (r.left // 5, r.top // 5, r.width() // 5, r.height() // 5)
                if box_key not in seen_boxes:
                    seen_boxes.add(box_key)
                    unique_candidates.append(c)

            # Sort primarily vertically (by top), secondarily horizontally (by left)
            if item_type_lower in ("tab", "tabitem"):
                unique_candidates.sort(key=lambda c: (c.BoundingRectangle.top // 15, c.BoundingRectangle.left))
            else:
                unique_candidates.sort(key=lambda c: (c.BoundingRectangle.top, c.BoundingRectangle.left))

            total = len(unique_candidates)
            logger.info("Found %d candidate controls for ordinal index %d (type='%s')", total, ordinal_index, item_type)

            selected_ctrl = None
            if ordinal_index >= 0 and ordinal_index < total:
                selected_ctrl = unique_candidates[ordinal_index]
            elif ordinal_index < 0 and abs(ordinal_index) <= total:
                selected_ctrl = unique_candidates[ordinal_index]
            elif ordinal_index >= total and total > 0:
                selected_ctrl = unique_candidates[-1]
            elif abs(ordinal_index) > total and total > 0:
                selected_ctrl = unique_candidates[0]

            if selected_ctrl:
                r = selected_ctrl.BoundingRectangle
                cx = r.left + r.width() // 2
                cy = r.top + r.height() // 2
                logger.info(
                    "Selected ordinal [%d] item (name='%s', type='%s') at (%d, %d)",
                    ordinal_index, selected_ctrl.Name, selected_ctrl.ControlTypeName, cx, cy
                )
                return (cx, cy)

        except Exception as exc:
            logger.debug("Error in find_ordinal_element: %s", exc)

        return None

    def find_accessibility_element(
        self, query: str, control_type: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """Priority 2: Traverses Microsoft UI Automation tree to locate control coordinates."""
        if not HAS_UIA:
            logger.debug("uiautomation package not available for Priority 2 search.")
            return None

        clean_query = query.lower().strip()
        if not clean_query:
            return None

        try:
            # Search in the currently active foreground window first
            root = None
            if HAS_WIN32:
                try:
                    hwnd = win32gui.GetForegroundWindow()
                    if hwnd:
                        root = auto.ControlFromHandle(hwnd)
                except Exception:
                    pass
            if not root:
                root = auto.GetForegroundControl()
            if not root:
                root = auto.GetRootControl()

            exact_matches: List[auto.Control] = []
            button_matches: List[auto.Control] = []
            other_matches: List[auto.Control] = []

            # Match function
            def _check_control(ctrl: auto.Control, depth: int) -> bool:
                name = (ctrl.Name or "").strip().lower()
                auto_id = (ctrl.AutomationId or "").strip().lower()
                help_text = (ctrl.HelpText or "").strip().lower()
                ctrl_type = (ctrl.ControlTypeName or "").lower()

                is_match = False
                is_btn = "button" in ctrl_type or "menuitem" in ctrl_type or "tabitem" in ctrl_type or "link" in ctrl_type

                # Priority A: Exact match on Name, AutoId, or HelpText
                if clean_query == name or clean_query == auto_id or clean_query == help_text:
                    is_match = True
                    bucket = exact_matches
                # Priority B: Word boundary match
                elif re.search(rf"\b{re.escape(clean_query)}\b", name) or re.search(rf"\b{re.escape(clean_query)}\b", auto_id):
                    is_match = True
                    bucket = button_matches if is_btn else other_matches
                # Priority C: Substring match (only if query is at least 3 chars)
                elif len(clean_query) >= 3 and (clean_query in name or clean_query in auto_id):
                    is_match = True
                    bucket = button_matches if is_btn else other_matches

                if is_match:
                    rect = ctrl.BoundingRectangle
                    if rect.width() > 0 and rect.height() > 0:
                        bucket.append(ctrl)
                        if bucket is exact_matches and is_btn:
                            return True
                return False

            # Search up to depth 16 for comprehensive coverage across Electron / IDEs
            auto.WalkControl(root, _check_control, maxDepth=16)

            all_candidates = exact_matches + button_matches + other_matches
            if all_candidates:
                best = all_candidates[0]
                rect = best.BoundingRectangle
                center_x = rect.left + rect.width() // 2
                center_y = rect.top + rect.height() // 2
                logger.info(
                    "Priority 2 UIA found element '%s' (name='%s', type='%s') at (%d, %d)",
                    query, best.Name, best.ControlTypeName, center_x, center_y
                )
                return (center_x, center_y)

        except Exception as exc:
            logger.debug("Priority 2 UIA traversal error for '%s': %s", query, exc)

        return None


    def move_mouse(self, x: int, y: int, smooth: bool = True) -> None:
        """Moves cursor to pixel coordinate (x, y)."""
        duration = 0.15 if smooth else 0.0
        try:
            pyautogui.moveTo(x, y, duration=duration)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI fail-safe intercepted mouse movement.")

    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        """Clicks at coordinates or at current position."""
        try:
            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI fail-safe intercepted click.")

    def key_combination(self, keys: List[str]) -> None:
        """Presses and releases keyboard hotkeys using native Win32 events if available."""
        if HAS_WIN32:
            try:
                vk_codes = []
                for k in keys:
                    kl = k.lower().strip()
                    if kl in WIN32_VK_MAP:
                        vk_codes.append(WIN32_VK_MAP[kl])
                    elif len(kl) == 1:
                        vk_codes.append(ord(kl.upper()))

                # Press down in order
                for vk in vk_codes:
                    win32api.keybd_event(vk, 0, 0, 0)
                time.sleep(0.02)
                # Release in reverse order
                for vk in reversed(vk_codes):
                    win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
                logger.info("Dispatched native Win32 hotkey combination: %s", keys)
                return
            except Exception as exc:
                logger.debug("Native keybd_event hotkey failed: %s; falling back to PyAutoGUI.", exc)

        try:
            pyautogui.hotkey(*keys)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI fail-safe intercepted hotkey: %s", keys)

    def press_key(self, key: str) -> None:
        """Presses and releases a single key using native Win32 events if available."""
        if HAS_WIN32:
            try:
                kl = key.lower().strip()
                vk = WIN32_VK_MAP.get(kl)
                if vk is None and len(kl) == 1:
                    vk = ord(kl.upper())
                if vk is not None:
                    win32api.keybd_event(vk, 0, 0, 0)
                    time.sleep(0.02)
                    win32api.keybd_event(vk, 0, win32con.KEYEVENTF_KEYUP, 0)
                    logger.info("Dispatched native Win32 key press: %s (0x%X)", key, vk)
                    return
            except Exception as exc:
                logger.debug("Native keybd_event press_key failed: %s; falling back to PyAutoGUI.", exc)

        try:
            pyautogui.press(key)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI fail-safe intercepted key press: %s", key)
