"""Global Hotkey and Emergency Kill-Switch Listener.

Handles Push-to-Talk activation (Alt+Space) and Emergency Abort (Esc)
using non-blocking global system listeners.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, Set

logger = logging.getLogger(__name__)


class PushToTalkListener:
    """Production-grade Global Hotkey & Emergency Abort Listener."""

    def __init__(
        self,
        on_press_callback: Optional[Callable[[], None]] = None,
        on_release_callback: Optional[Callable[[], None]] = None,
        on_abort_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.on_press_callback = on_press_callback
        self.on_release_callback = on_release_callback
        self.on_abort_callback = on_abort_callback

        self._listener = None
        self._is_running = False
        self._pressed_keys: Set[str] = set()
        self._is_active = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Starts the global key listener in a daemon background thread."""
        with self._lock:
            if self._is_running:
                return True

            try:
                from pynput import keyboard

                def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
                    self._handle_key_press(key)

                def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
                    self._handle_key_release(key)

                self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
                self._listener.daemon = True
                self._listener.start()
                self._is_running = True
                logger.info("Global Push-To-Talk & Emergency Kill-Switch listener started.")
                return True
            except Exception as exc:
                logger.warning("Could not start pynput global keyboard hook: %s", exc)
                return False

    def stop(self) -> None:
        """Stops the global listener."""
        with self._lock:
            if self._listener is not None:
                try:
                    self._listener.stop()
                except Exception:
                    pass
                self._listener = None
            self._is_running = False
            self._pressed_keys.clear()
            self._is_active = False
            logger.info("Global hotkey listener stopped.")

    def _normalize_key(self, key: object) -> str:
        """Extracts standard string identifier from pynput key event."""
        try:
            from pynput import keyboard

            if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                return "shift"
            elif key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                return "ctrl"
            elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, getattr(keyboard.Key, "alt_gr", None)):
                return "alt"
            elif key == keyboard.Key.space:
                return "space"
            elif key == keyboard.Key.esc:
                return "esc"
            elif key == keyboard.Key.pause:
                return "pause"
            elif hasattr(key, "char") and key.char:
                return key.char.lower()
            return str(key).lower()
        except Exception:
            return ""

    def _is_ptt_combo_pressed(self) -> bool:
        """Checks if any supported Push-To-Talk hotkey combination is held."""
        # 1. Primary requested: Ctrl + Shift
        if "ctrl" in self._pressed_keys and "shift" in self._pressed_keys:
            return True
        # 2. Alt + Shift
        if "alt" in self._pressed_keys and "shift" in self._pressed_keys:
            return True
        # 3. Alt + Space
        if "alt" in self._pressed_keys and "space" in self._pressed_keys:
            return True
        return False

    def _handle_key_press(self, key: object) -> None:
        key_name = self._normalize_key(key)
        if not key_name:
            return

        self._pressed_keys.add(key_name)

        # 1. Emergency Kill-Switch (Esc or Pause)
        if key_name in ("esc", "pause"):
            logger.critical("EMERGENCY KILL-SWITCH KEY DETECTED (%s). Halting execution.", key_name)
            if self.on_abort_callback:
                try:
                    self.on_abort_callback()
                except Exception as exc:
                    logger.error("Error invoking abort callback: %s", exc)
            return

        # 2. Push-to-Talk (Ctrl+Shift / Alt+Shift / Alt+Space)
        if self._is_ptt_combo_pressed():
            if not self._is_active:
                self._is_active = True
                logger.info("Push-to-Talk triggered (%s).", "+".join(sorted(self._pressed_keys)))
                if self.on_press_callback:
                    try:
                        self.on_press_callback()
                    except Exception as exc:
                        logger.error("Error in on_press_callback: %s", exc)

    def _handle_key_release(self, key: object) -> None:
        key_name = self._normalize_key(key)
        self._pressed_keys.discard(key_name)

        # If Push-to-Talk combo is no longer held down, deactivate
        if self._is_active and not self._is_ptt_combo_pressed():
            self._is_active = False
            logger.info("Push-to-Talk released.")
            if self.on_release_callback:
                try:
                    self.on_release_callback()
                except Exception as exc:
                    logger.error("Error in on_release_callback: %s", exc)
