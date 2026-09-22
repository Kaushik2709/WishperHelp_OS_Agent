"""Lightweight Cross-Platform Text-To-Speech (TTS) Engine.

Provides instant verbal responses and safety prompts using native OS speech synthesis:
- Windows: Native SAPI.SpVoice via win32com (0 latency, 0 download)
- macOS: say command via subprocess
- Linux: spd-say / espeak fallback
"""

from __future__ import annotations

import logging
import platform
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class TextToSpeech:
    """Production non-blocking Text-To-Speech synthesizer."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._os_type = platform.system().lower()
        self._sapi_voice = None
        self._init_engine()

    def _init_engine(self) -> None:
        if not self.enabled:
            return
        if self._os_type == "windows":
            try:
                import win32com.client

                self._sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                logger.info("Initialized native Windows SAPI TTS voice.")
            except Exception as exc:
                logger.debug("Could not initialize SAPI voice: %s", exc)

    def speak_async(self, text: str) -> None:
        """Speaks text in a non-blocking background thread."""
        if not self.enabled or not text:
            return

        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()

    def speak(self, text: str) -> None:
        """Speaks text synchronously."""
        if not self.enabled or not text:
            return

        try:
            if self._os_type == "windows" and self._sapi_voice is not None:
                # Flag 1 is SVSFlagsAsync in SAPI
                self._sapi_voice.Speak(text, 0)
            elif self._os_type == "darwin":
                subprocess.run(["say", text], check=False, timeout=5.0)
            elif self._os_type == "linux":
                subprocess.run(["spd-say", text], check=False, timeout=5.0)
            else:
                logger.info("[Spoken]: %s", text)
        except Exception as exc:
            logger.debug("TTS speak error: %s", exc)
