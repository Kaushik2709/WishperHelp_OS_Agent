"""Acoustic and Auditory Feedback Cues.

Provides sub-50ms acoustic confirmations for hands-free operation:
- Listening started
- Thinking / Speech detected
- Success completion chime
- Safety confirmation alert
- Emergency abort tone
"""

from __future__ import annotations

import logging
import platform
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class AudioFeedback:
    """Non-blocking auditory feedback coordinator."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._os_type = platform.system().lower()

    def _play_async(self, func) -> None:
        if not self.enabled:
            return
        thread = threading.Thread(target=func, daemon=True)
        thread.start()

    def play_listen_start(self) -> None:
        """Acoustic cue when listening starts."""
        def _beep():
            if self._os_type == "windows":
                try:
                    import winsound
                    winsound.Beep(880, 80)
                except Exception:
                    pass
        self._play_async(_beep)

    def play_speech_end(self) -> None:
        """Acoustic cue when user finishes speaking."""
        def _beep():
            if self._os_type == "windows":
                try:
                    import winsound
                    winsound.Beep(1175, 80)
                except Exception:
                    pass
        self._play_async(_beep)

    def play_success(self) -> None:
        """Ascending chime on successful task completion."""
        def _beep():
            if self._os_type == "windows":
                try:
                    import winsound
                    import time
                    winsound.Beep(659, 60)  # E5
                    time.sleep(0.02)
                    winsound.Beep(880, 100)  # A5
                except Exception:
                    pass
        self._play_async(_beep)

    def play_warning(self) -> None:
        """Double low-frequency alert for destructive confirmation."""
        def _beep():
            if self._os_type == "windows":
                try:
                    import winsound
                    import time
                    winsound.Beep(440, 120)
                    time.sleep(0.05)
                    winsound.Beep(440, 150)
                except Exception:
                    pass
        self._play_async(_beep)

    def play_abort(self) -> None:
        """Descending tone on emergency halt."""
        def _beep():
            if self._os_type == "windows":
                try:
                    import winsound
                    import time
                    winsound.Beep(900, 80)
                    time.sleep(0.02)
                    winsound.Beep(400, 150)
                except Exception:
                    pass
        self._play_async(_beep)
