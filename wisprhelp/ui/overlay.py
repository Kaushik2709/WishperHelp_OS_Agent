"""Floating Status Pill HUD (PySide6 with Headless Fallback).

Displays an always-on-top, translucent dark-glass pill at the top of the monitor.
Configured with WA_TransparentForMouseEvents so mouse clicks pass through
to underlying desktop applications.
"""

from __future__ import annotations

import enum
import logging
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class HUDState(str, enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    CONFIRM = "confirm"
    ERROR = "error"


# Color & visual tokens for dark-glass aesthetic
STATE_THEMES = {
    HUDState.IDLE: {"dot": "#888899", "badge": "WisprHelp", "glow": "rgba(136, 136, 153, 0.2)"},
    HUDState.LISTENING: {"dot": "#3b82f6", "badge": "Listening", "glow": "rgba(59, 130, 246, 0.4)"},
    HUDState.THINKING: {"dot": "#f59e0b", "badge": "Thinking", "glow": "rgba(245, 158, 11, 0.4)"},
    HUDState.EXECUTING: {"dot": "#10b981", "badge": "Executing", "glow": "rgba(16, 185, 129, 0.4)"},
    HUDState.CONFIRM: {"dot": "#ef4444", "badge": "Action Risk", "glow": "rgba(239, 68, 68, 0.5)"},
    HUDState.ERROR: {"dot": "#dc2626", "badge": "Error", "glow": "rgba(220, 38, 38, 0.4)"},
}


class StatusOverlayHUD:
    """Production floating HUD coordinator with PySide6 dark-glass widget & headless fallback."""

    def __init__(self, enable_gui: bool = True) -> None:
        self.enable_gui = enable_gui
        self.current_state = HUDState.IDLE
        self.current_message = "Ready"
        self._app = None
        self._window = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts the HUD in a background thread if GUI is enabled."""
        if not self.enable_gui:
            logger.info("HUD initialized in headless/console mode.")
            return

        with self._lock:
            if self._running:
                return
            self._running = True

            def _qt_thread_run():
                try:
                    from PySide6.QtCore import Qt, Signal, QObject
                    from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel

                    # Create or get existing QApplication
                    app = QApplication.instance()
                    if app is None:
                        app = QApplication(sys.argv if sys.argv else ["wisprhelp"])
                    self._app = app

                    class HUDBridge(QObject):
                        status_signal = Signal(object, str)

                    bridge = HUDBridge()
                    self._bridge = bridge

                    class PillWidget(QWidget):
                        def __init__(self):
                            super().__init__()
                            self.setWindowFlags(
                                Qt.WindowStaysOnTopHint
                                | Qt.FramelessWindowHint
                                | Qt.Tool
                            )
                            self.setAttribute(Qt.WA_TranslucentBackground, True)
                            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                            self.setAttribute(Qt.WA_ShowWithoutActivating, True)

                            self.setStyleSheet("""
                                QWidget#Container {
                                    background-color: rgba(18, 18, 26, 0.92);
                                    border: 1px solid rgba(255, 255, 255, 0.15);
                                    border-radius: 18px;
                                }
                                QLabel#Dot {
                                    border-radius: 5px;
                                    min-width: 10px;
                                    max-width: 10px;
                                    min-height: 10px;
                                    max-height: 10px;
                                }
                                QLabel#Badge {
                                    color: #f1f5f9;
                                    font-family: 'Segoe UI', system-ui, sans-serif;
                                    font-size: 11px;
                                    font-weight: 600;
                                    letter-spacing: 0.5px;
                                    text-transform: uppercase;
                                }
                                QLabel#Message {
                                    color: #94a3b8;
                                    font-family: 'Segoe UI', system-ui, sans-serif;
                                    font-size: 12px;
                                    font-weight: 400;
                                }
                            """)

                            self.setFixedHeight(40)
                            self.setMinimumWidth(320)
                            self.setMaximumWidth(600)

                            layout = QHBoxLayout(self)
                            layout.setContentsMargins(0, 0, 0, 0)
                            layout.setSpacing(0)

                            self.container = QWidget(self)
                            self.container.setObjectName("Container")
                            inner_layout = QHBoxLayout(self.container)
                            inner_layout.setContentsMargins(14, 4, 16, 4)
                            inner_layout.setSpacing(10)

                            self.dot = QLabel()
                            self.dot.setObjectName("Dot")
                            self.dot.setStyleSheet("background-color: #888899; border-radius: 5px;")

                            self.badge = QLabel("WISPRHELP")
                            self.badge.setObjectName("Badge")

                            self.msg = QLabel("Ready")
                            self.msg.setObjectName("Message")

                            inner_layout.addWidget(self.dot)
                            inner_layout.addWidget(self.badge)
                            inner_layout.addWidget(self.msg)

                            layout.addWidget(self.container)

                            # Position at top-center of primary screen
                            screen = app.primaryScreen().geometry()
                            pill_w = 360
                            self.resize(pill_w, 40)
                            x = screen.x() + (screen.width() - pill_w) // 2
                            y = screen.y() + 14
                            self.move(x, y)

                        def update_status(self, state: HUDState, message: str):
                            theme = STATE_THEMES.get(state, STATE_THEMES[HUDState.IDLE])
                            self.dot.setStyleSheet(f"background-color: {theme['dot']}; border-radius: 5px;")
                            self.badge.setText(theme["badge"].upper())
                            self.msg.setText(message[:50] + ("..." if len(message) > 50 else ""))

                    self._window = PillWidget()
                    bridge.status_signal.connect(self._window.update_status, Qt.QueuedConnection)
                    self._window.show()
                    app.exec()
                except Exception as exc:
                    logger.debug("Could not start PySide6 GUI HUD: %s; falling back to console.", exc)

            self._thread = threading.Thread(target=_qt_thread_run, daemon=True)
            self._thread.start()

    def set_state(self, state: HUDState, message: str) -> None:
        """Updates the status HUD text and visual indicator."""
        self.current_state = state
        self.current_message = message
        logger.info("[HUD: %s] %s", state.value.upper(), message)

        bridge = getattr(self, "_bridge", None)
        if bridge is not None:
            try:
                bridge.status_signal.emit(state, message)
            except Exception:
                pass
        elif self._window is not None:
            try:
                self._window.update_status(state, message)
            except Exception:
                pass

    def stop(self) -> None:
        """Hides HUD and terminates Qt event loop."""
        with self._lock:
            self._running = False
            if self._app is not None:
                try:
                    self._app.quit()
                except Exception:
                    pass
            self._window = None
            self._app = None
