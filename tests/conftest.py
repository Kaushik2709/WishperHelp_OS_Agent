"""Unit tests are offline by default, independent of developer API keys/models.

Live desktop work belongs in the separate evaluation runner; setting
WISPRHELP_TEST_LIVE=1 fails this suite. Protocol tests inject mock responses.
"""
import os
import sys
from unittest.mock import MagicMock

if os.getenv("WISPRHELP_TEST_LIVE") != "1":
    for name in ("JEV_API_KEY", "TYPESAFE_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY",
                 "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        os.environ[name] = ""
    os.environ["WISPRHELP_MOCK_MODELS"] = "true"

# PyAutoGUI imports an X11 connection at import time. Unit tests don't need one.
if sys.platform.startswith("linux") and not os.getenv("DISPLAY"):
    sys.modules["pyautogui"] = MagicMock(FAILSAFE=True)

import pytest


@pytest.fixture(autouse=True)
def isolated_unit_io(monkeypatch, tmp_path):
    """Default tests never paste, move the mouse, capture user pixels, or call APIs.

    Genuine OS evaluations run separately with `python -m evaluations.run`.
    """
    if os.getenv("WISPRHELP_TEST_LIVE") == "1":
        pytest.fail("Live services/desktops belong in the explicit evaluation runner, not the unit suite")
    import pyautogui
    import pyperclip
    import mss
    from PIL import Image, ImageGrab
    from types import SimpleNamespace
    from enum import Enum
    from pathlib import Path
    import subprocess
    key_names = "shift shift_l shift_r ctrl ctrl_l ctrl_r alt alt_l alt_r alt_gr space esc pause"
    keyboard = SimpleNamespace(Key=Enum("TestKey", key_names.split()), KeyCode=type("TestKeyCode", (), {}), Listener=MagicMock())
    monkeypatch.setitem(sys.modules, "pynput", SimpleNamespace(keyboard=keyboard))
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)
    test_home = tmp_path / "home"
    for directory in ("Documents", "Downloads", "Desktop"):
        (test_home / directory).mkdir(parents=True, exist_ok=True)
    expanduser = os.path.expanduser
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: test_home))
    monkeypatch.setattr(os.path, "expanduser", lambda path: str(test_home) + path[1:] if path.startswith("~") else expanduser(path))
    for name in ("click", "doubleClick", "moveTo", "hotkey", "press", "scroll", "keyDown", "keyUp", "mouseDown", "mouseUp", "failSafeCheck"):
        monkeypatch.setattr(pyautogui, name, MagicMock(), raising=False)
    clip = [""]
    monkeypatch.setattr(pyperclip, "copy", lambda value: clip.__setitem__(0, value))
    monkeypatch.setattr(pyperclip, "paste", lambda: clip[0])
    screenshot = SimpleNamespace(width=320, height=200, bgra=bytes([0, 0, 0, 255]) * (320 * 200))
    factory = lambda *args, **kwargs: SimpleNamespace(monitors=[{"left": 0, "top": 0}, {"left": 0, "top": 0}], grab=lambda monitor: screenshot)
    monkeypatch.setattr(mss, "mss", factory)
    monkeypatch.setattr(mss, "MSS", factory, raising=False)
    monkeypatch.setattr(ImageGrab, "grab", lambda *args, **kwargs: Image.new("RGB", (320, 200)))
    try:
        import win32api
        monkeypatch.setattr(win32api, "keybd_event", MagicMock())
        monkeypatch.setattr(win32api, "mouse_event", MagicMock())
    except ImportError:
        pass
    def no_launch(*args, **kwargs):
        raise OSError("Native process/app launch disabled in offline tests")
    monkeypatch.setattr(subprocess, "Popen", no_launch)
    monkeypatch.setattr(subprocess, "run", no_launch)
    monkeypatch.setattr(subprocess, "check_output", no_launch)
    if hasattr(os, "startfile"):
        monkeypatch.setattr(os, "startfile", no_launch)
    # Windows volume APIs can bypass keyboard mocks.
    audio = MagicMock()
    audio.AudioUtilities.GetSpeakers.side_effect = no_launch
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", audio)
    import urllib.request
    import requests
    def no_network(*args, **kwargs):
        raise OSError("External network disabled in offline tests")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
