"""Mocked contracts for all adapters, runnable without the named operating system."""
import subprocess
from unittest.mock import MagicMock

import pyperclip
import pytest

from wisprhelp.hal.linux import LinuxOSAdapter
from wisprhelp.hal.macos import MacOSAdapter
from wisprhelp.hal.windows import WindowsOSAdapter


@pytest.mark.parametrize("adapter_type", [WindowsOSAdapter, MacOSAdapter, LinuxOSAdapter])
def test_unicode_clipboard_restored_per_adapter(adapter_type):
    pyperclip.copy("fixture-original")
    assert adapter_type().inject_unicode_text("नमस्ते 👋")
    assert pyperclip.paste() == "fixture-original"


def test_macos_launch_waits_for_launcher_exit_and_passes_literal_arguments(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    assert MacOSAdapter().launch_app("Fixture App", "literal;argument")
    run.assert_called_once_with(["open", "-a", "Fixture App", "--args", "literal;argument"], check=True, timeout=3.0)


def test_macos_launcher_failure_is_false(monkeypatch):
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=subprocess.CalledProcessError(1, "open")))
    assert not MacOSAdapter().launch_app("Missing Fixture App")


def test_macos_accessibility_escapes_untrusted_query(monkeypatch):
    query = 'bad\\name"\nignored'
    output = MagicMock(return_value="NOT_FOUND")
    monkeypatch.setattr(subprocess, "check_output", output)
    assert MacOSAdapter().find_accessibility_element(query) is None
    script = output.call_args.args[0][2]
    assert 'bad\\\\name\\" ignored' in script
    assert output.call_args.kwargs["timeout"] == 3.0


def test_linux_never_guesses_control_coordinates(monkeypatch):
    output = MagicMock(return_value="Position: 100,100")
    monkeypatch.setattr(subprocess, "check_output", output)
    assert LinuxOSAdapter().find_accessibility_element("Pay Now") is None
    output.assert_not_called()


def test_linux_unknown_media_command_has_no_effect(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    assert not LinuxOSAdapter().media_control("invalid-command")
    run.assert_not_called()


@pytest.mark.parametrize("adapter_type", [MacOSAdapter, LinuxOSAdapter])
@pytest.mark.parametrize("command", ["mute", "unmute"])
def test_mute_is_explicit_instead_of_toggle(monkeypatch, adapter_type, command):
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    assert adapter_type().media_control(command)
    assert "toggle" not in str(run.call_args)


@pytest.mark.parametrize("operation", ["key_combination", "press_key", "inject_unicode_text"])
def test_windows_partial_native_input_releases_keys_without_replay(monkeypatch, operation):
    import wisprhelp.hal.windows as windows
    monkeypatch.setattr(windows, "HAS_WIN32", True)
    native = MagicMock()
    def event(vk, scan, flags, extra):
        if flags == 0 and vk in (ord("V"), ord("A")):
            raise OSError("Simulated dispatch failure")
    native.keybd_event.side_effect = event
    monkeypatch.setattr(windows, "win32api", native, raising=False)
    monkeypatch.setattr(windows, "win32con", MagicMock(KEYEVENTF_KEYUP=2), raising=False)
    fallback = MagicMock()
    monkeypatch.setattr(windows.pyautogui, "hotkey", fallback)
    monkeypatch.setattr(windows.pyautogui, "press", fallback)
    adapter = WindowsOSAdapter()
    if operation == "inject_unicode_text":
        assert not adapter.inject_unicode_text("isolated text")
    else:
        with pytest.raises(OSError):
            getattr(adapter, operation)(["ctrl", "a"] if operation == "key_combination" else "a")
    fallback.assert_not_called()
    downs = [c.args[0] for c in native.keybd_event.call_args_list if c.args[2] == 0]
    ups = [c.args[0] for c in native.keybd_event.call_args_list if c.args[2] == 2]
    assert ups == list(reversed(downs))


def test_windows_native_input_checks_emergency_before_dispatch(monkeypatch):
    import wisprhelp.hal.windows as windows
    monkeypatch.setattr(windows, "HAS_WIN32", True)
    native = MagicMock()
    monkeypatch.setattr(windows, "win32api", native, raising=False)
    adapter = WindowsOSAdapter()
    monkeypatch.setattr(adapter, "check_emergency_stop", MagicMock(side_effect=RuntimeError("Emergency stop")))
    with pytest.raises(RuntimeError):
        adapter.key_combination(["ctrl", "a"])
    native.keybd_event.assert_not_called()
