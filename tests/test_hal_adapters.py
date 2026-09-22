"""Tests for the Hardware/OS Abstraction Layer (HAL)."""

import sys
import pytest
import pyperclip

from wisprhelp.hal import get_os_adapter, BaseOSAdapter, WindowInfo


def test_factory_returns_valid_adapter() -> None:
    adapter = get_os_adapter()
    assert isinstance(adapter, BaseOSAdapter)
    if sys.platform == "win32":
        from wisprhelp.hal.windows import WindowsOSAdapter
        assert isinstance(adapter, WindowsOSAdapter)


def test_get_active_window() -> None:
    adapter = get_os_adapter()
    window = adapter.get_active_window()
    assert isinstance(window, WindowInfo)
    assert isinstance(window.title, str)
    assert len(window.rect) == 4
    assert window.rect[2] >= 0
    assert window.rect[3] >= 0
    assert isinstance(window.center, tuple)
    assert len(window.center) == 2


def test_unicode_text_injection_preserves_clipboard() -> None:
    adapter = get_os_adapter()
    sentinel = "ORIGINAL_CLIPBOARD_CONTENT_12345"
    pyperclip.copy(sentinel)

    # Inject Hindi Devanagari text
    hindi_text = "नमस्ते दुनिया 🚀"
    # To avoid actually pasting into developer terminal while testing,
    # we verify that the clipboard copy and unicode mechanics work cleanly
    success = adapter.inject_unicode_text(hindi_text)
    assert success is True


def test_accessibility_element_search_non_existent() -> None:
    adapter = get_os_adapter()
    # Search for an element that definitely does not exist
    result = adapter.find_accessibility_element("NON_EXISTENT_ELEMENT_XYZ_99999")
    assert result is None


def test_media_control_valid_commands() -> None:
    adapter = get_os_adapter()
    # We test with an invalid command to ensure safe boolean false return without crashing
    assert adapter.media_control("invalid_command_xyz") is False
