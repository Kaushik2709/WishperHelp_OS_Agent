"""Exercise the UIA iterator API without touching a real desktop."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from wisprhelp.hal import windows


def control(name="", kind="EditControl", value="old"):
    c = Mock()
    c.Name = name
    c.AutomationId = "editor" if kind == "EditControl" else "button"
    c.HelpText = ""
    c.ControlTypeName = kind
    c.IsEnabled = True
    c.IsOffscreen = False
    c.HasKeyboardFocus = kind == "EditControl"
    c.BoundingRectangle = SimpleNamespace(left=10, top=20, width=lambda: 100, height=lambda: 40)
    c.GetValuePattern.return_value = SimpleNamespace(IsReadOnly=False, Value=value)
    return c


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(windows, "HAS_UIA", True)
    monkeypatch.setattr(windows, "HAS_WIN32", False)
    return windows.WindowsOSAdapter()


def test_uia_iterator_discovers_named_and_unnamed_controls(adapter, monkeypatch):
    edit, button = control(), control("Save", "ButtonControl")
    api = Mock()
    api.WalkControl.side_effect = lambda *args, **kwargs: iter([(edit, 1), (button, 1)])
    monkeypatch.setattr(windows, "auto", api)
    assert adapter.find_accessibility_element("Save") == (60, 40)
    assert adapter.find_accessibility_element("Save", "edit") is None
    assert any(e["auto_id"] == "editor" for e in adapter.get_visible_elements())


def test_focus_and_read_editor_without_select_all_copy(adapter, monkeypatch):
    edit = control(value="नमस्ते")
    api = Mock()
    api.WalkControl.return_value = iter([(edit, 1)])
    api.GetFocusedControl.return_value = edit
    monkeypatch.setattr(windows, "auto", api)
    assert adapter.focus_text_input("editor")
    edit.SetFocus.assert_called_once()
    assert adapter.read_focused_text() == "नमस्ते"


def test_multiple_unfocused_editors_are_ambiguous(adapter, monkeypatch):
    editors = [control(), control()]
    for edit in editors:
        edit.HasKeyboardFocus = False
    api = Mock()
    api.WalkControl.return_value = iter([(e, 1) for e in editors])
    monkeypatch.setattr(windows, "auto", api)
    assert not adapter.focus_text_input("editor")
    for edit in editors:
        edit.SetFocus.assert_not_called()
