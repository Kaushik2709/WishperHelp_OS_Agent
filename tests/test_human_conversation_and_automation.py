"""Test suite for Natural Conversation, Generative Writing, and Browser Automation.

Verifies:
1. Conversational Queries: "who are you", "what can you do", "tell me a joke", etc.
2. Generative Writing: "write something about AI", "write an email...", "write a poem..."
3. Browser Automation: "go to github", "search google for weather in Tokyo", "open a new tab", "close tab"
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from wisprhelp.core.router import RouteTarget
from wisprhelp.core.runtime import RuntimeCoordinator
from wisprhelp.hal.base import BaseOSAdapter, WindowInfo


class MockOSAdapter(BaseOSAdapter):
    """Deterministic mock for OS interactions."""

    def __init__(self) -> None:
        self.launched_apps: list[str] = []
        self.typed_texts: list[str] = []
        self.hotkeys: list[list[str]] = []
        self.pressed_keys: list[str] = []
        self.scrolls: list[tuple[str, int]] = []
        self.clicks: list[tuple[int | None, int | None, str]] = []
        self.active_window = WindowInfo(
            title="Google Chrome",
            process_name="chrome.exe",
            pid=123,
            rect=(0, 0, 1920, 1080),
        )

    def get_active_window(self) -> WindowInfo:
        return self.active_window

    def get_open_windows(self) -> list[WindowInfo]:
        return [self.active_window]

    def launch_app(self, app_identifier: str, parameter: str = None) -> bool:
        self.launched_apps.append(app_identifier)
        return True

    def inject_unicode_text(self, text: str) -> bool:
        self.typed_texts.append(text)
        return True

    def key_combination(self, keys: list[str]) -> None:
        self.hotkeys.append(keys)

    def press_key(self, key: str) -> None:
        self.pressed_keys.append(key)

    def scroll(self, direction: str = "down", amount: int = 3) -> None:
        self.scrolls.append((direction, amount))

    def media_control(self, command: str) -> bool:
        return True

    def find_accessibility_element(self, query: str, control_type: str = None) -> tuple[int, int] | None:
        return None

    def find_ordinal_element(
        self, ordinal_index: int, item_type: str = "item", target_app: str = None
    ) -> tuple[int, int] | None:
        return (220, 140)

    def move_mouse(self, x: int, y: int, smooth: bool = True) -> None:
        pass

    def click(self, x: int = None, y: int = None, button: str = "left", clicks: int = 1) -> None:
        self.clicks.append((x, y, button))

    def wait_for_active_window(self, titles_or_processes: list[str], timeout: float = 2.0) -> bool:
        return True


class TestHumanConversationAndAutomation(unittest.TestCase):
    """Tests natural conversation, generative writing, and browser automation."""

    def setUp(self) -> None:
        self.os_adapter = MockOSAdapter()
        # Initialize runtime with mock OS and offline planner to ensure instant execution
        from wisprhelp.core.groq_planner import GroqStrategicPlanner
        self.mock_planner = GroqStrategicPlanner(api_key="")
        self.coordinator = RuntimeCoordinator(
            os_adapter=self.os_adapter,
            planner=self.mock_planner,
        )

    def test_normal_conversation_who_are_you(self) -> None:
        """Conversational query asking who the assistant is."""
        res = self.coordinator.process_command("Hello, who are you?")
        self.assertTrue(res.success)
        self.assertEqual(res.route, RouteTarget.CONVERSATIONAL_QUERY)
        self.assertIn("WisprHelp", res.message)

    def test_normal_conversation_capabilities(self) -> None:
        """Conversational query asking what the assistant can do."""
        res = self.coordinator.process_command("What can you do?")
        self.assertTrue(res.success)
        self.assertEqual(res.route, RouteTarget.CONVERSATIONAL_QUERY)
        self.assertTrue(len(res.message) > 10)

    def test_normal_conversation_joke(self) -> None:
        """Conversational query asking for a joke."""
        res = self.coordinator.process_command("Tell me a programmer joke")
        self.assertTrue(res.success)
        self.assertEqual(res.route, RouteTarget.CONVERSATIONAL_QUERY)
        self.assertIn("bugs", res.message.lower())

    def test_generative_writing_something_about_ai(self) -> None:
        """User asks to 'write something about AI' -> should generate content and type it."""
        res = self.coordinator.process_command("write something about AI")
        self.assertTrue(res.success)
        self.assertEqual(res.route, RouteTarget.GUI_WORKFLOW_AUTOMATION)
        # Should have injected text into the active window
        self.assertTrue(len(self.os_adapter.typed_texts) > 0)
        # The typed text must NOT be literally "something about AI"
        typed = self.os_adapter.typed_texts[-1]
        self.assertNotEqual(typed, "something about AI")
        self.assertTrue(len(typed) > 20)

    def test_generative_writing_email_composition(self) -> None:
        """User asks to 'write an email to Sarah' -> should generate email format and type it."""
        res = self.coordinator.process_command("write an email to Sarah saying I'll be 10 minutes late")
        self.assertTrue(res.success)
        self.assertEqual(res.route, RouteTarget.GUI_WORKFLOW_AUTOMATION)
        self.assertTrue(len(self.os_adapter.typed_texts) > 0)
        typed = self.os_adapter.typed_texts[-1]
        self.assertIn("Subject:", typed)

    def test_browser_automation_go_to_website(self) -> None:
        """User asks to 'go to github' -> should navigate to github.com using address bar."""
        res = self.coordinator.process_command("go to github")
        self.assertTrue(res.success)
        self.assertIn(["ctrl", "l"], self.os_adapter.hotkeys)
        self.assertIn("github.com", self.os_adapter.typed_texts)
        self.assertIn("enter", self.os_adapter.pressed_keys)

    def test_browser_automation_search_google(self) -> None:
        """User asks to 'search google for weather in Tokyo' -> should focus address bar and search."""
        res = self.coordinator.process_command("search google for weather in Tokyo")
        self.assertTrue(res.success)
        self.assertIn(["ctrl", "l"], self.os_adapter.hotkeys)
        self.assertIn("weather in Tokyo", self.os_adapter.typed_texts)
        self.assertIn("enter", self.os_adapter.pressed_keys)

    def test_browser_automation_new_tab(self) -> None:
        """User asks to 'open a new tab' -> should press Ctrl+T."""
        res = self.coordinator.process_command("open a new tab")
        self.assertTrue(res.success)
        self.assertIn(["ctrl", "t"], self.os_adapter.hotkeys)

    def test_browser_automation_close_tab(self) -> None:
        """User asks to 'close this tab' -> should press Ctrl+W."""
        res = self.coordinator.process_command("close this tab")
        self.assertTrue(res.success)
        self.assertIn(["ctrl", "w"], self.os_adapter.hotkeys)

    def test_browser_automation_refresh_page(self) -> None:
        """User asks to 'refresh page' -> should press Ctrl+R."""
        res = self.coordinator.process_command("refresh page")
        self.assertTrue(res.success)
        self.assertIn(["ctrl", "r"], self.os_adapter.hotkeys)

    def test_browser_automation_scroll_down(self) -> None:
        """User asks to 'scroll down' -> should dispatch scroll or pagedown."""
        res = self.coordinator.process_command("scroll down")
        self.assertTrue(res.success)
        self.assertTrue(len(self.os_adapter.scrolls) > 0 or "pagedown" in self.os_adapter.pressed_keys)

    def test_whatsapp_open_first_chat(self) -> None:
        """User asks to 'Open WhatsApp and open the first chat which is appearing on the tab'."""
        res = self.coordinator.process_command("Open WhatsApp and open the first chat which is appearing on the tab")
        self.assertTrue(res.success)
        self.assertTrue(any("whatsapp" in app.lower() for app in self.os_adapter.launched_apps))
        self.assertTrue(len(self.os_adapter.clicks) > 0)
        cx, cy, _ = self.os_adapter.clicks[-1]
        self.assertEqual((cx, cy), (220, 140))


if __name__ == "__main__":
    unittest.main()
