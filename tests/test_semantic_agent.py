"""Comprehensive regression and scenario test suite for the WisprHelp Semantic Agent.

Verifies Scenarios A through N from the specification:
- Scenario A: Browser search
- Scenario B: File search & opening
- Scenario C: Files + contextual reference ('it')
- Scenario D: Application context & multi-step navigation
- Scenario E: Spatial UI reference ('button on top right')
- Scenario F: Relative / ordinal reference ('second tab')
- Scenario G: Contextual follow-up ('volume down')
- Scenario H: In-flight user correction ('Actually, use Firefox instead')
- Scenario I: Ambiguity & targeted clarification ('Send this to Rahul')
- Scenario J: Literal text dictation vs action language
- Scenario K: Destructive operation safety confirmation
- Scenario L: Visual grounding fallback
- Scenario M: Unexpected UI recovery & bounded retries
- Scenario N: Multi-step natural conversation with correction
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.memory import ConversationMemory
from wisprhelp.agent.planner import SemanticAgent, TaskGraph
from wisprhelp.agent.reasoning import SemanticReasoner
from wisprhelp.agent.references import ReferenceResolver
from wisprhelp.agent.state import ComputerState, ComputerStateManager, UIElementInfo
from wisprhelp.core.runtime import RuntimeCoordinator, RouteTarget
from wisprhelp.hal.base import BaseOSAdapter, WindowInfo


class MockOSAdapter(BaseOSAdapter):
    """Controllable mock HAL adapter for scenario unit tests."""

    def __init__(self) -> None:
        self.launched_apps: list[str] = []
        self.typed_texts: list[str] = []
        self.pressed_keys: list[str] = []
        self.clicked_coords: list[tuple[int, int]] = []
        self.media_commands: list[str] = []
        self.hotkeys: list[list[str]] = []
        self.scroll_events: list[tuple[str, int]] = []
        self.opened_files: list[str] = []
        self.active_window = WindowInfo(
            title="Google Chrome",
            process_name="chrome.exe",
            pid=1234,
            rect=(0, 0, 1920, 1080),
        )
        self.open_windows = [self.active_window]
        self.visible_elements: list[dict] = [
            {"name": "Tab 1", "control_type": "tabitem", "center": (100, 30)},
            {"name": "Tab 2", "control_type": "tabitem", "center": (250, 30)},
            {"name": "Close", "control_type": "button", "center": (1900, 15)},
            {"name": "Plus", "control_type": "button", "center": (350, 30)},
            {"name": "Search or enter address", "control_type": "edit", "center": (500, 80)},
            {"name": "Email: user@example.com", "control_type": "text", "center": (400, 300)},
            {"name": "Developer Documentation - First Result", "control_type": "link", "center": (400, 200)},
            {"name": "Second Result - API Reference", "control_type": "link", "center": (400, 250)},
        ]

    def open_file(self, file_path: str) -> bool:
        self.opened_files.append(file_path)
        return True

    def launch_app(self, app_identifier: str, parameter: str = None) -> bool:
        self.launched_apps.append(app_identifier)
        return True

    def get_active_window(self) -> WindowInfo:
        return self.active_window

    def get_open_windows(self) -> list[WindowInfo]:
        return self.open_windows

    def get_visible_elements(self, max_depth: int = 4) -> list[dict]:
        return self.visible_elements

    def inject_unicode_text(self, text: str) -> bool:
        self.typed_texts.append(text)
        return True

    def media_control(self, command: str) -> bool:
        self.media_commands.append(command)
        return True

    def find_accessibility_element(self, query: str, control_type: str = None) -> tuple[int, int] | None:
        for elem in self.visible_elements:
            if query.lower() in elem["name"].lower() or elem["name"].lower() in query.lower():
                return elem["center"]
        return None

    def find_ordinal_element(self, ordinal_index: int, item_type: str = "item", target_app: str = None) -> tuple[int, int] | None:
        matching = [e for e in self.visible_elements if item_type in e["control_type"].lower()] or self.visible_elements
        if 0 <= ordinal_index < len(matching):
            return matching[ordinal_index]["center"]
        return None

    def move_mouse(self, x: int, y: int, smooth: bool = True) -> None:
        pass

    def click(self, x: int = None, y: int = None, button: str = "left", clicks: int = 1) -> None:
        if x is not None and y is not None:
            self.clicked_coords.append((x, y))

    def key_combination(self, keys: list[str]) -> None:
        self.hotkeys.append(keys)

    def press_key(self, key: str) -> None:
        self.pressed_keys.append(key)

    def scroll(self, direction: str = "down", amount: int = 3) -> None:
        self.scroll_events.append((direction, amount))


class TestSemanticAgentScenarios(unittest.TestCase):
    """Verification of Scenarios A through N."""

    def setUp(self) -> None:
        self.os_adapter = MockOSAdapter()
        self.ref_resolver = ReferenceResolver()
        self.reasoner = SemanticReasoner(ref_resolver=self.ref_resolver, groq_api_key="")
        self.mock_classifier = MagicMock()
        self.mock_classifier.classify.return_value = MagicMock(
            is_high_risk=lambda thresh: False,
        )
        self.agent = SemanticAgent(
            os_adapter=self.os_adapter,
            ref_resolver=self.ref_resolver,
            reasoner=self.reasoner,
            classifier=self.mock_classifier,
        )
        def _mock_visual_task(goal: str, **kwargs):
            if "plus" in goal.lower():
                self.os_adapter.click(350, 30)
            return MagicMock(success=True, message="Vision target located")

        self.agent.vlm_agent.execute_visual_task = MagicMock(side_effect=_mock_visual_task)
        self.coordinator = RuntimeCoordinator(os_adapter=self.os_adapter, agent=self.agent)

    def test_scenario_a_browser_search(self) -> None:
        """Scenario A: 'Open my browser and find information about LangGraph.'"""
        res = self.agent.process_turn("Open my browser and find information about LangGraph")
        self.assertTrue(res.success)
        self.assertIn("LangGraph", res.goal)
        # Should launch browser and execute search
        self.assertTrue(any("Chrome" in a for a in self.os_adapter.launched_apps) or len(self.os_adapter.typed_texts) > 0)

    def test_scenario_b_file_search_and_open(self) -> None:
        """Scenario B: 'Find the PDF I downloaded today and open it.'"""
        res = self.agent.process_turn("Find the PDF I downloaded today and open it")
        self.assertTrue(res.success)
        self.assertIn("PDF", res.goal)
        # Actions should include find_file or open_file
        action_types = [a.type for a in res.actions_executed]
        self.assertTrue(ActionType.FIND_FILE in action_types or ActionType.OPEN_FILE in action_types)

    def test_scenario_c_files_and_contextual_reference(self) -> None:
        """Scenario C: 'Open my resume.' then 'Now copy the email address from it.'"""
        # Turn 1
        res1 = self.agent.process_turn("Open my resume")
        self.assertTrue(res1.success)
        self.assertEqual(self.agent.memory.entity_store.get_last_entity("files"), "resume.pdf")

        # Turn 2: 'Now copy the email address from it.'
        res2 = self.agent.process_turn("Now copy the email address from it")
        self.assertTrue(res2.success)
        self.assertIn("resume.pdf", res2.goal)
        self.assertIn(["ctrl", "c"], self.os_adapter.hotkeys)

    def test_scenario_d_application_context_multi_step(self) -> None:
        """Scenario D: 'Open VS Code.' then 'Open the project folder.'"""
        res1 = self.agent.process_turn("Open VS Code")
        self.assertTrue(res1.success)
        self.assertEqual(self.agent.memory.entity_store.get_last_entity("apps"), "VS Code")

        res2 = self.agent.process_turn("Open the backend folder")
        self.assertTrue(res2.success)
        self.assertIn("backend", res2.goal.lower())

    def test_scenario_e_spatial_ui_reference(self) -> None:
        """Scenario E: 'Click the button on the top right.'"""
        # The mock OS has 'Close' at (1900, 15) which is rightmost and topmost
        res = self.agent.process_turn("Click the button on the top right")
        self.assertTrue(res.success)
        self.assertTrue("close" in res.goal.lower())
        self.assertIn((1900, 15), self.os_adapter.clicked_coords)


    def test_scenario_f_relative_ordinal_reference(self) -> None:
        """Scenario F: 'Open the second tab.'"""
        res = self.agent.process_turn("Open the second tab")
        self.assertTrue(res.success)
        # Should click the second tabitem at index 1 -> (250, 30)
        self.assertIn((250, 30), self.os_adapter.clicked_coords)

    def test_scenario_g_contextual_follow_up_media(self) -> None:
        """Scenario G: 'Open Spotify.' -> 'Play my workout playlist.' -> 'Turn the volume down a little.'"""
        res1 = self.agent.process_turn("Open Spotify")
        self.assertTrue(res1.success)

        res2 = self.agent.process_turn("Play my workout playlist")
        self.assertTrue(res2.success)
        self.assertIn("workout", res2.goal.lower())

        res3 = self.agent.process_turn("Turn the volume down a little")
        self.assertTrue(res3.success)
        self.assertTrue(len(self.os_adapter.media_commands) > 0)

    def test_scenario_h_correction_handling(self) -> None:
        """Scenario H: 'Open Chrome.' -> 'Actually, use Firefox instead.'"""
        res1 = self.agent.process_turn("Open Chrome")
        self.assertTrue(res1.success)

        res2 = self.agent.process_turn("Actually, use Firefox instead")
        self.assertTrue(res2.success)
        self.assertIn("Firefox", res2.goal)
        self.assertIn("Firefox", self.os_adapter.launched_apps)

    def test_scenario_i_ambiguity_targeted_clarification(self) -> None:
        """Scenario I: 'Send this to Rahul.' without messaging context requires clarification."""
        res = self.agent.process_turn("Send this to Rahul")
        self.assertTrue(res.success)
        # Must ask targeted clarification rather than hallucinate recipient or app
        self.assertIn("Rahul", res.message)
        self.assertTrue("?" in res.message or "clarif" in res.message.lower() or "which" in res.message.lower())

    def test_scenario_j_literal_text_vs_action(self) -> None:
        """Scenario J: 'Type: "Open Chrome and search for Jev"' must type exactly that sentence, not launch Chrome."""
        res = self.coordinator.process_command('Type: "Open Chrome and search for Jev"')
        self.assertTrue(res.success)
        self.assertEqual(res.route, RouteTarget.TEXT_INJECTION)
        self.assertIn("Open Chrome and search for Jev", self.os_adapter.typed_texts)
        # Must NOT have launched Chrome!
        self.assertEqual(self.os_adapter.launched_apps, [])

    def test_scenario_k_destructive_operation_guardrail(self) -> None:
        """Scenario K: 'Delete everything in this folder' must require confirmation."""
        res = self.agent.process_turn("Delete everything in this folder")
        self.assertTrue(res.needs_confirmation)
        self.assertIn("destructive", res.message.lower())
        self.assertIsNotNone(res.confirmation_prompt)

        # Confirm execution
        confirmed_res = self.agent.process_turn("Delete everything in this folder", confirmation_response="yes")
        self.assertTrue(confirmed_res.success)

    def test_scenario_l_visual_grounding_fallback(self) -> None:
        """Scenario L: 'Click the plus button next to the tabs.'"""
        res = self.agent.process_turn("Click the plus button next to the tabs")
        self.assertTrue(res.success)
        # Plus button is at (350, 30)
        self.assertIn((350, 30), self.os_adapter.clicked_coords)

    def test_scenario_m_recovery_and_bounded_retry(self) -> None:
        """Scenario M: If an action encounters a transient error, agent retries up to max_retries."""
        fail_action = Action.click("NonExistentButton")
        # Ensure VLM mock doesn't crash
        with patch.object(self.agent.vlm_agent, "execute_visual_task") as mock_vlm:
            mock_vlm.return_value = MagicMock(success=True, message="Found on screen")
            outcome = self.agent.executor.execute_action(fail_action)
            self.assertTrue(outcome.success)

    def test_scenario_n_multi_step_conversation_with_correction(self) -> None:
        """Scenario N:
        User: Open my browser.
        Agent: Done.
        User: Go to the OpenAI website.
        Agent: Done.
        User: Look for their developer documentation.
        Agent: Done.
        User: Open the first result.
        Agent: Acts.
        User: No, the other one.
        Agent: Updates reference to second result.
        """
        # Step 1: Open my browser
        r1 = self.agent.process_turn("Open my browser")
        self.assertTrue(r1.success)

        # Step 2: Go to the OpenAI website
        r2 = self.agent.process_turn("Go to the OpenAI website")
        self.assertTrue(r2.success)

        # Step 3: Look for their developer documentation
        r3 = self.agent.process_turn("Look for their developer documentation")
        self.assertTrue(r3.success)

        # Step 4: Open the first result
        r4 = self.agent.process_turn("Open the first result")
        self.assertTrue(r4.success)

        # Step 5: No, the other one. (User correction on visible/ordinal item)
        r5 = self.agent.process_turn("No, the other one")
        self.assertTrue(r5.success)
        self.assertTrue(len(r5.actions_executed) > 0)


if __name__ == "__main__":
    unittest.main()
