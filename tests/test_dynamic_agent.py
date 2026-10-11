"""End-to-end tests for the LLM-first dynamic cognitive pipeline.

Verifies the dynamic OS-agent behaviors:
- Compound + conditional workflows (Notepad: launch -> clear-if-content -> type)
- Chrome search automation
- Ordinal UI targets ("first chat")
- Conversational replies with no OS actions
- Low-confidence clarification gating
- Destructive safety hold + user confirmation
- Replan-on-surprise closed-loop recovery
- Deictic multi-turn follow-ups grounded in live screen inventory

The Groq LLM is mocked at the HTTP layer; the offline plan classifier
(keyless) is injected so no real Jev calls are made.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.planner import SemanticAgent
from wisprhelp.agent.plan_classifier import PlanClassifier
from wisprhelp.agent.reasoning import SemanticReasoner
from wisprhelp.agent.references import ReferenceResolver
from wisprhelp.classification.client import JevClient
from wisprhelp.hal.base import BaseOSAdapter, WindowInfo


def keyless_plan_classifier() -> PlanClassifier:
    """PlanClassifier guaranteed to make no real Jev API calls."""
    client = JevClient()
    client.api_key = ""
    return PlanClassifier(client=client)


class DynamicMockOSAdapter(BaseOSAdapter):
    """Mock HAL with clipboard simulation for conditional workflow tests."""

    def __init__(self) -> None:
        self.launched_apps: list[str] = []
        self.typed_texts: list[str] = []
        self.pressed_keys: list[str] = []
        self.clicked_coords: list[tuple[int, int]] = []
        self.hotkeys: list[list[str]] = []
        self.launch_succeeds = True
        # Simulated active document + clipboard content
        self.document_content = ""
        self.clipboard = "user clipboard content"
        self.active_window = WindowInfo(
            title="Untitled - Notepad",
            process_name="notepad.exe",
            pid=99,
            rect=(0, 0, 1920, 1080),
        )
        self.visible_elements: list[dict] = [
            {"name": "File", "control_type": "menuitem", "center": (30, 20)},
            {"name": "Rahul chat", "control_type": "listitem", "center": (300, 200)},
            {"name": "Priya chat", "control_type": "listitem", "center": (300, 260)},
            {"name": "Type a message", "control_type": "edit", "center": (600, 900)},
            {"name": "OpenCode - Official Site", "control_type": "link", "center": (400, 300)},
            {"name": "OpenCode GitHub Repository", "control_type": "link", "center": (400, 360)},
            {"name": "Delete all", "control_type": "button", "center": (700, 500)},
        ]

    # --- HAL interface ---
    def launch_app(self, app_identifier: str, parameter: str = None) -> bool:
        if not self.launch_succeeds:
            return False
        self.launched_apps.append(app_identifier)
        self.active_window = WindowInfo(app_identifier, app_identifier + ".exe", 99, (0, 0, 1920, 1080))
        return True

    def get_active_window(self) -> WindowInfo:
        return self.active_window

    def get_open_windows(self) -> list[WindowInfo]:
        return [self.active_window]

    def get_visible_elements(self, max_depth: int = 4) -> list[dict]:
        return self.visible_elements

    def inject_unicode_text(self, text: str) -> bool:
        self.typed_texts.append(text)
        return True

    def read_focused_text(self):
        return self.document_content

    def media_control(self, command: str) -> bool:
        return True

    def find_accessibility_element(self, query: str, control_type: str = None):
        for elem in self.visible_elements:
            if query.lower() in elem["name"].lower():
                return elem["center"]
        return None

    def find_ordinal_element(self, ordinal_index: int, item_type: str = "item", target_app: str = None):
        matching = [
            e
            for e in self.visible_elements
            if item_type in e["control_type"].lower() or item_type in e["name"].lower()
        ]
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
        # Simulate copy: copying the (selected) document puts its content on the clipboard
        if [k.lower() for k in keys] in (["ctrl", "c"], ["command", "c"]):
            self.clipboard = self.document_content

    def press_key(self, key: str) -> None:
        self.pressed_keys.append(key)
        if key.lower() in ("delete", "backspace"):
            self.document_content = ""

    def scroll(self, direction: str = "down", amount: int = 3) -> None:
        pass

    def get_clipboard_text(self) -> str:
        return self.clipboard

    def set_clipboard_text(self, text: str) -> bool:
        self.clipboard = text
        return True


def make_groq_response(payload: dict) -> MagicMock:
    """Builds a fake Groq chat-completions HTTP response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    return resp


NOTEPAD_PLAN = {
    "primary_goal": "Open Notepad, clear any existing content, and type 'hey there'",
    "is_conversational": False,
    "reply": None,
    "needs_clarification": False,
    "clarification_question": None,
    "target_app": "Notepad",
    "steps": [
        {
            "action": "launch_app",
            "target": "Notepad",
            "payload": None,
            "params": {},
            "condition": None,
            "is_destructive": False,
            "confidence": 0.98,
            "rationale": "Launch Notepad",
        },
        {
            "action": "clear_text",
            "target": "active_document",
            "payload": None,
            "params": {},
            "condition": "document_has_content",
            "is_destructive": False,
            "confidence": 0.95,
            "rationale": "Clean any existing content if present",
        },
        {
            "action": "type_text",
            "target": "active_document",
            "payload": "hey there",
            "params": {"press_enter": False},
            "condition": None,
            "is_destructive": False,
            "confidence": 0.99,
            "rationale": "Type the greeting",
        },
    ],
}

CHROME_SEARCH_PLAN = {
    "primary_goal": "Open Chrome and search for OpenCode",
    "is_conversational": False,
    "reply": None,
    "needs_clarification": False,
    "clarification_question": None,
    "target_app": "Chrome",
    "steps": [
        {
            "action": "launch_app",
            "target": "Chrome",
            "payload": None,
            "params": {},
            "condition": None,
            "is_destructive": False,
            "confidence": 0.97,
            "rationale": "Launch the browser",
        },
        {
            "action": "search",
            "target": "Chrome",
            "payload": "OpenCode",
            "params": {},
            "condition": None,
            "is_destructive": False,
            "confidence": 0.96,
            "rationale": "Search the web for OpenCode",
        },
    ],
}

FIRST_CHAT_PLAN = {
    "primary_goal": "Open the first chat in the active messaging app",
    "is_conversational": False,
    "reply": None,
    "needs_clarification": False,
    "clarification_question": None,
    "target_app": None,
    "steps": [
        {
            "action": "click",
            "target": "ordinal:0:chat",
            "payload": None,
            "params": {},
            "condition": None,
            "is_destructive": False,
            "confidence": 0.93,
            "rationale": "Click the first chat in the list",
        },
        {
            "action": "click",
            "target": "Type a message",
            "payload": None,
            "params": {},
            "condition": None,
            "is_destructive": False,
            "confidence": 0.94,
            "rationale": "Focus the message input field",
        },
    ],
}

DESTRUCTIVE_PLAN = {
    "primary_goal": "Delete the contents of the Downloads folder",
    "is_conversational": False,
    "reply": None,
    "needs_clarification": False,
    "clarification_question": None,
    "target_app": None,
    "steps": [
        {
            "action": "click",
            "target": "Delete all",
            "payload": None,
            "params": {},
            "condition": None,
            "is_destructive": True,
            "confidence": 0.92,
            "rationale": "Delete everything - destructive",
        },
    ],
}

LOW_CONFIDENCE_PLAN = {
    "primary_goal": "Do that thing",
    "is_conversational": False,
    "reply": None,
    "needs_clarification": False,
    "clarification_question": None,
    "target_app": None,
    "steps": [
        {
            "action": "click",
            "target": "maybe-this",
            "payload": None,
            "params": {},
            "condition": None,
            "is_destructive": False,
            "confidence": 0.30,
            "rationale": "Unsure what the user means",
        },
    ],
}

CLARIFICATION_PLAN = {
    "primary_goal": "Send something somewhere",
    "is_conversational": False,
    "reply": None,
    "needs_clarification": True,
    "clarification_question": "Which application should I use to send this?",
    "target_app": None,
    "steps": [],
}

CONVERSATIONAL_PLAN = {
    "primary_goal": "Answer the capital-of-France question",
    "is_conversational": True,
    "reply": "The capital of France is Paris.",
    "needs_clarification": False,
    "clarification_question": None,
    "target_app": None,
    "steps": [],
}


class TestDynamicAgentWorkflows(unittest.TestCase):
    """LLM-first dynamic workflow scenarios with mocked Groq + keyless Jev."""

    def setUp(self) -> None:
        self.os_adapter = DynamicMockOSAdapter()
        self.reasoner = SemanticReasoner(
            ref_resolver=ReferenceResolver(),
            groq_api_key="test-llm-key",
        )
        self.offline_classifier = keyless_plan_classifier()
        self.mock_jev = MagicMock()
        self.mock_jev.classify.return_value = MagicMock(is_high_risk=lambda t: False)
        self.agent = SemanticAgent(
            os_adapter=self.os_adapter,
            reasoner=self.reasoner,
            classifier=self.mock_jev,
            plan_classifier=self.offline_classifier,
        )
        self.agent.vlm_agent.execute_visual_task = MagicMock(
            return_value=MagicMock(success=False, message="vision unavailable")
        )
        # Default: replan returns nothing (tests override when testing recovery).
        # Prevents accidental real network calls if a step unexpectedly fails.
        self.replan_patcher = patch.object(
            self.reasoner, "replan", return_value=[]
        )
        self.replan_patcher.start()
        self.addCleanup(self.replan_patcher.stop)

    def _run_with_llm(self, plan: dict, query: str):
        """Runs one agent turn with the Groq LLM mocked to return `plan`."""
        with patch(
            "wisprhelp.agent.reasoning.requests.post",
            return_value=make_groq_response(plan),
        ) as mock_post:
            res = self.agent.process_turn(query)
            mock_post.assert_called_once()
            return res

    # ------------------------------------------------------------------
    # Scenario 1: Notepad conditional workflow (the headline use case)
    # ------------------------------------------------------------------

    def test_notepad_clear_condition_met_then_type(self) -> None:
        """Doc has content -> clear executes (select-all + delete) then types."""
        self.os_adapter.document_content = "old junk text"
        res = self._run_with_llm(NOTEPAD_PLAN, "Open Notepad. If there is something in it, clean it and type hey there")

        self.assertTrue(res.needs_confirmation)
        self.assertEqual(self.os_adapter.document_content, "old junk text")
        res = self.agent.process_turn("yes")
        self.assertTrue(res.success)
        self.assertIn("Notepad", self.os_adapter.launched_apps)
        self.assertIn(["ctrl", "a"], self.os_adapter.hotkeys)
        # Clear executed: delete pressed
        self.assertIn("delete", self.os_adapter.pressed_keys)
        # Greeting typed
        self.assertIn("hey there", self.os_adapter.typed_texts)
        # Clipboard restored after the document probe
        self.assertEqual(self.os_adapter.clipboard, "user clipboard content")

    def test_notepad_condition_not_met_skips_clear(self) -> None:
        """Empty document -> clear step is skipped but typing still happens."""
        self.os_adapter.document_content = ""
        res = self._run_with_llm(NOTEPAD_PLAN, "Open Notepad, clean it if needed, and type hey there")

        self.assertTrue(res.success)
        self.assertIn("Notepad", self.os_adapter.launched_apps)
        # Accessibility inspected content without changing clipboard or selection.
        self.assertNotIn(["ctrl", "a"], self.os_adapter.hotkeys)
        # ...but the clear itself was skipped: no delete pressed
        self.assertNotIn("delete", self.os_adapter.pressed_keys)
        # Typing still executed
        self.assertIn("hey there", self.os_adapter.typed_texts)

    # ------------------------------------------------------------------
    # Scenario 2: Chrome compound search workflow
    # ------------------------------------------------------------------

    def test_chrome_search_workflow(self) -> None:
        res = self._run_with_llm(CHROME_SEARCH_PLAN, "Open Chrome browser and search for Open code")

        self.assertTrue(res.success)
        self.assertIn("Chrome", self.os_adapter.launched_apps)
        # Search executed via address bar: Ctrl+L, typed query, Enter
        self.assertIn(["ctrl", "l"], self.os_adapter.hotkeys)
        self.assertIn("OpenCode", self.os_adapter.typed_texts)
        self.assertIn("enter", self.os_adapter.pressed_keys)

    # ------------------------------------------------------------------
    # Scenario 3: Ordinal UI interaction ("first chat") + focus-before-type
    # ------------------------------------------------------------------

    def test_first_chat_ordinal_and_focus(self) -> None:
        res = self._run_with_llm(FIRST_CHAT_PLAN, "Open the first chat and write something there")

        self.assertTrue(res.success)
        # ordinal:0:chat -> first chat listitem (Rahul chat at (300, 200))
        self.assertIn((300, 200), self.os_adapter.clicked_coords)
        # Then focused the message input at (600, 900)
        self.assertIn((600, 900), self.os_adapter.clicked_coords)

    # ------------------------------------------------------------------
    # Scenario 4: Conversational reply without OS actions
    # ------------------------------------------------------------------

    def test_conversational_answer_no_os_actions(self) -> None:
        res = self._run_with_llm(CONVERSATIONAL_PLAN, "What is the capital of France?")

        self.assertTrue(res.success)
        self.assertIn("Paris", res.message)
        self.assertEqual(self.os_adapter.launched_apps, [])
        self.assertEqual(self.os_adapter.clicked_coords, [])
        self.assertEqual(self.os_adapter.typed_texts, [])

    # ------------------------------------------------------------------
    # Scenario 5: Clarification flows
    # ------------------------------------------------------------------

    def test_llm_requested_clarification(self) -> None:
        res = self._run_with_llm(CLARIFICATION_PLAN, "Send this")
        self.assertTrue(res.success)
        self.assertIn("Which application", res.message)
        self.assertEqual(self.os_adapter.clicked_coords, [])

    def test_low_confidence_step_blocks_execution(self) -> None:
        """Weak-confidence plans must not execute; user is asked to clarify."""
        res = self._run_with_llm(LOW_CONFIDENCE_PLAN, "Do that thing")
        self.assertFalse(res.success)
        self.assertIn("clarif", res.message.lower())
        self.assertEqual(self.os_adapter.clicked_coords, [])

    # ------------------------------------------------------------------
    # Scenario 6: Destructive safety interlock + confirmation
    # ------------------------------------------------------------------

    def test_destructive_step_holds_for_confirmation(self) -> None:
        res = self._run_with_llm(DESTRUCTIVE_PLAN, "Delete everything in downloads")
        self.assertTrue(res.needs_confirmation)
        self.assertIn("destructive", res.message.lower())
        # Nothing executed yet
        self.assertEqual(self.os_adapter.clicked_coords, [])

        # User confirms -> held action executes
        confirmed = self.agent.process_turn("Delete everything in downloads", confirmation_response="yes")
        self.assertTrue(confirmed.success)
        self.assertEqual(len(self.os_adapter.clicked_coords), 1)

    def test_spoken_confirmation_resolves_held_action(self) -> None:
        """Voice-mode flow: the NEXT spoken utterance ('Yes, we do') must resolve
        the pending safety hold as an affirmation - not be parsed as a new command."""
        res = self._run_with_llm(DESTRUCTIVE_PLAN, "Delete everything in downloads")
        self.assertTrue(res.needs_confirmation)

        # User speaks a natural affirmative - NO explicit confirmation_response param
        confirmed = self.agent.process_turn("Yes, we do")
        self.assertTrue(confirmed.success)
        # Must NOT have been misinterpreted as a fresh workflow: exactly one click
        self.assertEqual(len(self.os_adapter.clicked_coords), 1)

    def test_spoken_rejection_cancels_held_action(self) -> None:
        """A spoken 'no' cancels the held destructive action."""
        res = self._run_with_llm(DESTRUCTIVE_PLAN, "Delete everything in downloads")
        self.assertTrue(res.needs_confirmation)

        rejected = self.agent.process_turn("No, don't do that")
        self.assertFalse(rejected.success)
        self.assertIn("cancelled", rejected.message.lower())
        self.assertEqual(self.os_adapter.clicked_coords, [])

    def test_new_command_supersedes_held_action(self) -> None:
        """A non-confirmation utterance after a hold drops the held action and
        processes the new command normally."""
        res = self._run_with_llm(DESTRUCTIVE_PLAN, "Delete everything in downloads")
        self.assertTrue(res.needs_confirmation)

        superseding = {
            "primary_goal": "Open Notepad",
            "is_conversational": False,
            "reply": None,
            "needs_clarification": False,
            "clarification_question": None,
            "target_app": "Notepad",
            "steps": [
                {
                    "action": "launch_app",
                    "target": "Notepad",
                    "payload": None,
                    "params": {},
                    "condition": None,
                    "is_destructive": False,
                    "confidence": 0.97,
                    "rationale": "Launch Notepad",
                },
            ],
        }
        with patch(
            "wisprhelp.agent.reasoning.requests.post",
            return_value=make_groq_response(superseding),
        ):
            new_res = self.agent.process_turn("Open Notepad")

        self.assertTrue(new_res.success)
        # Held destructive action dropped - nothing clicked
        self.assertEqual(self.os_adapter.clicked_coords, [])
        # New command executed instead
        self.assertIn("Notepad", self.os_adapter.launched_apps)
        # The pending hold is cleared so future utterances aren't misread
        self.assertIsNone(self.agent._pending_held_action)

    # ------------------------------------------------------------------
    # Scenario 7: Replan-on-surprise closed-loop recovery
    # ------------------------------------------------------------------

    def test_replan_after_step_failure(self) -> None:
        """A failing step retries, then triggers LLM replan with recovery steps."""
        plan = {
            "primary_goal": "Open GhostApp",
            "is_conversational": False,
            "reply": None,
            "needs_clarification": False,
            "clarification_question": None,
            "target_app": "GhostApp",
            "steps": [
                {
                    "action": "launch_app",
                    "target": "GhostApp",
                    "payload": None,
                    "params": {},
                    "condition": None,
                    "is_destructive": False,
                    "confidence": 0.95,
                    "rationale": "Launch the app",
                },
            ],
        }
        self.os_adapter.launch_succeeds = False
        recovery = [Action.respond("Recovered by falling back to a conversational reply")]
        with patch(
            "wisprhelp.agent.reasoning.requests.post",
            return_value=make_groq_response(plan),
        ):
            with patch.object(self.reasoner, "replan", return_value=recovery) as mock_replan:
                res = self.agent.process_turn("Open GhostApp")

        mock_replan.assert_called_once()
        self.assertFalse(res.success)
        self.assertEqual(res.failure_code.value, "target_app_missing")
        # A conversational fallback does not complete an OS launch task.
        self.assertLessEqual(len(res.actions_executed), 3)

    # ------------------------------------------------------------------
    # Scenario 8: Deictic multi-turn follow-up with live screen grounding
    # ------------------------------------------------------------------

    def test_multi_turn_deictic_follow_up(self) -> None:
        """Turn 1 searches; turn 2 'click the first result' grounds via UIA."""
        with patch(
            "wisprhelp.agent.reasoning.requests.post",
            return_value=make_groq_response(CHROME_SEARCH_PLAN),
        ):
            r1 = self.agent.process_turn("Open Chrome and search for Open code")
        self.assertTrue(r1.success)

        follow_up = {
            "primary_goal": "Click the OpenCode official site result",
            "is_conversational": False,
            "reply": None,
            "needs_clarification": False,
            "clarification_question": None,
            "target_app": "Chrome",
            "steps": [
                {
                    "action": "click",
                    "target": "OpenCode - Official Site",
                    "payload": None,
                    "params": {},
                    "condition": None,
                    "is_destructive": False,
                    "confidence": 0.94,
                    "rationale": "Click the first search result",
                },
            ],
        }
        with patch(
            "wisprhelp.agent.reasoning.requests.post",
            return_value=make_groq_response(follow_up),
        ):
            r2 = self.agent.process_turn("Click the first result")

        self.assertTrue(r2.success)
        self.assertIn((400, 300), self.os_adapter.clicked_coords)

    # ------------------------------------------------------------------
    # Scenario 9: Unknown condition fails safe (skip)
    # ------------------------------------------------------------------

    def test_unknown_condition_fails_safe(self) -> None:
        from wisprhelp.agent.executor import ToolBasedActionExecutor

        executor = ToolBasedActionExecutor(os_adapter=self.os_adapter)
        action = Action.clear_text(condition="some_unknown_condition")
        outcome = executor.execute_action(action)
        # Fail-safe: unknown condition -> step skipped (no delete pressed)
        self.assertTrue(outcome.success)
        self.assertIn("Skipped", outcome.message)
        self.assertNotIn("delete", self.os_adapter.pressed_keys)


if __name__ == "__main__":
    unittest.main()
