"""Semantic Equivalence & Generalization Test Suite.

Verifies Section 30 of the specification:
The agent must understand diverse human expressions mapping to the same underlying goal
WITHOUT relying on brittle keyword databases or dozens of hardcoded regex branches.
"""

from __future__ import annotations

import unittest

from wisprhelp.agent.actions import ActionType
from wisprhelp.agent.memory import ConversationMemory
from wisprhelp.agent.reasoning import SemanticReasoner
from wisprhelp.agent.references import ReferenceResolver
from wisprhelp.agent.state import ComputerState, WindowInfo


class TestSemanticGeneralization(unittest.TestCase):
    """Evaluates semantic generalization across diverse phrasings."""

    def setUp(self) -> None:
        self.ref_resolver = ReferenceResolver()
        self.reasoner = SemanticReasoner(ref_resolver=self.ref_resolver, groq_api_key="")
        self.memory = ConversationMemory()

        self.state = ComputerState(
            active_window=WindowInfo(title="Google Chrome", process_name="chrome.exe", pid=100, rect=(0, 0, 1000, 800)),
        )

    def test_browser_launch_semantic_equivalence(self) -> None:
        """Verifies diverse phrasings for launching the browser map to the same goal."""
        phrasings = [
            "Launch the browser",
            "Bring up my browser",
            "Open Chrome",
            "Get Chrome running",
            "Can you open the browser?",
            "I want to use Chrome",
        ]
        for phrase in phrasings:
            with self.subTest(phrase=phrase):
                understanding = self.reasoner.analyze(phrase, self.memory, self.state)
                # Must not be classified as literal text
                self.assertFalse(understanding.is_literal_text)
                # Primary goal or actions must indicate opening browser/Chrome
                action_types = [a.type for a in understanding.initial_actions]
                self.assertTrue(
                    ActionType.LAUNCH_APP in action_types
                    or ActionType.SEARCH in action_types
                    or "chrome" in understanding.primary_goal.lower()
                    or "browser" in understanding.primary_goal.lower(),
                    f"Failed on phrasing: '{phrase}' (Actions: {action_types})",
                )

    def test_close_window_semantic_equivalence(self) -> None:
        """Verifies diverse phrasings for closing the active window."""
        phrasings = [
            "Close this",
            "Close this window",
            "Shut this",
            "Get rid of this",
            "Exit the current window",
        ]
        for phrase in phrasings:
            with self.subTest(phrase=phrase):
                understanding = self.reasoner.analyze(phrase, self.memory, self.state)
                self.assertFalse(understanding.is_literal_text)
                action_types = [a.type for a in understanding.initial_actions]
                self.assertIn(ActionType.CLOSE, action_types, f"Failed on phrasing: '{phrase}'")

    def test_search_query_semantic_equivalence(self) -> None:
        """Verifies diverse search request verbs map to a unified search action with payload."""
        phrasings = [
            "Find information about quantum computing",
            "Look for quantum computing",
            "Search for quantum computing",
            "Look up quantum computing",
            "Google quantum computing",
            "Pull up information about quantum computing",
        ]
        for phrase in phrasings:
            with self.subTest(phrase=phrase):
                understanding = self.reasoner.analyze(phrase, self.memory, self.state)
                self.assertFalse(understanding.is_literal_text)
                action_types = [a.type for a in understanding.initial_actions]
                self.assertIn(ActionType.SEARCH, action_types, f"Failed on phrasing: '{phrase}'")

                # Payload must contain 'quantum computing'
                search_actions = [a for a in understanding.initial_actions if a.type == ActionType.SEARCH]
                self.assertTrue(len(search_actions) > 0)
                self.assertIn("quantum computing", search_actions[0].payload.lower())

    def test_volume_control_semantic_equivalence(self) -> None:
        """Verifies audio adjustment phrasings map to system media control."""
        phrasings = [
            "Turn down the volume",
            "Lower the sound",
            "Volume down",
            "Awaaz kam karo",
            "Make the volume lower",
        ]
        for phrase in phrasings:
            with self.subTest(phrase=phrase):
                understanding = self.reasoner.analyze(phrase, self.memory, self.state)
                action_types = [a.type for a in understanding.initial_actions]
                self.assertIn(ActionType.MEDIA_CONTROL, action_types, f"Failed on phrasing: '{phrase}'")

    def test_pronoun_reference_across_turns(self) -> None:
        """Verifies 'it' resolves to the previously opened document."""
        self.memory.entity_store.record_file("annual_budget.xlsx")
        resolved = self.ref_resolver.resolve("it", self.memory, self.state)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.resolved_target, "annual_budget.xlsx")
        self.assertEqual(resolved.category, "file")

    def test_contrastive_reference_resolution(self) -> None:
        """Verifies 'the other one' resolves to the alternative candidate."""
        self.memory.entity_store.record_ui_target("Search Button A")
        self.memory.entity_store.record_ui_target("Search Button B")  # newer

        resolved = self.ref_resolver.resolve("the other one", self.memory, self.state)
        self.assertIsNotNone(resolved)
        # Should resolve to the previous target (Search Button A)
        self.assertEqual(resolved.resolved_target, "Search Button A")


if __name__ == "__main__":
    unittest.main()
