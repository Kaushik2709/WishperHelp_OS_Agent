"""Test suite for the batched per-step TypeSafe Jev plan classifier.

Verifies question construction, LLM/Jev confidence merging, graceful
degradation without an API key, tier mapping, and plan-level gating.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.plan_classifier import ClassifiedStep, PlanClassifier
from wisprhelp.classification.client import JevClient


def _keyless_client() -> JevClient:
    """A JevClient that is guaranteed to have NO API key (never hits the network)."""
    client = JevClient()
    client.api_key = ""
    return client


def _sample_plan() -> list[Action]:
    return [
        Action.launch_app("Notepad", rationale="Launch Notepad"),
        Action.clear_text(
            "active_document",
            rationale="Clear existing content",
            condition="document_has_content",
        ),
        Action.type_text("hey there", rationale="Type the greeting"),
    ]


class TestPlanClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.keyless_client = _keyless_client()
        self.classifier = PlanClassifier(client=self.keyless_client)

    # ------------------------------------------------------------------
    # Batched Jev evaluation (mocked API)
    # ------------------------------------------------------------------

    def test_batched_questions_and_merging(self) -> None:
        """A single batched Jev call classifies every step with merged metrics."""
        mock_client = MagicMock(spec=JevClient)
        mock_client.api_key = "apikey_test"

        def fake_evaluate(state, questions):
            # Verify the batched question structure
            self.assertIn("step_1_complexity", questions)
            self.assertIn("step_2_is_destructive", questions)
            self.assertIn("step_3_alignment", questions)
            self.assertEqual(len(questions), 9)  # 3 questions x 3 steps
            self.assertEqual(state["user_goal"], "Prepare Notepad")
            self.assertEqual(len(state["plan_steps"]), 3)
            return {
                "answers": {
                    "step_1_alignment": {"choice": "correct_step", "confidence": 0.97},
                    "step_1_complexity": {"score": 0.0, "confidence": 0.95},
                    "step_1_is_destructive": {"noul": 0.02},
                    "step_2_alignment": {"choice": "correct_step", "confidence": 0.93},
                    "step_2_complexity": {"score": 1.0, "confidence": 0.90},
                    "step_2_is_destructive": {"noul": 0.71},
                    "step_3_alignment": {"choice": "correct_step", "confidence": 0.99},
                    "step_3_complexity": {"score": 1.0, "confidence": 0.92},
                    "step_3_is_destructive": {"noul": 0.03},
                }
            }

        mock_client.evaluate.side_effect = fake_evaluate
        clf = PlanClassifier(client=mock_client)
        plan = _sample_plan()
        plan[0].confidence = 0.98   # LLM says 0.98, Jev alignment 0.97 -> merged 0.97
        plan[1].confidence = 0.60   # LLM says 0.60, Jev alignment 0.93 -> merged 0.60
        plan[2].confidence = 0.95

        classified = clf.classify_plan(plan, goal="Prepare Notepad")

        self.assertEqual(len(classified), 3)
        self.assertTrue(all(c.jev_evaluated for c in classified))

        c0, c1, c2 = classified
        self.assertEqual(c0.alignment, "correct_step")
        self.assertAlmostEqual(c0.confidence, 0.97)
        self.assertEqual(c0.tier, 1)  # complexity 0.0 -> P1
        self.assertAlmostEqual(c0.destructive_probability, 0.02)

        # Step 2: destructive 0.71 and complexity 1.0 -> tier 2
        self.assertAlmostEqual(c1.confidence, 0.60)
        self.assertEqual(c1.tier, 2)
        self.assertAlmostEqual(c1.destructive_probability, 0.95)  # clear_text is inherently destructive
        self.assertTrue(c1.jev_evaluated)

        self.assertAlmostEqual(c2.confidence, 0.95)
        self.assertEqual(c2.alignment, "correct_step")

    def test_wrong_step_alignment_penalized(self) -> None:
        """A step Jev judges 'wrong_step' is heavily penalized and gates the plan."""
        mock_client = MagicMock(spec=JevClient)
        mock_client.api_key = "apikey_test"
        mock_client.evaluate.return_value = {
            "answers": {
                "step_1_alignment": {"choice": "wrong_step", "confidence": 0.9},
                "step_1_complexity": {"score": 1.0, "confidence": 0.9},
                "step_1_is_destructive": {"noul": 0.1},
            }
        }
        clf = PlanClassifier(client=mock_client)
        plan = [Action.click("Maybe not this")]
        plan[0].confidence = 0.95
        classified = clf.classify_plan(plan, goal="Do something else")
        self.assertAlmostEqual(classified[0].confidence, 0.20)
        self.assertIsNotNone(clf.gate_plan(classified))

    def test_llm_destructive_flag_inflates_probability(self) -> None:
        """LLM-flagged destructive steps carry high destructive probability even offline."""
        plan = [
            Action(
                type=ActionType.CLEAR_TEXT,
                target="active_document",
                is_destructive=True,
                confidence=0.9,
            )
        ]
        classified = self.classifier.classify_plan(plan, goal="Clear the doc")
        self.assertAlmostEqual(classified[0].destructive_probability, 0.95)
        self.assertFalse(classified[0].jev_evaluated)

    def test_graceful_degradation_without_key(self) -> None:
        """No Jev key: LLM-only metrics are used and no exception is raised."""
        plan = _sample_plan()
        plan[0].confidence = 0.90
        classified = self.classifier.classify_plan(plan, goal="Prepare Notepad")
        self.assertEqual(len(classified), 3)
        self.assertFalse(any(c.jev_evaluated for c in classified))
        self.assertAlmostEqual(classified[0].confidence, 0.90)
        self.assertEqual(classified[0].alignment, "correct_step")

    def test_jev_api_failure_degrades_gracefully(self) -> None:
        """Jev API exceptions must not abort the turn."""
        mock_client = MagicMock(spec=JevClient)
        mock_client.api_key = "apikey_test"
        mock_client.evaluate.side_effect = RuntimeError("network down")
        clf = PlanClassifier(client=mock_client)
        plan = _sample_plan()
        classified = clf.classify_plan(plan, goal="Prepare Notepad")
        self.assertEqual(len(classified), 3)
        self.assertFalse(any(c.jev_evaluated for c in classified))

    def test_respond_actions_skip_classification(self) -> None:
        """Conversational respond/wait steps never trigger Jev classification."""
        plan = [Action.respond("The capital of France is Paris.")]
        mock_client = MagicMock(spec=JevClient)
        mock_client.api_key = "apikey_test"
        clf = PlanClassifier(client=mock_client)
        classified = clf.classify_plan(plan, goal="Answer question")
        mock_client.evaluate.assert_not_called()
        self.assertEqual(len(classified), 1)
        self.assertFalse(classified[0].jev_evaluated)

    # ------------------------------------------------------------------
    # Tier mapping and gating
    # ------------------------------------------------------------------

    def test_tier_mapping(self) -> None:
        self.assertEqual(PlanClassifier._tier_for_complexity(0.0), 1)
        self.assertEqual(PlanClassifier._tier_for_complexity(0.5), 1)
        self.assertEqual(PlanClassifier._tier_for_complexity(1.0), 2)
        self.assertEqual(PlanClassifier._tier_for_complexity(1.5), 2)
        self.assertEqual(PlanClassifier._tier_for_complexity(2.0), 3)

    def test_gate_plan_low_confidence_blocks_execution(self) -> None:
        """A weak-confidence step must produce a clarification prompt."""
        plan = [Action.launch_app("Notepad", rationale="Launch Notepad")]
        plan[0].confidence = 0.35
        classified = self.classifier.classify_plan(plan, goal="Open Notepad")
        gate = self.classifier.gate_plan(classified)
        self.assertIsNotNone(gate)
        self.assertIn("clarif", gate.lower())

    def test_gate_plan_high_confidence_passes(self) -> None:
        plan = [Action.launch_app("Notepad", rationale="Launch Notepad")]
        plan[0].confidence = 0.95
        classified = self.classifier.classify_plan(plan, goal="Open Notepad")
        self.assertIsNone(self.classifier.gate_plan(classified))

    def test_apply_to_actions_merges_flags(self) -> None:
        """Destructive probability over threshold sets the action flag; tier is written back."""
        plan = _sample_plan()
        plan[0].grounding_tier = None
        plan[1].grounding_tier = None
        mock_client = MagicMock(spec=JevClient)
        mock_client.api_key = "apikey_test"
        mock_client.evaluate.return_value = {
            "answers": {
                "step_1_alignment": {"choice": "correct_step", "confidence": 0.95},
                "step_1_complexity": {"score": 0.0, "confidence": 0.95},
                "step_1_is_destructive": {"noul": 0.05},
                "step_2_alignment": {"choice": "correct_step", "confidence": 0.90},
                "step_2_complexity": {"score": 1.0, "confidence": 0.90},
                "step_2_is_destructive": {"noul": 0.80},
                "step_3_alignment": {"choice": "correct_step", "confidence": 0.95},
                "step_3_complexity": {"score": 1.0, "confidence": 0.90},
                "step_3_is_destructive": {"noul": 0.05},
            }
        }
        clf = PlanClassifier(client=mock_client)
        classified = clf.classify_plan(plan, goal="Prepare Notepad")
        clf.apply_to_actions(classified)

        self.assertFalse(plan[0].is_destructive)
        self.assertTrue(plan[1].is_destructive)  # 0.80 >= 0.65 threshold
        self.assertFalse(plan[2].is_destructive)
        self.assertEqual(plan[0].grounding_tier, 1)  # tier written from complexity


if __name__ == "__main__":
    unittest.main()
