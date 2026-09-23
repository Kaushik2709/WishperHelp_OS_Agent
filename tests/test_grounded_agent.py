"""Unit Tests for GroundedActionAgent.

Verifies candidate decision parsing, dispatch to exact bounding box centers,
and bounded step cap limits.
"""

from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from wisprhelp.vision.capture import ScreenFrame
from wisprhelp.vision.grounded_agent import GroundedActionAgent
from wisprhelp.vision.grounded_schemas import ActionType, CandidateCatalog, GroundedDecision, UICandidateNode


def test_grounded_decision_parsing():
    raw_data = {
        "thought": "Click on the Search input box to focus.",
        "action_type": "click",
        "chosen_candidate_id": 3,
        "typed_text": None,
        "expected_mutation": "Search bar receives focus cursor.",
        "confidence": 0.95,
    }
    decision = GroundedDecision.from_dict(raw_data)

    assert decision.thought == raw_data["thought"]
    assert decision.action_type == ActionType.CLICK.value
    assert decision.chosen_candidate_id == 3
    assert decision.confidence == 0.95


def test_grounded_dispatch_clicks_exact_candidate_center():
    mock_hal = MagicMock()
    agent = GroundedActionAgent(os_adapter=mock_hal)

    catalog = CandidateCatalog()
    catalog.add(
        UICandidateNode(
            candidate_id=2,
            label="Send Message",
            control_type="Button",
            bbox=(500, 800, 600, 850),
            center=(550, 825),
        )
    )

    decision = GroundedDecision(
        thought="Send message button",
        action_type=ActionType.CLICK.value,
        chosen_candidate_id=2,
    )

    agent._dispatch_grounded_action(decision, catalog)
    # HAL click MUST be called with exact center (550, 825)
    mock_hal.click.assert_called_once_with(550, 825)


def test_grounded_dispatch_type_text_focuses_first():
    mock_hal = MagicMock()
    agent = GroundedActionAgent(os_adapter=mock_hal)

    catalog = CandidateCatalog()
    catalog.add(
        UICandidateNode(
            candidate_id=5,
            label="Chat Input",
            control_type="Edit",
            bbox=(100, 700, 400, 750),
            center=(250, 725),
        )
    )

    decision = GroundedDecision(
        thought="Type hello world into input",
        action_type=ActionType.TYPE_TEXT.value,
        chosen_candidate_id=5,
        typed_text="hello world",
    )

    agent._dispatch_grounded_action(decision, catalog)
    mock_hal.click.assert_called_once_with(250, 725)
    mock_hal.inject_unicode_text.assert_called_once_with("hello world")


def test_grounded_agent_emergency_abort():
    mock_hal = MagicMock()
    agent = GroundedActionAgent(os_adapter=mock_hal, max_steps=10)
    agent.abort()

    res = agent.execute_goal("Some risky automation")
    assert res["success"] is False
    assert res["aborted"] is True
    assert res["steps_taken"] == 0
