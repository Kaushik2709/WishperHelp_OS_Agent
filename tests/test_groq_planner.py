"""Unit Tests for GroqStrategicPlanner.

Verifies milestone plan decomposition, heuristic fallback, and conversational responses.
"""

from unittest.mock import MagicMock, patch
import pytest

from wisprhelp.core.groq_planner import GroqStrategicPlanner, StrategicPlan


def test_groq_planner_heuristic_fallback_when_no_key():
    planner = GroqStrategicPlanner(api_key="")
    plan = planner.plan_workflow("Open Chrome and search Python 3.14", active_app="Chrome")

    assert plan is not None
    assert isinstance(plan, StrategicPlan)
    assert len(plan.milestones) == 1
    assert "Open Chrome" in plan.milestones[0].description


def test_groq_planner_successful_decomposition():
    planner = GroqStrategicPlanner(api_key="mock_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"is_complex": true, "milestones": [{"step_number": 1, "description": "Open WhatsApp", "target_app": "WhatsApp", "expected_outcome": "WhatsApp window open"}, {"step_number": 2, "description": "Click search bar and type Kaushik", "target_app": "WhatsApp", "expected_outcome": "Kaushik chat appears"}]}'
            }
        }]
    }

    with patch("requests.post", return_value=mock_resp):
        plan = planner.plan_workflow("Open WhatsApp and find Kaushik")

        assert plan.is_complex is True
        assert len(plan.milestones) == 2
        assert plan.milestones[0].target_app == "WhatsApp"
        assert plan.milestones[1].step_number == 2


def test_groq_planner_conversational_reply():
    planner = GroqStrategicPlanner(api_key="mock_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": "Antigravity is Google's advanced agentic coding assistant."
            }
        }]
    }

    with patch("requests.post", return_value=mock_resp):
        reply = planner.answer_conversational("What is Antigravity?")
        assert "Antigravity" in reply
