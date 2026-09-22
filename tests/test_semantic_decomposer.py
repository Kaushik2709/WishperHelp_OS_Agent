"""Automated unit tests for SemanticDecomposer and Pre-Jev Compound Query Pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from wisprhelp.classification.classifier import JevClassifier
from wisprhelp.classification.decomposer import SemanticDecomposer
from wisprhelp.classification.schemas import (
    DecomposedQuery,
    DecomposedStep,
    IntentType,
    JevDecision,
    TargetApp,
)
from wisprhelp.core.runtime import RuntimeCoordinator
from wisprhelp.core.router import RouteTarget


def test_heuristic_fallback_compound_browser_search() -> None:
    """Verifies that heuristic fallback cleanly decomposes launch + search queries."""
    decomposer = SemanticDecomposer(api_key="")  # Force heuristic fallback
    res = decomposer.decompose("Open Chrome and search tobey maguire")

    assert res.is_compound is True
    assert res.target_app.lower() == "chrome"
    assert len(res.sub_queries) == 2

    step1 = res.sub_queries[0]
    assert step1.action_type == "launch"
    assert "chrome" in step1.query_text.lower()

    step2 = res.sub_queries[1]
    assert step2.action_type == "in_app_search"
    assert step2.payload == "tobey maguire"


def test_heuristic_fallback_volume_command() -> None:
    """Verifies that volume command is parsed with exact payload."""
    decomposer = SemanticDecomposer(api_key="")
    res = decomposer.decompose("put the volume to 60%")

    assert res.is_compound is False
    assert res.normalized_intent == IntentType.SYSTEM_MEDIA_CONTROL.value
    assert len(res.sub_queries) == 1
    assert res.sub_queries[0].action_type == "media_control"
    assert res.sub_queries[0].payload == "set_volume:60"


def test_heuristic_fallback_text_typing() -> None:
    """Verifies compound editor launch and typing decomposition."""
    decomposer = SemanticDecomposer(api_key="")
    res = decomposer.decompose("open notepad and write meeting notes")

    assert res.is_compound is True
    assert "notepad" in res.target_app.lower()
    assert len(res.sub_queries) == 2
    assert res.sub_queries[1].action_type == "type_text"
    assert res.sub_queries[1].payload == "meeting notes"


def test_groq_json_parsing() -> None:
    """Verifies that JSON returned by Groq LPU parses into typed DecomposedQuery."""
    decomposer = SemanticDecomposer(api_key="test_key")
    mock_json = {
        "target_app": "Google Chrome",
        "is_compound": True,
        "normalized_intent": "gui_workflow_automation",
        "sub_queries": [
            {"step_number": 1, "query_text": "Open Google Chrome", "action_type": "launch", "target": "Google Chrome", "payload": None},
            {"step_number": 2, "query_text": "Search for tobey maguire", "action_type": "in_app_search", "target": "Google Chrome", "payload": "tobey maguire"},
        ],
    }

    parsed = decomposer._parse_groq_json("Open Chrom and search tobey maguire", mock_json)
    assert parsed.is_compound is True
    assert parsed.target_app == "Google Chrome"
    assert len(parsed.sub_queries) == 2
    assert parsed.sub_queries[0].action_type == "launch"
    assert parsed.sub_queries[1].payload == "tobey maguire"


def test_jev_classifier_confidence_boost_on_compound() -> None:
    """Verifies that JevClassifier preserves >= 0.90 confidence for compound commands."""
    mock_decomposer = MagicMock()
    mock_decomposer.decompose.return_value = DecomposedQuery(
        original_query="Open Chrom and search tobey maguire",
        target_app="Chrome",
        is_compound=True,
        normalized_intent=IntentType.GUI_WORKFLOW_AUTOMATION.value,
        sub_queries=[
            DecomposedStep(step_number=1, query_text="Open Chrome", action_type="launch", target="Chrome"),
            DecomposedStep(step_number=2, query_text="Search for tobey maguire", action_type="in_app_search", target="Chrome", payload="tobey maguire"),
        ],
    )

    classifier = JevClassifier(client=None, decomposer=mock_decomposer)
    decision = classifier.classify("Open Chrom and search tobey maguire")

    assert decision.intent == IntentType.GUI_WORKFLOW_AUTOMATION.value
    assert decision.intent_confidence >= 0.90
    assert decision.target_app == TargetApp.BROWSER.value
    assert decision.decomposed_plan is not None
    assert decision.decomposed_plan.is_compound is True


def test_runtime_coordinator_compound_execution() -> None:
    """Verifies that RuntimeCoordinator executes sequential steps cleanly using the decomposed plan."""
    mock_os = MagicMock()
    mock_os.get_active_window.return_value = MagicMock(title="Desktop", process_name="explorer.exe")
    mock_os.launch_app.return_value = True

    mock_classifier = MagicMock()
    decomposed = DecomposedQuery(
        original_query="Open Chrom and search tobey maguire",
        target_app="chrome",
        is_compound=True,
        normalized_intent=IntentType.GUI_WORKFLOW_AUTOMATION.value,
        sub_queries=[
            DecomposedStep(step_number=1, query_text="Open Chrome", action_type="launch", target="chrome"),
            DecomposedStep(step_number=2, query_text="Search for tobey maguire", action_type="in_app_search", target="chrome", payload="tobey maguire"),
        ],
    )
    mock_classifier.classify.return_value = JevDecision(
        intent=IntentType.GUI_WORKFLOW_AUTOMATION.value,
        intent_confidence=0.95,
        target_app=TargetApp.BROWSER.value,
        target_app_confidence=0.92,
        complexity_score=2.0,
        complexity_confidence=0.90,
        is_destructive=False,
        destructive_probability=0.05,
        has_multilingual_content=False,
        multilingual_probability=0.05,
        decomposed_plan=decomposed,
    )

    coordinator = RuntimeCoordinator(os_adapter=mock_os, classifier=mock_classifier)
    result = coordinator.process_command("Open Chrom and search tobey maguire")

    assert result.success is True
    assert result.route == RouteTarget.GUI_WORKFLOW_AUTOMATION
    # Verify Step 1: launch Chrome
    mock_os.launch_app.assert_called_with("chrome")
    # Verify Step 2: address bar shortcut and exact search query injection
    mock_os.key_combination.assert_called_with(["ctrl", "l"])
    mock_os.inject_unicode_text.assert_called_with("tobey maguire")
    mock_os.press_key.assert_called_with("enter")


def test_live_groq_decomposition_if_key_available() -> None:
    """Verifies that live Groq API decomposes speech query if API key is configured."""
    from wisprhelp.config import config

    if not config.groq_api_key:
        pytest.skip("No GROQ_API_KEY available for live test")

    decomposer = SemanticDecomposer()
    res = decomposer.decompose("Open Chrom and search tobey maguire")
    assert res.is_compound is True
    assert res.target_app is not None
    assert "chrome" in res.target_app.lower()
    assert len(res.sub_queries) >= 2
