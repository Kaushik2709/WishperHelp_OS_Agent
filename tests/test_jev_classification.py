"""Tests for TypeSafe JEV classification engine."""

import pytest
from wisprhelp.classification import (
    JevClassifier,
    JevDecision,
    IntentType,
    TargetApp,
)
from wisprhelp.config import config


@pytest.fixture
def classifier() -> JevClassifier:
    return JevClassifier()


def test_empty_query_returns_safe_default(classifier: JevClassifier) -> None:
    decision = classifier.classify("   ")
    assert decision.intent == IntentType.CONVERSATIONAL_QUERY.value
    assert decision.is_fast_path() is False
    assert decision.is_high_risk() is False


def test_launch_app_classification(classifier: JevClassifier) -> None:
    decision = classifier.classify("Launch Google Chrome")
    assert decision.intent == IntentType.DIRECT_APP_LAUNCH.value
    assert decision.target_app == TargetApp.BROWSER.value
    assert decision.complexity_score <= 0.5
    assert decision.is_fast_path() is True
    assert decision.is_high_risk() is False


def test_media_control_classification(classifier: JevClassifier) -> None:
    decision = classifier.classify("Mute the sound")
    assert decision.intent == IntentType.SYSTEM_MEDIA_CONTROL.value
    assert decision.complexity_score <= 0.5
    assert decision.is_fast_path() is True
    assert decision.is_high_risk() is False


def test_multilingual_text_injection(classifier: JevClassifier) -> None:
    decision = classifier.classify("Type नमस्ते Kaushik, कल मिलते हैं into the chat")
    assert decision.intent == IntentType.TEXT_INPUT_INJECTION.value
    assert decision.has_multilingual_content is True
    assert decision.multilingual_probability >= 0.5
    assert decision.is_text_injection() is True


def test_destructive_risk_detection(classifier: JevClassifier) -> None:
    decision = classifier.classify("Delete all files in my downloads folder permanently")
    assert decision.is_destructive is True
    assert decision.destructive_probability >= 0.65
    assert decision.is_high_risk() is True


def test_complex_gui_workflow(classifier: JevClassifier) -> None:
    decision = classifier.classify("Open WhatsApp, search for Rahul, and send him 'I will be there in 5 minutes'")
    assert decision.intent == IntentType.GUI_WORKFLOW_AUTOMATION.value
    assert decision.target_app == TargetApp.WHATSAPP.value
    assert decision.complexity_score >= 1.0
    assert decision.is_gui_agent_required() is True


def test_offline_fallback_mode() -> None:
    # Test that fallback works even if client has invalid key
    from wisprhelp.classification.client import JevClient
    bad_client = JevClient(api_key="invalid_key_for_testing", timeout=1.0, max_retries=1)
    fallback_classifier = JevClassifier(client=bad_client)

    # Should fall back cleanly without raising unhandled exceptions
    decision = fallback_classifier.classify("Launch Chrome")
    assert decision.intent == IntentType.DIRECT_APP_LAUNCH.value
    assert decision.target_app == TargetApp.BROWSER.value
    assert decision.is_fast_path() is True
