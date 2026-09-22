"""Tests for router, safety interlocks, and runtime coordination."""

import pytest
from wisprhelp.classification.schemas import JevDecision, IntentType, TargetApp
from wisprhelp.core.router import CommandRouter, RouteTarget
from wisprhelp.core.safety import SafetyGate
from wisprhelp.core.runtime import RuntimeCoordinator


@pytest.fixture
def router() -> CommandRouter:
    return CommandRouter(confidence_threshold=0.60, destructive_threshold=0.65)


@pytest.fixture
def safety_gate() -> SafetyGate:
    return SafetyGate(risk_threshold=0.65)


def test_router_clarification(router: CommandRouter) -> None:
    # Low confidence decision
    decision = JevDecision(
        intent="unknown_intent",
        intent_confidence=0.45,
        target_app="generic",
        target_app_confidence=0.5,
        complexity_score=1.0,
        complexity_confidence=0.5,
        is_destructive=False,
        destructive_probability=0.1,
        has_multilingual_content=False,
        multilingual_probability=0.1,
    )
    route, reason = router.route(decision)
    assert route == RouteTarget.CLARIFY


def test_router_safety_interlock(router: CommandRouter) -> None:
    decision = JevDecision(
        intent=IntentType.DIRECT_APP_LAUNCH.value,
        intent_confidence=0.95,
        target_app="terminal_code",
        target_app_confidence=0.95,
        complexity_score=0.0,
        complexity_confidence=0.95,
        is_destructive=True,
        destructive_probability=0.85,  # High risk
        has_multilingual_content=False,
        multilingual_probability=0.1,
    )
    route, reason = router.route(decision)
    assert route == RouteTarget.SAFETY_INTERLOCK


def test_safety_gate_confirmation_flow(safety_gate: SafetyGate) -> None:
    risky_decision = JevDecision(
        intent="delete_database",
        intent_confidence=0.95,
        target_app="file_manager",
        target_app_confidence=0.95,
        complexity_score=1.0,
        complexity_confidence=0.95,
        is_destructive=True,
        destructive_probability=0.92,
        has_multilingual_content=False,
        multilingual_probability=0.1,
    )
    assert safety_gate.check_risk(risky_decision) is True
    prompt = safety_gate.hold_for_confirmation(risky_decision)
    assert "High-risk action detected" in prompt

    # Affirmative confirmation
    confirmed = safety_gate.resolve_confirmation("yes, proceed")
    assert confirmed is not None
    assert confirmed.intent == "delete_database"

    # Decline flow
    safety_gate.hold_for_confirmation(risky_decision)
    declined = safety_gate.resolve_confirmation("no, cancel that")
    assert declined is None


def test_router_priority_1_launch(router: CommandRouter) -> None:
    decision = JevDecision(
        intent=IntentType.DIRECT_APP_LAUNCH.value,
        intent_confidence=0.95,
        target_app=TargetApp.BROWSER.value,
        target_app_confidence=0.95,
        complexity_score=0.1,  # Fast path <= 0.5
        complexity_confidence=0.95,
        is_destructive=False,
        destructive_probability=0.02,
        has_multilingual_content=False,
        multilingual_probability=0.01,
    )
    route, _ = router.route(decision)
    assert route == RouteTarget.PRIORITY_1_LAUNCH


def test_router_text_injection(router: CommandRouter) -> None:
    decision = JevDecision(
        intent=IntentType.TEXT_INPUT_INJECTION.value,
        intent_confidence=0.92,
        target_app=TargetApp.GENERIC_OR_ACTIVE.value,
        target_app_confidence=0.85,
        complexity_score=1.0,
        complexity_confidence=0.90,
        is_destructive=False,
        destructive_probability=0.05,
        has_multilingual_content=True,
        multilingual_probability=0.98,
    )
    route, _ = router.route(decision)
    assert route == RouteTarget.TEXT_INJECTION


def test_runtime_text_extraction() -> None:
    extracted_1 = RuntimeCoordinator._extract_text_content("Type hello world into active window")
    assert extracted_1 == "hello world"

    extracted_2 = RuntimeCoordinator._extract_text_content("लिखो 'नमस्ते Kaushik' in chat")
    assert extracted_2 == "नमस्ते Kaushik"


def test_runtime_coordinator_end_to_end_media() -> None:
    coordinator = RuntimeCoordinator()
    # Test volume adjustment end-to-end
    result = coordinator.process_command("Volume up")
    assert result.success is True
    assert result.route == RouteTarget.PRIORITY_1_MEDIA
    assert result.latency_ms > 0.0
