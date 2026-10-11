"""Mocked pipeline integration tests for WisprHelp; no genuine voice/desktop E2E.

Tests the full cognitive loop: Hearing (VAD/STT) -> Reflex Brain (Jev) ->
Safety Interlock -> 3-Priority Tool Execution -> HUD & Feedback.
"""

from unittest.mock import MagicMock
import pytest

from wisprhelp.classification.classifier import JevClassifier
from wisprhelp.classification.schemas import JevDecision, TargetApp
from wisprhelp.core.router import CommandRouter, RouteTarget
from wisprhelp.core.runtime import RuntimeCoordinator
from wisprhelp.core.safety import SafetyGate
from wisprhelp.ui.overlay import StatusOverlayHUD, HUDState
from wisprhelp.vision.vlm_agent import VLMAgent
from wisprhelp.voice.feedback import AudioFeedback
from wisprhelp.voice.stt import WhisperSTT
from wisprhelp.voice.vad import SileroVAD

pytestmark = pytest.mark.mocked


def test_full_pipeline_voice_to_action():
    # 1. Simulate VAD and STT
    stt = WhisperSTT(api_key=None)
    mock_stt_text = "open whatsapp"

    # 2. Mock HAL and HUD
    mock_hal = MagicMock()
    mock_hal.get_active_window.return_value = MagicMock(title="Desktop", process_name="explorer.exe")
    mock_hal.launch_app.return_value = True

    hud = StatusOverlayHUD(enable_gui=False)
    feedback = AudioFeedback(enabled=False)

    coordinator = RuntimeCoordinator(
        classifier=JevClassifier(),
        router=CommandRouter(),
        safety_gate=SafetyGate(),
        os_adapter=mock_hal,
        hud=hud,
        feedback=feedback,
    )

    # 3. Process the query
    result = coordinator.process_command(mock_stt_text)

    assert result.success is True
    assert result.route == RouteTarget.PRIORITY_1_LAUNCH
    assert mock_hal.launch_app.called
    assert result.latency_ms > 0


def test_end_to_end_hindi_text_injection():
    mock_hal = MagicMock()
    mock_hal.get_active_window.return_value = MagicMock(title="Notepad", process_name="notepad.exe")
    mock_hal.inject_unicode_text.return_value = True

    hud = StatusOverlayHUD(enable_gui=False)
    feedback = AudioFeedback(enabled=False)

    coordinator = RuntimeCoordinator(
        classifier=JevClassifier(),
        router=CommandRouter(),
        safety_gate=SafetyGate(),
        os_adapter=mock_hal,
        hud=hud,
        feedback=feedback,
    )

    query = "टाइप करो नमस्ते दुनिया"
    result = coordinator.process_command(query)

    assert result.success is True
    assert result.route == RouteTarget.TEXT_INJECTION
    assert mock_hal.inject_unicode_text.called
    injected_arg = mock_hal.inject_unicode_text.call_args[0][0]
    assert "नमस्ते दुनिया" in injected_arg


def test_end_to_end_destructive_safety_interlock():
    mock_hal = MagicMock()
    mock_hal.get_active_window.return_value = MagicMock(title="Terminal", process_name="cmd.exe")

    coordinator = RuntimeCoordinator(
        classifier=JevClassifier(),
        router=CommandRouter(),
        safety_gate=SafetyGate(),
        os_adapter=mock_hal,
        hud=StatusOverlayHUD(enable_gui=False),
        feedback=AudioFeedback(enabled=False),
    )

    query = "Delete all system files and format disk"
    res1 = coordinator.process_command(query)

    # Step 1: Intercepted by safety interlock
    assert res1.route == RouteTarget.SAFETY_INTERLOCK
    assert "High-risk action detected" in res1.message
    assert not mock_hal.launch_app.called

    # Step 2: Negative confirmation
    res_cancel = coordinator.process_command(query, confirmation_response="cancel")
    assert res_cancel.success is False
    assert "cancelled by user" in res_cancel.message

    # Step 3: Positive confirmation
    res2 = coordinator.process_command(query)  # Re-hold
    res_confirm = coordinator.process_command(query, confirmation_response="yes I confirm")
    assert res_confirm.success is False
    assert "Confirmed high-risk action" in res_confirm.message


def test_end_to_end_p2_to_p3_escalation():
    """Verifies that when Priority 2 UIA finds no element, system escalates to Priority 3 VLM."""
    mock_hal = MagicMock()
    mock_hal.get_active_window.return_value = MagicMock(title="CustomCanvasApp", process_name="canvas.exe")
    mock_hal.launch_app.return_value = True
    # Priority 2 UIA element search returns None (element not exposed in accessibility tree)
    mock_hal.find_accessibility_element.return_value = None

    mock_vlm = MagicMock(spec=VLMAgent)
    from wisprhelp.vision.vlm_agent import VisionActionResult
    mock_vlm.execute_visual_task.return_value = VisionActionResult(
        success=True,
        steps_taken=2,
        message="Found button and clicked via vision grounding.",
    )

    coordinator = RuntimeCoordinator(
        classifier=JevClassifier(),
        router=CommandRouter(),
        safety_gate=SafetyGate(),
        os_adapter=mock_hal,
        vlm_agent=mock_vlm,
        hud=StatusOverlayHUD(enable_gui=False),
        feedback=AudioFeedback(enabled=False),
    )

    # A complex workflow query
    query = "In CanvasApp click the green help icon in the corner"
    result = coordinator.process_command(query)

    assert result.success is True
    assert result.route == RouteTarget.GUI_WORKFLOW_AUTOMATION
    # Verified Priority 3 was invoked
    assert mock_vlm.execute_visual_task.called
    assert "Priority 3 Vision" in result.message
