"""Unit Tests for WisprHelp Phase 5 Vision Engine.

Tests screen capture, token-efficient downscaling, perceptual dHash diffing,
coordinate grounding, and bounded VLM ReAct execution.
"""

from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from wisprhelp.vision.capture import ScreenCapture, ScreenFrame
from wisprhelp.vision.grounding import CoordinateGrounding
from wisprhelp.vision.vlm_agent import VLMAgent


def test_screen_capture_and_dhash():
    capture = ScreenCapture(max_dimension=1920)
    frame = capture.capture()

    assert frame is not None
    assert frame.width > 0
    assert frame.height > 0
    assert frame.scale_x > 0
    assert frame.scale_y > 0
    assert isinstance(frame.dhash, int)

    # Base64 serialization
    b64 = frame.to_base64_jpeg()
    assert isinstance(b64, str)
    assert len(b64) > 100


def test_perceptual_dhash_diffing():
    img1 = Image.new("RGB", (100, 100), color=(0, 0, 0))
    img2 = Image.new("RGB", (100, 100), color=(0, 0, 0))
    # Different image with vertical stripes
    img3 = Image.new("RGB", (100, 100), color=(255, 255, 255))
    for x in range(0, 100, 10):
        for y in range(100):
            img3.putpixel((x, y), (0, 0, 0))

    hash1 = ScreenCapture.compute_dhash(img1)
    hash2 = ScreenCapture.compute_dhash(img2)
    hash3 = ScreenCapture.compute_dhash(img3)

    # Identical images have distance 0
    assert ScreenCapture.hamming_distance(hash1, hash2) == 0

    # Different image has non-zero distance
    dist = ScreenCapture.hamming_distance(hash1, hash3)
    assert dist > 0


def test_coordinate_grounding():
    # Test normalized 1000 coordinates
    px, py = CoordinateGrounding.normalized_1000_to_pixels(500, 500, 1920, 1080)
    assert px == 960
    assert py == 540

    # Test clamping
    px_clamped, py_clamped = CoordinateGrounding.normalized_1000_to_pixels(1500, -200, 1920, 1080)
    assert px_clamped == 1920
    assert py_clamped == 0

    # Test relative float coordinates
    fx, fy = CoordinateGrounding.normalized_float_to_pixels(0.25, 0.75, 1000, 1000)
    assert fx == 250
    assert fy == 750

    # Test bbox center calculation
    cx, cy = CoordinateGrounding.bbox_to_center((100, 200, 50, 80), format="xywh")
    assert cx == 125
    assert cy == 240


def test_vlm_agent_bounded_execution():
    mock_hal = MagicMock()
    agent = VLMAgent(os_adapter=mock_hal, max_steps=5)

    mock_res = {
        "success": True,
        "steps_taken": 2,
        "message": "Found search box",
        "history": [{"step": 1, "action": "click"}],
    }
    with patch.object(agent.grounded_agent, "execute_goal", return_value=mock_res):
        with patch("wisprhelp.vision.vlm_agent.config") as cfg:
            cfg.enable_set_of_marks = True
            cfg.gemini_api_key = "test-key"
            result = agent.execute_visual_task(goal="Find search box", allowed_actions={"click"})
        assert result is not None
        assert result.steps_taken <= 5
        assert len(result.action_history) > 0


def test_vlm_agent_emergency_abort():
    mock_hal = MagicMock()
    agent = VLMAgent(os_adapter=mock_hal, max_steps=10)

    # Trigger abort before execution
    agent.abort()
    result = agent.execute_visual_task(goal="Run risky automation")

    assert result.aborted is True
    assert result.success is False
    assert result.steps_taken == 0


def test_scoped_visual_click_rejects_unrelated_action():
    hal = MagicMock()
    capture = MagicMock()
    capture.capture.return_value = MagicMock()
    agent = VLMAgent(os_adapter=hal, capture=capture)
    agent._decide_next_action = MagicMock(return_value={"type": "type_text", "params": {"text": "surprise"}})
    with patch("wisprhelp.vision.vlm_agent.config") as cfg:
        cfg.enable_set_of_marks = False
        cfg.anthropic_api_key = "test-key"
        result = agent.execute_visual_task("Click Save", allowed_actions={"click"})
    assert not result.success
    hal.inject_unicode_text.assert_not_called()
    hal.click.assert_not_called()


def test_scoped_visual_click_dispatches_once():
    hal = MagicMock()
    capture = MagicMock()
    frame = MagicMock()
    frame.width = frame.screen_width = 1000
    frame.height = frame.screen_height = 1000
    frame.original_width = frame.original_height = 1000
    frame.scale_x = frame.scale_y = 1
    capture.capture.return_value = frame
    agent = VLMAgent(os_adapter=hal, capture=capture)
    agent._decide_next_action = MagicMock(return_value={"type": "click", "params": {"x": 100, "y": 200}})
    with patch("wisprhelp.vision.vlm_agent.config") as cfg:
        cfg.enable_set_of_marks = False
        cfg.anthropic_api_key = "test-key"
        result = agent.execute_visual_task("Click Save", allowed_actions={"click"})
    assert result.success
    assert result.steps_taken == 1
    hal.click.assert_called_once()


def test_visual_model_failure_never_simulates_click():
    hal = MagicMock()
    capture = MagicMock()
    agent = VLMAgent(os_adapter=hal, capture=capture)
    with patch("wisprhelp.vision.vlm_agent.config") as cfg:
        cfg.enable_set_of_marks = False
        cfg.anthropic_api_key = "test-key"
        cfg.gemini_api_key = None
        with patch.object(agent, "_call_anthropic_computer_use", side_effect=RuntimeError("offline")):
            result = agent.execute_visual_task("Click Save", allowed_actions={"click"})
    assert not result.success
    hal.click.assert_not_called()


def test_vlm_agent_429_quota_handling():
    """Verify that HTTP 429 quota exhaustion from Gemini returns success=False with clear message."""
    mock_hal = MagicMock()
    agent = VLMAgent(os_adapter=mock_hal, max_steps=5)

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = '{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota)."}}'

    mock_config = MagicMock()
    mock_config.enable_set_of_marks = False  # Exercise this module's coordinate-based HTTP client.
    mock_config.gemini_api_key = "mock_gemini_key"
    mock_config.anthropic_api_key = None

    with patch("requests.post", return_value=mock_resp):
        with patch("wisprhelp.vision.vlm_agent.config", mock_config):
            result = agent.execute_visual_task(goal="Click submit button")
            assert result.success is False
            assert "quota exceeded" in result.message.lower()
