"""Tests for dynamic volume scaling, ordinal element resolution, and universal click protection."""

import pytest
from wisprhelp.core.runtime import RuntimeCoordinator
from wisprhelp.core.router import RouteTarget
from wisprhelp.classification.schemas import JevDecision, IntentType, TargetApp
from wisprhelp.hal.factory import get_os_adapter


def test_volume_percentage_extraction():
    """Verify natural language volume queries extract exact percentage commands."""
    assert RuntimeCoordinator._extract_media_command("put the volume to 60%") == "set_volume:60"
    assert RuntimeCoordinator._extract_media_command("set volume to 100%") == "set_volume:100"
    assert RuntimeCoordinator._extract_media_command("lower volume to 30%") == "set_volume:30"
    assert RuntimeCoordinator._extract_media_command("volume 80") == "set_volume:80"
    assert RuntimeCoordinator._extract_media_command("awaaz 50 karo") == "set_volume:50"
    assert RuntimeCoordinator._extract_media_command("sound at 75 percent") == "set_volume:75"
    assert RuntimeCoordinator._extract_media_command("mute") == "mute"
    assert RuntimeCoordinator._extract_media_command("unmute") == "unmute"
    assert RuntimeCoordinator._extract_media_command("volume up") == "vol_up"
    assert RuntimeCoordinator._extract_media_command("turn it down") == "vol_down"


def test_volume_relative_extraction():
    """Verify relative volume adjustment queries."""
    assert RuntimeCoordinator._extract_media_command("increase volume by 20%") == "adjust_volume:+20"
    assert RuntimeCoordinator._extract_media_command("lower the volume by 15%") == "adjust_volume:-15"


def test_ordinal_target_parsing():
    """Verify ordinal parsing for chats, tabs, and list items."""
    # First / Top
    assert RuntimeCoordinator._parse_ordinal_target("click on the first chat on WhatsApp") == (0, "chat")
    assert RuntimeCoordinator._parse_ordinal_target("click first chat") == (0, "chat")
    assert RuntimeCoordinator._parse_ordinal_target("select the top item") == (0, "item")
    assert RuntimeCoordinator._parse_ordinal_target("pehla chat click karo") == (0, "chat")

    # Second / Third
    assert RuntimeCoordinator._parse_ordinal_target("click on the second chat") == (1, "chat")
    assert RuntimeCoordinator._parse_ordinal_target("open 3rd tab") == (2, "tab")

    # Negative / from bottom: last, second-last, third-last
    assert RuntimeCoordinator._parse_ordinal_target("click the last chat on WhatsApp") == (-1, "chat")
    assert RuntimeCoordinator._parse_ordinal_target("click on the second-last chat on WhatsApp") == (-2, "chat")
    assert RuntimeCoordinator._parse_ordinal_target("click second last chat") == (-2, "chat")
    assert RuntimeCoordinator._parse_ordinal_target("click 2nd last item") == (-2, "item")
    assert RuntimeCoordinator._parse_ordinal_target("click third-last row") == (-3, "row")
    assert RuntimeCoordinator._parse_ordinal_target("penultimate message") == (-2, "message")


def test_hal_media_set_volume():
    """Verify HAL adapter handles set_volume command without errors."""
    adapter = get_os_adapter()
    # Native effects are mocked. Support varies by adapter; never change volume.
    ok = adapter.media_control("set_volume:50")
    assert isinstance(ok, bool)


def test_hal_find_ordinal_element_interface():
    """Verify find_ordinal_element interface handles invalid/empty calls gracefully."""
    adapter = get_os_adapter()
    # Searching non-existent window or out-of-range should return None safely without raising exception
    res = adapter.find_ordinal_element(9999, "chat")
    assert res is None or isinstance(res, tuple)


def test_click_not_misrouted_to_text_injection():
    """Verify commands with click/button terms are never treated as text injection."""
    coordinator = RuntimeCoordinator()
    
    # Simulate a decision where Jev returned text injection
    decision = JevDecision(
        intent=IntentType.TEXT_INPUT_INJECTION.value,
        intent_confidence=0.9,
        target_app=TargetApp.GENERIC_OR_ACTIVE.value,
        target_app_confidence=0.8,
        complexity_score=0.5,
        complexity_confidence=0.9,
        is_destructive=False,
        destructive_probability=0.05,
        has_multilingual_content=False,
        multilingual_probability=0.05,
    )

    # If user says "Click the Enter button on Antigravity", it must NOT inject text "Click the Enter..."
    result = coordinator.execute_query("Click the Enter button on Antigravity", decision=decision)
    # The route must have been overridden away from text injection
    assert result.route != RouteTarget.TEXT_INJECTION
