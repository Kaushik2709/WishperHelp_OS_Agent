"""Unit Tests for Closed-Loop Postcondition Verification Contract.

Verifies mutation detection, focus validation, dHash difference checks, and failure reporting.
"""

from unittest.mock import MagicMock
from PIL import Image
import pytest

from wisprhelp.vision.capture import ScreenCapture, ScreenFrame
from wisprhelp.vision.grounded_schemas import ActionType
from wisprhelp.vision.postcondition import PostconditionVerifier


def _create_frame(color: tuple, dhash: int) -> ScreenFrame:
    img = Image.new("RGB", (200, 200), color=color)
    return ScreenFrame(
        image=img,
        width=200,
        height=200,
        original_width=1920,
        original_height=1080,
        scale_x=9.6,
        scale_y=5.4,
        dhash=dhash,
        timestamp=100.0,
    )


def test_postcondition_terminating_actions():
    verifier = PostconditionVerifier()
    f1 = _create_frame((0, 0, 0), 100)
    f2 = _create_frame((0, 0, 0), 100)

    res_finish = verifier.verify(f1, f2, action_type=ActionType.FINISH.value)
    assert res_finish.verified is True

    res_abort = verifier.verify(f1, f2, action_type=ActionType.ABORT.value)
    assert res_abort.verified is True


def test_postcondition_type_text_success_on_screen_mutation():
    verifier = PostconditionVerifier()
    # Different dHash indicates screen responded
    f1 = _create_frame((0, 0, 0), 0b1111)
    f2 = _create_frame((255, 255, 255), 0b0000)

    res = verifier.verify(f1, f2, action_type=ActionType.TYPE_TEXT.value, typed_text="hello")
    assert res.verified is True
    assert res.text_matched is True
    assert res.dhash_distance > 0


def test_postcondition_type_text_fails_when_frozen():
    verifier = PostconditionVerifier()
    # Identical dHash (distance 0)
    f1 = _create_frame((0, 0, 0), 12345)
    f2 = _create_frame((0, 0, 0), 12345)

    res = verifier.verify(f1, f2, action_type=ActionType.TYPE_TEXT.value, typed_text="hello")
    assert res.verified is False
    assert "No visual response" in res.details


def test_postcondition_click_mutation():
    verifier = PostconditionVerifier()
    f1 = _create_frame((0, 0, 0), 0b1010)
    f2 = _create_frame((255, 255, 255), 0b0101)

    res = verifier.verify(f1, f2, action_type=ActionType.CLICK.value)
    assert res.verified is True

    # Identical click
    f3 = _create_frame((0, 0, 0), 0b1010)
    res_miss = verifier.verify(f1, f3, action_type=ActionType.CLICK.value)
    assert res_miss.verified is False
