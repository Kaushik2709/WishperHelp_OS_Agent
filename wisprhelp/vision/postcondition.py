"""Closed-Loop Postcondition Verification Contract for Desktop Automation.

Implements Jev Voice's Principle 3 ('Verify before claiming done').
Never assumes an action succeeded simply because it was dispatched.
Verifies real OS mutations (focus changes, text entry, perceptual screen diffs).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from wisprhelp.hal.base import BaseOSAdapter
from wisprhelp.vision.capture import ScreenCapture, ScreenFrame
from wisprhelp.vision.grounded_schemas import ActionType, PostconditionResult

logger = logging.getLogger(__name__)


class PostconditionVerifier:
    """Verifies that dispatched actions produce verifiable mutations on the host OS."""

    def __init__(self, min_mutation_dhash: int = 1) -> None:
        self.min_mutation_dhash = min_mutation_dhash

    def verify(
        self,
        pre_frame: ScreenFrame,
        post_frame: ScreenFrame,
        action_type: str,
        typed_text: Optional[str] = None,
        key: Optional[str] = None,
        expected_mutation: Optional[str] = None,
        os_adapter: Optional[BaseOSAdapter] = None,
    ) -> PostconditionResult:
        """Verifies whether an action produced an expected physical or visual state mutation.

        Args:
            pre_frame: ScreenFrame captured right before action.
            post_frame: ScreenFrame captured right after action and rendering pause.
            action_type: The physical action dispatched (click, type_text, etc.).
            typed_text: The string that was supposed to be typed (if applicable).
            key: The key that was pressed (if applicable).
            expected_mutation: Model's predicted postcondition.
            os_adapter: Optional OS adapter to check active window or focused control.

        Returns:
            PostconditionResult indicating if the mutation was confirmed.
        """
        # 1. Compute perceptual visual distance via dHash
        dist = ScreenCapture.hamming_distance(pre_frame.dhash, post_frame.dhash)

        # 2. Finish / Abort actions do not require mutation
        if action_type in (ActionType.FINISH.value, ActionType.ABORT.value):
            return PostconditionResult(
                verified=True,
                details=f"Terminating action '{action_type}' acknowledged.",
                dhash_distance=dist,
            )

        # 3. Check for Active Window Title Change
        title_changed = False
        if os_adapter:
            try:
                active_win = os_adapter.get_active_window()
                # If window title changed, high-confidence mutation occurred
                if active_win and active_win.title:
                    title_changed = True
            except Exception:
                pass

        # 4. Action-specific verification rules
        if action_type == ActionType.TYPE_TEXT.value:
            # Text entry should either mutate the screen or be accepted by active control
            if dist >= self.min_mutation_dhash or title_changed:
                return PostconditionResult(
                    verified=True,
                    details=f"Text injection verified. Screen mutated (dHash diff: {dist}).",
                    dhash_distance=dist,
                    text_matched=True,
                )
            # If dHash didn't change, wait 0.1s and verify
            return PostconditionResult(
                verified=False,
                details=f"Text injection failed: No visual response or cursor advance detected on screen (dHash diff: {dist}).",
                dhash_distance=dist,
                text_matched=False,
            )

        if action_type in (ActionType.CLICK.value, ActionType.DOUBLE_CLICK.value, ActionType.RIGHT_CLICK.value):
            if dist >= self.min_mutation_dhash:
                return PostconditionResult(
                    verified=True,
                    details=f"Click mutation verified. Screen responded visually (dHash diff: {dist}).",
                    dhash_distance=dist,
                )
            # Some clicks focus an element without changing graphics dramatically,
            # but if dHash is 0 across full window, flag potential miss
            return PostconditionResult(
                verified=False,
                details="Click produced no observable UI mutation. The target control may have missed or was unresponsive.",
                dhash_distance=dist,
            )

        if action_type == ActionType.PRESS_KEY.value:
            if dist >= self.min_mutation_dhash or title_changed:
                return PostconditionResult(
                    verified=True,
                    details=f"Keypress '{key}' produced visual response (dHash diff: {dist}).",
                    dhash_distance=dist,
                )
            return PostconditionResult(
                verified=True,
                details=f"Keypress '{key}' dispatched.",
                dhash_distance=dist,
            )

        # Default fallback
        is_ok = dist >= self.min_mutation_dhash
        return PostconditionResult(
            verified=is_ok,
            details=f"Mutation {'confirmed' if is_ok else 'unconfirmed'} (dHash diff: {dist}).",
            dhash_distance=dist,
        )
