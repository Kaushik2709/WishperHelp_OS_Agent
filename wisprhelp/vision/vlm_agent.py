"""System Two Bounded Multimodal Visual ReAct Agent Loop (Priority 3).

Executes iterative visual computer use (Capture -> VLM Reason -> OS Act -> Verify)
with hard 15-step bounds, emergency kill-switch halts, and token-saving screen diffing.
"""

from __future__ import annotations

import logging
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from wisprhelp.config import config
from wisprhelp.hal.base import BaseOSAdapter
from wisprhelp.hal.factory import get_os_adapter
from wisprhelp.vision.capture import ScreenCapture, ScreenFrame
from wisprhelp.vision.grounded_agent import GroundedActionAgent
from wisprhelp.vision.grounding import CoordinateGrounding
from wisprhelp.vision.postcondition import PostconditionVerifier
from wisprhelp.core.outcomes import ExecutionFault, FailureCode
from wisprhelp.hal.base import WindowInfo

logger = logging.getLogger(__name__)


@dataclass
class VisionActionResult:
    """Outcome of a multi-step vision agent execution."""
    success: bool
    steps_taken: int
    message: str
    aborted: bool = False
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    failure_code: FailureCode = FailureCode.NONE
    postcondition_verified: Optional[bool] = None


class VLMAgent:
    """Production-grade System Two Multimodal Visual Computer Use Agent."""

    def __init__(
        self,
        os_adapter: Optional[BaseOSAdapter] = None,
        max_steps: int = 15,
        capture: Optional[ScreenCapture] = None,
        grounded_agent: Optional[GroundedActionAgent] = None,
    ) -> None:
        self.os_adapter = os_adapter or get_os_adapter()
        self.max_steps = max(1, min(max_steps, 15))  # Strict contract: max 15 steps
        self.capture = capture or ScreenCapture()
        self.grounded_agent = grounded_agent or GroundedActionAgent(
            os_adapter=self.os_adapter,
            capture=self.capture,
            max_steps=self.max_steps,
        )
        self._abort_flag = False

    def abort(self) -> None:
        """Signals immediate emergency abort."""
        logger.critical("Emergency abort signal received in VLMAgent.")
        self._abort_flag = True
        self.grounded_agent.abort()

    def reset_abort(self) -> None:
        self._abort_flag = False
        self.grounded_agent.reset_abort()

    def execute_visual_task(
        self, goal: str, target_app: Optional[str] = None, step_callback=None,
        reset_abort_on_start: bool = False, allowed_actions: Optional[Set[str]] = None,
        approved_target: Optional[str] = None,
    ) -> VisionActionResult:
        try:
            result = self._execute_visual_task(goal, target_app, step_callback, reset_abort_on_start,
                                               allowed_actions, approved_target)
            if result.aborted:
                result.failure_code = FailureCode.EMERGENCY_STOP
            elif not result.success and result.failure_code == FailureCode.NONE:
                result.failure_code = FailureCode.INCORRECT_GROUNDING
            return result
        except Exception as exc:
            code = exc.code if isinstance(exc, ExecutionFault) else (
                FailureCode.EMERGENCY_STOP if type(exc).__name__ in ("FailSafeException", "EmergencyKillSwitchTriggered")
                else FailureCode.ACTION_TIMEOUT if isinstance(exc, TimeoutError) else FailureCode.API_UNAVAILABLE)
            return VisionActionResult(False, 0, f"Vision stopped: {type(exc).__name__}",
                                      aborted=code == FailureCode.EMERGENCY_STOP, failure_code=code)

    def _execute_visual_task(
        self,
        goal: str,
        target_app: Optional[str] = None,
        step_callback: Optional[callable] = None,
        reset_abort_on_start: bool = False,
        allowed_actions: Optional[Set[str]] = None,
        approved_target: Optional[str] = None,
    ) -> VisionActionResult:
        """Runs the bounded visual ReAct loop to complete the desktop user goal.

        Args:
            goal: High-level user command.
            target_app: Target application context if available.
            step_callback: Optional observer hook called on each step.
            reset_abort_on_start: If True, clears any previous abort state.

        Returns:
            VisionActionResult: Execution outcome and audit trace.
        """
        if reset_abort_on_start:
            self.reset_abort()

        # Immediate check for abort flag
        if self._abort_flag:
            logger.warning("Visual task called but abort flag is active.")
            return VisionActionResult(
                success=False,
                steps_taken=0,
                message="Aborted via emergency kill-switch.",
                aborted=True,
                action_history=[],
            )

        if not (config.gemini_api_key or config.anthropic_api_key):
            return VisionActionResult(False, 0, "No vision model is configured for a grounded action.",
                                      failure_code=FailureCode.API_UNAVAILABLE)

        logger.info("Starting System Two Visual Loop for goal: '%s' (max_steps=%d)", goal, self.max_steps)

        # 1. Grounded Set-of-Marks Execution Path (Decide, Don't Generate)
        if config.enable_set_of_marks and config.gemini_api_key:
            res = self.grounded_agent.execute_goal(
                goal=goal,
                target_app=target_app,
                step_callback=step_callback,
                allowed_actions=allowed_actions,
                approved_target=approved_target,
            )
            return VisionActionResult(
                success=res.get("success", False),
                steps_taken=res.get("steps_taken", 0),
                message=res.get("message", "Grounded execution complete."),
                aborted=res.get("aborted", False),
                action_history=res.get("history", []),
                failure_code=res.get("failure_code", FailureCode.NONE),
                postcondition_verified=res.get("success", False),
            )

        # 2. Fallback to iterative coordinate-based ReAct loop
        if approved_target is not None:
            return VisionActionResult(False, 0,
                                      "A named target requires labeled candidate grounding; coordinate-only fallback was refused.",
                                      failure_code=FailureCode.INCORRECT_GROUNDING)
        history: List[Dict[str, Any]] = []
        consecutive_stagnant_steps = 0
        last_frame: Optional[ScreenFrame] = None

        for step in range(1, self.max_steps + 1):
            # 1. Kill-Switch check before every step
            if self._abort_flag:
                logger.warning("Step %d aborted via emergency kill-switch.", step)
                return VisionActionResult(
                    success=False,
                    steps_taken=step - 1,
                    message="Aborted via emergency kill-switch.",
                    aborted=True,
                    action_history=history,
                    failure_code=FailureCode.CANCELLED,
                )

            # 2. Capture and downscale current screen
            window_before = self.os_adapter.get_active_window()
            current_frame = self.capture.capture()

            # 3. Check for UI stagnation
            if last_frame is not None:
                changed = self.capture.has_screen_changed(last_frame, current_frame)
                if not changed:
                    consecutive_stagnant_steps += 1
                    if consecutive_stagnant_steps >= 3:
                        logger.warning("Screen has not changed for 3 consecutive steps. Halting loop to prevent spinning.")
                        return VisionActionResult(
                            success=False,
                            steps_taken=step,
                            message="Task halted: No visual response detected on screen.",
                            action_history=history,
                        )
                else:
                    consecutive_stagnant_steps = 0

            # 4. Invoke VLM reasoning
            action = self._decide_next_action(goal, current_frame, history, target_app)

            if step_callback:
                try:
                    step_callback(step, action)
                except Exception:
                    pass
            if self._abort_flag:
                return VisionActionResult(False, step - 1, "Emergency stop before dispatch.", aborted=True, action_history=history)
            if isinstance(window_before, WindowInfo) and self.os_adapter.get_active_window() != window_before:
                return VisionActionResult(False, step - 1, "Foreground changed while grounding.", action_history=history,
                                          failure_code=FailureCode.FOCUS_CHANGED)
            if self.os_adapter.has_unexpected_modal() is True:
                return VisionActionResult(False, step - 1, "Unexpected dialog before vision input.", action_history=history,
                                          failure_code=FailureCode.UNEXPECTED_MODAL)

            history.append({
                "step": step,
                "action": action.get("type"),
                "params": action.get("params"),
                "thought": action.get("thought", ""),
            })

            if allowed_actions is not None and action.get("type") not in allowed_actions | {"finish", "error"}:
                return VisionActionResult(False, step, "Vision proposed an action outside the approved step.", action_history=history)
            if allowed_actions is not None and action.get("type") not in {"finish", "error"}:
                params = action.get("params") or {}
                if action.get("type") not in {"click", "double_click"} or not all(
                    isinstance(params.get(axis), (int, float)) for axis in ("x", "y")
                ):
                    return VisionActionResult(False, step, "Vision did not return a supported grounded click.", action_history=history)
            if action.get("type") not in {"finish", "error"} and allowed_actions is None:
                return VisionActionResult(False, step - 1, "Vision requires a scoped approved action.", action_history=history,
                                          failure_code=FailureCode.UNAUTHORIZED)
            if action.get("type") in {"click", "double_click", "right_click"}:
                params = action.get("params") or {}
                if any(isinstance(params.get(axis), bool) or not isinstance(params.get(axis), (float, int))
                       or not math.isfinite(params[axis]) or not 0 <= params[axis] < limit
                       for axis, limit in (("x", current_frame.width), ("y", current_frame.height))):
                    return VisionActionResult(False, step - 1, "Coordinates outside observed frame.", action_history=history,
                                              failure_code=FailureCode.INCORRECT_GROUNDING)

            # 5. Check if goal is finished or encountered fatal error
            if action.get("type") in ("finish", "error"):
                msg = action.get("params", {}).get("message", "Done")
                is_ok = False  # A model's 'done' is not an independently verified goal.
                if allowed_actions is not None:
                    is_ok = False  # A scoped step requires an actual dispatched action.
                logger.info("VLMAgent %s at step %d: %s", "completed goal" if is_ok else "stopped with error", step, msg)
                return VisionActionResult(
                    success=is_ok,
                    steps_taken=step,
                    message=msg,
                    action_history=history,
                    failure_code=FailureCode.API_UNAVAILABLE if action.get("type") == "error" else FailureCode.POSTCONDITION_FAILED,
                )

            # 6. Execute hardware action via OS HAL
            self._dispatch_hal_action(action, current_frame)
            if allowed_actions is not None:
                time.sleep(0.3)
                post = self.capture.capture()
                verification = PostconditionVerifier().verify(current_frame, post, action.get("type"), os_adapter=self.os_adapter)
                history[-1]["verified"] = verification.verified
                return VisionActionResult(verification.verified, step, verification.details, action_history=history,
                                          failure_code=FailureCode.NONE if verification.verified else FailureCode.POSTCONDITION_FAILED,
                                          postcondition_verified=verification.verified)
            last_frame = current_frame

            # Bounded sleep to allow OS window animation/rendering
            time.sleep(0.3)

        return VisionActionResult(
            success=False,
            steps_taken=self.max_steps,
            message=f"Reached maximum step limit ({self.max_steps}) without finishing.",
            action_history=history,
        )

    def _decide_next_action(
        self,
        goal: str,
        frame: ScreenFrame,
        history: List[Dict[str, Any]],
        target_app: Optional[str],
    ) -> Dict[str, Any]:
        """Decides next action via Claude 3.5 Sonnet / Gemini 2.5 Flash, or offline fallback."""
        # 1. Anthropic Computer Use API (if key configured)
        if config.anthropic_api_key:
            try:
                return self._call_anthropic_computer_use(goal, frame, history)
            except Exception as exc:
                logger.error("Anthropic Computer Use error: %s; falling back.", exc)

        # 2. Google Gemini 2.5 Flash Multimodal Vision API (if key configured)
        if config.gemini_api_key:
            try:
                return self._call_gemini_vision(goal, frame, history)
            except Exception as exc:
                logger.error("Gemini VLM error: %s; falling back.", exc)

        return {"type": "error", "params": {
            "message": "No vision model could ground the requested screen action."
        }}

    def _call_anthropic_computer_use(
        self,
        goal: str,
        frame: ScreenFrame,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calls Anthropic Claude 3.5 Sonnet Computer Use API."""
        import anthropic

        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        # Convert frame to base64 jpeg
        b64_img = frame.to_base64_jpeg(quality=80)

        # Standard Anthropic Computer Use tool definition
        response = client.beta.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=[{
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": frame.width,
                "display_height_px": frame.height,
            }],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"User Goal: {goal}"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_img,
                        },
                    },
                ],
            }],
            betas=["computer-use-2024-10-22"],
        )

        for content_block in response.content:
            if content_block.type == "tool_use" and content_block.name == "computer":
                tool_input = content_block.input
                action_name = tool_input.get("action")
                coords = tool_input.get("coordinate")
                text = tool_input.get("text")

                if action_name == "left_click":
                    return {"type": "click", "params": {"x": coords[0], "y": coords[1]}}
                elif action_name == "type":
                    return {"type": "type_text", "params": {"text": text}}
                elif action_name == "key":
                    return {"type": "key_combination", "params": {"keys": [text]}}

        return {"type": "finish", "params": {"message": "Goal achieved."}}

    def _call_gemini_vision(
        self,
        goal: str,
        frame: ScreenFrame,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calls Google Gemini 2.5 Flash Vision API for visual element grounding and action generation."""
        import json
        import requests

        api_key = config.gemini_api_key

        prompt = (
            f"You are WisprHelp's System Two Vision Grounding Agent acting as an autonomous human user.\n"
            f"Goal: {goal}\n"
            f"Screen Resolution: {frame.width}x{frame.height}\n"
            f"Action History: {json.dumps(history[-3:]) if history else 'None'}\n\n"
            f"Instructions:\n"
            f"1. Examine the screenshot carefully to locate the target control, button, chat, icon, or list item.\n"
            f"2. Human-like Semantic & Phonetic Tolerance: Spoken user speech may contain slight phonetic variations or typos (e.g. 'show mode' -> 'Show more' button or 'AI Mode' tab, 'cross' -> 'X / Close', 'magnifier' -> 'Search icon'). As an intelligent human user, locate the most visually and semantically plausible button or control on screen that fulfills the user's intended action.\n"
            f"3. For ordinal or positional goals (e.g. 'first chat', 'second-last chat', 'third item'): find the list on screen, count the items in order (first at top, last at bottom), and target the center of that specific item.\n"
            f"4. For visual attributes (e.g. 'blue button', 'green checkmark', 'plus icon'): find the matching colored element.\n"
            f"5. Target the EXACT center coordinates of the button or control.\n"
            f"6. Output strictly valid JSON matching one schema:\n"
            f"- Click: {{\"type\": \"click\", \"params\": {{\"x\": <int>, \"y\": <int>}}, \"thought\": \"<reason>\"}}\n"
            f"- Double Click: {{\"type\": \"double_click\", \"params\": {{\"x\": <int>, \"y\": <int>}}, \"thought\": \"<reason>\"}}\n"
            f"- Type Text: {{\"type\": \"type_text\", \"params\": {{\"text\": \"<string>\"}}, \"thought\": \"<reason>\"}}\n"
            f"- Key: {{\"type\": \"key_combination\", \"params\": {{\"keys\": [\"<key>\"]}}, \"thought\": \"<reason>\"}}\n"
            f"- Finish: {{\"type\": \"finish\", \"params\": {{\"message\": \"<summary>\"}}, \"thought\": \"<reason>\"}}"
        )

        b64_image = frame.to_base64_jpeg(quality=75)
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_image,
                        }
                    },
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
            },
        }

        # Candidate models: verified-fast 1080p vision models first (grounded decisions
        # measured in <5s), slow/legacy models last as rescue options. Dead models
        # (gemini-2.0-flash, gemini-1.5-flash, gemini-2.5-flash-lite: HTTP 404) removed.
        candidate_models = [
            config.gemini_model,
            "gemini-3.1-flash-lite",
            "gemini-flash-latest",
            "gemini-3.6-flash",
            "gemini-2.5-flash",
        ]
        seen = set()
        candidate_models = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

        last_error_msg = "No response from Gemini API"
        for attempt_idx, model_name in enumerate(candidate_models):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            try:
                resp = requests.post(url, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict) and "type" in parsed:
                        return parsed
                    return {"type": "finish", "params": {"message": "Goal achieved."}}
                elif resp.status_code == 429:
                    logger.warning("Gemini model '%s' quota exceeded (HTTP 429). Falling back to next model...", model_name)
                    last_error_msg = f"Gemini Vision API quota exceeded (HTTP 429 for {model_name}). Please check your Gemini API plan or billing details at https://ai.google.dev"
                    continue
                elif resp.status_code in (503, 500):
                    logger.warning("Gemini model '%s' temporarily unavailable (%d). Trying next candidate...", model_name, resp.status_code)
                    last_error_msg = f"Gemini API temporary service error ({resp.status_code})"
                    continue
                else:
                    logger.error("Gemini model '%s' API error (%d): %s", model_name, resp.status_code, resp.text[:200])
                    last_error_msg = f"Gemini API returned status {resp.status_code}"
                    continue
            except Exception as exc:
                logger.warning("Gemini request to '%s' failed (%s). Trying next...", model_name, exc)
                last_error_msg = str(exc)
                continue

        logger.error("All candidate Gemini models failed or exceeded quota.")
        return {"type": "error", "params": {"message": last_error_msg}}

    def _dispatch_hal_action(self, action: Dict[str, Any], frame: ScreenFrame) -> None:
        """Maps high-level action to OS HAL operations with coordinate translation."""
        action_type = action.get("type")
        params = action.get("params", {})
        if self._abort_flag:
            raise ExecutionFault(FailureCode.EMERGENCY_STOP, "Emergency stop before visual input")
        self.os_adapter.check_emergency_stop()

        if action_type == "click":
            raw_x = params.get("x", 0)
            raw_y = params.get("y", 0)
            screen_x, screen_y = CoordinateGrounding.frame_coords_to_screen_pixels(raw_x, raw_y, frame)
            self.os_adapter.click(screen_x, screen_y)

        elif action_type == "double_click":
            raw_x = params.get("x", 0)
            raw_y = params.get("y", 0)
            screen_x, screen_y = CoordinateGrounding.frame_coords_to_screen_pixels(raw_x, raw_y, frame)
            self.os_adapter.click(screen_x, screen_y, clicks=2)

        elif action_type == "right_click":
            screen_x, screen_y = CoordinateGrounding.frame_coords_to_screen_pixels(params["x"], params["y"], frame)
            self.os_adapter.click(screen_x, screen_y, button="right")

        elif action_type == "type_text":
            text = params.get("text", "")
            # Universal Unicode clipboard injection prevents IME/Devanagari corruption
            self.os_adapter.inject_unicode_text(text)

        elif action_type == "key_combination":
            keys = params.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            self.os_adapter.key_combination(keys)

        elif action_type == "wait":
            duration = min(params.get("seconds", 1.0), 3.0)
            time.sleep(duration)
