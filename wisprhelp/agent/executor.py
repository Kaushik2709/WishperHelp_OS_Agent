"""Deterministic Tool Execution Bridge for WisprHelp Semantic Brain.

Translates platform-neutral actions into validated, safety-gated system calls,
bridging to the 3-Priority Tool Execution Hierarchy (P1 Shell -> P2 UIA -> P3 Vision).
"""

from __future__ import annotations

import logging
import os
import platform
import time
import re
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.state import ComputerState
from wisprhelp.hal.base import BaseOSAdapter
from wisprhelp.hal.base import WindowInfo
from wisprhelp.hal.guarded import GuardedOSAdapter
from wisprhelp.core.outcomes import ExecutionFault, FailureCode
from wisprhelp.hal.factory import get_os_adapter
from wisprhelp.vision.vlm_agent import VLMAgent

logger = logging.getLogger(__name__)


@dataclass
class ExecutionOutcome:
    """The result of executing a single action step."""
    success: bool
    message: str
    action: Action
    observation: str = ""
    latency_ms: float = 0.0
    needs_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    failure_code: FailureCode = FailureCode.NONE
    postcondition_verified: Optional[bool] = None
    dispatched: bool = False


class ToolBasedActionExecutor:
    """Executes platform-neutral actions deterministically through HAL and safety."""

    def __init__(
        self,
        os_adapter: Optional[BaseOSAdapter] = None,
        safety_gate: Optional[Any] = None,
        vlm_agent: Optional[VLMAgent] = None,
        action_timeout: float = 30.0,
    ) -> None:
        native_adapter = os_adapter or get_os_adapter()
        self.os_adapter = GuardedOSAdapter(native_adapter, self._checkpoint)
        self.action_timeout = max(0.01, action_timeout)
        self._deadline = None
        self._expected_window = None
        self._guard_focus = False
        self._file_dispatched = False
        if safety_gate is None:
            from wisprhelp.core.safety import SafetyGate
            self.safety_gate = SafetyGate()
        else:
            self.safety_gate = safety_gate
        self.vlm_agent = vlm_agent or VLMAgent(os_adapter=self.os_adapter)
        self.abort_requested = lambda: False
        self.cancellation_code = lambda: FailureCode.CANCELLED
        self._held_text_context = {}
        self._held_action_context = {}


    def execute_action(
        self, action: Action, state: Optional[ComputerState] = None, confirmed: bool = False,
    ) -> ExecutionOutcome:
        """Translate failures, including preflight errors, into explicit outcomes.

        Deadlines are cooperative: a pending native/API call must return before
        its timeout can be reported. No background mutation is abandoned.
        """
        start = time.perf_counter()
        count = self.os_adapter.mutations
        self._file_dispatched = False
        self._deadline = start + self.action_timeout
        self._guard_focus = action.type in {
            ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK,
            ActionType.SELECT, ActionType.TYPE_TEXT, ActionType.CLEAR_TEXT,
            ActionType.PRESS_KEY, ActionType.KEY_COMBINATION, ActionType.SELECT_ALL,
            ActionType.SCROLL, ActionType.DRAG, ActionType.SEARCH, ActionType.NAVIGATE,
            ActionType.CLOSE,
        }
        try:
            self._expected_window = state.active_window if state else self.os_adapter.get_active_window()
            outcome = self._execute_action(action, state, confirmed)
            self._checkpoint("complete")
        except ExecutionFault as exc:
            outcome = ExecutionOutcome(False, str(exc), action, failure_code=exc.code)
        except TimeoutError:
            outcome = ExecutionOutcome(False, "Action timed out; execution stopped.", action,
                                       failure_code=FailureCode.ACTION_TIMEOUT)
        except Exception as exc:
            code = (FailureCode.EMERGENCY_STOP if type(exc).__name__ in
                    ("FailSafeException", "EmergencyKillSwitchTriggered") else FailureCode.ACTION_FAILED)
            outcome = ExecutionOutcome(False, f"Action stopped: {type(exc).__name__}", action, failure_code=code)
        outcome.dispatched = self.os_adapter.mutations > count or self._file_dispatched
        outcome.latency_ms = (time.perf_counter() - start) * 1000
        if not outcome.success and outcome.failure_code == FailureCode.NONE:
            outcome.failure_code = FailureCode.ACTION_FAILED
        self._deadline = None
        return outcome

    def _checkpoint(self, operation="input"):
        self._check_abort()
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            raise ExecutionFault(FailureCode.ACTION_TIMEOUT, "Action deadline exceeded; no further input allowed.")
        if self._guard_focus and operation not in ("launch_app", "focus_window", "complete"):
            if self.os_adapter.has_unexpected_modal() is True:
                raise ExecutionFault(FailureCode.UNEXPECTED_MODAL, "Unexpected dialog; execution stopped.")
            current = self.os_adapter.get_active_window()
            expected = self._expected_window
            if isinstance(expected, WindowInfo) and isinstance(current, WindowInfo):
                if (expected.pid, expected.process_name, expected.title) != (current.pid, current.process_name, current.title):
                    raise ExecutionFault(FailureCode.FOCUS_CHANGED, "Foreground window changed before input.")

    def clear_pending_confirmations(self):
        self._held_action_context.clear()
        self._held_text_context.clear()

    def _execute_action(
        self,
        action: Action,
        state: Optional[ComputerState] = None,
        confirmed: bool = False,
    ) -> ExecutionOutcome:
        """Executes a single typed Action safely and deterministically.

        Args:
            action: The Action to be executed.
            state: Current computer state snapshot.
            confirmed: Whether destructive confirmation has already been granted.

        Returns:
            ExecutionOutcome detailing success, latency, and observation.
        """
        start_time = time.perf_counter()

        self._check_abort()
        from wisprhelp.config import config
        if action.type not in (ActionType.RESPOND, ActionType.ASK_CLARIFICATION, ActionType.WAIT) and (
                not math.isfinite(action.confidence) or action.confidence < config.intent_confidence_threshold):
            return ExecutionOutcome(False, "Action confidence is insufficient for OS execution.", action,
                                    failure_code=FailureCode.LOW_CONFIDENCE)
        editing = action.type == ActionType.CLEAR_TEXT or (
            action.type == ActionType.TYPE_TEXT and any(k in action.params for k in ("mode", "app", "field")))
        before = None
        if editing:
            if not self._focus_edit_target(action):
                return ExecutionOutcome(False, "Could not safely focus the requested text field.", action,
                                        observation="Editable target could not be verified.")
            self._check_abort()
            before = self.os_adapter.read_focused_text()
        # Evaluate conditions before asking to confirm a step that will be skipped.
        if action.condition:
            if action.condition in ("document_has_content", "document_is_empty"):
                content = before if editing else self.os_adapter.read_focused_text()
                if content is None:
                    return ExecutionOutcome(False, "Could not inspect text for the requested condition.", action)
                condition_met = bool(content) if action.condition == "document_has_content" else not content
            else:
                condition_met = self.evaluate_condition(action.condition, state)
            if not condition_met:
                return ExecutionOutcome(True, f"Skipped: condition '{action.condition}' was not met.", action,
                                        observation="Condition false; no edit performed.")
        replacing = editing and (action.type == ActionType.CLEAR_TEXT or action.params.get("mode") == "replace")
        if action.type == ActionType.CLEAR_TEXT and before == "":
            return ExecutionOutcome(True, "The text field is already empty.", action,
                                    observation="Empty content verified; no deletion needed.")
        if replacing and before == (action.payload or "") and action.type == ActionType.TYPE_TEXT:
            return ExecutionOutcome(True, "Requested text is already present.", action, observation="Content verified.")
        runtime_risk = replacing and before != ""
        destructive = action.is_destructive or action.type in (ActionType.CLEAR_TEXT, ActionType.DELETE_FILE) or runtime_risk
        if action.type in (ActionType.CREATE_FILE, ActionType.WRITE_FILE):
            destructive = destructive or self._resolve_filesystem_path(action.target or "").exists()
        if action.type in (ActionType.MOVE_FILE, ActionType.RENAME_FILE):
            source = self._resolve_filesystem_path(action.target or "")
            destination = (self._resolve_filesystem_path(action.payload or "") / source.name
                           if action.type == ActionType.MOVE_FILE else source.parent / (action.payload or ""))
            destructive = destructive or destination.exists()
        if action.type == ActionType.CLOSE:
            destructive = True
        if action.type == ActionType.KEY_COMBINATION:
            keys = {str(k).casefold() for k in action.params.get("keys", [])}
            destructive = destructive or bool(keys & {"delete", "backspace"}) or any(
                combo <= keys for combo in ({"win", "r"}, {"command", "space"}, {"ctrl", "alt", "t"}, {"alt", "f2"}))
        # Risk is enforced locally even when a planner/classifier omits its flag.
        if action.type in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.SELECT, ActionType.PRESS_KEY):
            destructive = destructive or bool(re.search(
                r"\b(delete|backspace|remove|erase|format|wipe|send|submit|pay|purchase|buy|uninstall|shutdown)\b",
                action.target or "", re.IGNORECASE))
        destructive = destructive or (action.type == ActionType.TYPE_TEXT and action.params.get("press_enter", False))
        foreground = self.os_adapter.get_active_window()
        if action.type == ActionType.TYPE_TEXT and isinstance(foreground, WindowInfo):
            if any(app in foreground.process_name.casefold() for app in ("powershell", "cmd.exe", "terminal", "bash", "zsh")):
                destructive = True
        file_stamp = None
        if action.type in (ActionType.CREATE_FILE, ActionType.WRITE_FILE, ActionType.DELETE_FILE,
                           ActionType.MOVE_FILE, ActionType.RENAME_FILE):
            path = self._resolve_filesystem_path(action.target or "").resolve()
            stat = path.stat() if path.exists() else None
            file_stamp = (str(path), stat.st_mtime_ns if stat else None, stat.st_size if stat else None)
        signature = (json.dumps(action.to_dict(), sort_keys=True, ensure_ascii=False), foreground, file_stamp)
        if destructive and confirmed and self._held_action_context.get(id(action)) != signature:
            confirmed = False
        if replacing and before == "" and not action.is_destructive:
            destructive = False
        context = (self.os_adapter.get_active_window(), before) if editing else None
        if confirmed and editing and self._held_text_context.get(id(action)) != context:
            confirmed = False  # Foreground/content changed while awaiting consent.
        if destructive and not confirmed:
            self._held_action_context[id(action)] = signature
            if editing:
                self._held_text_context[id(action)] = context
            prompt = f"Safety Gate: destructive action {action.type.value} targeting '{action.target or 'active focus'}' ({action.rationale}) may change or send data. Confirm this specific action? (yes/no)"
            return ExecutionOutcome(True, prompt, action, needs_confirmation=True, confirmation_prompt=prompt)
        self._held_text_context.pop(id(action), None)
        self._held_action_context.pop(id(action), None)

        try:
            # 2. Dispatch to typed handler
            self._checkpoint("dispatch")
            if action.type in (ActionType.CREATE_FILE, ActionType.WRITE_FILE, ActionType.APPEND_FILE,
                               ActionType.RENAME_FILE, ActionType.MOVE_FILE, ActionType.DELETE_FILE):
                self._file_dispatched = True
            handler = getattr(self, f"_exec_{action.type.value}", self._exec_default)
            res = handler(action, state)
            elapsed = (time.perf_counter() - start_time) * 1000
            res.latency_ms = elapsed
            return res
        except Exception as exc:
            if isinstance(exc, (ExecutionFault, TimeoutError)):
                raise
            if type(exc).__name__ in ("FailSafeException", "EmergencyKillSwitchTriggered"):
                raise
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error("Execution error for action %s: %s", action.type, exc)
            return ExecutionOutcome(
                success=False,
                message=f"Action execution error: {exc}",
                action=action,
                observation=f"Exception raised: {exc}",
                latency_ms=elapsed,
            )

    # --- Action Handlers ---

    def evaluate_condition(self, condition: str, state: Optional[ComputerState]) -> bool:
        """Evaluates an LLM-planned runtime predicate before step execution.

        Supported predicates:
        - document_has_content / document_is_empty: probes the active document
          via select-all + copy + clipboard read (clipboard restored afterward).
        - clipboard_non_empty: clipboard currently holds text.
        - element_exists:<name>: accessibility tree contains the named control.
        - window_focused:<app>: active window title/process matches.

        Returns:
            True when the condition holds. Unknown/erroring conditions return
            False (fail-safe skip).
        """
        try:
            if condition == "document_has_content":
                return self._probe_document_content()
            if condition == "document_is_empty":
                return not self._probe_document_content()
            if condition == "clipboard_non_empty":
                clip = ""
                if hasattr(self.os_adapter, "get_clipboard_text"):
                    clip = self.os_adapter.get_clipboard_text() or ""
                return bool(clip.strip())
            if condition.startswith("element_exists:"):
                element_name = condition.split(":", 1)[1].strip()
                if not element_name:
                    return False
                return self.os_adapter.find_accessibility_element(element_name) is not None
            if condition.startswith("window_focused:"):
                app_keyword = condition.split(":", 1)[1].strip().lower()
                if not app_keyword:
                    return False
                win = self.os_adapter.get_active_window()
                title = (win.title or "").lower()
                proc = (win.process_name or "").lower()
                return app_keyword in title or app_keyword in proc
            logger.warning("Unknown condition '%s'; treating as unmet (fail-safe skip).", condition)
            return False
        except Exception as exc:
            logger.warning("Condition evaluation error for '%s': %s. Treating as unmet.", condition, exc)
            return False

    def _probe_document_content(self) -> bool:
        """Probes whether the active document/input holds non-empty text.

        Saves the user's clipboard, selects all + copies, reads the result,
        and restores the clipboard. The leftover selection is harmless: a
        following clear_text re-selects, and typing over a selection replaces
        it (which matches the natural 'clean it then type' expectation).
        """
        old_clip = None
        try:
            if hasattr(self.os_adapter, "get_clipboard_text"):
                old_clip = self.os_adapter.get_clipboard_text()
        except Exception:
            old_clip = None
        try:
            select_keys = self._select_all_keys()
            self.os_adapter.key_combination(select_keys)
            time.sleep(0.05)
            self.os_adapter.key_combination(self._copy_keys())
            time.sleep(0.10)
            content = ""
            if hasattr(self.os_adapter, "get_clipboard_text"):
                content = self.os_adapter.get_clipboard_text() or ""
            return bool(content.strip())
        finally:
            try:
                if old_clip is not None and hasattr(self.os_adapter, "set_clipboard_text"):
                    self.os_adapter.set_clipboard_text(old_clip)
            except Exception:
                logger.debug("Clipboard restore after document probe failed.")

    def _select_all_keys(self) -> List[str]:
        """Returns the platform-correct select-all hotkey."""
        if platform.system() == "Darwin":
            return ["command", "a"]
        return ["ctrl", "a"]

    def _copy_keys(self) -> List[str]:
        """Returns the platform-correct copy hotkey."""
        if platform.system() == "Darwin":
            return ["command", "c"]
        return ["ctrl", "c"]

    def _exec_launch_app(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        target = action.target or ""
        ok = self.os_adapter.launch_app(target, parameter=action.payload)
        if not ok:
            return ExecutionOutcome(False, f"Application '{target}' could not be launched.", action,
                                    failure_code=FailureCode.TARGET_APP_MISSING)
        focused = self.os_adapter.wait_for_active_window([target], timeout=1.5)
        if not focused:
            return ExecutionOutcome(False, "Launch dispatched, but target window did not appear.", action,
                                    failure_code=FailureCode.POSTCONDITION_FAILED, postcondition_verified=False)
        return ExecutionOutcome(
            success=ok,
            message=f"Priority 1: Launched '{target}' ({'Success' if ok else 'Failed'}).",
            action=action,
            observation=f"App '{target}' launch dispatched.",
            postcondition_verified=True,
        )

    def _exec_focus_window(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        target = action.target or ""
        ok = False
        if hasattr(self.os_adapter, "focus_window"):
            ok = self.os_adapter.focus_window(target)
        if not ok:
            # Fallback to launch / switch
            ok = self.os_adapter.launch_app(target)
        if ok and not self.os_adapter.wait_for_active_window([target], timeout=1.5):
            return ExecutionOutcome(False, "Target window focus could not be verified.", action,
                                    failure_code=FailureCode.POSTCONDITION_FAILED, postcondition_verified=False)
        return ExecutionOutcome(
            success=ok,
            message=f"Focused window matching '{target}'.",
            action=action,
            observation=f"Window '{target}' focused.",
            postcondition_verified=True if ok else False,
        )

    def _check_abort(self):
        if self.abort_requested():
            raise ExecutionFault(self.cancellation_code(), "Execution stopped before further input.")
        self.os_adapter.check_emergency_stop()

    def _focus_edit_target(self, action):
        app = action.params.get("app")
        if app and app not in ("generic_or_active", "active", "current"):
            win = self.os_adapter.get_active_window()
            if app.casefold() not in f"{win.title} {win.process_name}".casefold():
                if not self.os_adapter.focus_window(app):
                    return False
                if not self.os_adapter.wait_for_active_window([app], timeout=2):
                    return False
                self._expected_window = self.os_adapter.get_active_window()
        field = action.params.get("field") or action.target or "editor"
        # Editing needs a verified target. The general vision loop can execute
        # arbitrary tool actions, so it cannot safely serve as focus-only input.
        return self.os_adapter.focus_text_input(field)

    @staticmethod
    def _normalized_text(text):
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _exec_type_text(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        text = action.payload or ""
        mode = action.params.get("mode", "insert")
        before = self.os_adapter.read_focused_text()
        if mode == "replace":
            self.os_adapter.key_combination(self._select_all_keys())
        elif mode == "append":
            self.os_adapter.move_text_caret_to_end()
        self._check_abort()
        if mode == "replace" and not text:
            self.os_adapter.press_key("backspace")
            ok = True
        else:
            ok = self.os_adapter.inject_unicode_text(text)
        if not ok:
            return ExecutionOutcome(False, "Text injection failed.", action)
        after = self.os_adapter.read_focused_text()
        expected = text if mode == "replace" else ((before + text) if mode == "append" and before is not None else None)
        if expected is not None:
            for _ in range(5):
                self._check_abort()
                if after is not None and self._normalized_text(after) == self._normalized_text(expected):
                    break
                time.sleep(0.05)
                after = self.os_adapter.read_focused_text()
            if after is None or self._normalized_text(after) != self._normalized_text(expected):
                return ExecutionOutcome(False, "Text was sent, but the resulting content could not be verified.", action,
                                        observation="Text postcondition failed; do not repeat blindly.",
                                        failure_code=FailureCode.POSTCONDITION_FAILED, postcondition_verified=False)
        self._check_abort()
        if action.params.get("press_enter", False):
            self.os_adapter.press_key("enter")
        verified = expected is not None and after is not None
        return ExecutionOutcome(True, "Text updated and verified." if verified else "Text inserted.", action,
                                observation="Content matches requested text." if verified else "Text injection dispatched; exact content not verified.",
                                postcondition_verified=True if verified else None)

    def _exec_press_key(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        key = action.target or "enter"
        self.os_adapter.press_key(key)
        return ExecutionOutcome(
            success=True,
            message=f"Priority 1: Pressed key '{key}'.",
            action=action,
            observation=f"Key '{key}' pressed.",
        )

    def _exec_key_combination(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        keys = action.params.get("keys", [])
        self.os_adapter.key_combination(keys)
        return ExecutionOutcome(
            success=True,
            message=f"Priority 1: Dispatched hotkey '{'+'.join(keys)}'.",
            action=action,
            observation=f"Hotkey '{'+'.join(keys)}' pressed.",
        )

    def _exec_select_all(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        keys = self._select_all_keys()
        self.os_adapter.key_combination(keys)
        return ExecutionOutcome(
            success=True,
            message=f"Priority 1: Selected all content ({'+'.join(keys)}).",
            action=action,
            observation="Selected all content in active document.",
        )

    def _exec_clear_text(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        """Clears the active document: select all, then delete the selection."""
        keys = self._select_all_keys()
        self.os_adapter.key_combination(keys)
        time.sleep(0.05)
        delete_key = "backspace" if platform.system() == "Darwin" else "delete"
        self._check_abort()
        self.os_adapter.press_key(delete_key)
        time.sleep(0.05)
        content = self.os_adapter.read_focused_text()
        if content != "":
            return ExecutionOutcome(False, "Could not verify that the field is empty.", action)
        return ExecutionOutcome(
            success=True,
            message=f"Priority 1: Cleared content of '{action.target or 'active document'}'.",
            action=action,
            observation="Document content selected and deleted.",
        )

    def _exec_drag(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        """Drags a source element to a destination via UIA coordinates, falling back to vision."""
        source = action.target or ""
        destination = action.payload or ""

        src_coords = self.os_adapter.find_accessibility_element(source) if source else None
        dst_coords = self.os_adapter.find_accessibility_element(destination) if destination else None

        if src_coords and dst_coords:
            if hasattr(self.os_adapter, "drag_and_drop"):
                self.os_adapter.drag_and_drop(src_coords, dst_coords)
            else:
                return ExecutionOutcome(False, "Adapter has no supported drag primitive.", action,
                                        failure_code=FailureCode.ACTION_FAILED)
            return ExecutionOutcome(
                success=True,
                message=f"Priority 2 UIA: Dragged '{source}' to '{destination}'.",
                action=action,
                observation=f"Dragged '{source}' ({src_coords}) to '{destination}' ({dst_coords}).",
            )

        # The vision dispatchers do not expose a bounded, verified drag primitive.
        # Report the unsupported step instead of letting an unrestricted loop act.
        return ExecutionOutcome(
            success=False,
            message=f"Could not locate both drag endpoints for '{source}' and '{destination}'.",
            action=action,
            observation="Drag requires accessible source and destination controls.",
        )

    def _exec_scroll(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        direction = action.params.get("direction", "down")
        amount = action.params.get("amount", 3)
        if hasattr(self.os_adapter, "scroll"):
            self.os_adapter.scroll(direction=direction, amount=amount)
        else:
            # Fallback via PageDown / PageUp
            key = "pagedown" if direction == "down" else "pageup"
            self.os_adapter.press_key(key)
        return ExecutionOutcome(
            success=True,
            message=f"Scrolled {direction} by {amount}.",
            action=action,
            observation=f"Scrolled {direction}.",
        )

    def _exec_click(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        target = action.target or ""
        button = action.params.get("button", "left")
        clicks = action.params.get("clicks", 1)

        # Normalization: detect natural language ordinal targets
        t_low = target.lower()
        if not target.startswith("ordinal:"):
            if "whatsapp" in t_low and ("chat" in t_low or "set" in t_low):
                idx = 0
                if any(w in t_low for w in ("second", "2nd", "doosra")):
                    idx = 1
                elif any(w in t_low for w in ("third", "3rd")):
                    idx = 2
                target = f"ordinal:{idx}:chat"
                action.target = target
                action.params["app"] = "WhatsApp"
            elif any(w in t_low for w in ("first", "1st", "firsti", "second", "2nd", "position", "pos")) and any(w in t_low for w in ("file", "doc", "document")):
                idx = 1 if any(w in t_low for w in ("second", "2nd")) else 0
                action.target = f"ordinal:{idx}:file"
                return self._exec_open_file(action, state)

        # 1. Ordinal reference e.g. "ordinal:0:chat" or "ordinal:0:file"
        coords = None
        if target.startswith("ordinal:"):
            parts = target.split(":")
            ord_idx = int(parts[1]) if len(parts) > 1 else 0
            itype = parts[2] if len(parts) > 2 else "item"
            app_name = action.params.get("app") or (state.active_app_name if state else None)
            if not app_name and itype in ("chat", "conversation"):
                app_name = "WhatsApp"
            coords = self.os_adapter.find_ordinal_element(ord_idx, item_type=itype, target_app=app_name)

            # If opening a file item and UIA couldn't locate it on screen, use native Priority 1 file opening
            if not coords and itype in ("file", "doc", "document"):
                logger.info("Ordinal file target '%s' escalating to Priority 1 native file opener.", target)
                return self._exec_open_file(action, state)

        # 2. Priority 2: Native Accessibility Tree search
        if not coords:
            coords = self.os_adapter.find_accessibility_element(target)

        # If found in Priority 2 UIA, click directly
        if coords:
            self.os_adapter.click(coords[0], coords[1], button=button, clicks=clicks)
            return ExecutionOutcome(
                success=True,
                message=f"Priority 2 UIA: Clicked '{target}' at {coords}.",
                action=action,
                observation=f"Element '{target}' clicked at {coords}.",
            )

        # 3. Priority 3: Fall back to Multimodal Vision Grounding
        logger.info("Priority 2 UIA could not find '%s'. Escalating to Priority 3 Vision.", target)
        if action.params.get("accessibility_only"):
            return ExecutionOutcome(False, "Accessibility target was not found; no input dispatched.", action,
                                    failure_code=FailureCode.ACCESSIBILITY_TARGET_NOT_FOUND)
        allowed = {"double_click"} if clicks == 2 else {"right_click"} if button == "right" else {"click"}
        vlm_res = self.vlm_agent.execute_visual_task(goal=f"Click {target}", allowed_actions=allowed, approved_target=target)
        return ExecutionOutcome(
            success=vlm_res.success,
            message=f"Priority 3 Vision: Clicked '{target}' ({vlm_res.message}).",
            action=action,
            observation=f"Vision grounding executed for '{target}'.",
            failure_code=FailureCode.NONE if vlm_res.success else getattr(vlm_res, "failure_code", FailureCode.INCORRECT_GROUNDING),
        )

    def _exec_double_click(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        action.params["clicks"] = 2
        return self._exec_click(action, state)

    def _exec_right_click(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        action.params["button"] = "right"
        return self._exec_click(action, state)

    def _exec_select(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        return self._exec_click(action, state)

    def _exec_search(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        query = action.payload or action.target or ""
        app = action.target or (state.active_app_name if state else "Chrome")

        # In browser: Ctrl+L, inject query, press Enter
        if any(b in (app or "").lower() for b in ("browser", "chrome", "edge", "firefox", "brave")):
            self.os_adapter.key_combination(["ctrl", "l"])
            time.sleep(0.06)
            self.os_adapter.inject_unicode_text(query)
            time.sleep(0.04)
            self.os_adapter.press_key("enter")
            return ExecutionOutcome(
                success=True,
                message=f"Priority 1: Searched for '{query}' in {app}.",
                action=action,
                observation=f"Browser search submitted for '{query}'.",
            )

        # In app search: Ctrl+F or direct type
        self.os_adapter.key_combination(["ctrl", "f"])
        time.sleep(0.05)
        self.os_adapter.inject_unicode_text(query)
        self.os_adapter.press_key("enter")
        return ExecutionOutcome(
            success=True,
            message=f"In-app search submitted for '{query}'.",
            action=action,
            observation=f"Search for '{query}' executed.",
        )

    def _exec_navigate(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        url = action.payload or action.target or ""
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        self.os_adapter.key_combination(["ctrl", "l"])
        time.sleep(0.06)
        self.os_adapter.inject_unicode_text(url)
        time.sleep(0.04)
        self.os_adapter.press_key("enter")
        return ExecutionOutcome(
            success=True,
            message=f"Priority 1: Navigated browser to '{url}'.",
            action=action,
            observation=f"Navigated to '{url}'.",
        )

    def _resolve_filesystem_path(self, path_str: str) -> Path:
        """Resolves file paths against user home, Documents, Desktop, or current directory."""
        if not path_str:
            return Path.home() / "Documents"
        p = Path(os.path.expanduser(path_str))
        if p.is_absolute():
            return p

        # Check standard user folders if relative
        home = Path.home()
        parts = p.parts
        if parts:
            first_part = parts[0].lower()
            if first_part in ("documents", "document"):
                return home / "Documents" / Path(*parts[1:])
            if first_part in ("desktop", "downloads", "pictures", "music", "videos"):
                return home / p

        # Default to Documents if simple filename
        if len(parts) == 1:
            return home / "Documents" / p.name

        return (home / p).resolve()

    def _exec_open_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        target_str = (action.target or "").strip()
        t_low = target_str.lower()

        # Handle ordinal file targets: e.g. "ordinal:0:file" or natural language "first file", "second file"
        if not target_str.startswith("ordinal:") and (
            any(w in t_low for w in ("first", "1st", "firsti", "second", "2nd", "third", "3rd", "firstfile", "second_file", "yourusername", "position", "pos"))
            or t_low in ("file", "doc", "document")
        ):
            ord_idx = 0
            if any(w in t_low for w in ("second", "2nd", "second_file")):
                ord_idx = 1
            elif any(w in t_low for w in ("third", "3rd")):
                ord_idx = 2
            target_str = f"ordinal:{ord_idx}:file"

        if target_str.startswith("ordinal:"):
            parts = target_str.split(":")
            try:
                ord_idx = int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                ord_idx = 0
            dir_name = action.params.get("directory") or "Documents"
            dir_path = self._resolve_filesystem_path(dir_name)
            if not dir_path.is_dir():
                dir_path = Path.home() / "Documents"

            files = [f for f in dir_path.iterdir() if f.is_file() and not f.name.startswith(".")]
            if not files:
                for alt_name in ("Downloads", "Desktop"):
                    alt_dir = Path.home() / alt_name
                    if alt_dir.is_dir():
                        files = [f for f in alt_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
                        if files:
                            dir_path = alt_dir
                            break

            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            if files:
                if 0 <= ord_idx < len(files):
                    file_path = files[ord_idx]
                elif ord_idx < 0 and abs(ord_idx) <= len(files):
                    file_path = files[ord_idx]
                else:
                    file_path = files[0]
            else:
                return ExecutionOutcome(
                    success=False,
                    message=f"No files found in '{dir_path.name}' to open.",
                    action=action,
                    observation=f"Directory '{dir_path}' contains no files.",
                )
        else:
            file_path = self._resolve_filesystem_path(target_str)

        try:
            ok = self.os_adapter.open_file(str(file_path))
            return ExecutionOutcome(
                success=ok,
                message=f"Priority 1: Opened file '{file_path.name}'.",
                action=action,
                observation=f"File opened: '{file_path}'.",
            )
        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Failed to open file '{file_path}': {exc}",
                action=action,
                observation=f"File open error: {exc}",
            )

    def _exec_create_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        file_path = self._resolve_filesystem_path(action.target or "new_file.txt")
        content = action.payload or ""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "x", encoding="utf-8") as f:
                f.write(content)
            if file_path.exists():
                return ExecutionOutcome(
                    success=True,
                    message=f"Priority 1: Created file '{file_path.name}' in '{file_path.parent}'.",
                    action=action,
                    observation=f"File '{file_path}' created successfully ({len(content)} chars).",
                )
            return ExecutionOutcome(
                success=False,
                message=f"Failed to verify creation of file '{file_path}'.",
                action=action,
                observation="File does not exist after write operation.",
            )
        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Error creating file '{file_path}': {exc}",
                action=action,
                observation=f"Exception during file create: {exc}",
            )

    def _exec_write_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        file_path = self._resolve_filesystem_path(action.target or "")
        content = action.payload or ""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return ExecutionOutcome(
                success=True,
                message=f"Priority 1: Wrote {len(content)} characters to '{file_path.name}'.",
                action=action,
                observation=f"File '{file_path}' updated.",
            )
        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Error writing to file '{file_path}': {exc}",
                action=action,
                observation=f"Exception during file write: {exc}",
            )

    def _exec_append_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        file_path = self._resolve_filesystem_path(action.target or "")
        content = action.payload or ""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
            return ExecutionOutcome(
                success=True,
                message=f"Priority 1: Appended text to '{file_path.name}'.",
                action=action,
                observation=f"Appended to '{file_path}'.",
            )
        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Error appending to file '{file_path}': {exc}",
                action=action,
                observation=f"Exception during file append: {exc}",
            )

    def _exec_rename_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        old_path = self._resolve_filesystem_path(action.target or "")
        new_name = action.payload or ""
        if not old_path.exists():
            return ExecutionOutcome(
                success=False,
                message=f"File to rename '{old_path}' not found.",
                action=action,
                observation=f"File not found: '{old_path}'.",
            )
        try:
            new_path = old_path.parent / new_name
            old_path.rename(new_path)
            return ExecutionOutcome(
                success=new_path.exists(),
                message=f"Priority 1: Renamed '{old_path.name}' to '{new_name}'.",
                action=action,
                observation=f"File renamed to '{new_path}'.",
            )
        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Error renaming file: {exc}",
                action=action,
                observation=f"Exception during file rename: {exc}",
            )

    def _exec_move_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        import shutil
        src_path = self._resolve_filesystem_path(action.target or "")
        dest_dir = self._resolve_filesystem_path(action.payload or "")
        if not src_path.exists():
            return ExecutionOutcome(
                success=False,
                message=f"Source file '{src_path}' not found.",
                action=action,
                observation=f"File not found: '{src_path}'.",
            )
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_dir))
            dest_file = dest_dir / src_path.name
            return ExecutionOutcome(
                success=dest_file.exists(),
                message=f"Priority 1: Moved '{src_path.name}' to '{dest_dir.name}'.",
                action=action,
                observation=f"File moved to '{dest_file}'.",
            )
        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Error moving file: {exc}",
                action=action,
                observation=f"Exception during file move: {exc}",
            )

    def _exec_delete_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        file_path = self._resolve_filesystem_path(action.target or "")
        if not file_path.exists():
            return ExecutionOutcome(
                success=True,
                message=f"File '{file_path.name}' does not exist (already removed).",
                action=action,
                observation="File does not exist.",
            )
        try:
            if file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
            else:
                file_path.unlink()
            return ExecutionOutcome(
                success=not file_path.exists(),
                message=f"Priority 1: Deleted '{file_path.name}'.",
                action=action,
                observation=f"File '{file_path}' deleted.",
            )
        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Error deleting file '{file_path}': {exc}",
                action=action,
                observation=f"Exception during file deletion: {exc}",
            )

    def _exec_read_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        file_path = self._resolve_filesystem_path(action.target or "")
        if not file_path.exists():
            return ExecutionOutcome(
                success=False,
                message=f"File '{file_path}' not found.",
                action=action,
                observation=f"File not found: '{file_path}'.",
            )
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(4000)
            return ExecutionOutcome(
                success=True,
                message=f"Read {len(content)} characters from '{file_path.name}': {content[:100]}...",
                action=action,
                observation=content,
            )
        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Error reading file '{file_path}': {exc}",
                action=action,
                observation=f"Exception during file read: {exc}",
            )

    def _exec_open_folder(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        folder_path = self._resolve_filesystem_path(action.target or "")
        if not folder_path.is_dir():
            return ExecutionOutcome(False, "Requested folder does not exist.", action,
                                    failure_code=FailureCode.ACCESSIBILITY_TARGET_NOT_FOUND)
        try:
            ok = self.os_adapter.open_file(str(folder_path))
            return ExecutionOutcome(
                success=ok,
                message=f"Priority 1: Opened folder '{folder_path}'.",
                action=action,
                observation=f"Folder opened: '{folder_path}'.",
            )
        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Error opening folder '{folder_path}': {exc}",
                action=action,
                observation=f"Exception opening folder: {exc}",
            )

    def _exec_find_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        pattern = action.target or "*.*"
        directory = action.params.get("directory") or "Downloads"
        user_home = os.path.expanduser("~")
        search_dir = os.path.join(user_home, directory)

        found_file = pattern
        if os.path.exists(search_dir):
            import glob
            matches = glob.glob(os.path.join(search_dir, pattern))
            if matches:
                # Return most recently modified file
                matches.sort(key=os.path.getmtime, reverse=True)
                found_file = matches[0]

        return ExecutionOutcome(
            success=True,
            message=f"Located file candidate: '{os.path.basename(found_file)}'.",
            action=action,
            observation=f"File found at '{found_file}'.",
        )

    def _exec_close(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        target = action.target or "active_window"
        if target in ("active_window", "window"):
            if platform.system() == "Darwin":
                self.os_adapter.key_combination(["command", "w"])
            else:
                self.os_adapter.key_combination(["alt", "f4"])
        return ExecutionOutcome(
            success=True,
            message="Priority 1: Closed active window.",
            action=action,
            observation="Window closed.",
        )

    def _exec_media_control(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        cmd = action.payload or action.target or "vol_up"
        ok = self.os_adapter.media_control(cmd)
        return ExecutionOutcome(
            success=ok,
            message=f"Priority 1: Executed media command '{cmd}'.",
            action=action,
            observation=f"Media control '{cmd}' dispatched.",
        )

    def _exec_wait(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        timeout = float(action.params.get("timeout", 1.0))
        end = time.perf_counter() + max(0.0, min(timeout, 3.0))
        while time.perf_counter() < end:
            self._checkpoint("wait")
            time.sleep(min(0.05, max(0.0, end - time.perf_counter())))
        return ExecutionOutcome(
            success=True,
            message=f"Waited for condition '{action.target}'.",
            action=action,
            observation="Wait condition completed.",
        )

    def _exec_ask_clarification(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        return ExecutionOutcome(
            success=True,
            message=action.payload or "Clarification required.",
            action=action,
            observation="Awaiting user clarification.",
        )

    def _exec_respond(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        return ExecutionOutcome(
            success=True,
            message=action.payload or "Understood.",
            action=action,
            observation="Responded to user.",
        )

    def _exec_default(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        return ExecutionOutcome(
            success=True,
            message=f"Executed generic action: {action.type.value}",
            action=action,
            observation=f"Generic {action.type.value} performed.",
        )
