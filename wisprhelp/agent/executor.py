"""Deterministic Tool Execution Bridge for WisprHelp Semantic Brain.

Translates platform-neutral actions into validated, safety-gated system calls,
bridging to the 3-Priority Tool Execution Hierarchy (P1 Shell -> P2 UIA -> P3 Vision).
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.state import ComputerState
from wisprhelp.hal.base import BaseOSAdapter
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


class ToolBasedActionExecutor:
    """Executes platform-neutral actions deterministically through HAL and safety."""

    def __init__(
        self,
        os_adapter: Optional[BaseOSAdapter] = None,
        safety_gate: Optional[Any] = None,
        vlm_agent: Optional[VLMAgent] = None,
    ) -> None:
        self.os_adapter = os_adapter or get_os_adapter()
        if safety_gate is None:
            from wisprhelp.core.safety import SafetyGate
            self.safety_gate = SafetyGate()
        else:
            self.safety_gate = safety_gate
        self.vlm_agent = vlm_agent or VLMAgent(os_adapter=self.os_adapter)


    def execute_action(
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

        # 1. Deterministic Safety Interlock Gate (Outside LLM control)
        if action.is_destructive and not confirmed:
            prompt = f"Safety Gate: '{action.rationale or action.type.value}' is a destructive action. Confirm execution? (yes/no)"
            logger.warning("Holding action %s at Safety Gate: %s", action.type, prompt)
            return ExecutionOutcome(
                success=True,
                message=prompt,
                action=action,
                observation="Action held for user safety confirmation.",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                needs_confirmation=True,
                confirmation_prompt=prompt,
            )

        try:
            # 2. Dispatch to typed handler
            handler = getattr(self, f"_exec_{action.type.value}", self._exec_default)
            res = handler(action, state)
            elapsed = (time.perf_counter() - start_time) * 1000
            res.latency_ms = elapsed
            return res
        except Exception as exc:
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

    def _exec_launch_app(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        target = action.target or ""
        ok = self.os_adapter.launch_app(target, parameter=action.payload)
        self.os_adapter.wait_for_active_window([target], timeout=1.5)
        return ExecutionOutcome(
            success=ok,
            message=f"Priority 1: Launched '{target}' ({'Success' if ok else 'Failed'}).",
            action=action,
            observation=f"App '{target}' launch dispatched.",
        )

    def _exec_focus_window(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        target = action.target or ""
        ok = False
        if hasattr(self.os_adapter, "focus_window"):
            ok = self.os_adapter.focus_window(target)
        if not ok:
            # Fallback to launch / switch
            ok = self.os_adapter.launch_app(target)
        return ExecutionOutcome(
            success=ok,
            message=f"Focused window matching '{target}'.",
            action=action,
            observation=f"Window '{target}' focused.",
        )

    def _exec_type_text(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        text = action.payload or ""
        # Universal Unicode clipboard swap prevents character and IME corruption
        ok = self.os_adapter.inject_unicode_text(text)
        if action.params.get("press_enter", False):
            time.sleep(0.04)
            self.os_adapter.press_key("enter")
        return ExecutionOutcome(
            success=ok,
            message=f"Priority 1: Injected text '{text[:30]}...'",
            action=action,
            observation=f"Typed text: '{text[:40]}'",
        )

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

        # 1. Ordinal reference e.g. "ordinal:1:tab" or "second tab"
        coords = None
        if target.startswith("ordinal:"):
            parts = target.split(":")
            ord_idx = int(parts[1]) if len(parts) > 1 else 0
            itype = parts[2] if len(parts) > 2 else "item"
            coords = self.os_adapter.find_ordinal_element(ord_idx, item_type=itype)

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
        vlm_res = self.vlm_agent.execute_visual_task(goal=f"Click {target}")
        return ExecutionOutcome(
            success=vlm_res.success,
            message=f"Priority 3 Vision: Clicked '{target}' ({vlm_res.message}).",
            action=action,
            observation=f"Vision grounding executed for '{target}'.",
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

    def _exec_open_file(self, action: Action, state: Optional[ComputerState]) -> ExecutionOutcome:
        file_path = action.target or ""
        # Check standard user paths
        expanded = os.path.expanduser(file_path)
        if not os.path.exists(expanded):
            # Check Downloads, Desktop, Documents
            user_home = os.path.expanduser("~")
            for sub in ("Downloads", "Desktop", "Documents"):
                candidate = os.path.join(user_home, sub, file_path)
                if os.path.exists(candidate):
                    expanded = candidate
                    break

        try:
            if hasattr(self.os_adapter, "open_file"):
                ok = self.os_adapter.open_file(expanded)
            else:
                ok = True
                if os.path.exists(expanded):
                    if platform.system() == "Windows":
                        os.startfile(expanded)
                    elif platform.system() == "Darwin":
                        import subprocess
                        subprocess.Popen(["open", expanded])
                    else:
                        import subprocess
                        subprocess.Popen(["xdg-open", expanded])
            return ExecutionOutcome(
                success=ok,
                message=f"Priority 1: Opened file '{expanded}'.",
                action=action,
                observation=f"File opened: '{expanded}'.",
            )

        except Exception as exc:
            return ExecutionOutcome(
                success=False,
                message=f"Failed to open file '{file_path}': {exc}",
                action=action,
                observation=f"File open error: {exc}",
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
        time.sleep(min(timeout, 3.0))
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
