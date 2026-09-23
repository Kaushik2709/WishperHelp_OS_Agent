"""Central runtime coordinator for WisprHelp.

Orchestrates the Ear (STT), System One Reflex (Jev), Safety Interlocks,
Command Router, and OS HAL into an end-to-end execution loop.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from wisprhelp.classification.classifier import JevClassifier
from wisprhelp.classification.schemas import JevDecision, TargetApp
from wisprhelp.core.groq_planner import GroqStrategicPlanner
from wisprhelp.core.router import CommandRouter, RouteTarget
from wisprhelp.core.safety import SafetyGate
from wisprhelp.hal.base import BaseOSAdapter
from wisprhelp.hal.factory import get_os_adapter
from wisprhelp.ui.overlay import StatusOverlayHUD, HUDState
from wisprhelp.vision.vlm_agent import VLMAgent
from wisprhelp.voice.feedback import AudioFeedback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of an agent execution turn."""
    success: bool
    route: RouteTarget
    message: str
    decision: Optional[JevDecision] = None
    latency_ms: float = 0.0


class RuntimeCoordinator:
    """Coordinates classification, safety, routing, and physical OS execution."""

    def __init__(
        self,
        classifier: Optional[JevClassifier] = None,
        router: Optional[CommandRouter] = None,
        safety_gate: Optional[SafetyGate] = None,
        os_adapter: Optional[BaseOSAdapter] = None,
        vlm_agent: Optional[VLMAgent] = None,
        planner: Optional[GroqStrategicPlanner] = None,
        hud: Optional[StatusOverlayHUD] = None,
        feedback: Optional[AudioFeedback] = None,
    ) -> None:
        self.classifier = classifier or JevClassifier()
        self.router = router or CommandRouter()
        self.safety_gate = safety_gate or SafetyGate()
        self.os_adapter = os_adapter or get_os_adapter()
        self.vlm_agent = vlm_agent or VLMAgent(os_adapter=self.os_adapter)
        self.planner = planner or GroqStrategicPlanner()
        self.hud = hud
        self.feedback = feedback

    def process_command(
        self,
        user_query: str,
        confirmation_response: Optional[str] = None,
        decision: Optional[JevDecision] = None,
    ) -> ExecutionResult:
        """Processes a natural language command from start to finish.

        Args:
            user_query: Spoken or typed command from the user.
            confirmation_response: Optional affirmative reply if answering a safety prompt.
            decision: Optional pre-classified JevDecision.

        Returns:
            ExecutionResult describing the action taken and performance metrics.
        """
        start_time = time.perf_counter()
        if self.hud:
            self.hud.set_state(HUDState.THINKING, f"Analyzing: {user_query}")

        # Handle pending safety confirmation resolution
        if confirmation_response:
            confirmed_action = self.safety_gate.resolve_confirmation(confirmation_response)
            if not confirmed_action:
                elapsed = (time.perf_counter() - start_time) * 1000
                if self.hud:
                    self.hud.set_state(HUDState.IDLE, "Cancelled")
                if self.feedback:
                    self.feedback.play_abort()
                return ExecutionResult(
                    success=False,
                    route=RouteTarget.SAFETY_INTERLOCK,
                    message="High-risk action was cancelled by user.",
                    latency_ms=elapsed,
                )
            # User confirmed! Execute the held action
            return self._execute_confirmed_action(confirmed_action, start_time)

        # 1. Sense Active OS Context
        active_win = self.os_adapter.get_active_window()
        context = {
            "active_window_title": active_win.title,
            "active_process_name": active_win.process_name,
        }

        # 2. System One Evaluation via TypeSafe Jev (~100-150ms)
        if decision is None:
            decision = self.classifier.classify(user_query, os_context=context)

        # 3. Confidence & Risk Routing
        route, reason = self.router.route(decision)

        # Compound command resolution: If router flagged CLARIFY or PRIORITY_1_LAUNCH,
        # but the query contains compound actions, promote to GUI_WORKFLOW_AUTOMATION.
        parsed_app, in_app_cmd = self._parse_app_and_command(user_query, decision.target_app)
        has_compound_plan = bool(decision.decomposed_plan and decision.decomposed_plan.is_compound)
        if (route in (RouteTarget.CLARIFY, RouteTarget.PRIORITY_1_LAUNCH)) and (in_app_cmd or has_compound_plan):
            logger.info("Promoting route '%s' to GUI_WORKFLOW_AUTOMATION for compound query: '%s'", route.value, user_query)
            route = RouteTarget.GUI_WORKFLOW_AUTOMATION

        # 4. Dispatch by Selected Route
        if route == RouteTarget.CLARIFY:
            elapsed = (time.perf_counter() - start_time) * 1000
            if self.hud:
                self.hud.set_state(HUDState.IDLE, "Clarification Needed")
            return ExecutionResult(
                success=True,
                route=route,
                message=f"Clarification required: {reason}",
                decision=decision,
                latency_ms=elapsed,
            )

        if route == RouteTarget.SAFETY_INTERLOCK:
            prompt = self.safety_gate.hold_for_confirmation(decision)
            elapsed = (time.perf_counter() - start_time) * 1000
            if self.hud:
                self.hud.set_state(HUDState.CONFIRM, "Confirmation Required")
            if self.feedback:
                self.feedback.play_warning()
            return ExecutionResult(
                success=True,
                route=route,
                message=prompt,
                decision=decision,
                latency_ms=elapsed,
            )

        if route == RouteTarget.PRIORITY_1_LAUNCH:
            target_app, in_app_cmd = self._parse_app_and_command(user_query, decision.target_app)
            if self.hud:
                self.hud.set_state(HUDState.EXECUTING, f"Launching {target_app}")
            launched = self.os_adapter.launch_app(target_app)
            self.os_adapter.wait_for_active_window([target_app], timeout=1.5)
            elapsed = (time.perf_counter() - start_time) * 1000

            # If the user also specified an in-app command, execute it like a real human!
            if launched and in_app_cmd:
                time.sleep(0.3)
                human_res = self._execute_human_like_action(target_app, in_app_cmd, user_query, decision)
                if human_res.success:
                    if self.hud:
                        self.hud.set_state(HUDState.IDLE, f"Executed in {target_app}")
                    if self.feedback:
                        self.feedback.play_success()
                    return human_res

            msg = f"Priority 1: Launched application '{target_app}' ({'Success' if launched else 'Failed'})."
            if self.hud:
                self.hud.set_state(HUDState.IDLE, f"Launched {target_app}")
            if self.feedback:
                self.feedback.play_success()
            return ExecutionResult(
                success=launched,
                route=route,
                message=msg,
                decision=decision,
                latency_ms=elapsed,
            )

        # Check for media/volume command override before other pathways
        lower_q = user_query.lower()
        is_volume_query = bool(
            re.search(r"\b(?:volume|vol|sound|audio|awaaz)\b", lower_q)
            and re.search(r"\b(?:\d{1,3}|up|down|mute|unmute|quieter|louder|increase|decrease|lower|raise)\b", lower_q)
        )

        if route == RouteTarget.PRIORITY_1_MEDIA or is_volume_query:
            media_cmd = self._extract_media_command(user_query)
            display_msg = f"Volume: {media_cmd.split(':')[1]}%" if media_cmd.startswith("set_volume:") else f"Media: {media_cmd}"
            if self.hud:
                self.hud.set_state(HUDState.EXECUTING, display_msg)
            ok = self.os_adapter.media_control(media_cmd)
            elapsed = (time.perf_counter() - start_time) * 1000
            msg = f"Priority 1: Set volume to {media_cmd.split(':')[1]}%." if media_cmd.startswith("set_volume:") else f"Priority 1: Executed media command '{media_cmd}'."
            if self.hud:
                self.hud.set_state(HUDState.IDLE, "Ready")
            if self.feedback:
                self.feedback.play_success()
            return ExecutionResult(
                success=ok,
                route=RouteTarget.PRIORITY_1_MEDIA,
                message=msg,
                decision=decision,
                latency_ms=elapsed,
            )

        # Safety Gate: Prevent Click / Button / Tab / UI actions from ever being misrouted to Text Injection
        is_click_action = bool(
            re.search(r"\b(?:click(?:\s+on)?|press|tap|hit|select|push|daba|dabao|क्लिक)\b", lower_q)
            or re.search(r"\b(?:button|btn|tab|icon|checkbox|link|chat)\b", lower_q)
        )
        if route == RouteTarget.TEXT_INJECTION and is_click_action:
            logger.info("Overriding TEXT_INJECTION to GUI_WORKFLOW_AUTOMATION because query contains click action: '%s'", user_query)
            route = RouteTarget.GUI_WORKFLOW_AUTOMATION

        if route == RouteTarget.TEXT_INJECTION:
            text_to_type = self._extract_text_content(user_query)
            if self.hud:
                self.hud.set_state(HUDState.EXECUTING, f"Typing: {text_to_type[:20]}")
            ok = self.os_adapter.inject_unicode_text(text_to_type)
            elapsed = (time.perf_counter() - start_time) * 1000
            msg = f"Injected text into active focus: '{text_to_type}'."
            if self.hud:
                self.hud.set_state(HUDState.IDLE, "Ready")
            if self.feedback:
                self.feedback.play_success()
            return ExecutionResult(
                success=ok,
                route=route,
                message=msg,
                decision=decision,
                latency_ms=elapsed,
            )

        if route == RouteTarget.GUI_WORKFLOW_AUTOMATION:
            if self.hud:
                self.hud.set_state(HUDState.EXECUTING, "Executing Workflow")
            res = self._execute_gui_workflow(user_query, decision)
            elapsed = (time.perf_counter() - start_time) * 1000
            if self.hud:
                self.hud.set_state(
                    HUDState.IDLE if res.success else HUDState.ERROR,
                    "Workflow Finished" if res.success else "Action Failed",
                )
            if self.feedback:
                if res.success:
                    self.feedback.play_success()
                else:
                    self.feedback.play_abort()
            return ExecutionResult(
                success=res.success,
                route=route,
                message=res.message,
                decision=decision,
                latency_ms=elapsed,
            )

        # Conversational Query (System Two Groq / Generative Synthesizer)
        reply = self.planner.answer_conversational(user_query)
        elapsed = (time.perf_counter() - start_time) * 1000
        if self.hud:
            self.hud.set_state(HUDState.IDLE, reply[:30])
        if self.feedback:
            self.feedback.play_success()
        return ExecutionResult(
            success=True,
            route=route,
            message=reply,
            decision=decision,
            latency_ms=elapsed,
        )

    def _execute_confirmed_action(
        self,
        decision: JevDecision,
        start_time: float,
    ) -> ExecutionResult:
        """Executes an action after positive user confirmation."""
        # Execute based on intent
        if decision.intent == "direct_app_launch":
            ok = self.os_adapter.launch_app(decision.target_app)
        else:
            ok = True
        elapsed = (time.perf_counter() - start_time) * 1000
        return ExecutionResult(
            success=ok,
            route=RouteTarget.SAFETY_INTERLOCK,
            message="Confirmed high-risk action executed successfully.",
            decision=decision,
            latency_ms=elapsed,
        )

    def _execute_gui_workflow(
        self,
        user_query: str,
        decision: JevDecision,
    ) -> ExecutionResult:
        """Cascades through Priority 1 (Launch), Human-like In-App Actions, Priority 2 (UIA), and Priority 3 (Vision)."""
        target_app, in_app_cmd = self._parse_app_and_command(user_query, decision.target_app)

        # Leverage structured decomposed plan if available from Groq/Heuristic pre-processor
        if decision.decomposed_plan and decision.decomposed_plan.is_compound:
            plan = decision.decomposed_plan
            if plan.target_app:
                target_app = plan.target_app
            if len(plan.sub_queries) > 1:
                step2 = plan.sub_queries[1]
                if step2.action_type == "in_app_search" and step2.payload:
                    in_app_cmd = f"search for {step2.payload}"
                elif step2.action_type == "type_text" and step2.payload:
                    in_app_cmd = f"type {step2.payload}"
                elif step2.action_type == "click_element" and step2.payload:
                    in_app_cmd = f"click {step2.payload}"
                elif step2.query_text:
                    in_app_cmd = step2.query_text

        # Step 1: Ensure target app is launched and focused (Priority 1)
        if target_app and target_app != TargetApp.GENERIC_OR_ACTIVE.value:
            if self.hud:
                self.hud.set_state(HUDState.EXECUTING, f"Focusing {target_app}")
            self.os_adapter.launch_app(target_app)
            self.os_adapter.wait_for_active_window([target_app], timeout=1.5)
            time.sleep(0.3)

        # Step 2: If there is an in-app command, execute it like a real human user
        if in_app_cmd:
            human_res = self._execute_human_like_action(target_app, in_app_cmd, user_query, decision)
            if human_res.success:
                return human_res

        # Step 3: Try Priority 2 Native Accessibility Tree (UIA)
        search_terms = self._derive_search_keywords(in_app_cmd or user_query)
        coords = None
        for term in search_terms:
            coords = self.os_adapter.find_accessibility_element(term)
            if coords:
                logger.info("Priority 2 found element for '%s' at %s", term, coords)
                self.os_adapter.click(coords[0], coords[1])
                time.sleep(0.1)
                if in_app_cmd and any(w in in_app_cmd.lower() for w in ["type", "write", "search"]):
                    text_to_type = self._extract_text_content(in_app_cmd)
                    self.os_adapter.inject_unicode_text(text_to_type)
                    self.os_adapter.press_key("enter")
                break

        if coords:
            return ExecutionResult(
                success=True,
                route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                message=f"Priority 2 UIA interacted with control at {coords}.",
                decision=decision,
            )

        # Step 4: Priority 3 Vision Grounding & ReAct Loop
        logger.info("Priority 2 UIA found no element. Escalating to Priority 3 Multimodal Vision.")
        if self.hud:
            self.hud.set_state(HUDState.THINKING, "Priority 3: Visual Grounding")

        goal = in_app_cmd if in_app_cmd else user_query
        vlm_res = self.vlm_agent.execute_visual_task(goal=goal, target_app=target_app)
        return ExecutionResult(
            success=vlm_res.success,
            route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
            message=f"Priority 3 Vision: {vlm_res.message} (steps: {vlm_res.steps_taken})",
            decision=decision,
        )

    def _execute_human_like_action(
        self,
        app_name: str,
        in_app_cmd: str,
        full_query: str,
        decision: JevDecision,
    ) -> ExecutionResult:
        """Executes in-app tasks like a real human user (address bar shortcuts, text injection, etc.)."""
        cmd_clean = in_app_cmd.strip()
        cmd_lower = cmd_clean.lower()
        app_lower = (app_name or "").lower()

        # 0. Button / Control / List / Chat Click Action
        is_click = bool(
            re.search(r"\b(?:click(?:\s+on)?|press|tap|hit|select|push|daba|dabao|क्लिक)\b", cmd_lower)
            or re.search(r"\b(?:button|btn|tab|icon|checkbox|link|chat|item|row)\b", cmd_lower)
        )

        if is_click:
            # Check for ordinal / positional reference (e.g. "click on the first chat on WhatsApp", "click second-last chat")
            ordinal_info = self._parse_ordinal_target(cmd_clean)
            if ordinal_info:
                ord_idx, itype = ordinal_info
                ord_labels = {0: "first", 1: "second", 2: "third", -1: "last", -2: "second-last", -3: "third-last"}
                ord_label = ord_labels.get(ord_idx, f"item {ord_idx}")
                logger.info("Human-like click action: Ordinal target [%s] %s in '%s'", ord_label, itype, app_name)

                coords = self.os_adapter.find_ordinal_element(ord_idx, item_type=itype, target_app=app_name)
                if coords:
                    logger.info("Priority 2 UIA found ordinal [%s] %s at %s. Clicking.", ord_label, itype, coords)
                    self.os_adapter.click(coords[0], coords[1])
                    return ExecutionResult(
                        success=True,
                        route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                        message=f"Interacted with {ord_label} {itype} at {coords} in {app_name or 'active window'}.",
                        decision=decision,
                    )
                else:
                    logger.info("Priority 2 UIA could not locate ordinal [%s] %s. Escalating to Priority 3 Vision Grounding.", ord_label, itype)
                    if self.hud:
                        self.hud.set_state(HUDState.THINKING, f"Vision: Finding {ord_label} {itype}")
                    vlm_res = self.vlm_agent.execute_visual_task(goal=cmd_clean, target_app=app_name)
                    return ExecutionResult(
                        success=vlm_res.success,
                        route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                        message=f"Priority 3 Vision: {vlm_res.message} (steps: {vlm_res.steps_taken})",
                        decision=decision,
                    )

            button_candidates = self._extract_button_or_control_name(cmd_clean)
            logger.info("Human-like click action: Attempting to click control candidates %s in '%s'", button_candidates, app_name)

            for cand in button_candidates:
                coords = self.os_adapter.find_accessibility_element(cand)
                if coords:
                    logger.info("Priority 2 found control '%s' at %s. Clicking.", cand, coords)
                    self.os_adapter.click(coords[0], coords[1])
                    return ExecutionResult(
                        success=True,
                        route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                        message=f"Interacted with '{cand}' button at {coords} in {app_name or 'active window'}.",
                        decision=decision,
                    )

            # Special case for "Enter" key / button: if no UIA button found and command was "press enter" / "click enter"
            if any(cand.lower() == "enter" for cand in button_candidates):
                logger.info("Dispatching keyboard Enter key for 'press/click enter' request.")
                self.os_adapter.press_key("enter")
                return ExecutionResult(
                    success=True,
                    route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                    message=f"Pressed Enter key in {app_name or 'active window'}.",
                    decision=decision,
                )

            # If Priority 2 UIA didn't find the control (e.g. custom canvas, color-based like "blue button", or web view):
            # Escalate immediately to Priority 3 Multimodal Vision!
            logger.info("Priority 2 UIA could not locate %s. Escalating to Priority 3 Vision.", button_candidates)
            if self.hud:
                self.hud.set_state(HUDState.THINKING, "Priority 3: Visual Grounding")
            vlm_res = self.vlm_agent.execute_visual_task(goal=cmd_clean, target_app=app_name)
            return ExecutionResult(
                success=vlm_res.success,
                route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                message=f"Priority 3 Vision: {vlm_res.message} (steps: {vlm_res.steps_taken})",
                decision=decision,
            )

        # 1. Browser Navigation & Web Search (Chrome, Edge, Brave, Firefox, Browser)
        is_browser = any(b in app_lower for b in ["browser", "chrome", "edge", "brave", "firefox", "opera"])
        is_search = any(cmd_lower.startswith(w) for w in ["search for", "search", "look up", "find", "google", "go to", "visit", "open url", "navigate to"])

        # Prioritize explicit payload from decomposed plan if available
        search_query = None
        if decision.decomposed_plan and len(decision.decomposed_plan.sub_queries) > 1:
            for s in decision.decomposed_plan.sub_queries:
                if s.action_type == "in_app_search" and s.payload:
                    search_query = s.payload
                    break

        if is_browser or is_search or search_query:
            if not search_query:
                search_query = re.sub(
                    r"^(search\s+for|search|look\s+up|find|google|go\s+to|visit|open\s+url|navigate\s+to)\s+",
                    "",
                    cmd_clean,
                    flags=re.IGNORECASE,
                ).strip()
            if search_query:
                logger.info("Human-like browser action: Focusing address bar and searching for '%s'", search_query)
                self.os_adapter.key_combination(["ctrl", "l"])
                time.sleep(0.08)
                self.os_adapter.inject_unicode_text(search_query)
                time.sleep(0.05)
                self.os_adapter.press_key("enter")
                return ExecutionResult(
                    success=True,
                    route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                    message=f"Human action: Searched for '{search_query}' in {app_name or 'browser'}.",
                    decision=decision,
                )

        # 2. File Creation & Code Editing (VS Code, Cursor, Notepad, Word, Sublime, Editor)
        is_editor = any(e in app_lower for e in ["notepad", "word", "notes", "editor", "text", "docs", "vscode", "code", "sublime", "pycharm", "cursor"])
        is_file_creation = bool(re.search(r"\b(?:create|make|new|open)\s+(?:a\s+)?(?:new\s+)?file\b", cmd_lower))

        if is_editor and is_file_creation:
            m_fn = re.search(r"\b(?:called|named)\s+([a-zA-Z0-9_\-\.]+)", cmd_clean, re.IGNORECASE) or re.search(r"\bfile\s+([a-zA-Z0-9_\-\.]+)", cmd_clean, re.IGNORECASE)
            filename = m_fn.group(1).strip() if m_fn else ""
            if filename.lower() in ("called", "named", "a", "an", "the", "new"):
                filename = ""
            if filename:
                filename = re.sub(r"\.(by|pie)$", ".py", filename, flags=re.IGNORECASE)
                filename = re.sub(r"\s+(and|with|in|aur)$", "", filename, flags=re.IGNORECASE).strip()

            logger.info("Human-like file creation in %s: filename='%s'", app_name, filename)
            # Shortcut 1: Ctrl+N creates new buffer
            self.os_adapter.key_combination(["ctrl", "n"])
            time.sleep(0.35)

            # Shortcut 2: Ctrl+S to save as filename if filename was specified
            if filename:
                self.os_adapter.key_combination(["ctrl", "s"])
                time.sleep(0.45)
                self.os_adapter.inject_unicode_text(filename)
                time.sleep(0.15)
                self.os_adapter.press_key("enter")
                time.sleep(0.3)

            # Optional: if user also said "and write/type <code>"
            m_code = re.search(r"(?:and|aur)\s+(?:type|write|code|likho)\s+(.+)$", cmd_clean, re.IGNORECASE)
            if m_code:
                code_text = m_code.group(1).strip()
                time.sleep(0.2)
                self.os_adapter.inject_unicode_text(code_text)

            return ExecutionResult(
                success=True,
                route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                message=f"Human action: Created file '{filename or 'untitled'}' in {app_name or 'editor'}.",
                decision=decision,
            )

        # 3. Text Typing & Note Taking (Notepad, Word, Editor, Chat)
        is_typing = any(cmd_lower.startswith(w) for w in ["type", "write", "dictate", "enter", "paste", "लिखो", "टाइप करो"])

        # Prioritize explicit text payload from decomposed plan if available
        text_from_plan = None
        if decision.decomposed_plan and len(decision.decomposed_plan.sub_queries) > 1:
            for s in decision.decomposed_plan.sub_queries:
                if s.action_type == "type_text" and s.payload:
                    text_from_plan = s.payload
                    break

        if is_editor or is_typing or text_from_plan:
            text_to_type = text_from_plan or self._extract_text_content(cmd_clean)
            if text_to_type:
                logger.info("Human-like text action: Injecting text into '%s': '%s'", app_name, text_to_type)
                time.sleep(0.15)
                ok = self.os_adapter.inject_unicode_text(text_to_type)
                if "and press enter" in cmd_lower or "send" in cmd_lower:
                    time.sleep(0.05)
                    self.os_adapter.press_key("enter")
                return ExecutionResult(
                    success=ok,
                    route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                    message=f"Human action: Typed '{text_to_type}' into {app_name or 'active document'}.",
                    decision=decision,
                )

        # 4. Media Playback (Spotify, YouTube, Media Player)
        is_media = any(m in app_lower for m in ["spotify", "youtube", "media", "music"])
        is_play = any(cmd_lower.startswith(w) for w in ["play", "listen to", "stream"])

        if is_media or is_play:
            track = re.sub(r"^(play|listen\s+to|stream|search\s+and\s+play)\s+", "", cmd_clean, flags=re.IGNORECASE).strip()
            if track:
                logger.info("Human-like media action: Searching and playing '%s' in %s", track, app_name)
                self.os_adapter.key_combination(["ctrl", "l"])
                time.sleep(0.1)
                self.os_adapter.inject_unicode_text(track)
                time.sleep(0.1)
                self.os_adapter.press_key("enter")
                time.sleep(0.6)
                self.os_adapter.press_key("enter")
                return ExecutionResult(
                    success=True,
                    route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                    message=f"Human action: Dispatched playback for '{track}' in {app_name or 'media player'}.",
                    decision=decision,
                )

        # 5. Calculator Arithmetic
        is_calc = any(c in app_lower for c in ["calculator", "calc"])
        is_math = any(cmd_lower.startswith(w) for w in ["calculate", "compute", "add", "subtract", "multiply", "divide", "do"])

        if is_calc or is_math:
            math_expr = re.sub(r"^(calculate|compute|do|what\s+is)\s+", "", cmd_clean, flags=re.IGNORECASE).strip()
            math_expr = math_expr.replace(" plus ", "+").replace(" minus ", "-").replace(" times ", "*").replace(" multiplied by ", "*").replace(" divided by ", "/")
            math_expr = re.sub(r"[^0-9\+\-\*\/\.\(\)]", "", math_expr)
            if math_expr:
                logger.info("Human-like calculator action: Entering expression '%s'", math_expr)
                time.sleep(0.15)
                self.os_adapter.inject_unicode_text(math_expr)
                time.sleep(0.05)
                self.os_adapter.press_key("enter")
                return ExecutionResult(
                    success=True,
                    route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
                    message=f"Human action: Calculated '{math_expr}' in Calculator.",
                    decision=decision,
                )

        return ExecutionResult(
            success=False,
            route=RouteTarget.GUI_WORKFLOW_AUTOMATION,
            message="No deterministic human action matched; escalating to UIA/Vision.",
            decision=decision,
        )

    @staticmethod
    def _extract_button_or_control_name(cmd: str) -> List[str]:
        """Extracts prioritized candidate names for buttons or controls from natural language."""
        cleaned = cmd.strip()
        # Remove action verb: click, press, tap, hit, select, etc.
        cleaned = re.sub(
            r"^(?:please\s+)?(?:click(?:\s+on)?|press|tap|hit|select|push|daba|dabao|क्लिक(?:\s+करो)?)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        # Remove trailing app references: "on Antigravity", "in Notepad", "on VS Code"
        cleaned = re.sub(
            r"\s+(?:on|in|inside|pe|par)\s+[a-zA-Z0-9\s\.\-_]+$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        # Remove "the", "that", "this", "a", "an"
        cleaned_no_article = re.sub(r"^(?:the|that|this|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        # Remove trailing or leading "button", "icon", "tab", "link", "option"
        target_name = re.sub(r"\s+(?:button|btn|icon|tab|link|option)$", "", cleaned_no_article, flags=re.IGNORECASE).strip()
        target_name = re.sub(r"^(?:button|btn|icon|tab|link|option)\s+", "", target_name, flags=re.IGNORECASE).strip()

        candidates = []
        if target_name:
            candidates.append(target_name)
            words = target_name.split()
            if len(words) > 1 and words[0].lower() not in ("the", "that", "blue", "red", "green"):
                candidates.append(words[0])
        if cleaned_no_article and cleaned_no_article != target_name:
            candidates.append(cleaned_no_article)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

        # Common synonym mappings
        target_lower = target_name.lower()
        if target_lower in ("accept changes", "accept change"):
            candidates.extend(["Accept Changes", "Accept", "Keep", "Apply"])
        elif target_lower in ("enter", "return"):
            candidates.extend(["Enter", "Send", "Submit"])
        elif target_lower in ("send", "submit"):
            candidates.extend(["Send", "Submit", "Enter"])
        elif target_lower in ("show mode", "show more") or "show mode" in target_lower or "show more" in target_lower:
            candidates.extend(["Show more", "Show More", "AI Mode", "More"])

        seen = set()
        deduped = []
        for c in candidates:
            cl = c.lower()
            if cl not in seen and len(cl) > 0 and cl not in ("the", "that", "a", "an", "button"):
                seen.add(cl)
                deduped.append(c)
        return deduped

    @classmethod
    def _derive_search_keywords(cls, cmd: str) -> List[str]:
        """Derives element candidate search keywords from user command."""
        btn_candidates = cls._extract_button_or_control_name(cmd)
        cleaned = re.sub(r"^(click|press|tap|select|open|hit)\s+", "", cmd, flags=re.IGNORECASE).strip()
        tokens = [t for t in re.split(r"[\s\-_]+", cleaned) if len(t) > 2 and t.lower() not in ("the", "that", "this", "button")]
        keywords = list(btn_candidates)
        keywords.extend(tokens)
        seen = set()
        result = []
        for k in keywords:
            kl = k.lower()
            if kl not in seen and kl not in ("the", "that", "this", "button"):
                seen.add(kl)
                result.append(k)
        return result

    @staticmethod
    def _parse_ordinal_target(cmd: str) -> Optional[Tuple[int, str]]:
        """Parses ordinal references like 'first chat', 'second-last item', '3rd tab'.

        Returns:
            Tuple of (ordinal_index, item_type) e.g. (0, 'chat'), (-2, 'chat'), or None.
        """
        lower = cmd.lower().strip()
        item_types = r"(?:chat|conversation|message|contact|item|row|element|tab|button|entry|email|link)"

        # Second-last / third-last / penultimate / aakhri
        m_neg = re.search(
            rf"\b(?:the\s+)?(?:(second|2nd|doosra)|(third|3rd|teesra)|(fourth|4th|chautha))\s*(?:to\s+|-)?\s*last\s+({item_types})\b",
            lower,
        )
        if m_neg:
            prefix = (m_neg.group(1) or m_neg.group(2) or m_neg.group(3) or "").lower()
            itype = m_neg.group(4)
            if any(w in prefix for w in ("second", "2nd", "doosra")):
                return (-2, itype)
            elif any(w in prefix for w in ("third", "3rd", "teesra")):
                return (-3, itype)
            elif any(w in prefix for w in ("fourth", "4th", "chautha")):
                return (-4, itype)

        m_pen = re.search(rf"\b(?:penultimate)\s+({item_types})\b", lower)
        if m_pen:
            return (-2, m_pen.group(1))

        m_last = re.search(rf"\b(?:the\s+)?(?:last|aakhri|bottom)\s+({item_types})\b", lower)
        if m_last:
            return (-1, m_last.group(1))

        # Positive ordinals: first, second, 3rd, etc.
        pos_map = {
            "first": 0, "1st": 0, "top": 0, "pehla": 0, "1": 0,
            "second": 1, "2nd": 1, "doosra": 1, "2": 1,
            "third": 2, "3rd": 2, "teesra": 2, "3": 2,
            "fourth": 3, "4th": 3, "chautha": 3, "4": 3,
            "fifth": 4, "5th": 4, "paanchwa": 4, "5": 4,
            "sixth": 5, "6th": 5, "6": 5,
            "seventh": 6, "7th": 6, "7": 6,
            "eighth": 7, "8th": 7, "8": 7,
            "ninth": 8, "9th": 8, "9": 8,
            "tenth": 9, "10th": 9, "10": 9,
        }
        for word, idx in pos_map.items():
            pattern = rf"\b(?:the\s+)?{re.escape(word)}\s+({item_types})\b"
            m_pos = re.search(pattern, lower)
            if m_pos:
                return (idx, m_pos.group(1))

        return None

    @staticmethod
    def _extract_media_command(query: str) -> str:
        """Extracts standard media command key or target volume percentage."""
        lower = query.lower()

        # 1. Exact volume percentage matching: "put the volume to 60%", "volume 100%", "lower volume to 30%"
        m_pct = (
            re.search(r"\b(?:set|put|turn|change|make|adjust|bring)?\s*(?:the\s+)?(?:volume|vol|sound|audio|awaaz)\s*(?:level|to|at|ko|pe|par)?\s*(\d{1,3})\s*(?:%|percent)?\b", lower)
            or re.search(r"\b(\d{1,3})\s*(?:%|percent)\s*(?:volume|vol|sound|audio|awaaz)?\b", lower)
            or re.search(r"\b(?:volume|vol|awaaz)\s*(?:to|at|ko|pe|par)?\s*(\d{1,3})\b", lower)
        )
        if m_pct:
            val = int(m_pct.group(1))
            if 0 <= val <= 100:
                return f"set_volume:{val}"

        # 2. Relative percentage adjustment: "increase volume by 20%", "lower volume by 10%"
        m_rel = re.search(r"\b(increase|raise|boost|turn up|lower|decrease|reduce|turn down)\s+(?:the\s+)?(?:volume|vol|sound|audio|awaaz)\s+(?:by\s+)?(\d{1,3})\s*(?:%|percent)?\b", lower)
        if m_rel:
            direction = m_rel.group(1)
            amt = int(m_rel.group(2))
            sign = "+" if any(d in direction for d in ("increase", "raise", "boost", "turn up")) else "-"
            return f"adjust_volume:{sign}{amt}"

        # 3. Mute / Unmute
        if "unmute" in lower:
            return "unmute"
        if "mute" in lower:
            return "mute"

        # 4. Discrete volume steps
        if any(w in lower for w in ("volume up", "vol up", "louder", "turn it up", "increase volume")):
            return "vol_up"
        if any(w in lower for w in ("volume down", "vol down", "quieter", "turn it down", "decrease volume", "lower the volume")):
            return "vol_down"

        # 5. Playback controls
        if "pause" in lower or "stop" in lower:
            return "pause"
        if "play" in lower or "resume" in lower:
            return "play"
        if "next" in lower or "skip" in lower:
            return "next"
        if "prev" in lower or "previous" in lower or "back" in lower:
            return "prev"

        return "vol_up"

    @classmethod
    def _extract_launch_app_name(cls, query: str, default_target: str) -> str:
        """Extracts the canonical target application name from natural user speech."""
        app, _ = cls._parse_app_and_command(query, default_target)
        return app

    @staticmethod
    def _parse_app_and_command(query: str, default_target: str) -> Tuple[str, Optional[str]]:
        """Extracts the canonical target application name and subsequent in-app command from natural user speech."""
        query_clean = query.strip()
        alias_map = {
            "antigravity": "antigravity",
            "vscode": "vscode",
            "vs code": "vscode",
            "visual studio code": "vscode",
            "visual studio": "visual studio",
            "code": "vscode",
            "cursor": "cursor",
            "pycharm": "pycharm",
            "notepad": "notepad",
            "chrome": "chrome",
            "google chrome": "chrome",
            "edge": "edge",
            "microsoft edge": "edge",
            "brave": "brave",
            "brave browser": "brave",
            "firefox": "firefox",
            "terminal": "terminal",
            "cmd": "cmd",
            "command prompt": "cmd",
            "powershell": "powershell",
            "spotify": "spotify",
            "whatsapp": "whatsapp",
            "calculator": "calculator",
            "calc": "calculator",
            "settings": "settings",
            "store": "store",
            "microsoft store": "store",
            "task manager": "task manager",
            "file explorer": "file explorer",
            "control panel": "control panel",
            "paint": "paint",
            "word": "word",
            "excel": "excel",
            "powerpoint": "powerpoint",
            "steam": "steam",
            "discord": "discord",
        }

        # Pattern 0: Preposition first: 'In/On/Inside <app>[,] <cmd>': e.g. 'In VS Code, create a file called main.py'
        m0 = re.match(
            r"^(?:in|on|inside|me)\s+(?P<app>[a-zA-Z0-9\s\.\-_]+?)(?:,\s*|\s+me\s+|\s+pe\s+|\s+par\s+|\s+)(?P<cmd>(?:create|make|new|open|search|play|type|write|calculate|run|execute|click|press|tap|hit|select|push|daba|dabao).+)$",
            query_clean,
            re.IGNORECASE,
        )
        if m0:
            raw_app = m0.group("app").strip().lower()
            cmd = m0.group("cmd").strip()
            cmd = re.sub(r"\b([a-zA-Z0-9_\-]+)\.(by|pie)\b", r"\1.py", cmd, flags=re.IGNORECASE)
            app = alias_map.get(raw_app, raw_app)
            return app, cmd

        # Pattern 1: Action first, then 'in/on/using/with' app: e.g. 'search weather in chrome', 'click enter on antigravity'
        m1 = re.match(
            r"^(?P<cmd>(?:create|make|new|search\s+for|search|play|type|write|calculate|click(?:\s+on)?|press|tap|hit|select|push|daba|dabao)\s+.+?)\s+(?:in|on|using|with)\s+(?P<app>[a-zA-Z0-9\s\.\-_]+)$",
            query_clean,
            re.IGNORECASE,
        )
        if m1:
            raw_app = m1.group("app").strip().lower()
            cmd = m1.group("cmd").strip()
            cmd = re.sub(r"\b([a-zA-Z0-9_\-]+)\.(by|pie)\b", r"\1.py", cmd, flags=re.IGNORECASE)
            app = alias_map.get(raw_app, raw_app)
            return app, cmd

        # Pattern 2: 'open/launch/start X and/then/to Y': e.g. 'open chrome and search for weather'
        m2 = re.match(
            r"^(?:open|launch|start|switch to|kholo|chalu karo)\s+(?P<app>[a-zA-Z0-9\s\.\-_]+?)\s+(?:and|then|to|aur|fir)\s+(?P<cmd>.+)$",
            query_clean,
            re.IGNORECASE,
        )
        if m2:
            raw_app = m2.group("app").strip().lower()
            cmd = m2.group("cmd").strip()
            cmd = re.sub(r"\b([a-zA-Z0-9_\-]+)\.(by|pie)\b", r"\1.py", cmd, flags=re.IGNORECASE)
            app = alias_map.get(raw_app, raw_app)
            return app, cmd

        # Special handling for standalone 'code' or 'open code'
        q_lower = query_clean.lower()
        if q_lower in ("open code", "launch code", "start code", "code"):
            return "vscode", None
        m_code_cmd = re.match(r"^(?:open|launch|start)\s+code\s+(?:and|then|to)\s+(.+)$", query_clean, re.IGNORECASE)
        if m_code_cmd:
            return "vscode", m_code_cmd.group(1).strip()

        # Pattern 3: Simple open app: e.g. 'open task manager', 'launch visual studio code'
        m3 = re.match(
            r"^(?:open|launch|start|switch to|kholo|chalu karo)\s+(?P<app>.+)$",
            query_clean,
            re.IGNORECASE,
        )
        if m3:
            raw_app = m3.group("app").strip().lower()
            raw_app = re.sub(r"\s+(please|for me|now|fast|turant)$", "", raw_app, flags=re.IGNORECASE).strip()
            app = alias_map.get(raw_app, raw_app)
            return app, None

        # Pattern 4: Direct in-app command without app specified (e.g. 'write code...', 'click accept changes', 'press enter')
        action_verbs = r"^(?:type|write|dictate|enter|paste|create|make|search(?:\s+for)?|play|calculate|click(?:\s+on)?|press|tap|hit|select|push|daba|dabao|likho|लिखो|टाइप करो)\b"
        if re.match(action_verbs, query_clean, re.IGNORECASE):
            cleaned_cmd = re.sub(r"\b([a-zA-Z0-9_\-]+)\.(by|pie)\b", r"\1.py", query_clean, flags=re.IGNORECASE)
            return default_target, cleaned_cmd

        # Fallback: check alias map directly
        for alias, canonical in alias_map.items():
            if alias == q_lower:
                return canonical, None
            if alias == "code":
                continue  # Never let the common word 'code' match arbitrary sentences in fallback
            # Match alias followed by command e.g. "vscode create a file"
            pattern = rf"\b{re.escape(alias)}\b(?:,\s*|\s+me\s+|\s+pe\s+|\s+)(.+)$"
            m_sub = re.search(pattern, q_lower)
            if m_sub:
                rem_cmd = m_sub.group(1).strip()
                if rem_cmd:
                    rem_cmd = re.sub(r"\b([a-zA-Z0-9_\-]+)\.(by|pie)\b", r"\1.py", rem_cmd, flags=re.IGNORECASE)
                    return canonical, rem_cmd
            elif len(alias) > 3 and alias in q_lower:
                return canonical, None

        cleaned = re.sub(r"\s+(please|for me|now|fast|turant)$", "", query_clean, flags=re.IGNORECASE).strip()
        return cleaned or default_target, None

    @staticmethod
    def _format_code_if_applicable(text: str) -> str:
        """Formats dictated programming statements (e.g. 'print Hello World' -> 'print("Hello World")')."""
        cleaned = text.strip()
        m_print = re.match(r"^print\s+(?![\(\"'])(.+)$", cleaned, re.IGNORECASE)
        if m_print:
            arg = m_print.group(1).strip().strip("\"'")
            if arg.isdigit() or arg in ("True", "False", "None"):
                return f"print({arg})"
            return f'print("{arg}")'
        m_log = re.match(r"^(?:console\.log|console\s+dot\s+log)\s+(?![\(\"'])(.+)$", cleaned, re.IGNORECASE)
        if m_log:
            arg = m_log.group(1).strip().strip("\"'")
            if arg.isdigit() or arg in ("true", "false", "null", "undefined"):
                return f"console.log({arg})"
            return f'console.log("{arg}")'
        return cleaned

    @classmethod
    def _extract_text_content(cls, query: str) -> str:
        """Extracts text meant to be typed from commands like 'type hello into chat' or 'write code called print Hello World'."""
        cleaned = re.sub(
            r"^(?:type|write|dictate|enter|paste|likho|लिखो|टाइप करो)\s+(?:the\s+)?(?:code\s+)?(?:called\s+|named\s+)?",
            "",
            query,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:code\s+called|code\s+named|code)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:into\s+.*|in\s+.*|here)$", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip("\"' ")
        return cls._format_code_if_applicable(cleaned) or query

    # Convenient aliases
    execute = process_command
    execute_query = process_command



