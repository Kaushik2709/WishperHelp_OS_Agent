"""Dynamic Task Planning, Mutable Task Graph, and Closed-Loop Replanning.

Implements the Plan -> Execute -> Observe -> Verify -> Update State -> Replan cycle,
enabling self-correction, dynamic recovery from unexpected UI states, and
graceful handling of user interruptions and corrections.
"""

from __future__ import annotations

import logging
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.executor import ExecutionOutcome, ToolBasedActionExecutor
from wisprhelp.agent.memory import ConversationMemory, ConversationTurn
from wisprhelp.agent.planning import StateAwarePlanner
from wisprhelp.agent.plan_classifier import PlanClassifier
from wisprhelp.agent.reasoning import GoalUnderstanding, SemanticReasoner
from wisprhelp.agent.references import ReferenceResolver
from wisprhelp.agent.state import ComputerState, ComputerStateManager
from wisprhelp.classification.schemas import JevDecision
from wisprhelp.hal.base import BaseOSAdapter
from wisprhelp.hal.factory import get_os_adapter
from wisprhelp.core.outcomes import FailureCode, ExecutionFault
from wisprhelp.core.safety import is_explicit_confirmation
from wisprhelp.agent.authorization import plan_authorization_error

logger = logging.getLogger(__name__)


@dataclass
class TaskNode:
    """A single node within the mutable task graph."""
    node_id: str
    action: Action
    status: str = "pending"  # "pending", "executing", "completed", "failed", "cancelled"
    outcome: Optional[ExecutionOutcome] = None
    observation: str = ""
    retries: int = 0
    max_retries: int = 2


@dataclass
class TaskGraph:
    """A mutable, dynamic plan representing an open-ended multi-step computer task."""
    goal: str
    nodes: List[TaskNode] = field(default_factory=list)
    active_index: int = 0
    is_cancelled: bool = False
    steps_used: int = 0
    replans_used: int = 0

    def add_action(self, action: Action) -> TaskNode:
        node_id = f"step_{len(self.nodes) + 1}"
        node = TaskNode(node_id=node_id, action=action)
        self.nodes.append(node)
        return node

    def get_current_node(self) -> Optional[TaskNode]:
        if 0 <= self.active_index < len(self.nodes):
            return self.nodes[self.active_index]
        return None

    def replace_remaining(self, new_actions: List[Action]) -> None:
        """Mutates the remaining steps of the task graph in response to a correction or replan."""
        del self.nodes[self.active_index:]
        for act in new_actions:
            self.add_action(act)

    def cancel(self) -> None:
        self.is_cancelled = True
        for node in self.nodes:
            if node.status in ("pending", "executing"):
                node.status = "cancelled"

    @property
    def is_finished(self) -> bool:
        if self.is_cancelled:
            return True
        return self.active_index >= len(self.nodes)


@dataclass
class TaskTrace:
    """Structured audit and debugging trace for observability."""
    user_input: str
    semantic_goal: str
    active_app: str
    active_window: str
    entities: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    replanning_events: List[str] = field(default_factory=list)
    final_result: str = ""
    success: bool = True
    total_latency_ms: float = 0.0
    failure_code: FailureCode = FailureCode.NONE


@dataclass
class AgentTurnResult:
    """The outcome of a complete agent conversational turn."""
    success: bool
    message: str
    goal: str
    trace: TaskTrace
    needs_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    pending_held_action: Optional[Action] = None
    actions_executed: List[Action] = field(default_factory=list)
    latency_ms: float = 0.0
    failure_code: FailureCode = FailureCode.NONE


class SemanticAgent:
    """The central cognitive brain of WisprHelp."""

    def __init__(
        self,
        state_mgr: Optional[ComputerStateManager] = None,
        memory: Optional[ConversationMemory] = None,
        ref_resolver: Optional[ReferenceResolver] = None,
        reasoner: Optional[SemanticReasoner] = None,
        executor: Optional[ToolBasedActionExecutor] = None,
        classifier: Optional[Any] = None,
        safety_gate: Optional[Any] = None,
        os_adapter: Optional[BaseOSAdapter] = None,
        plan_classifier: Optional[PlanClassifier] = None,
        max_loop_steps: int = 15,
        max_replans: int = 2,
    ) -> None:
        self.os_adapter = os_adapter or get_os_adapter()
        self.state_mgr = state_mgr or ComputerStateManager(os_adapter=self.os_adapter)
        self.memory = memory or ConversationMemory()
        self.ref_resolver = ref_resolver or ReferenceResolver()
        self.reasoner = reasoner or SemanticReasoner(ref_resolver=self.ref_resolver)
        self.planner = StateAwarePlanner()  # New state-aware planner

        if classifier is None:
            from wisprhelp.classification.classifier import JevClassifier
            self.classifier = JevClassifier()
        else:
            self.classifier = classifier

        if safety_gate is None:
            from wisprhelp.core.safety import SafetyGate
            self.safety_gate = SafetyGate()
        else:
            self.safety_gate = safety_gate

        self.executor = executor or ToolBasedActionExecutor(
            os_adapter=self.os_adapter,
            safety_gate=self.safety_gate,
        )
        self.plan_classifier = plan_classifier or PlanClassifier()
        self.max_loop_steps = max(1, min(max_loop_steps, 15))
        self.max_replans = max_replans
        self._active_graph: Optional[TaskGraph] = None
        self._pending_held_action: Optional[Action] = None
        self._pending_trace: Optional[TaskTrace] = None
        self._abort_event = threading.Event()
        self._turn_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._stop_generation = 0
        self._stop_code = FailureCode.CANCELLED
        self._emergency_latched = False
        self.executor.abort_requested = self._abort_event.is_set
        self.executor.cancellation_code = lambda: self._stop_code

    def abort(self, emergency: bool = True) -> None:
        """Cancel the semantic loop and the same vision instance used by it."""
        with self._stop_lock:
            self._stop_generation += 1
            self._abort_event.set()
            self._stop_code = FailureCode.EMERGENCY_STOP if emergency else FailureCode.CANCELLED
            self._emergency_latched = self._emergency_latched or emergency
        self.safety_gate.cancel_pending()
        self._pending_held_action = None
        self._pending_trace = None
        self.executor.clear_pending_confirmations()
        if self._active_graph:
            self._active_graph.cancel()
        self.executor.vlm_agent.abort()

    def reset_abort(self) -> None:
        """Explicitly re-arm after an emergency; never clear a running turn's stop."""
        if self._turn_lock.locked():
            raise RuntimeError("Cannot reset emergency stop during execution")
        self._emergency_latched = False
        self._abort_event.clear()
        self.executor.vlm_agent.reset_abort()

    @property
    def vlm_agent(self) -> Any:
        return self.executor.vlm_agent

    @staticmethod
    def _is_affirmative_reply(reply: str) -> bool:
        """Word-boundary detection of a spoken affirmative (English + Hinglish)."""
        return is_explicit_confirmation(reply)

    @staticmethod
    def _is_negative_reply(reply: str) -> bool:
        """Word-boundary detection of a spoken negative (English + Hinglish)."""
        return bool(
            re.search(
                r"\b(no|nope|nah|nahi|cancel|stop|abort|don't|dont|do\s+not|ruk|mat)\b",
                reply,
            )
        )

    def process_turn(
        self, user_query: str, confirmation_response: Optional[str] = None,
        step_observer: Optional[Callable[[int, Action, str], None]] = None,
    ) -> AgentTurnResult:
        # Cancellation is handled before state inspection/model calls and before
        # the execution lock, so it can stop an in-flight multi-step workflow.
        if SemanticReasoner._is_interruption(user_query.lower().strip()):
            self.abort(emergency=False)
            trace = TaskTrace(user_query, "Cancel", "", "", {}, success=False,
                              final_result="Execution cancelled.", failure_code=FailureCode.CANCELLED)
            return AgentTurnResult(False, trace.final_result, "Cancel", trace, failure_code=FailureCode.CANCELLED)
        with self._stop_lock:
            generation = self._stop_generation
        if not self._turn_lock.acquire(blocking=False):
            trace = TaskTrace(user_query, "", "", "", {}, success=False, failure_code=FailureCode.BUSY)
            return AgentTurnResult(False, "Another turn is executing.", "", trace, failure_code=FailureCode.BUSY)
        try:
            with self._stop_lock:
                if self._emergency_latched:
                    trace = TaskTrace(user_query, "", "", "", {}, success=False, failure_code=FailureCode.EMERGENCY_STOP)
                    return AgentTurnResult(False, "Emergency stop is latched; explicitly reset before continuing.", "", trace,
                                           failure_code=FailureCode.EMERGENCY_STOP)
                if generation != self._stop_generation:
                    trace = TaskTrace(user_query, "", "", "", {}, success=False, failure_code=self._stop_code)
                    return AgentTurnResult(False, "Execution stopped during turn startup.", "", trace, failure_code=self._stop_code)
                self._abort_event.clear()
                self.executor.vlm_agent.reset_abort()
            return self._process_turn(user_query, confirmation_response, step_observer)
        except Exception as exc:
            self.abort(emergency=type(exc).__name__ in ("FailSafeException", "EmergencyKillSwitchTriggered"))
            code = exc.code if isinstance(exc, ExecutionFault) else self._stop_code if self._emergency_latched else FailureCode.API_UNAVAILABLE
            trace = TaskTrace(user_query, "", "", "", {}, success=False, failure_code=code)
            return AgentTurnResult(False, f"Turn stopped: {type(exc).__name__}", "", trace, failure_code=code)
        finally:
            self._turn_lock.release()

    def _process_turn(
        self,
        user_query: str,
        confirmation_response: Optional[str] = None,
        step_observer: Optional[Callable[[int, Action, str], None]] = None,
    ) -> AgentTurnResult:
        """Processes a natural language turn from understanding to verified execution.

        Args:
            user_query: Spoken or typed user input.
            confirmation_response: Optional 'yes'/'no' reply if answering a safety prompt.
            step_observer: Optional callback hook invoked after each action step.

        Returns:
            AgentTurnResult detailing execution status, message, and structured trace.
        """
        start_time = time.perf_counter()

        if self._pending_held_action is not None:
            reply = (confirmation_response or user_query).lower().strip()
            held = self._pending_held_action
            if self._is_negative_reply(reply):
                self._pending_held_action = None
                if self._active_graph:
                    self._active_graph.cancel()
                trace = self._pending_trace
                trace.final_result = "Destructive action cancelled by user."
                trace.success = False
                self.memory.interrupt_goal()
                return AgentTurnResult(False, trace.final_result, trace.semantic_goal, trace)
            if self._is_affirmative_reply(reply):
                self._pending_held_action = None
                return self._run_graph(user_query, self._pending_trace, self._active_graph,
                                       start_time, step_observer, confirmed_action=held)
            if confirmation_response is not None:
                trace = self._pending_trace
                return AgentTurnResult(True, "Please explicitly confirm yes or cancel.", trace.semantic_goal, trace,
                                       needs_confirmation=True, pending_held_action=held)
            # A new command supersedes the held workflow.
            self._pending_held_action = None
            if self._active_graph:
                self._active_graph.cancel()

        # 2. Capture Current Computer State Snapshot (Adaptive Observation)
        state = self.state_mgr.capture_state(adaptive=True, include_elements=True)

        # 3. Semantic Goal & Intent Understanding
        understanding: GoalUnderstanding = self.reasoner.analyze(user_query, self.memory, state)

        # Trace initialization
        trace = TaskTrace(
            user_input=user_query,
            semantic_goal=understanding.primary_goal,
            active_app=state.active_app_name,
            active_window=state.active_window.title if state.active_window else "None",
            entities=understanding.entities,
        )

        # 4. Handle Emergency Interruption ("Wait, stop", "cancel")
        if understanding.is_interruption:
            if self._active_graph:
                self._active_graph.cancel()
            self.memory.interrupt_goal()
            msg = "Action halted immediately."
            trace.final_result = msg
            trace.success = True
            trace.total_latency_ms = (time.perf_counter() - start_time) * 1000
            self.memory.add_turn(user_query, msg)
            return AgentTurnResult(
                success=True,
                message=msg,
                goal="Emergency Stop",
                trace=trace,
                latency_ms=trace.total_latency_ms,
            )

        # 5. Handle Ambiguity / Clarification Requirement
        if understanding.needs_clarification:
            msg = understanding.clarification_question or "Could you clarify what you'd like to do?"
            trace.final_result = msg
            trace.total_latency_ms = (time.perf_counter() - start_time) * 1000
            self.memory.pending_clarification = msg
            self.memory.add_turn(user_query, msg)
            return AgentTurnResult(
                success=True,
                message=msg,
                goal=understanding.primary_goal,
                trace=trace,
                latency_ms=trace.total_latency_ms,
            )

        # Ordered model/fallback actions are authoritative when supplied. Compile
        # legacy goal specs only when no complete action plan is present.
        actions = understanding.initial_actions
        if not actions and understanding.goal_spec is not None:
            actions = self.planner.plan(understanding.goal_spec, state)
        if not actions:
            actions = [Action.ask_clarification("I could not build a complete plan. What should I do?")]
        if len(actions) > self.max_loop_steps:
            actions = [Action.ask_clarification("This task exceeds the step limit. Which part should I do first?")]
        for action in actions:
            action.confidence = min(action.confidence, understanding.confidence.overall)
        gate_prompt = plan_authorization_error(actions, user_query)
        if gate_prompt:
            trace.failure_code = FailureCode.UNAUTHORIZED
            trace.success = False
            return AgentTurnResult(False, gate_prompt, understanding.primary_goal, trace,
                                   failure_code=FailureCode.UNAUTHORIZED)
        gate_prompt = self._classify_actions(actions, user_query, state)
        if gate_prompt:
            self.memory.pending_clarification = gate_prompt
            code = FailureCode.API_UNAVAILABLE if self.plan_classifier.last_failure else FailureCode.LOW_CONFIDENCE
            trace.failure_code = code
            return AgentTurnResult(False, gate_prompt, understanding.primary_goal, trace, failure_code=code)
        task_graph = TaskGraph(goal=understanding.primary_goal)
        for action in actions:
            task_graph.add_action(action)

        self._active_graph = task_graph
        self.memory.set_current_goal(understanding.primary_goal)

        return self._run_graph(user_query, trace, task_graph, start_time, step_observer)

    def _classify_actions(self, actions, goal, state):
        classified = self.plan_classifier.classify_plan(
            steps=actions, goal=goal,
            os_context={"active_window_title": state.active_window.title if state.active_window else "",
                        "active_process_name": state.active_window.process_name if state.active_window else "",
                        "os_platform": state.os_platform})
        prompt = self.plan_classifier.gate_plan(classified)
        self.plan_classifier.apply_to_actions(classified)
        return prompt

    def _run_graph(self, user_query, trace, task_graph, start_time,
                   step_observer=None, confirmed_action=None):
        state = self.state_mgr.capture_state(adaptive=True)
        # 7. Action-Observation-Replanning Loop (closed-loop dynamic execution)
        actions_executed: List[Action] = [n.action for n in task_graph.nodes[:task_graph.active_index]
                                         if n.status == "completed"]
        step_count = task_graph.steps_used
        replans_done = task_graph.replans_used

        while not task_graph.is_finished and step_count < self.max_loop_steps:
            if self._abort_event.is_set():
                task_graph.cancel()
                break
            step_count += 1
            task_graph.steps_used = step_count
            node = task_graph.get_current_node()
            if not node:
                break

            action = node.action
            node.status = "executing"

            # 7a. Destructive risk was already merged from the batched Jev
            # classification in step 6 (PlanClassifier.apply_to_actions).
            # The SafetyGate holds any destructive step inside the executor.

            # 7b. Execute Action via Deterministic Tool Executor
            try:
                outcome = self.executor.execute_action(action, state, confirmed=action is confirmed_action)
            except Exception as exc:
                if type(exc).__name__ not in ("FailSafeException", "EmergencyKillSwitchTriggered"):
                    raise
                self.abort()
                break
            confirmed_action = None
            node.outcome = outcome
            node.observation = outcome.observation

            # Record in structured trace
            trace.steps.append({
                "step": step_count,
                "action": action.to_dict(),
                "success": outcome.success,
                "observation": outcome.observation,
                "message": outcome.message,
                "failure_code": outcome.failure_code.value,
                "postcondition_verified": outcome.postcondition_verified,
                "dispatched": outcome.dispatched,
            })

            # Check if action was held at Safety Gate
            if outcome.needs_confirmation:
                # A held action has not executed and must retain its budget
                # slot when the user resumes this same task.
                step_count -= 1
                task_graph.steps_used = step_count
                self._pending_held_action = action
                self._pending_trace = trace
                elapsed = (time.perf_counter() - start_time) * 1000
                trace.final_result = outcome.message
                trace.total_latency_ms = elapsed
                return AgentTurnResult(
                    success=True,
                    message=outcome.message,
                    goal=task_graph.goal,
                    trace=trace,
                    needs_confirmation=True,
                    confirmation_prompt=outcome.confirmation_prompt,
                    pending_held_action=action,
                    actions_executed=actions_executed,
                    latency_ms=elapsed,
                )

            actions_executed.append(action)

            if step_observer:
                step_observer(step_count, action, outcome.observation)

            # 7c. Observe State Mutation & Closed-Loop Verification
            time.sleep(0.08)  # OS state settling
            new_state = self.state_mgr.capture_state(adaptive=True)
            self.state_mgr.update_with_action(action, outcome.observation)
            state = new_state

            # 7d. Self-Correction & Replanning
            if not outcome.success:
                trace.failure_code = outcome.failure_code
                if outcome.failure_code in (FailureCode.CANCELLED, FailureCode.EMERGENCY_STOP):
                    self.abort(emergency=outcome.failure_code == FailureCode.EMERGENCY_STOP)
                    break
                safe_to_recover = not action.is_destructive and (
                    action.type in (ActionType.LAUNCH_APP, ActionType.FOCUS_WINDOW,
                                    ActionType.FIND_FILE, ActionType.READ_FILE, ActionType.WAIT)
                    or (not outcome.dispatched and outcome.failure_code in (
                        FailureCode.ACCESSIBILITY_TARGET_NOT_FOUND, FailureCode.INCORRECT_GROUNDING,
                        FailureCode.API_UNAVAILABLE)))
                if not safe_to_recover or outcome.failure_code in (
                    FailureCode.FOCUS_CHANGED, FailureCode.UNEXPECTED_MODAL,
                    FailureCode.ACTION_TIMEOUT, FailureCode.POSTCONDITION_FAILED):
                    # A failed postcondition can follow a partially applied edit.
                    # Preserve the observed document for the user's next turn.
                    node.status = "failed"
                    break
                if node.retries < node.max_retries and action.type not in (ActionType.TYPE_TEXT, ActionType.CLEAR_TEXT):
                    node.retries += 1
                    logger.warning("Action %s failed. Initiating retry %d/%d with alternative grounding.", action.type, node.retries, node.max_retries)
                    trace.replanning_events.append(f"Retry {node.retries} for step {step_count}: {action.type.value}")
                    # Switch grounding tier from P2 to P3 if element click failed
                    if action.type == ActionType.CLICK and action.grounding_tier != 3:
                        action.grounding_tier = 3
                    continue

                # Replan-on-surprise: LLM re-decomposes remaining work with fresh eyes
                if replans_done < self.max_replans:
                    fresh_state = self.state_mgr.capture_state(adaptive=True)
                    completed_summaries = [
                        f"{n.action.type.value} ({n.action.target or n.action.payload or ''}) -> {n.observation or 'ok'}"
                        for n in task_graph.nodes[: task_graph.active_index]
                        if n.status == "completed"
                    ]
                    failed_summary = (
                        f"{action.type.value} ({action.target or action.payload or ''}) -> "
                        f"{outcome.observation or outcome.message}"
                    )
                    recovery_actions = self.reasoner.replan(
                        goal=task_graph.goal,
                        completed_summaries=completed_summaries,
                        failed_summary=failed_summary,
                        memory=self.memory,
                        state=fresh_state,
                    )
                    if recovery_actions:
                        if plan_authorization_error(recovery_actions, trace.user_input):
                            node.status = "failed"
                            trace.failure_code = FailureCode.UNAUTHORIZED
                            break
                        if all(a.type in (ActionType.RESPOND, ActionType.ASK_CLARIFICATION) for a in recovery_actions):
                            # A conversational fallback cannot satisfy a failed OS task.
                            node.status = "failed"
                            break
                        if len(recovery_actions) > self.max_loop_steps - step_count:
                            break
                        gate_prompt = self._classify_actions(recovery_actions, task_graph.goal, fresh_state)
                        if gate_prompt:
                            node.observation = gate_prompt
                            break
                        replans_done += 1
                        task_graph.replans_used = replans_done
                        trace.replanning_events.append(
                            f"LLM replan {replans_done}: {len(recovery_actions)} recovery steps"
                        )
                        task_graph.replace_remaining(recovery_actions)
                        state = fresh_state
                        continue

                node.status = "failed"
                logger.error("Step %d exceeded retry and replan limits. Halting task.", step_count)
                break

            node.status = "completed"
            task_graph.active_index += 1

        # 8. Finalize Turn
        elapsed = (time.perf_counter() - start_time) * 1000
        trace.total_latency_ms = elapsed

        final_msg = "Task completed successfully."
        if actions_executed:
            last_out = task_graph.nodes[-1].outcome if task_graph.nodes and task_graph.nodes[-1].outcome else None
            final_msg = last_out.message if last_out else f"Executed {len(actions_executed)} steps for: '{task_graph.goal}'."

        trace.success = bool(task_graph.nodes) and not task_graph.is_cancelled and all(
            n.status == "completed" for n in task_graph.nodes)
        if not trace.success:
            last_node = task_graph.get_current_node()
            detail = last_node.observation if last_node else ""
            final_msg = ("Execution aborted." if task_graph.is_cancelled else
                         f"Task incomplete after {step_count} steps. {detail}").strip()
            if task_graph.is_cancelled:
                trace.failure_code = self._stop_code
            elif trace.failure_code == FailureCode.NONE:
                trace.failure_code = FailureCode.STEP_LIMIT
        else:
            trace.failure_code = FailureCode.NONE
        trace.final_result = final_msg

        self.memory.add_turn(trace.user_input, final_msg, actions_taken=actions_executed)
        self.memory.complete_goal()
        self._pending_trace = None

        return AgentTurnResult(
            success=trace.success,
            message=final_msg,
            goal=task_graph.goal,
            trace=trace,
            actions_executed=actions_executed,
            latency_ms=elapsed,
            failure_code=trace.failure_code,
        )
