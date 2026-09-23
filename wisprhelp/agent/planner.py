"""Dynamic Task Planning, Mutable Task Graph, and Closed-Loop Replanning.

Implements the Plan -> Execute -> Observe -> Verify -> Update State -> Replan cycle,
enabling self-correction, dynamic recovery from unexpected UI states, and
graceful handling of user interruptions and corrections.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.executor import ExecutionOutcome, ToolBasedActionExecutor
from wisprhelp.agent.memory import ConversationMemory, ConversationTurn
from wisprhelp.agent.reasoning import GoalUnderstanding, SemanticReasoner
from wisprhelp.agent.references import ReferenceResolver
from wisprhelp.agent.state import ComputerState, ComputerStateManager
from wisprhelp.classification.schemas import JevDecision
from wisprhelp.hal.base import BaseOSAdapter
from wisprhelp.hal.factory import get_os_adapter

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
        max_loop_steps: int = 15,
    ) -> None:
        self.os_adapter = os_adapter or get_os_adapter()
        self.state_mgr = state_mgr or ComputerStateManager(os_adapter=self.os_adapter)
        self.memory = memory or ConversationMemory()
        self.ref_resolver = ref_resolver or ReferenceResolver()
        self.reasoner = reasoner or SemanticReasoner(ref_resolver=self.ref_resolver)

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
        self.max_loop_steps = min(max_loop_steps, 15)
        self._active_graph: Optional[TaskGraph] = None
        self._pending_held_action: Optional[Action] = None

    @property
    def vlm_agent(self) -> Any:
        return self.executor.vlm_agent



    def process_turn(
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

        # 1. Handle Pending Safety Confirmation Resolution
        if confirmation_response and self._pending_held_action:
            reply = confirmation_response.lower().strip()
            if any(w in reply for w in ("yes", "y", "confirm", "proceed", "haan")):
                held_act = self._pending_held_action
                self._pending_held_action = None
                logger.info("User confirmed destructive action: %s", held_act)
                outcome = self.executor.execute_action(held_act, confirmed=True)
                elapsed = (time.perf_counter() - start_time) * 1000

                trace = TaskTrace(
                    user_input=user_query,
                    semantic_goal=held_act.rationale,
                    active_app=self.state_mgr.capture_state().active_app_name,
                    active_window=self.state_mgr.capture_state().active_window.title if self.state_mgr.capture_state().active_window else "",
                    entities={},
                    final_result=outcome.message,
                    success=outcome.success,
                    total_latency_ms=elapsed,
                )
                self.memory.add_turn(user_query, outcome.message, actions_taken=[held_act])
                return AgentTurnResult(
                    success=outcome.success,
                    message=outcome.message,
                    goal=held_act.rationale,
                    trace=trace,
                    actions_executed=[held_act],
                    latency_ms=elapsed,
                )
            else:
                self._pending_held_action = None
                elapsed = (time.perf_counter() - start_time) * 1000
                msg = "Destructive action cancelled by user."
                trace = TaskTrace(
                    user_input=user_query,
                    semantic_goal="Cancelled destructive action",
                    active_app="",
                    active_window="",
                    entities={},
                    final_result=msg,
                    success=False,
                    total_latency_ms=elapsed,
                )
                self.memory.add_turn(user_query, msg)
                return AgentTurnResult(
                    success=False,
                    message=msg,
                    goal="Cancelled",
                    trace=trace,
                    latency_ms=elapsed,
                )

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

        # 6. Initialize or Mutate Task Graph
        task_graph = TaskGraph(goal=understanding.primary_goal)
        for act in understanding.initial_actions:
            task_graph.add_action(act)
        self._active_graph = task_graph
        self.memory.set_current_goal(understanding.primary_goal)

        # 7. Action-Observation-Replanning Loop
        actions_executed: List[Action] = []
        step_count = 0

        while not task_graph.is_finished and step_count < self.max_loop_steps:
            step_count += 1
            node = task_graph.get_current_node()
            if not node:
                break

            action = node.action
            node.status = "executing"

            # 7a. Fast System One Reflex & Destructive Risk Gate (TypeSafe Jev)
            # Ensure safety interlock checks Jev probability before risky execution
            if not action.is_destructive:
                try:
                    jev_decision = self.classifier.classify(
                        user_query=action.rationale or action.type.value,
                        os_context={"active_window": state.active_window.title if state.active_window else ""},
                    )
                    if jev_decision.is_high_risk(self.safety_gate.risk_threshold):
                        action.is_destructive = True
                except Exception as exc:
                    logger.debug("Jev classification check failed: %s", exc)

            # 7b. Execute Action via Deterministic Tool Executor
            outcome = self.executor.execute_action(action, state)
            node.outcome = outcome
            node.observation = outcome.observation
            actions_executed.append(action)

            # Record in structured trace
            trace.steps.append({
                "step": step_count,
                "action": action.to_dict(),
                "success": outcome.success,
                "observation": outcome.observation,
                "message": outcome.message,
            })

            # Check if action was held at Safety Gate
            if outcome.needs_confirmation:
                self._pending_held_action = action
                elapsed = (time.perf_counter() - start_time) * 1000
                trace.final_result = outcome.message
                trace.total_latency_ms = elapsed
                return AgentTurnResult(
                    success=True,
                    message=outcome.message,
                    goal=understanding.primary_goal,
                    trace=trace,
                    needs_confirmation=True,
                    confirmation_prompt=outcome.confirmation_prompt,
                    pending_held_action=action,
                    actions_executed=actions_executed,
                    latency_ms=elapsed,
                )

            if step_observer:
                step_observer(step_count, action, outcome.observation)

            # 7c. Observe State Mutation & Closed-Loop Verification
            time.sleep(0.08)  # OS state settling
            new_state = self.state_mgr.capture_state(adaptive=True)
            self.state_mgr.update_with_action(action, outcome.observation)
            state = new_state

            # 7d. Self-Correction & Replanning
            if not outcome.success:
                if node.retries < node.max_retries:
                    node.retries += 1
                    logger.warning("Action %s failed. Initiating retry %d/%d with alternative grounding.", action.type, node.retries, node.max_retries)
                    trace.replanning_events.append(f"Retry {node.retries} for step {step_count}: {action.type.value}")
                    # Switch grounding tier from P2 to P3 if element click failed
                    if action.type == ActionType.CLICK and action.grounding_tier != 3:
                        action.grounding_tier = 3
                    continue
                else:
                    node.status = "failed"
                    logger.error("Step %d exceeded retry limit. Halting task.", step_count)
                    break

            node.status = "completed"
            task_graph.active_index += 1

        # 8. Finalize Turn
        elapsed = (time.perf_counter() - start_time) * 1000
        trace.total_latency_ms = elapsed

        final_msg = "Task completed successfully."
        if actions_executed:
            last_out = task_graph.nodes[-1].outcome if task_graph.nodes and task_graph.nodes[-1].outcome else None
            final_msg = last_out.message if last_out else f"Executed {len(actions_executed)} steps for: '{understanding.primary_goal}'."

        trace.final_result = final_msg
        trace.success = all(n.status == "completed" for n in task_graph.nodes) if task_graph.nodes else True

        self.memory.add_turn(user_query, final_msg, actions_taken=actions_executed)
        self.memory.complete_goal()

        return AgentTurnResult(
            success=trace.success,
            message=final_msg,
            goal=understanding.primary_goal,
            trace=trace,
            actions_executed=actions_executed,
            latency_ms=elapsed,
        )
