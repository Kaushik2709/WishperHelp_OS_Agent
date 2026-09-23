"""WisprHelp General Semantic Agent Layer.

Transforms raw speech and natural language into state-aware, goal-oriented
desktop automation through conversational memory, multi-modal reference resolution,
and closed-loop dynamic replanning.
"""

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.executor import ExecutionOutcome, ToolBasedActionExecutor
from wisprhelp.agent.memory import ConversationMemory, ConversationTurn, EntityStore
from wisprhelp.agent.planner import (
    AgentTurnResult,
    SemanticAgent,
    TaskGraph,
    TaskNode,
    TaskTrace,
)
from wisprhelp.agent.reasoning import (
    ConfidenceBreakdown,
    GoalUnderstanding,
    SemanticReasoner,
)
from wisprhelp.agent.references import ReferenceResolver, ResolvedReference
from wisprhelp.agent.state import (
    ComputerState,
    ComputerStateManager,
    UIElementInfo,
)

__all__ = [
    "Action",
    "ActionType",
    "AgentTurnResult",
    "ComputerState",
    "ComputerStateManager",
    "ConfidenceBreakdown",
    "ConversationMemory",
    "ConversationTurn",
    "EntityStore",
    "ExecutionOutcome",
    "GoalUnderstanding",
    "ReferenceResolver",
    "ResolvedReference",
    "SemanticAgent",
    "SemanticReasoner",
    "TaskGraph",
    "TaskNode",
    "TaskTrace",
    "ToolBasedActionExecutor",
    "UIElementInfo",
]
