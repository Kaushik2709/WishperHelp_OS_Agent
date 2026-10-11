"""Goal Specification for state-aware query processing.

Defines the desired end state that the agent should achieve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoalSpec:
    """Structured representation of the desired end state.

    Attributes:
        goal: The type of goal (e.g., "content_equals", "app_running", "file_exists").
        target: A dictionary describing the target of the goal (e.g., 
            {"app": "notepad", "document": "current", "field": "editor"}).
        value: The desired value for the goal (e.g., the text content for content_equals).
        constraints: List of constraints that must be satisfied (e.g., ["exact_content", "no_append"]).
        ambiguities: List of ambiguities encountered during parsing (if any).
        confidence: Confidence in the goal specification (0.0 to 1.0).
    """

    goal: str
    target: Dict[str, Any]
    value: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "goal": self.goal,
            "target": self.target,
            "value": self.value,
            "constraints": self.constraints,
            "ambiguities": self.ambiguities,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalSpec":
        """Create from a dictionary."""
        return cls(
            goal=data.get("goal", ""),
            target=data.get("target", {}),
            value=data.get("value"),
            constraints=data.get("constraints", []),
            ambiguities=data.get("ambiguities", []),
            confidence=data.get("confidence", 1.0),
        )