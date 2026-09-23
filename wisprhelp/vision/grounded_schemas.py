"""Grounded Computer-Use Schemas for Jev-Inspired 'Decide, Don't Generate' Paradigm.

Defines typed representations for Set-of-Marks UI candidates, candidate catalogs,
structured agent decisions, and closed-loop postcondition verification contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ActionType(str, Enum):
    """Permitted grounded physical actions."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    FINISH = "finish"
    ABORT = "abort"


@dataclass(frozen=True)
class UICandidateNode:
    """An interactive UI element observed directly on the host OS."""
    candidate_id: int
    label: str
    control_type: str
    bbox: Tuple[int, int, int, int]  # (left, top, right, bottom) in physical pixels
    center: Tuple[int, int]          # (center_x, center_y) in physical pixels
    is_focused: bool = False
    is_enabled: bool = True

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.candidate_id,
            "label": self.label,
            "type": self.control_type,
            "center": list(self.center),
            "bbox": list(self.bbox),
            "focused": self.is_focused,
        }


@dataclass
class CandidateCatalog:
    """Synchronized collection of observed UI candidates on screen."""
    candidates: Dict[int, UICandidateNode] = field(default_factory=dict)

    def add(self, node: UICandidateNode) -> None:
        self.candidates[node.candidate_id] = node

    def get(self, candidate_id: int) -> Optional[UICandidateNode]:
        return self.candidates.get(candidate_id)

    def __len__(self) -> int:
        return len(self.candidates)

    def to_prompt_context(self, max_items: int = 40) -> str:
        """Formats the candidates into a compact, human-readable list for LLM selection."""
        if not self.candidates:
            return "No discrete interactive controls identified."

        lines = []
        for cid in sorted(self.candidates.keys())[:max_items]:
            node = self.candidates[cid]
            name = node.label.strip() if node.label.strip() else f"Unnamed {node.control_type}"
            focused_flag = " [FOCUSED]" if node.is_focused else ""
            lines.append(f"[{cid}] {node.control_type}: '{name}'{focused_flag}")

        if len(self.candidates) > max_items:
            lines.append(f"... and {len(self.candidates) - max_items} more controls.")

        return "\n".join(lines)


@dataclass
class GroundedDecision:
    """A typed action decision returned by the model based on observed candidates."""
    thought: str
    action_type: str
    chosen_candidate_id: Optional[int] = None
    typed_text: Optional[str] = None
    key: Optional[str] = None
    expected_mutation: str = ""
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundedDecision:
        """Parses model response JSON safely into a typed GroundedDecision."""
        action = str(data.get("action_type") or data.get("action") or data.get("type") or "finish").lower()
        
        # Normalize action name
        if "click" in action and "double" not in action and "right" not in action:
            action = ActionType.CLICK.value
        elif "double" in action:
            action = ActionType.DOUBLE_CLICK.value
        elif "right" in action:
            action = ActionType.RIGHT_CLICK.value
        elif "type" in action or "write" in action:
            action = ActionType.TYPE_TEXT.value
        elif "key" in action or "press" in action:
            action = ActionType.PRESS_KEY.value
        elif "finish" in action or "done" in action:
            action = ActionType.FINISH.value
        elif "abort" in action or "error" in action:
            action = ActionType.ABORT.value

        cid = data.get("chosen_candidate_id") or data.get("candidate_id") or data.get("id")
        if cid is not None:
            try:
                cid = int(cid)
            except (ValueError, TypeError):
                cid = None

        text = data.get("typed_text") or data.get("text")
        key = data.get("key") or data.get("key_combination")
        expected = data.get("expected_mutation") or data.get("expected_outcome") or ""
        thought = data.get("thought") or data.get("reasoning") or ""
        conf = float(data.get("confidence", 1.0))

        return cls(
            thought=thought,
            action_type=action,
            chosen_candidate_id=cid,
            typed_text=text,
            key=key,
            expected_mutation=expected,
            confidence=conf,
        )


@dataclass(frozen=True)
class PostconditionResult:
    """Outcome of a post-action mutation verification check."""
    verified: bool
    details: str
    dhash_distance: int = 0
    text_matched: bool = False
