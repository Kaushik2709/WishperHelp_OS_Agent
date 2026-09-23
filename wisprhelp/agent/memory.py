"""Stateful Conversation Memory and Entity Store for WisprHelp.

Maintains multi-turn context, conversation history, referenced entities,
pronoun targets, and user preferences across turns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from wisprhelp.agent.actions import Action


@dataclass
class ConversationTurn:
    """A single exchange between user and agent."""
    turn_id: int
    user_input: str
    agent_response: str
    actions_taken: List[Action] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class EntityStore:
    """Tracks recently referenced entities for pronoun and reference resolution."""

    def __init__(self, max_items_per_category: int = 8) -> None:
        self.max_items = max_items_per_category
        self._categories: Dict[str, List[str]] = {
            "apps": [],
            "files": [],
            "urls": [],
            "ui_targets": [],
            "people": [],
            "concepts": [],
        }

    def record_entity(self, category: str, value: str) -> None:
        """Records an entity reference, moving it to most recent position."""
        if not value or not value.strip():
            return
        val_clean = value.strip()
        if category not in self._categories:
            self._categories[category] = []
        cat_list = self._categories[category]
        if val_clean in cat_list:
            cat_list.remove(val_clean)
        cat_list.insert(0, val_clean)
        if len(cat_list) > self.max_items:
            cat_list.pop()

    def record_app(self, app_name: str) -> None:
        self.record_entity("apps", app_name)

    def record_file(self, file_path: str) -> None:
        self.record_entity("files", file_path)

    def record_url(self, url: str) -> None:
        self.record_entity("urls", url)

    def record_ui_target(self, target: str) -> None:
        self.record_entity("ui_targets", target)

    def record_person(self, name: str) -> None:
        self.record_entity("people", name)

    def get_last_entity(self, category: str) -> Optional[str]:
        """Gets the most recently referenced entity in a category."""
        items = self._categories.get(category, [])
        return items[0] if items else None

    def get_previous_entity(self, category: str) -> Optional[str]:
        """Gets the second most recently referenced entity in a category."""
        items = self._categories.get(category, [])
        return items[1] if len(items) > 1 else None

    def get_all(self, category: str) -> List[str]:
        return list(self._categories.get(category, []))

    def to_dict(self) -> Dict[str, List[str]]:
        return dict(self._categories)


class ConversationMemory:
    """Maintains compact, task-oriented conversation context across turns."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.current_goal: Optional[str] = None
        self.active_task_status: str = "idle"  # idle, in_progress, interrupted, completed
        self.entity_store = EntityStore()
        self.pending_clarification: Optional[str] = None
        self.user_preferences: Dict[str, Any] = {
            "browser": "Chrome",
            "editor": "VS Code",
            "terminal": "PowerShell",
        }

    def add_turn(
        self,
        user_input: str,
        agent_response: str,
        actions_taken: Optional[List[Action]] = None,
    ) -> ConversationTurn:
        """Appends a new turn and updates entity trackers."""
        turn_id = len(self.turns) + 1
        turn = ConversationTurn(
            turn_id=turn_id,
            user_input=user_input,
            agent_response=agent_response,
            actions_taken=actions_taken or [],
            timestamp=time.time(),
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

        # Automatically extract referenced apps and targets from actions
        for act in turn.actions_taken:
            act_type = act.type.value if hasattr(act.type, "value") else str(act.type)
            if act_type == "launch_app" and act.target:
                self.entity_store.record_app(act.target)
            elif act_type in ("open_file", "find_file") and act.target:
                self.entity_store.record_file(act.target)
            elif act_type in ("click", "select") and act.target:
                self.entity_store.record_ui_target(act.target)
            elif act_type == "navigate" and act.payload:
                self.entity_store.record_url(act.payload)

        return turn

    def set_current_goal(self, goal: str) -> None:
        self.current_goal = goal
        self.active_task_status = "in_progress"

    def complete_goal(self) -> None:
        self.active_task_status = "completed"

    def interrupt_goal(self) -> None:
        self.active_task_status = "interrupted"

    def format_history_for_prompt(self, max_recent: int = 4) -> str:
        """Formats the last N conversation turns for LLM reasoning prompt."""
        if not self.turns:
            return "No previous conversation turns."

        recent = self.turns[-max_recent:]
        lines = []
        for t in recent:
            lines.append(f"User: {t.user_input}")
            lines.append(f"Agent: {t.agent_response}")
            if t.actions_taken:
                act_summaries = [
                    f"{a.type.value if hasattr(a.type, 'value') else a.type}({a.target or a.payload or ''})"
                    for a in t.actions_taken
                ]
                lines.append(f"  Actions: {', '.join(act_summaries)}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Resets conversation history."""
        self.turns.clear()
        self.current_goal = None
        self.active_task_status = "idle"
        self.pending_clarification = None
