"""Semantic Reference & Pronoun Resolver.

Resolves anaphoric pronouns ("it", "this", "that"), relative references
("the first one", "second tab", "the other one"), spatial descriptions
("button on the left", "top right"), and temporal references ("the app I was using earlier")
against combined conversation memory, active OS state, and accessibility trees.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from wisprhelp.agent.memory import ConversationMemory
from wisprhelp.agent.state import ComputerState, UIElementInfo

logger = logging.getLogger(__name__)


@dataclass
class ResolvedReference:
    """The outcome of reference resolution."""
    original_phrase: str
    resolved_target: str
    category: str  # "app", "file", "window", "ui_element", "url", "person"
    confidence: float = 1.0
    source: str = "direct"  # "memory_entity", "active_window", "ui_tree", "ordinal", "spatial"
    metadata: Dict[str, Any] = None


class ReferenceResolver:
    """Resolves natural language references against memory and world state."""

    def __init__(self) -> None:
        pass

    def resolve(
        self,
        phrase: str,
        memory: ConversationMemory,
        state: ComputerState,
    ) -> Optional[ResolvedReference]:
        """Resolves a reference phrase to a concrete entity or UI target.

        Args:
            phrase: Spoken phrase containing references (e.g. "this", "the other one", "my browser").
            memory: Conversation memory and entity history.
            state: Current workstation and window state.

        Returns:
            ResolvedReference if resolved, else None.
        """
        clean_phrase = phrase.lower().strip()

        # 1. Active Window / "this" / "this window" / "here"
        if clean_phrase in ("this", "this window", "the current window", "the current app", "here"):
            if state.active_window:
                return ResolvedReference(
                    original_phrase=phrase,
                    resolved_target=state.active_window.title,
                    category="window",
                    confidence=0.95,
                    source="active_window",
                    metadata={"process_name": state.active_window.process_name, "pid": state.active_window.pid},
                )

        # 2. Pronouns: "it", "that", "them"
        if clean_phrase in ("it", "that", "that one", "them"):
            # Check recently opened/referenced files first
            last_file = memory.entity_store.get_last_entity("files")
            if last_file:
                return ResolvedReference(
                    original_phrase=phrase,
                    resolved_target=last_file,
                    category="file",
                    confidence=0.88,
                    source="memory_entity",
                )

            # Check recently clicked or referenced UI target
            last_target = memory.entity_store.get_last_entity("ui_targets")
            if last_target:
                return ResolvedReference(
                    original_phrase=phrase,
                    resolved_target=last_target,
                    category="ui_element",
                    confidence=0.85,
                    source="memory_entity",
                )

            # Check recent app
            last_app = memory.entity_store.get_last_entity("apps")
            if last_app:
                return ResolvedReference(
                    original_phrase=phrase,
                    resolved_target=last_app,
                    category="app",
                    confidence=0.82,
                    source="memory_entity",
                )

            # Fall back to active window
            if state.active_window:
                return ResolvedReference(
                    original_phrase=phrase,
                    resolved_target=state.active_window.title,
                    category="window",
                    confidence=0.75,
                    source="active_window",
                )

        # 3. Contrastive / Correction: "the other one", "the other result", "the alternative"
        if any(w in clean_phrase for w in ("the other one", "other one", "the other", "the second one instead")):
            # Check if there is a previous UI candidate in memory or visible elements
            prev_ui = memory.entity_store.get_previous_entity("ui_targets")
            if prev_ui:
                return ResolvedReference(
                    original_phrase=phrase,
                    resolved_target=prev_ui,
                    category="ui_element",
                    confidence=0.90,
                    source="memory_entity",
                )
            # If in browser or list, "the other one" usually means candidate index 1 (second item)
            if state.visible_elements and len(state.visible_elements) > 1:
                return ResolvedReference(
                    original_phrase=phrase,
                    resolved_target=state.visible_elements[1].name,
                    category="ui_element",
                    confidence=0.85,
                    source="ui_tree",
                    metadata={"center": state.visible_elements[1].center},
                )

        # 4. Temporal / App Reference: "the app I was using earlier", "the previous app"
        if any(w in clean_phrase for w in ("earlier", "previous app", "app before this", "last app")):
            prev_app = memory.entity_store.get_previous_entity("apps")
            if prev_app:
                return ResolvedReference(
                    original_phrase=phrase,
                    resolved_target=prev_app,
                    category="app",
                    confidence=0.92,
                    source="memory_entity",
                )
            if state.open_windows and len(state.open_windows) > 1:
                return ResolvedReference(
                    original_phrase=phrase,
                    resolved_target=state.open_windows[1].title,
                    category="app",
                    confidence=0.85,
                    source="open_windows",
                )

        # 5. User Preferences: "my browser", "whatever browser I normally use", "my editor"
        if "browser" in clean_phrase:
            pref_browser = memory.user_preferences.get("browser", "Chrome")
            return ResolvedReference(
                original_phrase=phrase,
                resolved_target=pref_browser,
                category="app",
                confidence=0.95,
                source="user_preferences",
            )
        if "editor" in clean_phrase or "code editor" in clean_phrase or "my ide" in clean_phrase:
            pref_editor = memory.user_preferences.get("editor", "VS Code")
            return ResolvedReference(
                original_phrase=phrase,
                resolved_target=pref_editor,
                category="app",
                confidence=0.95,
                source="user_preferences",
            )

        # 6. Ordinal UI reference: "the first one", "the second one", "second tab", "first result"
        ordinal_info = self._extract_ordinal_index(clean_phrase)
        if ordinal_info is not None:
            idx, item_type = ordinal_info
            if state.visible_elements:
                matching_elems = [
                    e for e in state.visible_elements
                    if not item_type or item_type in e.control_type.lower() or item_type in e.name.lower()
                ] or state.visible_elements

                target_elem = None
                if 0 <= idx < len(matching_elems):
                    target_elem = matching_elems[idx]
                elif idx < 0 and abs(idx) <= len(matching_elems):
                    target_elem = matching_elems[idx]

                if target_elem:
                    return ResolvedReference(
                        original_phrase=phrase,
                        resolved_target=target_elem.name or f"item_{idx}",
                        category="ui_element",
                        confidence=0.88,
                        source="ordinal",
                        metadata={"center": target_elem.center, "control_type": target_elem.control_type},
                    )

            # If no elements cached in state, return structured ordinal representation
            return ResolvedReference(
                original_phrase=phrase,
                resolved_target=f"ordinal:{idx}:{item_type or 'item'}",
                category="ordinal",
                confidence=0.80,
                source="ordinal",
                metadata={"ordinal_index": idx, "item_type": item_type},
            )

        # 7. Spatial references: "button on the left", "top right button"
        spatial_match = self._extract_spatial_reference(clean_phrase, state.visible_elements)
        if spatial_match:
            return spatial_match

        return None

    @staticmethod
    def _extract_ordinal_index(phrase: str) -> Optional[Tuple[int, Optional[str]]]:
        """Extracts 0-based or negative index from ordinal phrase."""
        pos_map = {
            "first": 0, "1st": 0, "top": 0, "pehla": 0,
            "second": 1, "2nd": 1, "doosra": 1,
            "third": 2, "3rd": 2, "teesra": 2,
            "fourth": 3, "4th": 3,
            "fifth": 4, "5th": 4,
            "last": -1, "bottom": -1, "aakhri": -1,
            "second-last": -2, "penultimate": -2,
        }
        item_types = ["tab", "result", "chat", "button", "item", "row", "link", "window", "file", "folder"]
        found_type = None
        for it in item_types:
            if it in phrase:
                found_type = it
                break

        for word, idx in pos_map.items():
            if re.search(rf"\b{re.escape(word)}\b", phrase):
                return (idx, found_type)

        return None

    @staticmethod
    def _extract_spatial_reference(
        phrase: str, elements: List[UIElementInfo]
    ) -> Optional[ResolvedReference]:
        """Resolves spatial UI terms like 'on the left', 'on the right', 'at the top'."""
        if not elements:
            return None

        # Filter elements with center coordinates
        positioned = [e for e in elements if e.center is not None]
        if not positioned:
            return None

        if "left" in phrase:
            leftmost = min(positioned, key=lambda e: e.center[0])
            return ResolvedReference(
                original_phrase=phrase,
                resolved_target=leftmost.name,
                category="ui_element",
                confidence=0.85,
                source="spatial",
                metadata={"center": leftmost.center},
            )
        if "right" in phrase:
            rightmost = max(positioned, key=lambda e: e.center[0])
            return ResolvedReference(
                original_phrase=phrase,
                resolved_target=rightmost.name,
                category="ui_element",
                confidence=0.85,
                source="spatial",
                metadata={"center": rightmost.center},
            )
        if "top" in phrase:
            topmost = min(positioned, key=lambda e: e.center[1])
            return ResolvedReference(
                original_phrase=phrase,
                resolved_target=topmost.name,
                category="ui_element",
                confidence=0.85,
                source="spatial",
                metadata={"center": topmost.center},
            )
        if "bottom" in phrase:
            bottommost = max(positioned, key=lambda e: e.center[1])
            return ResolvedReference(
                original_phrase=phrase,
                resolved_target=bottommost.name,
                category="ui_element",
                confidence=0.85,
                source="spatial",
                metadata={"center": bottommost.center},
            )

        return None
