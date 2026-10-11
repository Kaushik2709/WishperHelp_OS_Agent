"""Compile supported end-state goals into ordered, executable plans.

General workflows use validated model steps. This compatibility compiler never
pretends an unobserved precondition is true or emits a partial unsupported plan.
"""
from typing import List
from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.goalspec import GoalSpec
from wisprhelp.agent.state import ComputerState


class StateAwarePlanner:
    def plan(self, spec: GoalSpec, state: ComputerState) -> List[Action]:
        if spec.ambiguities:
            return [Action.ask_clarification("; ".join(spec.ambiguities))]
        target = spec.target
        app = target.get("app")
        if app in ("generic_or_active", "active", "current", "desktop"):
            app = None
        goal = spec.goal
        if goal == "conversational_reply":
            return [Action.respond(spec.value or "")]
        if goal in ("needs_clarification", "empty_input", "unclear"):
            return [Action.ask_clarification("What would you like me to do?")]
        if goal == "app_running" and app:
            actions = [Action.focus_window(app) if self._active(app, state) else Action.launch_app(app)]
        elif goal in ("content_equals", "text_inserted", "content_appended", "document_empty"):
            actions = []
            if app:
                actions.append(Action.focus_window(app) if self._active(app, state) else Action.launch_app(app))
            edit = Action.type_text(spec.value or "")
            edit.params.update(app=app, field=target.get("field", "editor"),
                               mode={"content_equals": "replace", "document_empty": "replace",
                                     "content_appended": "append", "text_inserted": "insert"}[goal])
            actions.append(edit)
        elif goal == "element_interact" and target.get("element"):
            actions = [Action.click(target["element"])]
            if app and not self._active(app, state):
                actions.insert(0, Action.focus_window(app))
        elif goal == "folder_opened" and target.get("document"):
            actions = [Action.open_folder(target["document"])]
        elif goal == "file_exists" and target.get("document"):
            actions = [Action.create_file(target["document"], spec.value or "")]
        else:
            return [Action.ask_clarification(
                f"I could not build a complete plan for '{goal}'. Which action and target do you mean?")]
        for action in actions:
            action.confidence = spec.confidence
        return actions

    @staticmethod
    def _active(app: str, state: ComputerState) -> bool:
        window = state.active_window
        return bool(window and app.casefold() in
                    f"{window.title} {window.process_name}".casefold())
