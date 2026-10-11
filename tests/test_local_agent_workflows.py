"""Behavioral regressions for local plans and verified text workflows."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.executor import ToolBasedActionExecutor
from wisprhelp.agent.goalspec import GoalSpec
from wisprhelp.agent.memory import ConversationMemory
from wisprhelp.agent.planner import SemanticAgent
from wisprhelp.agent.planning import StateAwarePlanner
from wisprhelp.agent.reasoning import SemanticReasoner, GoalUnderstanding
from wisprhelp.agent.state import ComputerState
from wisprhelp.hal.base import BaseOSAdapter, WindowInfo


class EditorHAL(BaseOSAdapter):
    def __init__(self, text=""):
        self.text = text
        self.app = "Desktop"
        self.events = []
        self.selected = False
        self.focused = False
        self.can_focus = True
        self.drop_paste = False
        self.clipboard = "keep me"

    def launch_app(self, app_identifier, parameter=None):
        self.app = app_identifier
        self.events.append(("launch", app_identifier))
        return True

    def get_active_window(self):
        return WindowInfo(self.app, self.app + ".exe", 42, (0, 0, 800, 600))

    def focus_text_input(self, field=None):
        self.events.append(("focus", field))
        self.focused = self.can_focus
        return self.focused

    def read_focused_text(self):
        return self.text if self.focused else None

    def inject_unicode_text(self, text):
        assert self.focused
        self.events.append(("type", text))
        if not self.drop_paste:
            self.text = text if self.selected else self.text + text
        self.selected = False
        return True

    def media_control(self, command):
        self.events.append(("media", command))
        return True

    def key_combination(self, keys):
        self.events.append(("keys", keys))
        if keys[-1] == "a":
            self.selected = True
        if keys[-1] == "end":
            self.selected = False

    def press_key(self, key):
        if key in ("delete", "backspace") and self.selected:
            self.text = ""
            self.selected = False

    def find_accessibility_element(self, query, control_type=None):
        return None

    def move_mouse(self, x, y, smooth=True):
        pass

    def click(self, x=None, y=None, button="left", clicks=1):
        self.events.append(("click", x, y))


def make_agent(hal, understanding):
    reasoner = Mock()
    reasoner.analyze.return_value = understanding
    reasoner.replan.return_value = []
    agent = SemanticAgent(os_adapter=hal, reasoner=reasoner)
    agent.vlm_agent.execute_visual_task = Mock(return_value=SimpleNamespace(success=False))
    return agent


def plan(text, mode="replace"):
    edit = Action.type_text(text)
    edit.params.update(app="Notepad", field="editor", mode=mode)
    return GoalUnderstanding("write", "Write in Notepad", initial_actions=[Action.launch_app("Notepad"), edit])


@pytest.mark.parametrize("payload", ["I am a developer.", "Hey, I'm a developer.", "नमस्ते 👋\nI write code."])
def test_launch_then_exact_write_into_empty_editor(payload):
    hal = EditorHAL()
    result = make_agent(hal, plan(payload)).process_turn("Open Notepad and write: " + payload)
    assert result.success and not result.needs_confirmation
    assert hal.text == payload
    assert hal.events.index(("launch", "Notepad")) < hal.events.index(("type", payload))
    assert hal.clipboard == "keep me"


def test_confirmation_resumes_remaining_steps_and_preserves_original_text_until_yes():
    hal = EditorHAL("unsaved notes")
    understanding = plan("I am a developer.")
    understanding.initial_actions.append(Action.media_control("mute"))
    agent = make_agent(hal, understanding)
    held = agent.process_turn("Replace old notes and mute")
    assert held.needs_confirmation
    assert hal.text == "unsaved notes"
    assert not any(event[0] == "media" for event in hal.events)
    done = agent.process_turn("yes")
    assert done.success
    assert hal.text == "I am a developer."
    assert ("media", "mute") in hal.events
    assert hal.events.count(("launch", "Notepad")) == 1
    agent.reasoner.analyze.assert_called_once()


def test_changed_content_requires_new_confirmation():
    hal = EditorHAL("old")
    agent = make_agent(hal, plan("new"))
    assert agent.process_turn("replace").needs_confirmation
    hal.text = "newly edited by user"
    assert agent.process_turn("yes").needs_confirmation
    assert hal.text == "newly edited by user"


def test_negative_reply_wins_over_yes_word():
    hal = EditorHAL("old")
    agent = make_agent(hal, plan("new"))
    assert agent.process_turn("replace").needs_confirmation
    assert not agent.process_turn("no, don't say yes").success
    assert hal.text == "old"


def test_append_preserves_content_and_no_destructive_confirmation():
    hal = EditorHAL("old")
    result = make_agent(hal, plan(" new", "append")).process_turn("append")
    assert result.success and not result.needs_confirmation
    assert hal.text == "old new"


def test_paste_failure_is_not_reported_as_completion_or_retried_blindly():
    hal = EditorHAL()
    hal.drop_paste = True
    result = make_agent(hal, plan("new")).process_turn("write")
    assert not result.success
    assert hal.events.count(("type", "new")) == 1


def test_unresolved_editor_never_receives_text():
    hal = EditorHAL()
    hal.can_focus = False
    result = make_agent(hal, plan("new")).process_turn("write")
    assert not result.success
    assert not any(event[0] == "type" for event in hal.events)


def test_abort_prevents_next_semantic_action():
    hal = EditorHAL()
    agent = make_agent(hal, plan("new"))
    result = agent.process_turn("write", step_observer=lambda *args: agent.abort())
    assert not result.success
    assert not any(event[0] == "type" for event in hal.events)


def test_goal_spec_compiler_orders_focus_before_text_and_retains_payload():
    state = ComputerState(active_window=WindowInfo("Desktop", "explorer.exe", 1, (0, 0, 10, 10)))
    actions = StateAwarePlanner().plan(GoalSpec("content_equals", {"app": "Notepad"}, "exact\ntext"), state)
    assert [a.type for a in actions] == [ActionType.LAUNCH_APP, ActionType.TYPE_TEXT]
    assert actions[-1].payload == "exact\ntext"
    assert actions[-1].params["mode"] == "replace"


@pytest.mark.parametrize("steps", [
    [{"action": "launch_app", "target": "Notepad"}, {"action": "invented_tool"}],
    [{"action": "type_text"}],
    [{"action": "click", "target": "x", "confidence": float("nan")}],
    [{"action": "launch_app", "target": "Notepad"}] * 16,
])
def test_invalid_plan_rejected_as_a_whole(steps):
    with pytest.raises((TypeError, ValueError)):
        SemanticReasoner()._parse_llm_goal_spec("do work", {"steps": steps})


def test_offline_compound_literal_boundary():
    understanding = SemanticReasoner(groq_api_key="").analyze(
        "Open Notepad and write: open Chrome and delete nothing", ConversationMemory(), ComputerState())
    assert understanding.initial_actions[0].target == "Notepad"
    assert understanding.initial_actions[1].payload == "open Chrome and delete nothing"


def test_cloud_reasoner_is_used_for_desktop_task(monkeypatch):
    reasoner = SemanticReasoner(groq_api_key="test-key")
    expected = GoalUnderstanding(
        raw_query="Open Notepad",
        primary_goal="Open Notepad",
        initial_actions=[Action.launch_app("Notepad")],
    )
    cloud_call = Mock(return_value=expected)
    monkeypatch.setattr(reasoner, "_reason_with_llm", cloud_call)

    result = reasoner.analyze("Open Notepad", ConversationMemory(), ComputerState())

    assert result is expected
    cloud_call.assert_called_once()


def test_goal_specs_use_the_same_confidence_gate_as_model_actions():
    hal = EditorHAL()
    understanding = GoalUnderstanding("write", "write", goal_spec=GoalSpec(
        "content_equals", {"app": "Notepad"}, "text", confidence=0.1))
    result = make_agent(hal, understanding).process_turn("write")
    assert "clarif" in result.message.lower()
    assert hal.events == []


def test_confirmation_in_last_budget_slot_can_finish():
    hal = EditorHAL("old")
    agent = make_agent(hal, plan("new"))
    agent.max_loop_steps = 2
    assert agent.process_turn("replace").needs_confirmation
    assert agent.process_turn("yes").success
    assert agent._active_graph.steps_used == 2


def test_model_step_limit_rejects_whole_plan_before_execution():
    hal = EditorHAL()
    understanding = GoalUnderstanding("do work", "do work", initial_actions=[Action.launch_app("Notepad")] * 16)
    result = make_agent(hal, understanding).process_turn("do work")
    assert "step limit" in result.message
    assert not any(e[0] == "launch" for e in hal.events)


def test_clear_action_cannot_disable_safety_by_omitting_flag():
    hal = EditorHAL("old")
    executor = ToolBasedActionExecutor(os_adapter=hal)
    outcome = executor.execute_action(Action(type=ActionType.CLEAR_TEXT, is_destructive=False))
    assert outcome.needs_confirmation and hal.text == "old"


def test_offline_unknown_destructive_request_never_claims_completion():
    result = SemanticReasoner(groq_api_key="").analyze(
        "Delete everything in this folder", ConversationMemory(), ComputerState())
    assert result.needs_clarification and not result.initial_actions
