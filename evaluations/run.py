"""Safe, local task evaluation: mocked, native-filesystem, or opt-in desktop.

No model calls, microphone, screenshots, personal paths, payments or deletion.
Desktop input is restricted to the exact fixture PID and title before each call.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

# Do not consume developer credentials or models during evaluations.
for key in ("JEV_API_KEY", "TYPESAFE_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
    os.environ[key] = ""
os.environ["WISPRHELP_MOCK_MODELS"] = "true"
if sys.platform.startswith("linux") and not os.getenv("DISPLAY"):
    from unittest.mock import MagicMock
    sys.modules["pyautogui"] = MagicMock(FAILSAFE=True)

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.executor import ToolBasedActionExecutor
from wisprhelp.core.outcomes import ExecutionFault, FailureCode
from wisprhelp.hal.base import BaseOSAdapter, WindowInfo
from wisprhelp.hal.guarded import GuardedOSAdapter


class SimulatedDesktop(BaseOSAdapter):
    """In-memory desktop for mock tasks and filesystem-only integration."""
    def __init__(self):
        self.window = WindowInfo("Evaluation Fixture", "fixture", 1, (10, 10, 600, 400))
        self.text = ""
        self.page = 1
        self.calls = []
        self.selected = False
    def launch_app(self, app_identifier, parameter=None):
        self.calls.append("launch")
        return app_identifier == "Evaluation Fixture"
    def get_active_window(self): return self.window
    def check_emergency_stop(self): pass
    def inject_unicode_text(self, text):
        self.calls.append("type")
        self.text = text if self.selected else self.text + text
        self.selected = False
        return True
    def focus_text_input(self, field=None): return field in ("editor", "Evaluation Editor")
    def read_focused_text(self): return self.text
    def media_control(self, command): raise AssertionError("Media operations prohibited")
    def find_accessibility_element(self, query, control_type=None):
        return (100, 100) if query == "Next Test Page" else None
    def move_mouse(self, x, y, smooth=True): raise AssertionError("Mouse movement prohibited")
    def click(self, x=None, y=None, button="left", clicks=1):
        self.calls.append("click")
        self.page = 2
    def key_combination(self, keys):
        self.calls.append("hotkey")
        self.selected = keys in (["ctrl", "a"], ["command", "a"])
    def press_key(self, key): raise AssertionError("Unapproved key operation")


def record_task(name, executor, actions, oracle, expected_failure=None, after_step=None, recovery_actions=None):
    start = time.perf_counter()
    before = executor.os_adapter.mutations
    outcomes = []
    recovery_attempts = 0
    steps = list(actions)
    for index, action in enumerate(steps):
        result = executor.execute_action(action)
        outcomes.append(result)
        if after_step:
            after_step(index, result)
        if not result.success or result.needs_confirmation:
            if (recovery_actions and recovery_attempts == 0 and not result.dispatched
                    and result.failure_code == FailureCode.ACCESSIBILITY_TARGET_NOT_FOUND):
                recovery_attempts += 1
                steps[index + 1:] = recovery_actions
                continue
            break
    verified = bool(oracle(outcomes))
    completed = len(outcomes) == len(steps) and outcomes[-1].success and not outcomes[-1].needs_confirmation and all(
        o.success or (recovery_attempts and o.failure_code == FailureCode.ACCESSIBILITY_TARGET_NOT_FOUND and not o.dispatched) for o in outcomes)
    failure = next((o.failure_code for o in outcomes if not o.success), FailureCode.NONE)
    expected = verified and (failure == expected_failure if expected_failure else completed)
    applied = sum(o.dispatched for o in outcomes)
    status = "successful" if completed and verified else "partial" if applied else "failed"
    return {
        "task": name, "completion": status, "expected_behavior_met": expected,
        "postcondition_verified": verified, "tool_actions": len(outcomes),
        "native_input_calls": executor.os_adapter.mutations - before,
        "step_attempts": len(outcomes), "file_action_attempts": sum(o.action.type in {
            ActionType.CREATE_FILE, ActionType.APPEND_FILE, ActionType.WRITE_FILE} and o.dispatched for o in outcomes),
        "latency_ms": round((time.perf_counter() - start) * 1000, 3),
        "recovery_attempts": recovery_attempts, "unsafe_or_unauthorized_attempts": sum(o.needs_confirmation or o.failure_code == FailureCode.UNAUTHORIZED for o in outcomes),
        "failure_code": failure.value,
        "outcomes": [{"success": o.success, "failure_code": o.failure_code.value,
                      "needs_confirmation": o.needs_confirmation, "dispatched": o.dispatched} for o in outcomes],
    }


def mock_tasks():
    desktop = SimulatedDesktop()
    executor = ToolBasedActionExecutor(desktop)
    actions = [Action.launch_app("Evaluation Fixture"),
               Action(type=ActionType.TYPE_TEXT, payload="नमस्ते 👋", params={"mode": "replace", "field": "editor"}),
               Action.click("Next Test Page")]
    rows = [record_task("mock_multistep_unicode_navigation", executor, actions,
                        lambda _: desktop.text == "नमस्ते 👋" and desktop.page == 2)]
    missing = Action.click("Missing Button")
    missing.params["accessibility_only"] = True
    rows.append(record_task("mock_missing_accessibility_target", executor, [missing],
                            lambda _: desktop.page == 2, FailureCode.ACCESSIBILITY_TARGET_NOT_FOUND))
    desktop.page = 1
    rows.append(record_task("mock_approved_alternative_after_missing_target", executor, [missing],
                            lambda _: desktop.page == 2, recovery_actions=[Action.click("Next Test Page")]))
    cancel = threading.Event()
    executor.abort_requested = cancel.is_set
    rows.append(record_task("mock_cancel_between_steps", executor,
                            [Action.launch_app("Evaluation Fixture"), Action.click("Next Test Page")],
                            lambda results: len(results) == 2 and not results[-1].dispatched,
                            FailureCode.CANCELLED, after_step=lambda i, _: cancel.set() if i == 0 else None))
    return rows


def filesystem_tasks(root):
    executor = ToolBasedActionExecutor(SimulatedDesktop())
    path = root / "isolated-document.txt"
    rows = [record_task("native_create_append_read", executor,
                        [Action.create_file(str(path), "नमस्ते"), Action.append_file(str(path), " 👋"), Action.read_file(str(path))],
                        lambda _: path.read_text(encoding="utf-8") == "नमस्ते 👋")]
    rows.append(record_task("native_overwrite_requires_confirmation", executor, [Action.write_file(str(path), "replacement")],
                            lambda out: out[0].needs_confirmation and path.read_text(encoding="utf-8") == "नमस्ते 👋"))
    # Expected behavior is a held action, not completed file modification.
    rows[-1]["expected_behavior_met"] = rows[-1]["postcondition_verified"]
    return rows


def desktop_tasks(root):
    if sys.platform != "win32":
        return [{"task": "desktop_fixture", "completion": "not_run", "expected_behavior_met": False,
                 "reason": "The current GUI harness requires Windows UIA; macOS/Linux desktop tasks remain unverified."}]
    from wisprhelp.hal.windows import WindowsOSAdapter, HAS_WIN32, HAS_UIA
    if not HAS_WIN32 or not HAS_UIA:
        return [{"task": "desktop_fixture", "completion": "not_run", "expected_behavior_met": False,
                 "reason": "Windows desktop evaluation requires pywin32 and uiautomation",
                 "native_win32_available": HAS_WIN32, "uia_available": HAS_UIA}]
    fixture = Path(__file__).with_name("fixture_app.py").resolve()
    process = subprocess.Popen([sys.executable, str(fixture), str(root)])
    oracle_path = root / "oracle.json"
    def oracle():
        return json.loads(oracle_path.read_text(encoding="utf-8"))
    native = WindowsOSAdapter()
    def guard(operation):
        window = native.get_active_window()
        if window.pid != process.pid or window.title != "WisprHelp Evaluation Fixture":
            raise ExecutionFault(FailureCode.FOCUS_CHANGED, "Evaluation refused input outside fixture")
    rows = []
    try:
        deadline = time.perf_counter() + 10
        focus_attempted = False
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Fixture exited before becoming ready")
            if oracle_path.exists() and native.get_active_window().pid == process.pid:
                break
            if not focus_attempted and oracle_path.exists() and oracle()["pid"] == process.pid:
                native.focus_window("WisprHelp Evaluation Fixture")
                focus_attempted = True
            time.sleep(0.05)
        guard("ready")
        rows.append({"task": "desktop_fixture_startup", "completion": "successful",
                     "expected_behavior_met": oracle()["pid"] == process.pid, "postcondition_verified": True,
                     "tool_actions": 0, "latency_ms": None, "recovery_attempts": 0,
                     "unsafe_or_unauthorized_attempts": 0,
                     "note": "Harness subprocess launch; production launch_app was not exercised."})
        executor = ToolBasedActionExecutor(GuardedOSAdapter(native, guard))
        text = "WisprHelp नमस्ते 👋"
        rows.append(record_task("desktop_unicode_in_isolated_editor", executor,
                                [Action(type=ActionType.TYPE_TEXT, payload=text, params={"mode": "replace", "field": "Evaluation Editor"})],
                                lambda _: oracle()["text"] == text))
        action = Action.click("Next Test Page")
        action.params["accessibility_only"] = True
        rows.append(record_task("desktop_accessibility_navigation", executor, [action],
                                lambda _: oracle()["page"] == "Page 2" and oracle()["clicks"] == 1))
        action = Action.click("NON_EXISTENT_EVALUATION_CONTROL")
        action.params["accessibility_only"] = True
        rows.append(record_task("desktop_missing_element_stops", executor, [action],
                                lambda _: oracle()["clicks"] == 1, FailureCode.ACCESSIBILITY_TARGET_NOT_FOUND))
        cancelled = threading.Event()
        cancelled.set()
        executor.abort_requested = cancelled.is_set
        rows.append(record_task("desktop_cancel_blocks_input", executor, [Action.click("Next Test Page")],
                                lambda _: oracle()["clicks"] == 1, FailureCode.CANCELLED))
    except Exception as exc:
        rows.append({"task": "desktop_fixture", "completion": "failed", "expected_behavior_met": False,
                     "reason": str(exc), "failure_code": exc.code.value if isinstance(exc, ExecutionFault) else "action_failed",
                     "fixture_ready": oracle_path.exists(), "native_win32_available": HAS_WIN32, "uia_available": HAS_UIA})
    finally:
        (root / "stop").touch()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Terminate only the exact fixture process created above.
            process.terminate()
            process.wait(timeout=5)
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("mocked", "filesystem", "desktop"), default="mocked")
    parser.add_argument("--output", type=Path, default=Path("evaluation-results/latest.json"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(workspace):
        parser.error("Evaluation artifacts must stay inside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wisprhelp-eval-", dir=output.parent) as temp:
        root = Path(temp).resolve()
        assert root.is_relative_to(workspace) and root.name.startswith("wisprhelp-eval-")
        rows = mock_tasks() if args.mode == "mocked" else filesystem_tasks(root) if args.mode == "filesystem" else desktop_tasks(root)
    result = {"schema_version": 1, "mode": args.mode,
              "environment": {"os": platform.system(), "release": platform.release(), "python": platform.python_version(),
                              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                              "voice_tested": False, "models_tested": False}, "tasks": rows}
    result["summary"] = {status: sum(r["completion"] == status for r in rows)
                         for status in ("successful", "partial", "failed", "not_run")}
    result["summary"]["expected_behaviors_verified"] = sum(r["expected_behavior_met"] for r in rows)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    passed = sum(r["expected_behavior_met"] for r in rows)
    print(f"{args.mode}: {passed}/{len(rows)} expected behaviors verified; report: {output.relative_to(workspace)}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
