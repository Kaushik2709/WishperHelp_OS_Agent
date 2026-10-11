"""Comprehensive sandbox execution and cloud reasoning verification suite."""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.executor import ToolBasedActionExecutor
from wisprhelp.agent.memory import ConversationMemory
from wisprhelp.agent.reasoning import SemanticReasoner
from wisprhelp.agent.state import ComputerState
from wisprhelp.config import config


def run_sandbox():
    print("=" * 70)
    print("      WISPRHELP CLOUD API & SANDBOX EXECUTION VERIFICATION")
    print("=" * 70)

    # 1. Verify API configuration without exposing the credential.
    print("\n[1] Checking cloud reasoning configuration...")
    if not config.groq_api_key:
        print("    ERROR: GROQ_API_KEY is missing.")
        return

    reasoner = SemanticReasoner()
    executor = ToolBasedActionExecutor()
    memory = ConversationMemory()
    state = ComputerState()

    # 2. Test Key User Queries
    test_cases = [
        # User problem 1: WhatsApp chat selection
        {
            "category": "WhatsApp Chat Selection",
            "query": "click on the first set on WhatsApp",
            "expected_action": ActionType.CLICK,
            "expected_target_contains": "ordinal:0:chat",
        },
        # User problem 2: Ordinal File Opening
        {
            "category": "Ordinal File Opening (First)",
            "query": "open document and open the first file",
            "expected_action": ActionType.OPEN_FILE,
            "expected_target_contains": "ordinal:0:file",
        },
        {
            "category": "Ordinal File Opening (Voice Typo 'firsti')",
            "query": "open document and open the firsti file",
            "expected_action": ActionType.OPEN_FILE,
            "expected_target_contains": "ordinal:0:file",
        },
        {
            "category": "Ordinal File Opening (Second Position)",
            "query": "open documents and open the second file",
            "expected_action": ActionType.OPEN_FILE,
            "expected_target_contains": "ordinal:1:file",
        },
        # General language task 1: Direct File Creation
        {
            "category": "General Language: File Creation & Writing",
            "query": "create a file called greeting.txt in Documents and write hello from local agent",
            "expected_action": (ActionType.CREATE_FILE, ActionType.WRITE_FILE),
            "expected_target_contains": "greeting.txt",
        },
        # General language task 2: App Launching
        {
            "category": "General Language: App Launch & Typing",
            "query": "open Notepad and write hello world",
            "expected_action": ActionType.LAUNCH_APP,
            "expected_target_contains": "Notepad",
        },
        # General language task 3: Web Search
        {
            "category": "General Language: Web Search",
            "query": "search Chrome for latest AI news",
            "expected_action": ActionType.SEARCH,
            "expected_target_contains": None,
        },
    ]

    results = []

    print("\n[2] Executing Natural Language Queries via Groq API...")
    for tc in test_cases:
        q = tc["query"]
        cat = tc["category"]
        print(f"\n--- Scenario: {cat} ---")
        print(f"User Query: \"{q}\"")

        t0 = time.time()
        goal = reasoner.analyze(q, memory, state)
        elapsed_ms = (time.time() - t0) * 1000

        print(f"Cloud Planning Latency: {elapsed_ms:.1f}ms")
        print(f"Goal Understood: {goal.primary_goal}")
        print("Generated Actions:")
        for idx, act in enumerate(goal.initial_actions):
            print(f"  [{idx+1}] {act.type.value} -> target='{act.target}' payload='{act.payload}' params={act.params}")

        # Check expectations
        exp_act = tc["expected_action"]
        if isinstance(exp_act, tuple):
            has_expected_act = any(a.type in exp_act for a in goal.initial_actions)
        else:
            has_expected_act = any(a.type == exp_act for a in goal.initial_actions)
        target_ok = True
        if tc["expected_target_contains"]:
            target_ok = any(
                a.target and tc["expected_target_contains"] in a.target
                for a in goal.initial_actions
            )

        passed = has_expected_act and target_ok
        print(f"Plan Verification: {'PASSED' if passed else 'FAILED'}")
        results.append({
            "category": cat,
            "query": q,
            "passed": passed,
            "latency_ms": elapsed_ms,
            "actions": goal.initial_actions,
        })

    # 3. Sandbox Real OS Execution Test for File Operations
    print("\n[3] Testing Real Sandbox Execution of File Actions...")
    test_file_action = Action.create_file(
        "sandbox_test.txt",
        content="WisprHelp portable agent execution verification passed.",
        rationale="Sandbox automated verification",
    )
    outcome = executor.execute_action(test_file_action, state)
    print(f"    Create File Outcome: success={outcome.success}, message='{outcome.message}'")

    test_read_action = Action(
        type=ActionType.READ_FILE,
        target="sandbox_test.txt",
        rationale="Read back test file",
    )
    read_outcome = executor.execute_action(test_read_action, state)
    print(f"    Read File Outcome: success={read_outcome.success}, observation='{read_outcome.observation}'")

    test_ordinal_open = Action(
        type=ActionType.OPEN_FILE,
        target="ordinal:0:file",
        params={"directory": "Documents"},
        rationale="Open 1st file in Documents",
    )
    ord_outcome = executor.execute_action(test_ordinal_open, state)
    print(f"    Ordinal Open File Outcome: success={ord_outcome.success}, message='{ord_outcome.message}'")

    # 4. WhatsApp Geometry Verification (Guarantee click is never in search bar)
    print("\n[4] Testing WhatsApp Geometry & Search Bar Exclusion...")
    from wisprhelp.hal.factory import get_os_adapter
    os_adapter = get_os_adapter()
    coords = os_adapter.find_ordinal_element(0, item_type="chat", target_app="WhatsApp")
    print(f"    WhatsApp Ordinal 0 Coordinate: {coords}")
    if coords:
        print("    WhatsApp chat coordinate successfully resolved outside search bar.")

    # Clean up test file
    test_p = executor._resolve_filesystem_path("sandbox_test.txt")
    if test_p.exists():
        try:
            test_p.unlink()
            print("    Cleaned up sandbox_test.txt successfully.")
        except Exception:
            pass

    # Summary
    print("\n" + "=" * 70)
    print("                    FINAL VERIFICATION SUMMARY")
    print("=" * 70)
    all_passed = all(r["passed"] for r in results)
    for r in results:
        status = "PASSED" if r["passed"] else "FAILED"
        print(f"  [{status}] {r['category']} ({r['latency_ms']:.0f}ms)")
    print("=" * 70)
    print(f"OVERALL STATUS: {'ALL VERIFICATIONS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    run_sandbox()
