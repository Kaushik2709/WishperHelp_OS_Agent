"""Unit tests for query decomposition, multi-word app finding, and human-like command execution."""

import pytest
from unittest.mock import MagicMock, patch

from wisprhelp.core.runtime import RuntimeCoordinator
from wisprhelp.classification.schemas import JevDecision, IntentType, TargetApp
from wisprhelp.core.router import RouteTarget


def test_parse_app_and_command_direct_launch():
    """Verify clean extraction of single and multi-word applications without truncation."""
    test_cases = [
        ("open task manager", "task manager", None),
        ("open visual studio code", "vscode", None),
        ("launch vscode please", "vscode", None),
        ("open google chrome", "chrome", None),
        ("open brave browser", "brave", None),
        ("open android studio", "android studio", None),
        ("open epic games launcher", "epic games launcher", None),
        ("open file explorer", "file explorer", None),
        ("open control panel", "control panel", None),
        ("open calculator", "calculator", None),
        ("open discord", "discord", None),
        ("open steam", "steam", None),
        ("open obs studio", "obs studio", None),
        ("open notepad", "notepad", None),
    ]

    for query, expected_app, expected_cmd in test_cases:
        app, cmd = RuntimeCoordinator._parse_app_and_command(query, "generic_or_active")
        assert app == expected_app, f"Failed for query '{query}': got '{app}', expected '{expected_app}'"
        assert cmd == expected_cmd, f"Failed cmd for query '{query}': got '{cmd}', expected '{expected_cmd}'"


def test_parse_app_and_command_compound_actions():
    """Verify decomposition of 'open X and do Y' and 'do X in Y' commands."""
    test_cases = [
        ("open chrome and search for weather in tokyo", "chrome", "search for weather in tokyo"),
        ("open notepad and type hello world", "notepad", "type hello world"),
        ("open spotify and play starboy", "spotify", "play starboy"),
        ("open calculator and calculate 50 * 20", "calculator", "calculate 50 * 20"),
        ("search machine learning papers on google chrome", "chrome", "search machine learning papers"),
        ("play lofi beats in spotify", "spotify", "play lofi beats"),
        ("type meeting notes for today in notepad", "notepad", "type meeting notes for today"),
    ]

    for query, expected_app, expected_cmd in test_cases:
        app, cmd = RuntimeCoordinator._parse_app_and_command(query, "generic_or_active")
        assert app == expected_app, f"Failed app for '{query}': got '{app}', expected '{expected_app}'"
        assert cmd == expected_cmd, f"Failed cmd for '{query}': got '{cmd}', expected '{expected_cmd}'"


def test_human_like_browser_search_action():
    """Verify that browser searches use standard Ctrl+L navigation and Enter."""
    mock_os = MagicMock()
    mock_classifier = MagicMock()
    mock_router = MagicMock()
    coordinator = RuntimeCoordinator(classifier=mock_classifier, router=mock_router, os_adapter=mock_os)

    decision = JevDecision(
        intent="gui_workflow_automation",
        intent_confidence=0.9,
        target_app="browser",
        target_app_confidence=0.9,
        complexity_score=1.2,
        complexity_confidence=0.9,
        is_destructive=False,
        destructive_probability=0.0,
        has_multilingual_content=False,
        multilingual_probability=0.0,
    )

    result = coordinator._execute_human_like_action(
        app_name="chrome",
        in_app_cmd="search for weather today",
        full_query="open chrome and search for weather today",
        decision=decision,
    )

    assert result.success is True
    # Verify Ctrl+L was pressed to focus address bar
    mock_os.key_combination.assert_called_with(["ctrl", "l"])
    # Verify query was injected
    mock_os.inject_unicode_text.assert_called_with("weather today")
    # Verify Enter was pressed to initiate navigation
    mock_os.press_key.assert_called_with("enter")


def test_human_like_text_injection_action():
    """Verify that text typing actions in editors inject unicode text."""
    mock_os = MagicMock()
    mock_os.inject_unicode_text.return_value = True
    coordinator = RuntimeCoordinator(os_adapter=mock_os)

    decision = JevDecision(
        intent="direct_app_launch",
        intent_confidence=0.8,
        target_app="generic_or_active",
        target_app_confidence=0.8,
        complexity_score=1.0,
        complexity_confidence=0.8,
        is_destructive=False,
        destructive_probability=0.0,
        has_multilingual_content=False,
        multilingual_probability=0.0,
    )

    result = coordinator._execute_human_like_action(
        app_name="notepad",
        in_app_cmd="type Hello World into notepad",
        full_query="open notepad and type Hello World into notepad",
        decision=decision,
    )

    assert result.success is True
    mock_os.inject_unicode_text.assert_called_with("Hello World")


def test_human_like_calculator_action():
    """Verify that calculator commands format and inject mathematical expressions."""
    mock_os = MagicMock()
    coordinator = RuntimeCoordinator(os_adapter=mock_os)

    decision = JevDecision(
        intent="gui_workflow_automation",
        intent_confidence=0.8,
        target_app="generic_or_active",
        target_app_confidence=0.8,
        complexity_score=1.0,
        complexity_confidence=0.8,
        is_destructive=False,
        destructive_probability=0.0,
        has_multilingual_content=False,
        multilingual_probability=0.0,
    )

    result = coordinator._execute_human_like_action(
        app_name="calculator",
        in_app_cmd="calculate 50 plus 20",
        full_query="open calculator and calculate 50 plus 20",
        decision=decision,
    )

    assert result.success is True
    mock_os.inject_unicode_text.assert_called_with("50+20")
    mock_os.press_key.assert_called_with("enter")


def test_compound_launch_and_execute_flow():
    """Verify that RuntimeCoordinator executes both launch and in-app action for compound queries."""
    mock_os = MagicMock()
    mock_os.launch_app.return_value = True
    mock_os.inject_unicode_text.return_value = True

    mock_classifier = MagicMock()
    decision = JevDecision(
        intent="direct_app_launch",
        intent_confidence=0.9,
        target_app="generic_or_active",
        target_app_confidence=0.8,
        complexity_score=0.4,
        complexity_confidence=0.8,
        is_destructive=False,
        destructive_probability=0.0,
        has_multilingual_content=False,
        multilingual_probability=0.0,
    )
    mock_classifier.classify.return_value = decision

    coordinator = RuntimeCoordinator(classifier=mock_classifier, os_adapter=mock_os)

    # User query has both open and type
    result = coordinator.process_command("open notepad and type Welcome to WisprHelp")

    assert result.success is True
    # App was launched
    mock_os.launch_app.assert_called_with("notepad")
    # Text was typed like a real user
    mock_os.inject_unicode_text.assert_called_with("Welcome to WisprHelp")


def test_parse_app_and_command_preposition_first():
    """Verify parsing of 'In/On <app>[,] <cmd>' patterns."""
    test_cases = [
        ("In VS Code, create a file called main.py", "vscode", "create a file called main.py"),
        ("In VS Code, create a file called main.by", "vscode", "create a file called main.py"),
        ("In notepad, type Hello from WisprHelp", "notepad", "type Hello from WisprHelp"),
        ("On Chrome, search for latest AI research", "chrome", "search for latest AI research"),
        ("In VS Code create a new file named test.py", "vscode", "create a new file named test.py"),
        ("create a file called main.py in VS Code", "vscode", "create a file called main.py"),
    ]

    for query, expected_app, expected_cmd in test_cases:
        app, cmd = RuntimeCoordinator._parse_app_and_command(query, "generic_or_active")
        assert app == expected_app, f"Failed app for '{query}': got '{app}', expected '{expected_app}'"
        assert cmd == expected_cmd, f"Failed cmd for '{query}': got '{cmd}', expected '{expected_cmd}'"


def test_human_like_file_creation_in_editor():
    """Verify that file creation commands in code editors execute Ctrl+N, Ctrl+S, type filename, and Enter."""
    mock_os = MagicMock()
    mock_os.inject_unicode_text.return_value = True
    coordinator = RuntimeCoordinator(os_adapter=mock_os)

    decision = JevDecision(
        intent="gui_workflow_automation",
        intent_confidence=0.9,
        target_app="vscode",
        target_app_confidence=0.9,
        complexity_score=1.0,
        complexity_confidence=0.8,
        is_destructive=False,
        destructive_probability=0.0,
        has_multilingual_content=False,
        multilingual_probability=0.0,
    )

    result = coordinator._execute_human_like_action(
        app_name="vscode",
        in_app_cmd="create a file called main.by",
        full_query="In VS Code, create a file called main.by",
        decision=decision,
    )

    assert result.success is True
    # Verify Ctrl+N was pressed
    mock_os.key_combination.assert_any_call(["ctrl", "n"])
    # Verify Ctrl+S was pressed
    mock_os.key_combination.assert_any_call(["ctrl", "s"])
    # Verify filename was injected (with .by corrected to .py)
    mock_os.inject_unicode_text.assert_called_with("main.py")
    # Verify Enter was pressed
    mock_os.press_key.assert_called_with("enter")


def test_extract_text_content_and_code_formatting():
    """Verify clean extraction of programming statements and full sentences."""
    test_cases = [
        ("Write code called print Hello World", 'print("Hello World")'),
        ("Write code print Hello World", 'print("Hello World")'),
        ('write code print("Hello World")', 'print("Hello World")'),
        ("write Hey I am Kaushik and I am a developer", "Hey I am Kaushik and I am a developer"),
        ("write Hey I am Kaushik and I am a developer in notepad", "Hey I am Kaushik and I am a developer"),
        ("type hello into chat", "hello"),
        ("write code def add(a, b): return a + b", "def add(a, b): return a + b"),
        ("write code console.log Hello World", 'console.log("Hello World")'),
        ("write code print 42", "print(42)"),
    ]

    for raw, expected in test_cases:
        extracted = RuntimeCoordinator._extract_text_content(raw)
        assert extracted == expected, f"Failed for '{raw}': got '{extracted}', expected '{expected}'"


def test_parse_app_and_command_direct_action_without_app():
    """Verify that action verbs without an app target the active focus without launching an app."""
    test_cases = [
        ("write code called print Hello World", "generic_or_active", "write code called print Hello World"),
        ("write Hey I am Kaushik and I am a developer", "generic_or_active", "write Hey I am Kaushik and I am a developer"),
        ("create a file called test.py", "generic_or_active", "create a file called test.py"),
        ("open code", "vscode", None),
        ("open code and create a file called test.py", "vscode", "create a file called test.py"),
    ]

    for raw, expected_app, expected_cmd in test_cases:
        app, cmd = RuntimeCoordinator._parse_app_and_command(raw, "generic_or_active")
        assert app == expected_app, f"Failed app for '{raw}': got '{app}', expected '{expected_app}'"
        assert cmd == expected_cmd, f"Failed cmd for '{raw}': got '{cmd}', expected '{expected_cmd}'"


def test_human_like_code_dictation_execution():
    """Verify that 'write code called print Hello World' injects 'print(\"Hello World\")'."""
    mock_os = MagicMock()
    mock_os.inject_unicode_text.return_value = True
    coordinator = RuntimeCoordinator(os_adapter=mock_os)

    decision = JevDecision(
        intent="gui_workflow_automation",
        intent_confidence=0.9,
        target_app="generic_or_active",
        target_app_confidence=0.8,
        complexity_score=1.0,
        complexity_confidence=0.8,
        is_destructive=False,
        destructive_probability=0.0,
        has_multilingual_content=False,
        multilingual_probability=0.0,
    )

    result = coordinator._execute_human_like_action(
        app_name="vscode",
        in_app_cmd="write code called print Hello World",
        full_query="In VS Code, write code called print Hello World",
        decision=decision,
    )

    assert result.success is True
    mock_os.inject_unicode_text.assert_called_with('print("Hello World")')


def test_extract_button_or_control_name():
    """Verify clean extraction of target button names from natural speech."""
    test_cases = [
        ("Click the Enter button on Antigravity", ["Enter", "Enter button", "Send", "Submit"]),
        ("Click the Accept Changes button on Antigravity", ["Accept Changes", "Accept", "Accept Changes button", "Keep", "Apply"]),
        ("press the blue button in Notepad", ["blue", "blue button"]),
        ("Click on Save", ["Save"]),
        ("press that button", ["that button"]),
        ("Click Accept Changes", ["Accept Changes", "Accept", "Keep", "Apply"]),
    ]

    for raw, expected_candidates in test_cases:
        candidates = RuntimeCoordinator._extract_button_or_control_name(raw)
        for exp in expected_candidates:
            assert exp in candidates, f"Expected '{exp}' in candidates for '{raw}', got {candidates}"


def test_parse_app_and_command_button_clicking():
    """Verify that click and press commands decompose properly into app and in-app commands."""
    test_cases = [
        ("Click the Enter button on Antigravity", "antigravity", "Click the Enter button"),
        ("Click the Accept Changes button on Antigravity", "antigravity", "Click the Accept Changes button"),
        ("press the blue button in Notepad", "notepad", "press the blue button"),
        ("In Antigravity, click the Enter button", "antigravity", "click the Enter button"),
        ("Click Accept Changes", "generic_or_active", "Click Accept Changes"),
        ("press that button", "generic_or_active", "press that button"),
    ]

    for raw, expected_app, expected_cmd in test_cases:
        app, cmd = RuntimeCoordinator._parse_app_and_command(raw, "generic_or_active")
        assert app == expected_app, f"Failed app for '{raw}': got '{app}', expected '{expected_app}'"
        assert cmd == expected_cmd, f"Failed cmd for '{raw}': got '{cmd}', expected '{expected_cmd}'"


def test_human_like_button_click_never_types_in_editor():
    """Verify that clicking buttons in editors clicks UIA/Vision and NEVER types the click command as text."""
    mock_os = MagicMock()
    mock_os.find_accessibility_element.return_value = (600, 400)
    coordinator = RuntimeCoordinator(os_adapter=mock_os)

    decision = JevDecision(
        intent="gui_workflow_automation",
        intent_confidence=0.9,
        target_app="antigravity",
        target_app_confidence=0.9,
        complexity_score=1.0,
        complexity_confidence=0.8,
        is_destructive=False,
        destructive_probability=0.0,
        has_multilingual_content=False,
        multilingual_probability=0.0,
    )

    # 1. Accept Changes button in Antigravity
    res1 = coordinator._execute_human_like_action(
        app_name="antigravity",
        in_app_cmd="Click the Accept Changes button",
        full_query="Click the Accept Changes button on Antigravity",
        decision=decision,
    )
    assert res1.success is True
    mock_os.click.assert_called_with(600, 400)
    mock_os.inject_unicode_text.assert_not_called()

    # 2. Press blue button in Notepad: UIA not found -> escalates to Vision, never types text
    mock_os.find_accessibility_element.return_value = None
    mock_vlm = MagicMock()
    mock_vlm.execute_visual_task.return_value = MagicMock(success=True, message="Clicked blue button", steps_taken=1)
    coordinator_vlm = RuntimeCoordinator(os_adapter=mock_os, vlm_agent=mock_vlm)

    res2 = coordinator_vlm._execute_human_like_action(
        app_name="notepad",
        in_app_cmd="press the blue button",
        full_query="press the blue button in Notepad",
        decision=decision,
    )
    assert res2.success is True
    mock_vlm.execute_visual_task.assert_called_once()
    mock_os.inject_unicode_text.assert_not_called()


