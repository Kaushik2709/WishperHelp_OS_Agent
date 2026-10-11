"""Tests for Priority 1 Native Filesystem & Application Actions."""

import os
from pathlib import Path

import pytest
from unittest.mock import MagicMock
from wisprhelp.hal.base import BaseOSAdapter

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.executor import ToolBasedActionExecutor

pytestmark = pytest.mark.filesystem


def test_create_write_read_file(tmp_path: Path):
    """Verifies direct native file creation, writing, and reading without GUI."""
    executor = ToolBasedActionExecutor(os_adapter=MagicMock(spec=BaseOSAdapter))
    target_file = tmp_path / "test_notes.txt"

    # 1. Create file
    create_act = Action.create_file(str(target_file), content="Initial text content")
    out1 = executor.execute_action(create_act)
    assert out1.success is True
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == "Initial text content"

    # 2. Append to file
    append_act = Action.append_file(str(target_file), content="\nAppended line")
    out2 = executor.execute_action(append_act)
    assert out2.success is True
    assert "Appended line" in target_file.read_text(encoding="utf-8")

    # 3. Read file
    read_act = Action.read_file(str(target_file))
    out3 = executor.execute_action(read_act)
    assert out3.success is True
    assert "Initial text content" in out3.observation


def test_rename_and_delete_file(tmp_path: Path):
    """Verifies native file rename and safety-gated delete operations."""
    executor = ToolBasedActionExecutor(os_adapter=MagicMock(spec=BaseOSAdapter))
    old_file = tmp_path / "old_name.txt"
    old_file.write_text("Rename me", encoding="utf-8")

    # 1. Rename
    rename_act = Action.rename_file(str(old_file), "renamed.txt")
    out_rename = executor.execute_action(rename_act)
    assert out_rename.success is True
    new_file = tmp_path / "renamed.txt"
    assert new_file.exists()
    assert not old_file.exists()

    # 2. Delete (unconfirmed should hold at safety gate because is_destructive=True)
    delete_act = Action.delete_file(str(new_file))
    assert delete_act.is_destructive is True

    out_unconfirmed = executor.execute_action(delete_act, confirmed=False)
    assert out_unconfirmed.needs_confirmation is True
    assert new_file.exists()  # File still exists!

    # 3. Delete (confirmed should delete file natively)
    out_confirmed = executor.execute_action(delete_act, confirmed=True)
    assert out_confirmed.success is True
    assert not new_file.exists()  # Deleted!
