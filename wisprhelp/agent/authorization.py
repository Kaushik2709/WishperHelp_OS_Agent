"""Deterministic capability checks based only on the user's instruction.

Model plans and observed application text cannot grant additional privileges.
This is a narrow guard, not a general natural-language authorization proof.
"""
import re
from urllib.parse import urlparse

from wisprhelp.agent.actions import ActionType


FILE_MUTATIONS = {ActionType.CREATE_FILE, ActionType.WRITE_FILE, ActionType.APPEND_FILE,
                  ActionType.RENAME_FILE, ActionType.MOVE_FILE, ActionType.DELETE_FILE}


def plan_authorization_error(actions, user_instruction):
    words = user_instruction.casefold()
    for action in actions:
        if action.type in FILE_MUTATIONS | {ActionType.READ_FILE, ActionType.FIND_FILE}:
            if not re.search(r"\b(file|folder|directory|document|pdf|resume|cv|save|read|write|create|rename|move|delete|append)\b", words):
                return "File access was not requested by the user."
            # An explicit path cannot be introduced solely by observed content.
            target = action.target or ""
            if (":" in target or target.startswith(("/", "\\"))) and target.casefold() not in words:
                return "Confirm the exact file path in your instruction before accessing it."
        if action.type in (ActionType.LAUNCH_APP, ActionType.FOCUS_WINDOW):
            if any(term in (action.target or "").casefold() for term in ("terminal", "powershell", "cmd", "bash")):
                if (action.target or "").casefold() not in words:
                    return "A terminal cannot be opened on instructions from application content."
        if action.type == ActionType.TYPE_TEXT and action.params.get("app", "").casefold() in {
            "terminal", "powershell", "cmd", "bash", "terminal_code",
        }:
            action.is_destructive = True
        if action.type == ActionType.NAVIGATE:
            url = action.payload or action.target or ""
            host = (urlparse(url if "://" in url else "https://" + url).hostname or "").casefold()
            labels = [part for part in host.split(".") if len(part) > 2 and part != "www"]
            if not labels or not any(re.search(r"\b" + re.escape(label) + r"\b", words) for label in labels):
                return "Navigation requires a destination named in the user's instruction."
    return None
