"""Small read-only smoke test for configured cloud providers.

Runs real API calls without printing credentials or response bodies.
"""

from __future__ import annotations

import base64
import json
import sys

import requests

from wisprhelp.agent.memory import ConversationMemory
from wisprhelp.agent.reasoning import SemanticReasoner
from wisprhelp.agent.state import ComputerState
from wisprhelp.classification.client import JevClient
from wisprhelp.classification.schemas import JEV_QUESTIONS_SPEC
from wisprhelp.config import config


def main() -> int:
    outcomes = {}
    if config.groq_api_key:
        try:
            result = SemanticReasoner(timeout=20)._reason_with_llm(
                "Open Notepad and write: API smoke test",
                ConversationMemory(),
                ComputerState(),
            )
            outcomes["groq_task"] = "ok" if result.initial_actions or result.goal_spec else "no plan"
        except Exception as exc:
            outcomes["groq_task"] = type(exc).__name__
    else:
        outcomes["groq_task"] = "missing key"

    if config.jev_api_key:
        try:
            response = JevClient(timeout=15, max_retries=1).evaluate(
                {"user_query": "Open Notepad", "os_platform": "windows"},
                JEV_QUESTIONS_SPEC,
            )
            outcomes["jev_task"] = "ok" if response.get("answers") else "no answers"
        except Exception as exc:
            outcomes["jev_task"] = type(exc).__name__
    else:
        outcomes["jev_task"] = "missing key"

    if config.gemini_api_key:
        try:
            # A 1x1 PNG makes a real multimodal request without capturing a screen.
            image = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lXcAAAAASUVORK5CYII="
            )
            payload = {"contents": [{"parts": [
                {"text": "Identify this image in a few words."},
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image).decode("ascii")}},
            ]}]}
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{config.gemini_model}:generateContent",
                headers={"x-goog-api-key": config.gemini_api_key},
                json=payload, timeout=25,
            )
            outcomes["gemini_vision"] = (
                "ok" if response.ok and response.json().get("candidates")
                else f"HTTP {response.status_code}"
            )
        except Exception as exc:
            outcomes["gemini_vision"] = type(exc).__name__
    else:
        outcomes["gemini_vision"] = "missing key"

    print(json.dumps(outcomes, sort_keys=True))
    return 0 if all(value == "ok" for value in outcomes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
