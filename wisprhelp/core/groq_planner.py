"""Groq LPU Strategic Planner & Conversational Reasoner (System Two).

Fulfills the 'Astra' role from the MindStudio benchmark:
Provides fast (<100ms) high-level sub-goal planning, milestone sequencing,
and generative text responses to supervise the grounded execution layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from wisprhelp.config import config

logger = logging.getLogger(__name__)


@dataclass
class StrategicMilestone:
    """A high-level milestone to be achieved by the grounded action agent."""
    step_number: int
    description: str
    target_app: Optional[str] = None
    expected_outcome: str = ""


@dataclass
class StrategicPlan:
    """A sequenced collection of milestones for open-ended GUI tasks."""
    user_goal: str
    milestones: List[StrategicMilestone] = field(default_factory=list)
    is_complex: bool = False


class GroqStrategicPlanner:
    """System Two Strategic Planner powered by Groq LPU API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 3.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.groq_api_key
        self.model = model or config.groq_planner_model
        self.timeout = timeout
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def plan_workflow(
        self,
        user_goal: str,
        active_app: Optional[str] = None,
    ) -> StrategicPlan:
        """Decomposes a complex GUI goal into 2 to 4 strategic milestones.

        Args:
            user_goal: The high-level user request.
            active_app: The currently active or targeted application.

        Returns:
            StrategicPlan containing ordered milestones.
        """
        if not self.api_key:
            return self._heuristic_plan(user_goal, active_app)

        prompt = f"""You are WisprHelp's System Two Strategic Planner.
Break down this desktop computer task into 2 to 4 sequential, high-level milestones:
Goal: "{user_goal}"
Target App: {active_app or "Active Window"}

Output strictly valid JSON with this schema:
{{
  "is_complex": true,
  "milestones": [
    {{
      "step_number": 1,
      "description": "<what to do first>",
      "target_app": "<app name>",
      "expected_outcome": "<what visual state indicates success>"
    }}
  ]
}}"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert desktop task planner. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                milestones = []
                for m in parsed.get("milestones", []):
                    milestones.append(
                        StrategicMilestone(
                            step_number=int(m.get("step_number", len(milestones) + 1)),
                            description=str(m.get("description", "")),
                            target_app=m.get("target_app"),
                            expected_outcome=str(m.get("expected_outcome", "")),
                        )
                    )
                logger.info("Groq Strategic Planner generated %d milestones for: '%s'", len(milestones), user_goal)
                return StrategicPlan(
                    user_goal=user_goal,
                    milestones=milestones,
                    is_complex=bool(parsed.get("is_complex", len(milestones) > 1)),
                )
        except Exception as exc:
            logger.warning("Groq planner failed (%s); falling back to heuristic plan.", exc)

        return self._heuristic_plan(user_goal, active_app)

    def answer_conversational(self, user_query: str) -> str:
        """Generates a concise, natural spoken reply for conversational queries."""
        if not self.api_key:
            return f"I heard: {user_query}."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are WisprHelp's voice assistant. Give concise, helpful spoken responses (max 2 sentences).",
                },
                {"role": "user", "content": user_query},
            ],
            "temperature": 0.3,
            "max_tokens": 150,
        }

        try:
            resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Conversational reply failed: %s", exc)

        return f"Regarding '{user_query}': completed."

    def _heuristic_plan(self, goal: str, target_app: Optional[str]) -> StrategicPlan:
        """Fast offline fallback plan."""
        return StrategicPlan(
            user_goal=goal,
            milestones=[
                StrategicMilestone(
                    step_number=1,
                    description=goal,
                    target_app=target_app,
                    expected_outcome="Task finished",
                )
            ],
            is_complex=False,
        )
