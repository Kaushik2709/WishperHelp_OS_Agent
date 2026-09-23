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
        lower_q = user_query.lower().strip()
        if not self.api_key:
            if any(w in lower_q for w in ("who are you", "what is your name", "koun ho")):
                return "I'm WisprHelp, your autonomous universal computer assistant. I can open apps, search the web, manage browser tabs, and automate desktop workflows."
            if any(w in lower_q for w in ("what can you do", "help me", "capabilities", "kya kar sakte ho")):
                return "I can navigate your computer, launch applications, search the web, write notes and emails, adjust system volume, and execute multi-step workflows."
            if any(w in lower_q for w in ("how are you", "kya haal", "kaise ho")):
                return "I'm doing great, thank you! How can I help you on your computer right now?"
            if "joke" in lower_q:
                return "Why do programmers prefer dark mode? Because light attracts bugs!"
            return f"I understand: '{user_query}'. Let me know what you would like me to do on your computer."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are WisprHelp, a friendly, intelligent autonomous computer-use assistant. "
                        "Give warm, concise, natural spoken responses (max 2-3 sentences). "
                        "Do NOT use markdown headers, asterisks, or bullet points, as your response will be spoken aloud via text-to-speech. "
                        "If asked who you are or what you do, introduce yourself as WisprHelp, an AI computer-using agent."
                    ),
                },
                {"role": "user", "content": user_query},
            ],
            "temperature": 0.4,
            "max_tokens": 160,
        }

        try:
            resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                reply = data["choices"][0]["message"]["content"].strip()
                # Sanitize markdown formatting for clean speech and terminal display
                reply = reply.replace("**", "").replace("*", "").replace("#", "").strip()
                return reply
        except Exception as exc:
            logger.warning("Conversational reply failed: %s", exc)

        return f"Regarding '{user_query}': I'm ready to help you with your next task."

    def compose_text(self, prompt: str) -> str:
        """Composes creative, professional, or code text for direct insertion into documents or active apps."""
        lower_p = prompt.lower().strip()
        if not self.api_key:
            if "email" in lower_p:
                return "Subject: Quick Update\n\nHi there,\n\nI wanted to share a quick update regarding our discussion. Please let me know if you need any additional details.\n\nBest regards,\n[Your Name]"
            if "python" in lower_p or "function" in lower_p or "code" in lower_p:
                return "# Generated Python snippet\ndef process_data(items):\n    return [item.strip() for item in items if item]\n"
            return f"Artificial intelligence and automation are transforming productivity, enabling faster workflows and smarter computing for everyday tasks."

        system_msg = (
            "You are WisprHelp's text composer. The user has asked you to write or compose content "
            "directly into their active computer application (Notepad, Word, web editor, chat, or IDE).\n"
            "STRICT RULES:\n"
            "1. Output ONLY the exact text content to be typed.\n"
            "2. Do NOT include conversational filler like 'Sure! Here is the text:' or 'Hope this helps!'.\n"
            "3. Do NOT wrap in markdown code blocks unless the user explicitly requested code.\n"
            "4. Keep the content clear, natural, well-formatted, and human-like."
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 400,
        }

        try:
            resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # If wrapped in markdown code fences, unwrap unless code was requested
                if content.startswith("```") and content.endswith("```") and "code" not in lower_p and "python" not in lower_p:
                    lines = content.splitlines()
                    if len(lines) >= 2:
                        content = "\n".join(lines[1:-1]).strip()
                return content
        except Exception as exc:
            logger.warning("Generative text composition failed: %s", exc)

        return f"Notes on: {prompt}"

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
