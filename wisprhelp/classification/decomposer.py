"""Semantic Pre-Decomposer powered by Groq LPU API.

Pre-processes natural speech queries, corrects phonetic STT typos (e.g., 'chrom' -> 'Chrome'),
and decomposes compound commands into structured atomic steps before Jev classification.
Includes deterministic local fallback for zero-latency offline resilience.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

from wisprhelp.config import config
from wisprhelp.classification.schemas import (
    DecomposedQuery,
    DecomposedStep,
    IntentType,
    TargetApp,
)

logger = logging.getLogger(__name__)

GROQ_DECOMPOSER_SYSTEM_PROMPT = """You are WisprHelp's Semantic Pre-Classifier & Query Decomposer.
Your role is to understand user speech queries, correct phonetic STT typos (e.g. 'chrom' -> 'Google Chrome', 'watsap' -> 'WhatsApp', 'spotfy' -> 'Spotify'), and decompose compound or single commands into atomic executable steps.

CRITICAL RULES FOR COMPOUND ACTIONS:
- If a query involves opening/launching an app AND doing an in-app action (e.g. search, click, type), you MUST set "is_compound": true and produce AT LEAST TWO separate steps in "sub_queries":
  * Step 1: action_type "launch", target: the app name.
  * Step 2: action_type "in_app_search", "click_element", or "type_text", target: the app/control, and payload: the extracted search term or text to type.

You must output strictly valid JSON matching this structure:
{
  "target_app": "Google Chrome",
  "is_compound": true,
  "normalized_intent": "gui_workflow_automation",
  "sub_queries": [
    {
      "step_number": 1,
      "query_text": "Open Google Chrome",
      "action_type": "launch",
      "target": "Google Chrome",
      "payload": null
    },
    {
      "step_number": 2,
      "query_text": "Search for Tobey Maguire",
      "action_type": "in_app_search",
      "target": "Google Chrome",
      "payload": "tobey maguire"
    }
  ]
}
"""


class SemanticDecomposer:
    """Decomposes compound user queries into atomic steps using Groq LPU with heuristic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 3.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.groq_api_key
        self.model = model or config.groq_planner_model or "llama-3.3-70b-versatile"
        self.timeout = timeout
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def decompose(
        self,
        user_query: str,
        os_context: Optional[Dict[str, Any]] = None,
    ) -> DecomposedQuery:
        """Decomposes a query into atomic steps using Groq LPU or local heuristic fallback.

        Args:
            user_query: Spoken user query.
            os_context: Active window or OS context.

        Returns:
            DecomposedQuery with structured sub-actions.
        """
        clean_query = user_query.strip()
        if not clean_query:
            return DecomposedQuery(
                original_query="",
                target_app=None,
                is_compound=False,
                normalized_intent=IntentType.CONVERSATIONAL_QUERY.value,
                sub_queries=[],
            )

        # If no Groq API key configured, use local heuristic fallback immediately
        if not self.api_key:
            return self._fallback_heuristic_decompose(clean_query, os_context)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            context_hint = ""
            if os_context:
                win_title = os_context.get("active_window_title")
                proc_name = os_context.get("active_process_name")
                if win_title or proc_name:
                    context_hint = f" [Current active window: '{win_title}', process: '{proc_name}']"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": GROQ_DECOMPOSER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"{clean_query}{context_hint}\nRespond in valid JSON only."},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
                "max_tokens": 300,
            }

            t0 = time.perf_counter()
            r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if r.status_code == 200:
                raw_json = r.json()["choices"][0]["message"]["content"]
                parsed = json.loads(raw_json)
                logger.info(
                    "Groq Decomposer decomposed '%s' in %.1fms: is_compound=%s",
                    clean_query,
                    elapsed_ms,
                    parsed.get("is_compound", False),
                )
                return self._parse_groq_json(clean_query, parsed)
            else:
                logger.warning(
                    "Groq API decomposition error %s (%s). Falling back to local heuristic.",
                    r.status_code,
                    r.text[:100],
                )
                return self._fallback_heuristic_decompose(clean_query, os_context)

        except Exception as exc:
            logger.warning("Groq decomposer request failed: %s. Falling back to local heuristic.", exc)
            return self._fallback_heuristic_decompose(clean_query, os_context)

    def _parse_groq_json(self, original_query: str, data: Dict[str, Any]) -> DecomposedQuery:
        """Parses structured JSON response from Groq into a DecomposedQuery object."""
        target_app = data.get("target_app")
        is_compound = bool(data.get("is_compound", False))
        norm_intent = data.get("normalized_intent", IntentType.GUI_WORKFLOW_AUTOMATION.value)

        sub_queries: List[DecomposedStep] = []
        raw_steps = data.get("sub_queries") or data.get("steps") or []
        for idx, step_dict in enumerate(raw_steps, start=1):
            s_num = int(step_dict.get("step_number", idx))
            q_text = str(step_dict.get("query_text", "")).strip()
            act_type = str(step_dict.get("action_type", "in_app_action")).strip()
            tgt = step_dict.get("target")
            payload = step_dict.get("payload")
            if payload is not None and not isinstance(payload, str):
                payload = str(payload)

            sub_queries.append(
                DecomposedStep(
                    step_number=s_num,
                    query_text=q_text or original_query,
                    action_type=act_type,
                    target=tgt if tgt else target_app,
                    payload=payload,
                )
            )

        if not sub_queries:
            sub_queries.append(
                DecomposedStep(
                    step_number=1,
                    query_text=original_query,
                    action_type="in_app_action",
                    target=target_app,
                    payload=None,
                )
            )

        # Safety Interlock: If query was flagged compound but Groq returned < 2 steps,
        # supplement with local heuristic decomposer so multi-actions are never dropped
        if is_compound and len(sub_queries) < 2:
            fallback = self._fallback_heuristic_decompose(original_query)
            if len(fallback.sub_queries) > len(sub_queries):
                sub_queries = fallback.sub_queries
                if not target_app:
                    target_app = fallback.target_app

        return DecomposedQuery(
            original_query=original_query,
            target_app=target_app,
            is_compound=is_compound or len(sub_queries) > 1,
            normalized_intent=norm_intent,
            sub_queries=sub_queries,
        )

    def _fallback_heuristic_decompose(
        self,
        query: str,
        os_context: Optional[Dict[str, Any]] = None,
    ) -> DecomposedQuery:
        """High-speed local deterministic regex fallback when Groq is unreachable."""
        lower_q = query.lower().strip()

        # 0. Conversational & Knowledge Queries (who are you, what can you do, jokes, greetings)
        conversational_triggers = (
            "who are you", "what is your name", "what can you do", "what are your capabilities",
            "tell me a joke", "tell me about yourself", "how are you",
            "what is wisprhelp", "what can i ask", "koun ho", "kya kar sakte ho",
            "namaste", "good morning", "good afternoon", "good evening", "how's it going",
        )
        single_words = {"hi", "hello", "hey", "help", "who"}
        q_words = set(re.findall(r"\b\w+\b", lower_q))
        if any(w in lower_q for w in conversational_triggers) or (bool(q_words & single_words) and len(q_words) <= 2):
            return DecomposedQuery(
                original_query=query,
                target_app=None,
                is_compound=False,
                normalized_intent=IntentType.CONVERSATIONAL_QUERY.value,
                sub_queries=[
                    DecomposedStep(
                        step_number=1,
                        query_text=query,
                        action_type="conversational_query",
                        target=None,
                        payload=query,
                    )
                ],
            )

        # 1. Media commands
        if any(w in lower_q for w in ("volume", "vol", "sound", "awaaz", "mute", "unmute")):
            m_pct = re.search(r"\b(\d{1,3})\s*(?:%|percent)?\b", lower_q)
            vol_val = m_pct.group(1) if m_pct else None
            return DecomposedQuery(
                original_query=query,
                target_app=None,
                is_compound=False,
                normalized_intent=IntentType.SYSTEM_MEDIA_CONTROL.value,
                sub_queries=[
                    DecomposedStep(
                        step_number=1,
                        query_text=query,
                        action_type="media_control",
                        target="volume",
                        payload=f"set_volume:{vol_val}" if vol_val else lower_q,
                    )
                ],
            )

        # 2. Compound commands: "open/launch <app> and/then <action>"
        compound_match = re.match(
            r"^(?:please\s+)?(?:open|launch|start)\s+([a-zA-Z0-9\s]+?)\s+(?:and|then|aur|phir)\s+(.+)$",
            query,
            re.IGNORECASE,
        )
        if compound_match:
            app_raw = compound_match.group(1).strip()
            action_raw = compound_match.group(2).strip()
            action_lower = action_raw.lower()

            alias_map = {
                "chrom": "Chrome",
                "chrome": "Chrome",
                "google chrome": "Chrome",
                "watsap": "WhatsApp",
                "whatsapp": "WhatsApp",
                "spotfy": "Spotify",
                "spotify": "Spotify",
                "notepad": "Notepad",
                "calc": "Calculator",
                "calculator": "Calculator",
                "vscode": "VS Code",
                "vs code": "VS Code",
                "code": "VS Code",
            }
            canonical_app = alias_map.get(app_raw.lower(), app_raw)

            steps = [
                DecomposedStep(
                    step_number=1,
                    query_text=f"Open {canonical_app}",
                    action_type="launch",
                    target=canonical_app,
                    payload=None,
                )
            ]

            # Sub-action classification
            if any(action_lower.startswith(w) for w in ("search for", "search", "google", "look up", "find")):
                search_term = re.sub(r"^(?:search\s+for|search|google|look\s+up|find)\s+", "", action_raw, flags=re.IGNORECASE).strip()
                steps.append(
                    DecomposedStep(
                        step_number=2,
                        query_text=action_raw,
                        action_type="in_app_search",
                        target=canonical_app,
                        payload=search_term,
                    )
                )
            elif any(w in action_lower for w in ("click", "press", "tap", "select", "button", "chat", "link")):
                steps.append(
                    DecomposedStep(
                        step_number=2,
                        query_text=action_raw,
                        action_type="click_element",
                        target=canonical_app,
                        payload=action_raw,
                    )
                )
            elif any(action_lower.startswith(w) for w in ("type", "write", "likho")):
                text_content = re.sub(r"^(?:type|write|likho)\s+", "", action_raw, flags=re.IGNORECASE).strip()
                steps.append(
                    DecomposedStep(
                        step_number=2,
                        query_text=action_raw,
                        action_type="type_text",
                        target=canonical_app,
                        payload=text_content,
                    )
                )
            else:
                steps.append(
                    DecomposedStep(
                        step_number=2,
                        query_text=action_raw,
                        action_type="in_app_action",
                        target=canonical_app,
                        payload=action_raw,
                    )
                )

            return DecomposedQuery(
                original_query=query,
                target_app=canonical_app,
                is_compound=True,
                normalized_intent=IntentType.GUI_WORKFLOW_AUTOMATION.value,
                sub_queries=steps,
            )

        # 3. Direct App Launch
        launch_match = re.match(r"^(?:please\s+)?(?:open|launch|start|run)\s+([a-zA-Z0-9\s]+)$", query, re.IGNORECASE)
        if launch_match and len(lower_q.split()) <= 4:
            app_raw = launch_match.group(1).strip()
            return DecomposedQuery(
                original_query=query,
                target_app=app_raw,
                is_compound=False,
                normalized_intent=IntentType.DIRECT_APP_LAUNCH.value,
                sub_queries=[
                    DecomposedStep(
                        step_number=1,
                        query_text=query,
                        action_type="launch",
                        target=app_raw,
                        payload=None,
                    )
                ],
            )

        # 4. Click Action
        is_click = any(w in lower_q for w in ("click", "press", "tap", "hit", "select", "button", "chat", "tab"))
        if is_click:
            return DecomposedQuery(
                original_query=query,
                target_app=None,
                is_compound=False,
                normalized_intent=IntentType.GUI_WORKFLOW_AUTOMATION.value,
                sub_queries=[
                    DecomposedStep(
                        step_number=1,
                        query_text=query,
                        action_type="click_element",
                        target=None,
                        payload=query,
                    )
                ],
            )

        # 5. Default single action
        return DecomposedQuery(
            original_query=query,
            target_app=None,
            is_compound=False,
            normalized_intent=IntentType.GUI_WORKFLOW_AUTOMATION.value,
            sub_queries=[
                DecomposedStep(
                    step_number=1,
                    query_text=query,
                    action_type="in_app_action",
                    target=None,
                    payload=None,
                )
            ],
        )
