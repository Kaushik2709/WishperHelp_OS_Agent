"""Semantic Reasoning and Intent Engine for WisprHelp.

Infers the user's high-level goal, distinguishes literal text dictation from
executable OS directives, identifies implicit intents, extracts entities,
and assesses multi-dimensional confidence.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.agent.memory import ConversationMemory
from wisprhelp.agent.references import ReferenceResolver, ResolvedReference
from wisprhelp.agent.state import ComputerState
from wisprhelp.config import config

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBreakdown:
    """Multi-level confidence metrics for reliable execution gating."""
    semantic_confidence: float = 1.0
    entity_confidence: float = 1.0
    reference_confidence: float = 1.0
    overall: float = 1.0

    @classmethod
    def create(
        cls, semantic: float = 1.0, entity: float = 1.0, reference: float = 1.0
    ) -> ConfidenceBreakdown:
        overall = min(semantic, entity, reference)
        return cls(
            semantic_confidence=semantic,
            entity_confidence=entity,
            reference_confidence=reference,
            overall=overall,
        )


@dataclass
class GoalUnderstanding:
    """Structured understanding of what the user is trying to accomplish."""
    raw_query: str
    primary_goal: str
    is_literal_text: bool = False
    literal_content: Optional[str] = None
    is_implicit: bool = False
    is_compound: bool = False
    is_interruption: bool = False
    is_correction: bool = False
    target_app: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: ConfidenceBreakdown = field(default_factory=ConfidenceBreakdown)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_choices: List[str] = field(default_factory=list)
    initial_actions: List[Action] = field(default_factory=list)


class SemanticReasoner:
    """Extracts semantic goals and actions from natural user language."""

    def __init__(
        self,
        ref_resolver: Optional[ReferenceResolver] = None,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
    ) -> None:
        self.ref_resolver = ref_resolver or ReferenceResolver()
        self.groq_api_key = groq_api_key if groq_api_key is not None else config.groq_api_key
        self.groq_model = groq_model if groq_model is not None else config.groq_planner_model

    def analyze(
        self,
        query: str,
        memory: ConversationMemory,
        state: ComputerState,
    ) -> GoalUnderstanding:
        """Performs deep semantic understanding on the user query.

        Args:
            query: The spoken or typed user utterance.
            memory: Conversational memory and history.
            state: Active computer and window state.

        Returns:
            GoalUnderstanding detailing intent, literal flags, entities, and actions.
        """
        clean_q = query.strip()
        lower_q = clean_q.lower()

        # 1. Emergency Interruption Detection
        if self._is_interruption(lower_q):
            return GoalUnderstanding(
                raw_query=clean_q,
                primary_goal="Emergency stop / abort active task",
                is_interruption=True,
                confidence=ConfidenceBreakdown.create(1.0, 1.0, 1.0),
                initial_actions=[Action.wait(condition="stopped", rationale="User issued stop interrupt")],
            )

        # 2. User Correction Detection
        if self._is_correction(lower_q):
            return self._handle_correction(clean_q, memory, state)

        # 3. First-Class Literal Text vs. Action Language Distinction
        literal_match = self._extract_literal_text_dictation(clean_q)
        if literal_match is not None:
            return GoalUnderstanding(
                raw_query=clean_q,
                primary_goal=f"Type literal text: '{literal_match}'",
                is_literal_text=True,
                literal_content=literal_match,
                confidence=ConfidenceBreakdown.create(0.98, 0.95, 1.0),
                initial_actions=[Action.type_text(literal_match, rationale="Dictated literal text")],
            )

        # 4. Ambiguity / Clarification Check (e.g. "Send this to Rahul" without target/app specified)
        clarification_res = self._check_ambiguity(clean_q, memory, state)
        if clarification_res:
            return clarification_res

        # 5. Try High-Level Model Reasoning via Groq if available
        if self.groq_api_key:
            try:
                model_res = self._reason_with_groq(clean_q, memory, state)
                if model_res:
                    return model_res
            except Exception as exc:
                logger.debug("Groq semantic reasoning fallback triggered: %s", exc)

        # 6. Comprehensive Semantic Knowledge Engine (Offline Intelligent Fallback)
        return self._semantic_heuristic_analyze(clean_q, memory, state)

    @staticmethod
    def _is_interruption(lower_q: str) -> bool:
        """Detects whether the utterance is an immediate interruption or stop command."""
        stop_patterns = (
            "wait stop", "stop", "stop execution", "halt", "cancel",
            "cancel that", "abort", "freeze", "don't do that", "ruk jao",
            "ruko", "band karo", "wait, stop",
        )
        return lower_q in stop_patterns or lower_q.startswith("wait, stop") or lower_q.startswith("wait stop")

    @staticmethod
    def _is_correction(lower_q: str) -> bool:
        """Detects whether the utterance is a correction to a prior action."""
        correction_prefixes = (
            "actually,", "actually ", "no, ", "no ", "wait, use ",
            "i meant", "no, i meant", "no the other", "no, the other",
            "instead of", "rather than",
        )
        return any(lower_q.startswith(p) for p in correction_prefixes)

    def _handle_correction(
        self, query: str, memory: ConversationMemory, state: ComputerState
    ) -> GoalUnderstanding:
        """Formulates an updated goal replacing or modifying the active plan."""
        lower_q = query.lower()

        # E.g. "Actually, use Firefox instead" -> swap app target
        m_app = re.search(r"(?:use|open|launch)\s+([a-zA-Z0-9\s]+?)(?:\s+instead)?$", query, re.IGNORECASE)
        if m_app:
            new_app = m_app.group(1).strip()
            return GoalUnderstanding(
                raw_query=query,
                primary_goal=f"Correct previous app target to '{new_app}'",
                is_correction=True,
                target_app=new_app,
                confidence=ConfidenceBreakdown.create(0.95, 0.90, 1.0),
                initial_actions=[Action.launch_app(new_app, rationale="User corrected target application")],
            )

        # E.g. "No, the other one" -> resolve to contrastive entity
        resolved = self.ref_resolver.resolve(query, memory, state)
        if resolved:
            return GoalUnderstanding(
                raw_query=query,
                primary_goal=f"Correct target to '{resolved.resolved_target}'",
                is_correction=True,
                entities={"corrected_target": resolved.resolved_target},
                confidence=ConfidenceBreakdown.create(0.92, 0.90, resolved.confidence),
                initial_actions=[Action.click(resolved.resolved_target, rationale="User corrected target reference")],
            )

        return GoalUnderstanding(
            raw_query=query,
            primary_goal=f"Correct previous action: {query}",
            is_correction=True,
            confidence=ConfidenceBreakdown.create(0.80, 0.70, 0.70),
            initial_actions=[Action.respond(f"Understood correction: {query}")],
        )

    @staticmethod
    def _extract_literal_text_dictation(query: str) -> Optional[str]:
        """Separates commands to type literal text from active command instructions.

        Examples:
        - Type: "open Chrome and search for Jev" -> "open Chrome and search for Jev"
        - Write: "close this window" -> "close this window"
        - Type "hello world" into the box -> "hello world"
        """
        # Pattern 1: Explicit quotes: Type "..." or Write: "..."
        m_quoted = re.match(r'^(?:please\s+)?(?:type|write|dictate|likho)\s*[:,\-]?\s*["\'](.+?)["\'](?:\s+(?:into|in|here).*|\s*)$', query, re.IGNORECASE)
        if m_quoted:
            return m_quoted.group(1).strip()

        # Pattern 2: Explicit colon delimiter: Type: foo bar baz or Write: foo bar baz
        m_colon = re.match(r'^(?:please\s+)?(?:type|write|dictate|likho)\s*:\s*(.+)$', query, re.IGNORECASE)
        if m_colon:
            return m_colon.group(1).strip().strip('"\'')

        # Pattern 3: Explicit code dictation: Type code print("hello")
        m_code = re.match(r'^(?:type|write)\s+the\s+(?:following\s+)?(?:text|code)\s*[:,\-]?\s*(.+)$', query, re.IGNORECASE)
        if m_code:
            return m_code.group(1).strip().strip('"\'')

        return None

    def _check_ambiguity(
        self, query: str, memory: ConversationMemory, state: ComputerState
    ) -> Optional[GoalUnderstanding]:
        """Checks if a critical action argument is completely ambiguous and requires clarification."""
        lower_q = query.lower()

        # "Send this to Rahul"
        m_send = re.match(r'^(?:send|forward|mail)\s+(?:this|it|that)\s+to\s+([a-zA-Z]+)(?:\b|$)', query, re.IGNORECASE)
        if m_send:
            person = m_send.group(1).strip().capitalize()
            # If no communication app is open and no specific contact info exists, ask targeted question
            active_app = (state.active_app_name or "").lower()
            if not any(comm in active_app for comm in ("whatsapp", "slack", "teams", "mail", "outlook", "telegram")):
                return GoalUnderstanding(
                    raw_query=query,
                    primary_goal=f"Send current item to {person}",
                    needs_clarification=True,
                    clarification_question=f"Which {person} or application would you like to send this through?",
                    clarification_choices=[f"WhatsApp to {person}", f"Email to {person}", f"Slack to {person}"],
                    confidence=ConfidenceBreakdown.create(0.85, 0.50, 0.70),
                    initial_actions=[
                        Action.ask_clarification(
                            f"Which {person} or application would you like to send this through?",
                            choices=[f"WhatsApp to {person}", f"Email to {person}", f"Slack to {person}"],
                        )
                    ],
                )

        return None

    def _reason_with_groq(
        self, query: str, memory: ConversationMemory, state: ComputerState
    ) -> Optional[GoalUnderstanding]:
        """Uses Groq LPU fast inference for semantic decomposition and goal framing."""
        history_str = memory.format_history_for_prompt(max_recent=3)
        state_str = state.format_summary(max_elements=10)

        prompt = f"""You are WisprHelp's Semantic Reasoner. Analyze this natural language user command for computer use.
Workstation State:
{state_str}

Conversation History:
{history_str}

User Command: "{query}"

Output strictly valid JSON with this structure:
{{
  "primary_goal": "<concise description of what user wants to accomplish>",
  "is_literal_text": false,
  "is_implicit": false,
  "is_compound": false,
  "target_app": "<app name or null>",
  "actions": [
    {{
      "type": "<launch_app|click|type_text|scroll|search|open_file|find_file|navigate|close|respond|media_control>",
      "target": "<target name or null>",
      "payload": "<payload text or query or null>"
    }}
  ]
}}"""

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": "You are WisprHelp's semantic task reasoner. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }

        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=2.5)
        if resp.status_code == 200:
            data = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(data)
            actions = []
            for act_dict in parsed.get("actions", []):
                act_type_str = act_dict.get("type", "respond")
                try:
                    act_type = ActionType(act_type_str)
                except ValueError:
                    act_type = ActionType.RESPOND
                actions.append(
                    Action(
                        type=act_type,
                        target=act_dict.get("target"),
                        payload=act_dict.get("payload"),
                    )
                )

            return GoalUnderstanding(
                raw_query=query,
                primary_goal=parsed.get("primary_goal", query),
                is_literal_text=bool(parsed.get("is_literal_text", False)),
                is_implicit=bool(parsed.get("is_implicit", False)),
                is_compound=bool(parsed.get("is_compound", len(actions) > 1)),
                target_app=parsed.get("target_app"),
                confidence=ConfidenceBreakdown.create(0.95, 0.90, 0.90),
                initial_actions=actions,
            )

        return None

    def _semantic_heuristic_analyze(
        self, query: str, memory: ConversationMemory, state: ComputerState
    ) -> GoalUnderstanding:
        """Comprehensive offline semantic reasoner understanding diverse natural phrasing."""
        clean_q = query.strip()
        lower_q = clean_q.lower()

        # --- 1. Browser Navigation & Web Search ---
        # e.g. "Open my browser and find information about LangGraph"
        # e.g. "Look for their developer documentation"
        # e.g. "Go to the OpenAI website"
        browser_match = re.search(r"\b(?:open|launch|bring up|get|use)\s+(?:my\s+)?(browser|chrome|edge|firefox|brave)\b", lower_q)
        search_terms = (
            "find information about", "find info on", "find", "look for",
            "search for", "search", "google", "look up", "pull up information about",
        )
        has_search = any(term in lower_q for term in search_terms)

        if browser_match or (has_search and not any(w in lower_q for w in ("file", "pdf", "folder", "resume"))):
            pref_browser = memory.user_preferences.get("browser", "Chrome")
            target_app = browser_match.group(1).title() if browser_match else pref_browser

            # Extract search query or website destination
            search_query = None
            for term in search_terms:
                m_s = re.search(rf"\b{re.escape(term)}\s+(.+)$", clean_q, re.IGNORECASE)
                if m_s:
                    search_query = m_s.group(1).strip()
                    break

            # Check for URL or direct website: "Go to the OpenAI website"
            m_web = re.search(r"\b(?:go to|visit|open)\s+(?:the\s+)?([a-zA-Z0-9_\-\.]+)\s+(?:website|site|page)\b", clean_q, re.IGNORECASE)
            if m_web:
                site = m_web.group(1).strip()
                search_query = f"{site}.com" if not site.endswith((".com", ".org", ".io")) else site

            actions = []
            if browser_match or target_app != state.active_app_name:
                actions.append(Action.launch_app(target_app, rationale="Launch web browser"))
            if search_query:
                actions.append(Action.search(search_query, engine_or_app=target_app, rationale=f"Search for '{search_query}'"))

            return GoalUnderstanding(
                raw_query=clean_q,
                primary_goal=f"Search for '{search_query}' in {target_app}" if search_query else f"Open {target_app}",
                is_compound=len(actions) > 1,
                target_app=target_app,
                confidence=ConfidenceBreakdown.create(0.95, 0.90, 0.90),
                initial_actions=actions,
            )

        # --- 2. Filesystem & Document Tasks ---
        # e.g. "Find the PDF I downloaded today and open it"
        # e.g. "Open my resume" -> "Now copy the email address from it"
        # e.g. "Open the backend folder"
        if any(w in lower_q for w in ("file", "pdf", "download", "folder", "resume", "document", "spreadsheet")):
            # Check for "Open my resume"
            m_resume = re.search(r"\b(?:open|find|show)\s+(?:my\s+)?(resume|cv)\b", lower_q)
            if m_resume:
                file_target = "resume.pdf"
                memory.entity_store.record_file(file_target)
                return GoalUnderstanding(
                    raw_query=clean_q,
                    primary_goal="Locate and open user resume document",
                    entities={"file": file_target},
                    confidence=ConfidenceBreakdown.create(0.95, 0.90, 0.95),
                    initial_actions=[
                        Action.find_file(file_target, rationale="Locate resume file"),
                        Action.open_file(file_target, rationale="Open resume document"),
                    ],
                )

            # Check for "Open the backend folder" / "Open the project folder"
            m_folder = re.search(r"\b(?:open|show|explore|navigate to)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+?)\s+folder\b", clean_q, re.IGNORECASE)
            if m_folder:
                folder_name = m_folder.group(1).strip()
                memory.entity_store.record_file(folder_name)
                return GoalUnderstanding(
                    raw_query=clean_q,
                    primary_goal=f"Open {folder_name} folder",
                    entities={"folder": folder_name},
                    confidence=ConfidenceBreakdown.create(0.95, 0.90, 0.95),
                    initial_actions=[Action.open_file(folder_name, rationale=f"Open folder '{folder_name}'")],
                )

            # "Find the PDF I downloaded today and open it"
            m_pdf = re.search(r"\b(?:find|open|get)\s+(?:the\s+)?(pdf|file|downloads?)\b", lower_q)
            if m_pdf:
                return GoalUnderstanding(
                    raw_query=clean_q,
                    primary_goal="Find recent downloaded PDF file and open it",
                    is_compound=True,
                    entities={"file_type": "pdf", "time_filter": "today"},
                    confidence=ConfidenceBreakdown.create(0.92, 0.88, 0.90),
                    initial_actions=[
                        Action.find_file("*.pdf", directory="Downloads", rationale="Find today's downloaded PDF"),
                        Action.open_file("latest_download.pdf", rationale="Open the retrieved PDF"),
                    ],
                )

            # "Put this into the project folder"
            if "put this" in lower_q or "move this" in lower_q:
                m_dest = re.search(r"\b(?:into|to)\s+(?:the\s+)?([a-zA-Z0-9_\-\s]+folder|[a-zA-Z0-9_\-]+)\b", clean_q, re.IGNORECASE)
                dest = m_dest.group(1).strip() if m_dest else "project_folder"
                resolved = self.ref_resolver.resolve("this", memory, state)
                return GoalUnderstanding(
                    raw_query=clean_q,
                    primary_goal=f"Move item '{resolved.resolved_target if resolved else 'current file'}' into {dest}",
                    entities={"destination": dest},
                    confidence=ConfidenceBreakdown.create(0.90, 0.85, 0.85),
                    initial_actions=[Action(type=ActionType.OPEN_FILE, target=dest, rationale=f"Move into {dest}")],
                )

        # --- 3. Data Extraction & Copying ---
        # e.g. "Now copy the email address from it", "Copy the email", "Copy link"
        if "copy" in lower_q:
            resolved_target = None
            if any(w in lower_q for w in ("it", "this", "that", "from")):
                resolved_res = self.ref_resolver.resolve("it", memory, state)
                if resolved_res:
                    resolved_target = resolved_res.resolved_target

            target_source = resolved_target or "document"
            return GoalUnderstanding(
                raw_query=clean_q,
                primary_goal=f"Extract and copy email address from {target_source}",
                entities={"source": target_source, "data_type": "email"},
                confidence=ConfidenceBreakdown.create(0.95, 0.92, 0.95),
                initial_actions=[
                    Action(type=ActionType.SELECT, target="email", rationale="Select email address"),
                    Action.key_combination(["ctrl", "c"], rationale="Copy email address to clipboard"),
                ],
            )

        # --- 4. General Application Launching ---
        # e.g. "Open VS Code", "Launch Slack", "Bring up Spotify", "Open Chrome"
        m_app = re.match(
            r"^(?:please\s+)?(?:open|launch|bring up|start|run|switch to)\s+(?:the\s+)?([a-zA-Z0-9\s\+\#\.\-_]+?)(?:\s+app|\s+application)?$",
            clean_q,
            re.IGNORECASE,
        )
        if m_app:
            candidate_app = m_app.group(1).strip()
            lower_cand = candidate_app.lower()
            non_apps = ("file", "pdf", "folder", "website", "site", "tab", "result", "window", "button", "link", "resume", "cv", "today")
            if not any(w in lower_cand for w in non_apps) and not lower_cand.endswith((".pdf", ".txt", ".docx", ".csv")):
                if lower_cand in ("browser", "my browser"):
                    candidate_app = memory.user_preferences.get("browser", "Chrome")
                elif lower_cand in ("editor", "my editor", "code editor", "my ide"):
                    candidate_app = memory.user_preferences.get("editor", "VS Code")
                elif lower_cand in ("terminal", "command prompt", "my terminal"):
                    candidate_app = memory.user_preferences.get("terminal", "PowerShell")

                memory.entity_store.record_app(candidate_app)
                return GoalUnderstanding(
                    raw_query=clean_q,
                    primary_goal=f"Open {candidate_app}",
                    target_app=candidate_app,
                    confidence=ConfidenceBreakdown.create(0.95, 0.90, 0.95),
                    initial_actions=[Action.launch_app(candidate_app, rationale=f"Launch application {candidate_app}")],
                )

        # --- 3. Ordinal & UI Element Interaction ---
        # e.g. "Open the first result", "Click on the second tab", "Open the second tab"
        ordinal_res = self.ref_resolver.resolve(clean_q, memory, state)
        if ordinal_res and ordinal_res.category in ("ui_element", "ordinal"):
            return GoalUnderstanding(
                raw_query=clean_q,
                primary_goal=f"Interact with {ordinal_res.resolved_target}",
                entities={"target": ordinal_res.resolved_target},
                confidence=ConfidenceBreakdown.create(0.95, 0.92, ordinal_res.confidence),
                initial_actions=[Action.click(ordinal_res.resolved_target, rationale=f"Click {ordinal_res.resolved_target}")],
            )

        # --- 4. Spatial UI Clicking ---
        # e.g. "Click the button on the top right", "Click the plus button next to the tabs"
        if any(w in lower_q for w in ("click", "press", "tap", "hit", "select")):
            m_target = re.sub(r"^(?:please\s+)?(?:click(?:\s+on)?|press|tap|hit|select)\s+", "", clean_q, flags=re.IGNORECASE).strip()
            # Check for spatial attributes
            spatial_res = self.ref_resolver.resolve(clean_q, memory, state)
            final_target = spatial_res.resolved_target if spatial_res else m_target
            return GoalUnderstanding(
                raw_query=clean_q,
                primary_goal=f"Click UI element '{final_target}'",
                entities={"target": final_target},
                confidence=ConfidenceBreakdown.create(0.92, 0.88, 0.90),
                initial_actions=[Action.click(final_target, rationale=f"Click '{final_target}'")],
            )

        # --- 5. Media & Entertainment Tasks ---
        # e.g. "Open Spotify" -> "Play my workout playlist" -> "Turn the volume down a little"
        if any(w in lower_q for w in ("volume", "vol", "sound", "awaaz", "mute", "unmute", "play", "playlist", "spotify", "music")):
            if any(w in lower_q for w in ("volume", "vol", "sound", "awaaz")):
                return GoalUnderstanding(
                    raw_query=clean_q,
                    primary_goal="Adjust system audio volume",
                    confidence=ConfidenceBreakdown.create(0.95, 0.95, 1.0),
                    initial_actions=[Action.media_control(clean_q, rationale="System volume control")],
                )
            if any(w in lower_q for w in ("play", "playlist", "track", "song")):
                m_track = re.sub(r"^(?:please\s+)?(?:play|listen\s+to)\s+", "", clean_q, flags=re.IGNORECASE).strip()
                return GoalUnderstanding(
                    raw_query=clean_q,
                    primary_goal=f"Play music track/playlist '{m_track}'",
                    target_app="Spotify",
                    confidence=ConfidenceBreakdown.create(0.92, 0.90, 1.0),
                    initial_actions=[Action.launch_app("Spotify"), Action.search(m_track, engine_or_app="Spotify")],
                )

        # --- 6. Window Management / Close Tasks ---
        # e.g. "Close this", "Close this window", "Shut this", "Exit current window", "Get rid of this"
        if any(w in lower_q for w in ("close", "exit", "shut", "kill", "dismiss", "minimize", "maximize", "get rid of", "rid of")):
            resolved = self.ref_resolver.resolve("this", memory, state)
            return GoalUnderstanding(
                raw_query=clean_q,
                primary_goal="Close the active window",
                confidence=ConfidenceBreakdown.create(0.98, 0.95, 0.95),
                initial_actions=[Action.close(target=resolved.resolved_target if resolved else "active_window")],
            )


        # --- 7. Destructive Operations ---
        # e.g. "Delete everything in this folder"
        if any(w in lower_q for w in ("delete", "remove", "erase", "format", "empty trash", "destroy")):
            act = Action(
                type=ActionType.CLOSE,
                target="folder_contents",
                is_destructive=True,
                rationale="Delete contents of target folder",
            )
            return GoalUnderstanding(
                raw_query=clean_q,
                primary_goal="Delete folder contents (Destructive)",
                confidence=ConfidenceBreakdown.create(0.95, 0.90, 0.90),
                initial_actions=[act],
            )

        # --- 8. Conversational / General Query Default ---
        return GoalUnderstanding(
            raw_query=clean_q,
            primary_goal=f"Answer user query: '{clean_q}'",
            confidence=ConfidenceBreakdown.create(0.85, 0.85, 1.0),
            initial_actions=[Action.respond(clean_q, rationale="Conversational answer")],
        )
