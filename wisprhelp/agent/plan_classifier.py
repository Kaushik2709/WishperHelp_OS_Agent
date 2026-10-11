"""Per-step TypeSafe Jev plan classification.

Takes the LLM reasoner's decomposed action plan and evaluates Jev's calibrated
multi-question matrix for ALL steps in a single batched /v1/systemone call,
merging LLM confidence with Jev's calibrated probabilities per step.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from wisprhelp.agent.actions import Action, ActionType
from wisprhelp.classification.client import JevClient
from wisprhelp.config import config

logger = logging.getLogger(__name__)


# Action types that never need Jev classification (no OS side effects).
_NO_CLASSIFY_TYPES = (ActionType.RESPOND, ActionType.ASK_CLARIFICATION, ActionType.WAIT)


def _safe_float(value: Any, default: float) -> float:
    """Parses a float, treating None/invalid as the default (NOT falsy-zero)."""
    if value is None:
        return default
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


@dataclass
class ClassifiedStep:
    """A single plan step enriched with merged LLM + Jev calibrated metrics."""
    action: Action
    alignment: str = "correct_step"  # correct_step | possibly_related | wrong_step
    alignment_confidence: float = 1.0
    complexity: float = 1.0
    complexity_confidence: float = 1.0
    destructive_probability: float = 0.0
    confidence: float = 1.0  # merged = min(llm_confidence, jev_alignment_confidence)
    tier: int = 2  # 1 (Shell/deterministic), 2 (UIA), 3 (Vision loop)
    jev_evaluated: bool = False


class PlanClassifier:
    """Batch per-step classification of an LLM action plan via TypeSafe Jev."""

    def __init__(
        self,
        client: Optional[JevClient] = None,
        confidence_threshold: Optional[float] = None,
        destructive_threshold: Optional[float] = None,
    ) -> None:
        self.client = client or JevClient()
        self.confidence_threshold = confidence_threshold or config.intent_confidence_threshold
        self.destructive_threshold = destructive_threshold or config.destructive_risk_threshold
        self.last_failure = None

    def classify_plan(
        self,
        steps: List[Action],
        goal: str,
        os_context: Optional[Dict[str, Any]] = None,
    ) -> List[ClassifiedStep]:
        """Classifies every actionable step in a single batched Jev evaluation.

        Args:
            steps: The LLM reasoner's decomposed action plan.
            goal: The primary goal summary for context.
            os_context: Active window/process context.

        Returns:
            List of ClassifiedStep aligned 1:1 with the input steps. If Jev is
            unavailable, LLM-only metrics are returned (graceful degradation).
        """
        if not steps:
            return []
        self.last_failure = None

        actionable = [s for s in steps if s.type not in _NO_CLASSIFY_TYPES]
        if not actionable:
            return [ClassifiedStep(action=s, jev_evaluated=False) for s in steps]

        # Build the batched questions spec: one question set per step.
        questions: Dict[str, Dict[str, Any]] = {}
        step_ctx: List[Dict[str, Any]] = []
        idx = 0
        for step in steps:
            if step.type in _NO_CLASSIFY_TYPES:
                continue
            idx += 1
            step_ctx.append(
                {
                    "step": idx,
                    "action": step.type.value,
                    "target": step.target,
                    "payload": (step.payload[:80] if step.payload else None),
                    "rationale": step.rationale,
                }
            )
            questions[f"step_{idx}_alignment"] = {
                "type": "choice",
                "instructions": (
                    f"Step {idx}: does this action correctly and safely serve the user's goal?"
                ),
                "criteria": {
                    "correct_step": "The action directly, safely, or preparatory (e.g. launching or activating the target app) advances the user's stated goal.",
                    "possibly_related": "The action is only tangentially related or unclear in relation to the goal.",
                    "wrong_step": "The action contradicts or does not serve the user's goal.",
                },
            }
            questions[f"step_{idx}_complexity"] = {
                "type": "score",
                "instructions": f"Step {idx}: rate the operational complexity of this single step",
                "criteria": [
                    "Level 0: Deterministic single system call, launch, keystroke, or media hotkey.",
                    "Level 1: Single targeted interaction in the active window (type into focused field, press key).",
                    "Level 2: Interaction requiring element grounding by name or vision (clicking UI elements).",
                ],
            }
            questions[f"step_{idx}_is_destructive"] = {
                "type": "noul",
                "instructions": (
                    f"Step {idx}: does this action delete data, clear existing content, send messages, "
                    "make purchases, terminate processes, or modify critical state?"
                ),
            }

        state = {
            "user_goal": goal,
            "os_platform": (os_context or {}).get("os_platform", "windows"),
            "active_window_title": (os_context or {}).get("active_window_title", ""),
            "active_process_name": (os_context or {}).get("active_process_name", ""),
            "plan_steps": step_ctx,
        }

        results: Dict[int, Dict[str, Any]] = {}
        api_failed = False
        if self.client.api_key:
            try:
                raw = self.client.evaluate(state=state, questions=questions)
                answers = raw.get("answers", {})
                for i in range(1, idx + 1):
                    align_data = answers.get(f"step_{i}_alignment", {}) or {}
                    comp_data = answers.get(f"step_{i}_complexity", {}) or {}
                    dest_data = answers.get(f"step_{i}_is_destructive", {}) or {}
                    alignment = str(align_data.get("choice", "wrong_step"))
                    raw_conf = _safe_float(align_data.get("confidence"), 0.0)
                    probs = align_data.get("probabilities", {}) or {}
                    prob_wrong = _safe_float(probs.get("wrong_step"), 0.0)
                    prob_correct = _safe_float(probs.get("correct_step"), 0.0)

                    # Accurately derive alignment confidence from calibrated Jev output
                    if alignment == "wrong_step":
                        alignment_conf = min(prob_wrong if prob_wrong > 0 else raw_conf, 0.20)
                    elif alignment == "correct_step":
                        if probs:
                            alignment_conf = min(prob_correct, 1.0 - prob_wrong, raw_conf)
                        else:
                            alignment_conf = raw_conf
                    elif alignment == "possibly_related":
                        alignment_conf = min(raw_conf, 0.40)
                    else:
                        alignment_conf = 0.0

                    results[i] = {
                        "alignment": alignment,
                        "alignment_confidence": max(0.0, min(1.0, alignment_conf)),
                        "complexity": _safe_float(comp_data.get("score"), 1.0),
                        "complexity_confidence": _safe_float(comp_data.get("confidence"), 0.8),
                        "destructive_probability": _safe_float(dest_data.get("noul"), 1.0),
                    }
            except Exception as exc:
                api_failed = True
                self.last_failure = "api_unavailable"
                logger.warning("Batched Jev plan classification failed: %s. Using LLM-only metrics.", exc)
        else:
            logger.info("No Jev API key configured; plan classification uses LLM-only metrics.")

        classified: List[ClassifiedStep] = []
        running_idx = 0
        for step in steps:
            if step.type in _NO_CLASSIFY_TYPES:
                classified.append(ClassifiedStep(action=step, jev_evaluated=False))
                continue
            running_idx += 1
            jev = results.get(running_idx)
            if jev is None:
                # Graceful degradation: LLM-only metrics
                classified.append(
                    ClassifiedStep(
                        action=step,
                        alignment="correct_step",
                        alignment_confidence=step.confidence,
                        complexity=1.0,
                        destructive_probability=0.95 if step.is_destructive else 0.0,
                        confidence=0.0 if api_failed else step.confidence,
                        tier=self._tier_for_complexity(1.0),
                        jev_evaluated=False,
                    )
                )
                continue

            merged_confidence = min(step.confidence, jev["alignment_confidence"])
            destructive_prob = max(
                jev["destructive_probability"], 0.95 if step.is_destructive else 0.0
            )
            classified.append(
                ClassifiedStep(
                    action=step,
                    alignment=jev["alignment"],
                    alignment_confidence=jev["alignment_confidence"],
                    complexity=jev["complexity"],
                    complexity_confidence=jev["complexity_confidence"],
                    destructive_probability=destructive_prob,
                    confidence=merged_confidence,
                    tier=self._tier_for_complexity(jev["complexity"]),
                    jev_evaluated=True,
                )
            )

        return classified

    @staticmethod
    def _tier_for_complexity(complexity: float) -> int:
        """Maps Jev complexity score to the 3-Priority execution tier."""
        if complexity <= 0.5:
            return 1
        if complexity <= 1.5:
            return 2
        return 3

    def gate_plan(self, classified: List[ClassifiedStep]) -> Optional[str]:
        """Evaluates plan-level gates before any execution begins.

        Returns:
            A clarification prompt string when gating fails, else None.
        """
        actionable = [c for c in classified if c.action.type not in _NO_CLASSIFY_TYPES]
        if not actionable:
            return None

        weakest = min(actionable, key=lambda c: c.confidence)
        if not math.isfinite(weakest.confidence) or weakest.confidence < self.confidence_threshold:
            return (
                f"I'm not fully sure about the step '{weakest.action.rationale or weakest.action.type.value}' "
                f"(confidence {weakest.confidence:.2f}). Could you clarify what you'd like me to do?"
            )
        return None

    def apply_to_actions(self, classified: List[ClassifiedStep]) -> None:
        """Writes merged destructive flags and grounding tiers back onto actions."""
        for c in classified:
            c.action.confidence = c.confidence
            if c.destructive_probability >= self.destructive_threshold:
                c.action.is_destructive = True
            if c.action.grounding_tier is None:
                c.action.grounding_tier = c.tier
