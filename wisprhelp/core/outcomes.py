"""Stable execution failures shared by runtime, tools and evaluations."""
from enum import Enum


class FailureCode(str, Enum):
    NONE = "none"
    ACTION_FAILED = "action_failed"
    TARGET_APP_MISSING = "target_app_missing"
    ACCESSIBILITY_TARGET_NOT_FOUND = "accessibility_target_not_found"
    INCORRECT_GROUNDING = "incorrect_grounding"
    ACTION_TIMEOUT = "action_timeout"
    UNEXPECTED_MODAL = "unexpected_modal"
    FOCUS_CHANGED = "focus_changed"
    POSTCONDITION_FAILED = "postcondition_failed"
    API_UNAVAILABLE = "api_unavailable"
    INVALID_CLASSIFICATION = "invalid_classification"
    LOW_CONFIDENCE = "low_confidence"
    CANCELLED = "cancelled"
    EMERGENCY_STOP = "emergency_stop"
    BUSY = "busy"
    STEP_LIMIT = "step_limit"
    UNAUTHORIZED = "unauthorized"


class ExecutionFault(RuntimeError):
    def __init__(self, code: FailureCode, message: str):
        super().__init__(message)
        self.code = code
