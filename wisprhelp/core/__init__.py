"""Core routing, safety interlocks, and runtime coordination."""

from wisprhelp.core.router import CommandRouter, RouteTarget
from wisprhelp.core.safety import SafetyGate, EmergencyKillSwitchTriggered
def __getattr__(name):
    # Runtime imports the semantic agent; keep public runtime exports lazy so
    # low-level outcome/safety modules remain independently importable.
    if name in ("RuntimeCoordinator", "ExecutionResult"):
        from wisprhelp.core.runtime import RuntimeCoordinator, ExecutionResult
        return {"RuntimeCoordinator": RuntimeCoordinator, "ExecutionResult": ExecutionResult}[name]
    raise AttributeError(name)

__all__ = [
    "CommandRouter",
    "RouteTarget",
    "SafetyGate",
    "EmergencyKillSwitchTriggered",
    "RuntimeCoordinator",
    "ExecutionResult",
]
