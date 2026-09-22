"""Hardware and Operating System Abstraction Layer (HAL)."""

from wisprhelp.hal.base import BaseOSAdapter, WindowInfo
from wisprhelp.hal.factory import get_os_adapter

__all__ = [
    "BaseOSAdapter",
    "WindowInfo",
    "get_os_adapter",
]
