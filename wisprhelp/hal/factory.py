"""Factory for dynamically creating the host platform OS adapter."""

from __future__ import annotations

import sys
from typing import Optional

from wisprhelp.hal.base import BaseOSAdapter

_ADAPTER_INSTANCE: Optional[BaseOSAdapter] = None


def get_os_adapter() -> BaseOSAdapter:
    """Returns the singleton BaseOSAdapter implementation for the current host OS."""
    global _ADAPTER_INSTANCE
    if _ADAPTER_INSTANCE is not None:
        return _ADAPTER_INSTANCE

    if sys.platform == "win32":
        from wisprhelp.hal.windows import WindowsOSAdapter
        _ADAPTER_INSTANCE = WindowsOSAdapter()
    elif sys.platform == "darwin":
        from wisprhelp.hal.macos import MacOSAdapter
        _ADAPTER_INSTANCE = MacOSAdapter()
    else:
        from wisprhelp.hal.linux import LinuxOSAdapter
        _ADAPTER_INSTANCE = LinuxOSAdapter()

    return _ADAPTER_INSTANCE
