"""Dynamic USB Root and Portable Path Engine for WisprHelp.

This module guarantees that WisprHelp never hardcodes host paths (like C:\\ or D:\\).
It dynamically discovers the USB_ROOT mount point or project root at startup by
traversing upward until it finds the '.usb_root' anchor marker.

All directories (models, config, state, logs, cache, runtime) are derived strictly
relative to USB_ROOT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


class USBEnvironment:
    """Manages dynamic resolution of portable paths relative to USB_ROOT."""

    _instance: Optional["USBEnvironment"] = None

    def __init__(self, override_root: Optional[Path] = None) -> None:
        self._root = self._discover_root(override_root)
        self._ensure_standard_directories()

    @classmethod
    def get_instance(cls) -> "USBEnvironment":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _discover_root(override_root: Optional[Path] = None) -> Path:
        """Finds USB_ROOT dynamically without assuming drive letter or host path."""
        # 1. Environment variable override (useful for testing or custom launches)
        env_root = os.getenv("WISPRHELP_USB_ROOT")
        if env_root:
            p = Path(env_root).resolve()
            if p.exists():
                return p

        if override_root is not None:
            return override_root.resolve()

        # 2. Search upward from this file until we find '.usb_root'
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / ".usb_root").exists():
                return parent
            # Fallback marker: if folder has both 'wisprhelp' and ('models' or 'tests')
            if (parent / "wisprhelp").is_dir() and ((parent / "models").exists() or (parent / "tests").exists()):
                return parent

        # 3. Fallback: two levels above this file (wisprhelp/core/paths.py -> repo root)
        return Path(__file__).resolve().parent.parent.parent

    def _ensure_standard_directories(self) -> None:
        """Creates the required USB directory layout if it doesn't already exist."""
        dirs = [
            self.models_dir,
            self.reasoning_models_dir,
            self.vision_models_dir,
            self.config_dir,
            self.state_dir,
            self.machine_profiles_dir,
            self.app_profiles_dir,
            self.ui_memory_dir,
            self.logs_dir,
            self.cache_dir,
            self.runtime_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def usb_root(self) -> Path:
        """The dynamically discovered root directory of the USB drive / project."""
        return self._root

    @property
    def models_dir(self) -> Path:
        """Root directory for AI model weights on USB."""
        return self._root / "models"

    @property
    def reasoning_models_dir(self) -> Path:
        """Legacy location for optional downloaded model assets."""
        return self.models_dir / "reasoning"

    @property
    def vision_models_dir(self) -> Path:
        """Legacy location for optional downloaded vision assets."""
        return self.models_dir / "vision"

    @property
    def config_dir(self) -> Path:
        """Directory for agent and model configuration files."""
        return self._root / "config"

    @property
    def state_dir(self) -> Path:
        """Directory for persistent USB state."""
        return self._root / "state"

    @property
    def machine_profiles_dir(self) -> Path:
        """Directory storing host hardware capability profiles."""
        return self.state_dir / "machine_profiles"

    @property
    def app_profiles_dir(self) -> Path:
        """Directory storing learned application mappings & selectors."""
        return self.state_dir / "application_profiles"

    @property
    def ui_memory_dir(self) -> Path:
        """Directory storing cached accessibility controls and UI bounds."""
        return self.state_dir / "ui_memory"

    @property
    def logs_dir(self) -> Path:
        """Directory for structured execution audit logs."""
        return self._root / "logs"

    @property
    def cache_dir(self) -> Path:
        """Directory for temporary downscaled screenshots and transient files."""
        return self._root / "cache"

    @property
    def runtime_dir(self) -> Path:
        """Directory for optional portable runtimes."""
        return self._root / "runtime"

    def resolve_rel(self, relative_path: str | Path) -> Path:
        """Resolves any relative path against USB_ROOT."""
        return (self._root / relative_path).resolve()


# Singleton accessors
paths = USBEnvironment.get_instance()
get_usb_root = lambda: paths.usb_root
