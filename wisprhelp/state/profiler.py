"""Host Machine Hardware Profiling & Capability Engine.

Detects host PC specifications (RAM, CPU cores/threads, GPU, screen resolution)
without requiring permanent host installation. Generates a non-sensitive machine
profile hash stored in USB_ROOT/state/machine_profiles/<machine_hash>.json to adapt
model loading and inference parameters dynamically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil

from wisprhelp.core.paths import paths

logger = logging.getLogger(__name__)


@dataclass
class MachineProfile:
    """Non-sensitive capability snapshot of the current host computer."""

    machine_hash: str
    os_name: str
    os_release: str
    os_version: str
    cpu_arch: str
    cpu_cores: int
    cpu_threads: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_vram_gb: float
    screen_width: int
    screen_height: int
    optimal_threads: int
    supports_gpu_offload: bool
    created_at: float = field(default_factory=time.time)

    def can_load_model(self, model_size_gb: float, safety_margin_gb: float = 1.0) -> bool:
        """Determines if the host has sufficient free RAM to load a model safely."""
        return self.available_ram_gb >= (model_size_gb + safety_margin_gb)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MachineProfile":
        return cls(**data)


class HardwareProfiler:
    """Detects host machine capabilities and persists machine profiles on USB."""

    def __init__(self, profiles_dir: Optional[Path] = None) -> None:
        self.profiles_dir = profiles_dir or paths.machine_profiles_dir
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._cached_profile: Optional[MachineProfile] = None

    def get_profile(self, force_refresh: bool = False) -> MachineProfile:
        """Returns the current machine profile, profiling and caching to USB if needed."""
        if self._cached_profile and not force_refresh:
            return self._cached_profile

        profile = self._probe_hardware()
        self._cached_profile = profile
        self.save_profile(profile)
        return profile

    def _probe_hardware(self) -> MachineProfile:
        """Inspects host hardware parameters non-invasively."""
        os_name = platform.system()
        os_release = platform.release()
        os_version = platform.version()
        cpu_arch = platform.machine()

        # CPU Cores and Threads
        logical_threads = psutil.cpu_count(logical=True) or 4
        physical_cores = psutil.cpu_count(logical=False) or max(1, logical_threads // 2)

        # Allocate threads matching physical CPU cores to prevent SMT/hyperthreading cache thrashing
        optimal_threads = max(2, min(physical_cores, 8))

        # RAM in Gigabytes
        vm = psutil.virtual_memory()
        total_ram_gb = round(vm.total / (1024**3), 2)
        available_ram_gb = round(vm.available / (1024**3), 2)

        # GPU Detection
        gpu_available, gpu_name, gpu_vram_gb, supports_offload = self._probe_gpu()

        # Display Resolution
        screen_w, screen_h = self._probe_screen_resolution()

        # Generate non-sensitive anonymous machine hash
        raw_id = f"{os_name}-{cpu_arch}-{physical_cores}-{logical_threads}-{int(total_ram_gb)}-{gpu_name or 'none'}"
        machine_hash = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

        return MachineProfile(
            machine_hash=machine_hash,
            os_name=os_name,
            os_release=os_release,
            os_version=os_version,
            cpu_arch=cpu_arch,
            cpu_cores=physical_cores,
            cpu_threads=logical_threads,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            screen_width=screen_w,
            screen_height=screen_h,
            optimal_threads=optimal_threads,
            supports_gpu_offload=supports_offload,
        )

    def _probe_gpu(self) -> Tuple[bool, Optional[str], float, bool]:
        """Probes for GPU acceleration capabilities across platforms."""
        gpu_name: Optional[str] = None
        gpu_vram_gb: float = 0.0
        gpu_available = False
        supports_offload = False

        if sys.platform == "win32":
            try:
                cmd = 'powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object -Property Name, AdapterRAM | ConvertTo-Json"'
                out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL, timeout=4).strip()
                if out:
                    data = json.loads(out)
                    items = [data] if isinstance(data, dict) else data
                    if items:
                        first = items[0]
                        gpu_name = first.get("Name")
                        adapter_ram = first.get("AdapterRAM") or 0
                        if adapter_ram:
                            gpu_vram_gb = round(adapter_ram / (1024**3), 2)
                        gpu_available = True
                        # If NVIDIA GPU or dedicated AMD GPU with >= 4GB VRAM
                        if gpu_name and ("nvidia" in gpu_name.lower() or "geforce" in gpu_name.lower() or "rtx" in gpu_name.lower()):
                            supports_offload = True
            except Exception as e:
                logger.debug(f"Windows GPU probe via PowerShell failed: {e}")

        elif sys.platform == "darwin":
            gpu_available = True
            gpu_name = "Apple Silicon GPU"
            supports_offload = True  # Unified memory via Metal

        return gpu_available, gpu_name, gpu_vram_gb, supports_offload

    def _probe_screen_resolution(self) -> Tuple[int, int]:
        """Detects the primary display resolution."""
        if sys.platform == "win32":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                user32.SetProcessDPIAware()
                return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
            except Exception:
                pass
        return 1920, 1080

    def save_profile(self, profile: MachineProfile) -> Path:
        """Saves machine profile JSON to USB state directory."""
        path = self.profiles_dir / f"{profile.machine_hash}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, indent=2)
            logger.info(f"Persisted host machine profile: {path.name}")
        except Exception as e:
            logger.warning(f"Could not persist machine profile to USB: {e}")
        return path


# Default singleton instance
hardware_profiler = HardwareProfiler()
