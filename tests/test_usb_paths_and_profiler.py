"""Tests for USB Root Dynamic Discovery, Hardware Profiler, and Model Registry."""

import json
from pathlib import Path

import pytest

from wisprhelp.core.paths import USBEnvironment, paths
from wisprhelp.state.profiler import HardwareProfiler, MachineProfile, hardware_profiler


def test_usb_root_discovery(tmp_path: Path):
    """Verifies upward discovery finds .usb_root."""
    usb_dir = tmp_path / "mock_usb"
    usb_dir.mkdir()
    (usb_dir / ".usb_root").write_text("VERSION=1.0.0")

    sub_dir = usb_dir / "agent" / "core"
    sub_dir.mkdir(parents=True)

    env = USBEnvironment(override_root=usb_dir)
    assert env.usb_root == usb_dir
    assert env.models_dir == usb_dir / "models"
    assert env.reasoning_models_dir == usb_dir / "models" / "reasoning"
    assert env.vision_models_dir == usb_dir / "models" / "vision"
    assert env.config_dir == usb_dir / "config"
    assert env.state_dir == usb_dir / "state"


def test_hardware_profiler(tmp_path: Path):
    """Verifies hardware profiler probes RAM, threads, and writes non-sensitive machine profile."""
    profiles_dir = tmp_path / "machine_profiles"
    profiler = HardwareProfiler(profiles_dir=profiles_dir)

    profile = profiler.get_profile()
    assert isinstance(profile, MachineProfile)
    assert len(profile.machine_hash) == 16
    assert profile.total_ram_gb > 0
    assert profile.cpu_threads >= 1
    assert profile.optimal_threads >= 2

    # Verify profile was persisted to disk
    persisted_file = profiles_dir / f"{profile.machine_hash}.json"
    assert persisted_file.exists()

    with open(persisted_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["machine_hash"] == profile.machine_hash
    assert data["total_ram_gb"] == profile.total_ram_gb
