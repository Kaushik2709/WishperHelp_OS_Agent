# System Architecture & Technical Design: WisprHelp Universal OS Agent

**System:** WisprHelp Autonomous Universal Computer-Using Agent (CUA)  
**Document:** ARCHITECTURE.md  
**Version:** 1.1.0  

---

## 1. Architectural Overview

WisprHelp couples **Speech Perception (Silero VAD + Whisper Large v3)**, a **Two-Tier Cognitive Engine (TypeSafe Jev + Frontier VLM)**, a **3-Priority Tool Execution Hierarchy**, and a **Cross-Platform OS Abstraction Layer (HAL)**.

```mermaid
flowchart TB
    subgraph Layer1 ["Layer 1: Input & Sensory Perception"]
        Mic[Microphone Input Stream] --> VAD[Silero VAD Engine<br/>Sub-5ms Voice Activity Detection]
        Hotkey[Global Hotkey: Alt+Space] --> VAD
        VAD -->|Clean Audio Chunks| STT[Whisper Large v3 Engine<br/>Groq API or local faster-whisper]
        STT --> GoalText[Spoken Goal / Command]
        
        WinSensor[Active Window Sensor<br/>pywin32 / NSWorkspace / X11]
    end

    subgraph Layer2 ["Layer 2: System One - Fast Cognitive Classifier (TypeSafe Jev)"]
        GoalText & WinSensor --> StatePackager[State Context Enricher]
        StatePackager --> JevRequest[TypeSafe Jev API Client<br/>Endpoint: /v1/systemone | Model: jev-latest]
        
        subgraph JevMatrix ["Parallel Multi-Question Matrix (~100-150ms)"]
            Q1[Choice: intent<br/>direct_launch | quick_type | media_control | gui_workflow | query]
            Q2[Choice: target_app<br/>whatsapp | browser | store | terminal | settings | generic]
            Q3[Score: complexity<br/>0: Fast Path | 1: Focus Action | 2: Multi-step Vision]
            Q4[Noul: is_destructive<br/>Calibrated Risk Probability [0..1]]
            Q5[Noul: has_multilingual_content<br/>Hindi, Hinglish, non-Latin Script]
        end
        
        JevRequest --> JevMatrix
    end

    subgraph Layer3 ["Layer 3: Router & Guardrails"]
        JevMatrix --> Router{Confidence-Gated Router}
        Router -->|Confidence < 0.60| Clarification[Voice Clarification Handler]
        Router -->|is_destructive > 0.65| SafetyGate[Voice Confirmation Interlock]
        
        Router -->|Complexity 0: Deterministic| P1_Dispatch[Priority 1 Dispatcher]
        Router -->|Complexity 1: Active Window Typing| P1_Typing[Unicode Text Dispatcher]
        Router -->|Complexity 2: GUI Workflow| P2_P3_Pipeline[Tier-2 Execution Pipeline]
        Router -->|Intent: Conversational| ConversationalHandler[Voice Conversational LLM]
    end

    subgraph Layer4 ["Layer 4: 3-Priority Execution Hierarchy"]
        P1_Dispatch --> P1[Priority 1: Shell / Protocol / AppID<br/>whatsapp:, Get-StartApps, os.startfile, CLI]
        P1_Typing --> P1_Clip[Unicode Buffer Swap + Native Paste<br/>Instant Hindi, English, Emojis]
        
        P2_P3_Pipeline --> P2{Priority 2: Native Accessibility Tree<br/>Windows UIA / macOS AX / Linux AT-SPI}
        P2 -->|Element Found by Label/Pattern| ActionExecUIA[Execute Native Control Action<br/>Click/Invoke without Vision]
        P2 -->|Complex/Canvas/Missing UIA| P3[Priority 3: Full Multimodal Vision<br/>mss + PIL Downscaled + OmniParser / Claude 3.5 / Gemini 2.5]
        
        P3 --> ActionPlan[Action Generator: click, double_click, scroll, drag]
        ActionPlan --> OS_HAL
        ActionExecUIA --> OS_HAL
        P1 --> OS_HAL
        P1_Clip --> OS_HAL
        
        OS_HAL --> Verification{Screen Verification Loop}
        Verification -->|Pending / Retry| P2_P3_Pipeline
        Verification -->|Goal Satisfied| AudioDing[Completion Chime / Spoken Status]
    end

    subgraph Layer5 ["Layer 5: Hardware & OS Abstraction Layer (HAL)"]
        OS_HAL --> PlatformBridge{Host OS Auto-Detection}
        PlatformBridge -->|Windows| WinHAL[WindowsOSAdapter<br/>Win32, UIA, Get-StartApps, SendInput]
        PlatformBridge -->|macOS| MacHAL[MacOSAdapter<br/>AppleScript, Cocoa NSWorkspace, open]
        PlatformBridge -->|Linux| LinuxHAL[LinuxOSAdapter<br/>X11/Wayland, xdg-open, AT-SPI, xdotool]
    end

    subgraph UI_Layer ["Visual Feedback & Safety Supervision"]
        HUD[PySide6 Floating Pill HUD<br/>Always-on-top, click-through status pill]
        MouseFling[Screen Corner Mouse Fling] --> KillSwitch[Emergency Kill-Switch]
        EscKey[Escape Key Press] --> KillSwitch
        KillSwitch -.->|Instant SIGINT / Hard Abort| ActionPlan
        KillSwitch -.->|Release Controls| OS_HAL
    end
```

---

## 2. Audio Perception Pipeline: Silero VAD + Whisper Large v3

1. **Microphone Ring Buffer**:
   - Audio input stream captured at 16kHz mono.
2. **Silero VAD (Voice Activity Detection)**:
   - Evaluates audio chunks locally in Python in $< 5\text{ms}$.
   - Discards ambient room noise and background music.
   - Detects voice start and user speech termination (trailing silence threshold: 500ms).
3. **Whisper Large v3 (Ears)**:
   - High-accuracy multilingual transcription (Groq API for ~250ms roundtrip or local `faster-whisper`).
   - Handles Hindi, Hinglish code-switching, English, and 90+ languages.

---

## 3. System One: TypeSafe AI Jev Classification Model

### 3.1 Role & Benefits
- Sub-150ms parallel evaluation of intent, target app, complexity tier, destructive risk, and multilingual script presence.
- Uses **calibrated probabilities** and a **confidence score** on every dimension.
- Enables deterministic branching before touching the screen or invoking expensive vision models.

### 3.2 5-Question Specification Payload
```python
state = {
    "user_query": user_command,
    "os_platform": platform.system().lower(),  # "windows", "darwin", "linux"
    "active_window_title": active_window.title,
    "active_process_name": active_window.process_name,
}

questions = {
    "intent": Choice(
        instructions="What is the primary action requested by the user?",
        criteria={
            "direct_app_launch": "Opening, launching, or switching to an application or system tool.",
            "text_input_injection": "Typing, dictating, or entering text into the currently active or target field.",
            "system_media_control": "Controlling system settings like volume, brightness, media playback, window minimize/maximize, or locking.",
            "gui_workflow_automation": "Performing a multi-step task inside an application requiring navigation, searching, and clicking.",
            "conversational_query": "Asking a general question, calculation, or conversational request without expecting OS automation.",
        },
    ),
    "target_app": Choice(
        instructions="Which application or service is the target of this command?",
        criteria={
            "whatsapp": "WhatsApp desktop, web, or messaging app.",
            "browser": "Chrome, Edge, Safari, Firefox, or web browsing.",
            "app_store": "Microsoft Store, Mac App Store, package manager, or software center.",
            "file_manager": "Windows Explorer, macOS Finder, or Linux file manager.",
            "terminal_code": "Terminal, Command Prompt, PowerShell, VS Code, or IDE.",
            "system_settings": "OS Settings, Control Panel, Network/Bluetooth panels.",
            "media_player": "Spotify, YouTube, media player, music playback.",
            "generic_or_active": "Currently active window, desktop, or generic interface.",
        },
    ),
    "complexity": Score(
        instructions="Rate the operational complexity required to execute this command",
        criteria=[
            "Level 0: Deterministic command executable via single system call, URI protocol, or media hotkey.",
            "Level 1: Single targeted interaction in the active window (e.g. paste text, press enter).",
            "Level 2: Multi-step workflow requiring visual perception, element grounding, and iterative screen verification.",
        ],
    ),
    "is_destructive": Noul(
        instructions="Does this action delete files, make purchases, send unverified messages, terminate processes, or modify critical system settings?",
    ),
    "has_multilingual_content": Noul(
        instructions="Does the input contain text meant to be typed in Hindi, Hinglish, Spanish, Arabic, or another non-Latin/complex script?",
    ),
}
```

---

## 4. The 3-Priority Tool Execution Hierarchy

To achieve maximum speed and minimal cost, WisprHelp attempts action execution through three escalating tiers:

### Priority 1: Direct OS Launcher, Protocol URIs & CLI (< 50ms)
- **Windows**:
  - `Get-StartApps` PowerShell query matching AppID for Store and Desktop apps.
  - Windows URI Schemes: `whatsapp:`, `ms-windows-store:`, `chrome:`, `code:`, `spotify:`.
  - `os.startfile` and `win32api.ShellExecute` for registered executables.
- **macOS**:
  - `open -a "<AppName>"` and URL schemes (`whatsapp://`, `spotify://`).
- **Linux**:
  - `xdg-open`, `.desktop` file execution via `gio launch`, or binary path.

### Priority 2: Native Accessibility Tree Traversal (< 150ms)
- Skips visual screenshot inference when controls are cleanly exposed via OS accessibility:
  - **Windows**: Microsoft UI Automation via `uiautomation` and `pywinauto`. Finds buttons, text fields, menu bars, and list items by name/control type.
  - **macOS**: Accessibility API (AX) via Cocoa / PyObjC.
  - **Linux**: AT-SPI via `pyatspi2`.
- Directly triggers standard actions: `Invoke()`, `SetValue()`, `SetFocus()`, or computes control center coordinates to click.

### Priority 3: Full Multimodal Vision & Computer Use (1.5s - 2.5s)
- Triggered when:
  - An app is non-standard, custom-rendered, or canvas-based (e.g. Discord, Spotify, Photoshop, games).
  - Accessibility tree traversal yields no matching target.
  - Spatial visual reasoning is required (e.g. "click the blue button next to the green banner").
- **Pipeline**:
  1. Capture screen using `mss` (sub-40ms).
  2. Downscale frame with `PIL` (to 1080p or 1440p normalized bounds) to preserve vision tokens.
  3. Ground UI elements with OmniParser or query Claude 3.5 Sonnet Computer Use / Gemini 2.5 Flash for normalized coordinates `[0..1000]`.
  4. Dispatch mouse click / scroll / drag via OS HAL.
  5. Capture post-action frame to verify UI delta.

---

## 5. Cross-Platform Hardware & OS Abstraction Layer (HAL)

### 5.1 Abstract Base Class (`BaseOSAdapter`)
```python
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class WindowInfo:
    title: str
    process_name: str
    pid: int
    rect: Tuple[int, int, int, int]  # (x, y, width, height)
    is_minimized: bool

class BaseOSAdapter(ABC):
    @abstractmethod
    def launch_app(self, app_identifier: str, parameter: Optional[str] = None) -> bool:
        """Launches application by bundle ID, protocol URI, or executable path."""
        pass

    @abstractmethod
    def get_active_window(self) -> WindowInfo:
        """Queries currently focused window title, process ID, and boundary coordinates."""
        pass

    @abstractmethod
    def inject_unicode_text(self, text: str) -> bool:
        """Directly injects multilingual text into active input field without IME corruption."""
        pass

    @abstractmethod
    def media_control(self, command: str) -> bool:
        """Executes system-level audio/media adjustments (vol_up, vol_down, mute, play_pause)."""
        pass

    @abstractmethod
    def find_accessibility_element(self, query: str) -> Optional[Tuple[int, int]]:
        """Finds control center coordinate (x, y) via OS Accessibility Tree (UIA/AX/AT-SPI)."""
        pass

    @abstractmethod
    def move_mouse(self, x: int, y: int, smooth: bool = True) -> None:
        """Moves cursor to pixel coordinate (x, y)."""
        pass

    @abstractmethod
    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left") -> None:
        """Clicks at coordinates or at current position."""
        pass

    @abstractmethod
    def key_combination(self, keys: List[str]) -> None:
        """Executes hotkeys (e.g. ['ctrl', 'f'], ['alt', 'tab'], ['command', 'space'])."""
        pass
```

### 5.2 Universal Unicode Text Injection
Standard `pyautogui.typewrite()` fails on Hindi Devanagari, emojis, and non-ASCII scripts. WisprHelp uses the **Universal Clipboard Swap Engine**:
```python
import pyperclip, pyautogui, time

def inject_unicode_text(text: str) -> bool:
    old_clipboard = pyperclip.paste()
    try:
        pyperclip.copy(text)
        time.sleep(0.03)  # OS clipboard sync
        pyautogui.hotkey("ctrl" if platform.system() != "Darwin" else "command", "v")
        time.sleep(0.05)
        return True
    finally:
        pyperclip.copy(old_clipboard)
```

---

## 6. Floating UI / Status HUD (PySide6)

WisprHelp includes a click-through, always-on-top **PySide6 Pill HUD**:
- Floats centered at the top of the primary monitor.
- Frameless, translucent dark-glass aesthetic (`Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool`).
- Displays live agent states:
  - 🔵 **Listening**: Silero VAD active.
  - 🟡 **Thinking (Jev / VLM)**: Showing classified sub-goal.
  - 🟢 **Executing**: Action description (e.g., *"Clicking Rahul's chat"*).
  - 🔴 **Confirmation Required**: Destructive action warning.
- Passes all mouse clicks through to underlying OS applications (`Qt.WA_TransparentForMouseEvents`) so it never blocks desktop interaction.

---

## 7. Safety, Guardrails & Emergency Failsafes

1. **Emergency Kill-Switch (PyAutoGUI FailSafe + Esc Listener)**:
   - Flinging the mouse to any screen corner `(0, 0)` triggers an instant PyAutoGUI FailSafe exception.
   - Global keyboard hook intercepts `Esc` or `Pause/Break` to immediately abort running agent loops within 10ms.
2. **Destructive Action Interlock**:
   - Actions where Jev's `is_destructive.noul > 0.65` (e.g. deleting files, terminating processes, clearing data) halt execution immediately.
   - The agent requests verbal confirmation: *"High-risk action detected: Are you sure you want to proceed?"*
3. **Bounded Agent Execution**:
   - Multi-step VLA loops are strictly capped at a maximum of 15 steps to prevent runaway loops.
