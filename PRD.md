# Product Requirements Document (PRD): WisprHelp Universal OS AI Agent

**Document Version:** 1.1.0  
**Status:** Approved / In Execution  
**Target Platforms:** Windows 10/11, macOS (Ventura+), Linux (Ubuntu 22.04+, X11 & Wayland)  
**Primary Engine:** TypeSafe AI Jev (System One) + Frontier Multimodal Vision (System Two) + Silero VAD & Whisper Large v3 (Ears)  

---

## 1. Executive Summary & Vision

**WisprHelp** is an autonomous, hands-free, zero-touch Universal Computer-Using Agent (CUA). It enables users to operate their computer completely through natural multilingual speech (English, Hindi, Hinglish, etc.) or text commands.

Like a **human operator sitting in front of the workstation**:
- It listens with Voice Activity Detection (Silero VAD) and transcribes with Whisper Large v3.
- It observes the screen and accessibility tree (Eyes).
- It reasons with a Two-Tier Cognitive Engine (Brain).
- It moves the mouse, types Unicode text, and triggers system APIs (Hands).
- It interacts with **any application** (Chrome, Excel, VS Code, Slack, WhatsApp, File Explorer, System Settings, or custom software).

To eliminate latency and heavy inference costs, WisprHelp combines:
1. **TypeSafe AI Jev (System One Reflex)**: Sub-150ms calibrated classification of intent, complexity tier, target application domain, multilingual script presence, and safety risk. Over 65% of OS tasks execute in `< 50ms` via deterministic native calls without invoking vision models.
2. **3-Priority Tool Execution Hierarchy**:
   - **Priority 1: Direct Protocol / Shell / AppID**: Instant zero-vision launching (`whatsapp:`, `open -a`, `xdg-open`, `Get-StartApps`).
   - **Priority 2: OS Native Accessibility Tree**: Fast label/control matching via Windows UIA (`uiautomation` / `pywinauto`), macOS Accessibility (AX), and Linux AT-SPI without image inference.
   - **Priority 3: Full Multimodal Vision Grounding**: Screenshot capture (`mss` + `PIL` downscaled on demand) + OmniParser / Claude 3.5 Sonnet Computer Use / Gemini 2.5 Flash when visual spatial reasoning is necessary.
3. **Floating UI / HUD**: An unobtrusive, click-through PySide6 pill showing real-time agent status.

---

## 2. Target Personas & Core Use Cases

### Persona A: The Hands-Free Knowledge Worker / Power User
- **Need**: Operates multiple windows simultaneously (Excel, Chrome, Slack, VS Code). Wants to execute repetitive or context-switching workflows verbally without leaving their primary focus.
- **Example Scenario**: *"Open Chrome, find the Q3 revenue spreadsheet in my Drive, download the PDF version, and move it to my Reports folder on Desktop."*

### Persona B: The Multilingual Professional (English / Hindi / Hinglish)
- **Need**: Communicates in regional or mixed languages (Hinglish/Hindi) where OS keyboard input requires tedious IME switching.
- **Example Scenario**: *"Open WhatsApp, search for Rahul, and write 'कल दोपहर 2 बजे क्लाइंट मीटिंग है, प्रेजेंटेशन रेडी रखना' and confirm before sending."*

### Persona C: Accessibility & Hands-Free Computing
- **Need**: Users with motor impairments or situational limitations (hands occupied) who need complete, dependable control of their OS.
- **Example Scenario**: *"Open Spotify, play my Liked Songs, increase volume to 60%, and then minimize the window."*

---

## 3. High-Level Functional Requirements

### 3.1 Hearing & Speech Perception (Ear)
- **FR-01 (Voice Activity Detection - Silero VAD)**: Local sub-5ms Silero VAD running on audio stream to detect speech onset and end-of-turn, eliminating silence transmission and hallucinations.
- **FR-02 (Multilingual Speech-to-Text)**: Whisper Large v3 (via Groq API for < 300ms latency or local `faster-whisper`), transcribing English, Hindi, Hinglish, and 90+ languages.
- **FR-03 (Global Push-to-Talk / Hotkey Trigger)**: Configurable system-wide keyboard shortcut (`Ctrl + Space` or `Alt + Space`).
- **FR-04 (Streaming Audio Feedback)**: Audio cues (listening chime, processing cue, completion chime, and confirmation questions).

### 3.2 System One Classification & Gatekeeping (TypeSafe AI Jev)
- **FR-05 (Parallel Calibrated Evaluation)**: Every input command evaluated in a single sub-200ms request to TypeSafe Jev (`jev-latest`), answering 5 concurrent questions:
  1. `intent`: `direct_app_launch`, `text_input_injection`, `system_media_control`, `gui_workflow_automation`, `conversational_query`.
  2. `target_app`: `whatsapp`, `browser`, `app_store`, `file_manager`, `terminal_code`, `system_settings`, `media_player`, `generic_or_active`.
  3. `complexity`: Level 0 (deterministic), Level 1 (single-focus interaction), Level 2 (multi-step visual ReAct loop).
  4. `is_destructive`: Calibrated probability [0.0..1.0] of data loss, irreversible actions, payment, or system modification.
  5. `has_multilingual_content`: Detects non-Latin scripts (Hindi Devanagari, emojis) requiring unicode buffer injection.
- **FR-06 (Confidence-Gated Routing)**:
  - If `intent.confidence < 0.60`, halt OS interaction and verbally request clarification.
  - If `is_destructive > 0.65`, engage Safety Confirmation Interlock.
  - If `complexity == 0`, dispatch directly to OS HAL Priority 1 (Shell/AppID) in < 50ms.
  - If `complexity == 1` & `text_input_injection`, inject text into active focus in < 80ms.
  - If `complexity >= 2`, dispatch to System Two Multimodal Vision Loop.

### 3.3 3-Priority Tool Execution Hierarchy
- **FR-07 (Priority 1: Direct OS Launcher & CLI)**:
  - Windows: `Get-StartApps` + AppID, protocol URIs (`whatsapp:`, `ms-windows-store:`), `os.startfile`, and CLI binaries (`code`, `chrome`).
  - macOS: `open -a "<App>"`, custom URL schemes.
  - Linux: `xdg-open`, `.desktop` entries, CLI binaries.
- **FR-08 (Priority 2: Native Accessibility Tree Traversal)**:
  - Windows: Microsoft UI Automation via `uiautomation` and `pywinauto`. Fast element discovery for standard controls (search inputs, buttons, menu items).
  - macOS: macOS Accessibility API (AX) via PyObjC.
  - Linux: AT-SPI via `pyatspi2`.
- **FR-09 (Priority 3: Full Multimodal Vision & Computer Use)**:
  - Screen Capture: Fast screen grab via `mss` + `PIL`, downscaled on demand.
  - Visual Grounding: OmniParser / Claude 3.5 Sonnet Computer Use / Gemini 2.5 Flash for non-standard controls, web canvas, and complex layouts.
  - Verification: Screen capture after each step verifying visual state delta.

### 3.4 Hardware & OS Abstraction Layer (HAL)
- **FR-10 (Cross-Platform Parity)**: Identical core agent logic running across Windows, macOS, and Linux via `BaseOSAdapter`.
- **FR-11 (Universal Unicode Text Injection)**: Pastes multilingual characters (Hindi Devanagari, Arabic, emojis) into any focused field without character corruption or IME interference.
- **FR-12 (Human-like Physical OS Controls)**: Mouse click, double click, right click, smooth bezier drag, scroll, and keyboard hotkeys.
- **FR-13 (Window Management)**: `pywin32` (Windows), `NSWorkspace` / Quartz (macOS), `xdotool` / `wmctrl` (Linux) for active window tracking and focus control.

### 3.5 Floating Status HUD
- **FR-14 (PySide6 Floating Pill)**: Always-on-top, click-through pill overlay at the top of the display indicating agent state: `Listening`, `Thinking`, `Executing: [Action]`, `Waiting for Confirmation`.

### 3.6 Safety, Permissions & Fail-safes
- **FR-15 (Emergency Kill-Switch)**: Flinging the mouse to any screen corner or pressing `Esc` immediately terminates all active automation threads within $\le 10\text{ms}$.
- **FR-16 (Destructive Action Interlock)**: When Jev's `is_destructive > 0.65`, execution halts and requests verbal user confirmation.
- **FR-17 (Protected Credentials)**: Blocks automated interaction with password managers, OS credential dialogs (UAC/sudo), or private key directories.

---

## 4. Success Metrics & Acceptance Criteria

1. **Classification Accuracy**: $\ge 96\%$ correct intent and complexity routing on standard OS command benchmark.
2. **Speed Improvement**: $\ge 75\%$ reduction in average command execution time compared to a pure vision-agent baseline.
3. **Cost Reduction**: $\ge 85\%$ drop in token consumption by routing tier-0/tier-1 commands to Priority 1 (Shell) and Priority 2 (UIA).
4. **Multilingual Reliability**: 100% character fidelity when typing Hindi Devanagari and Hinglish across all OS applications.
5. **Safety Compliance**: Zero unconfirmed executions of actions classified as `is_destructive > 0.65`.
