# Architecture & Implementation Plan: WisprHelp Autonomous Universal OS Agent

**Project:** WisprHelp — Autonomous Universal Computer-Using Agent (CUA)  
**Classification Engine:** TypeSafe AI Jev (System One)  
**Vision Brain:** Frontier Multimodal VLM (Claude 3.5 Sonnet Computer Use / Gemini 2.5 Flash)  
**Speech Engine:** Silero VAD + Whisper Large v3 (Multilingual Ears)  
**Execution Hierarchy:** 1. Direct Shell/AppID $\rightarrow$ 2. OS Accessibility Tree (UIA/AX) $\rightarrow$ 3. Multimodal Vision  
**Supported Platforms:** Windows 10/11, macOS, Linux (X11 & Wayland)  

---

## 1. Project Overview & Philosophy

WisprHelp is an autonomous, zero-touch computer-using AI assistant. Like a human user with eyes on the monitor and hands on the mouse and keyboard, the agent interacts with **ANY application** and executes **ANY desktop workflow** across Windows, macOS, and Linux:
- Browsing websites, filing forms, and downloading documents.
- Organizing files, extracting archives, and managing folders.
- Navigating messaging apps (WhatsApp, Slack, Discord) and typing in any language (English, Hindi, Hinglish).
- Interacting with desktop software (Excel, Word, VS Code, Terminal, Photoshop, Media Players, System Settings).

### Two-Tier Cognitive Engine with 3-Priority Tool Execution
1. **Tier 1 (System One: TypeSafe Jev)**: Rapid classification (~100-150ms) of intent, complexity tier, target application, multilingual script, and destructive risk.
2. **3-Priority Execution Hierarchy**:
   - **Priority 1: Shell / Protocol / AppID**: Instant zero-vision launching (`whatsapp:`, `Get-StartApps`, `open -a`, `xdg-open`).
   - **Priority 2: OS Native Accessibility Tree**: Fast label/control matching via Windows UIA (`uiautomation` / `pywinauto`), macOS Accessibility (AX), and Linux AT-SPI without image inference.
   - **Priority 3: Full Multimodal Vision Grounding**: Screenshot capture (`mss` + `PIL` downscaled on demand) + OmniParser / Claude 3.5 Sonnet Computer Use / Gemini 2.5 Flash when visual spatial reasoning is necessary.
3. **UI Overlay**: PySide6 click-through floating pill HUD for real-time status.

---

## 2. Directory & Component Architecture

```
d:/WishperHelp/
├── .env                              # API keys (JEV_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY)
├── pyproject.toml                    # Package metadata & dependencies
├── README.md                         # Project documentation index & quickstart
├── PRD.md                            # Product Requirements Document
├── ARCHITECTURE.md                   # System architecture & technical design
├── AGENTS.md                         # Agent harness rules & technical contract
├── HARNESS.md                        # Runtime execution harness specification
├── IMPLEMENTATION_PLAN.md            # Phased execution roadmap
│
├── wisprhelp/
│   ├── __init__.py
│   ├── config.py                     # Environment config & threshold loader
│   │
│   ├── classification/               # === TYPE SAFE JEV (SYSTEM ONE) ===
│   │   ├── __init__.py
│   │   ├── client.py                 # Jev client wrapper (supports typesafe-sdk & HTTPS fallback)
│   │   ├── schemas.py                # Typed Question schemas (Choice, Score, Noul) & Decision models
│   │   └── classifier.py             # JevClassifier: context-aware state packager & decision evaluator
│   │
│   ├── hal/                          # === HARDWARE & OS ABSTRACTION LAYER (HAL) ===
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseOSAdapter abstract class & WindowInfo models
│   │   ├── windows.py                # Windows adapter (Win32, UIA, Get-StartApps, SendInput)
│   │   ├── macos.py                  # macOS adapter (AppleScript, Cocoa, open)
│   │   ├── linux.py                  # Linux adapter (X11/Wayland, xdg-open, AT-SPI)
│   │   └── factory.py                # Host OS auto-detection & adapter factory
│   │
│   ├── core/                         # === ROUTING & SAFETY ENGINE ===
│   │   ├── __init__.py
│   │   ├── router.py                 # Confidence-gated routing (Priority 1 vs 2 vs 3)
│   │   ├── safety.py                 # Destructive action interlock & confirmation gate
│   │   └── runtime.py                # Central orchestrator coordinating Ear, Brain & Hands
│   │
│   ├── voice/                        # === MULTILINGUAL HEARING & SPEECH ===
│   │   ├── __init__.py
│   │   ├── vad.py                    # Silero VAD local voice activity filter
│   │   ├── stt.py                    # Whisper Large v3 (Groq API or local faster-whisper)
│   │   ├── hotkey.py                 # Global push-to-talk listener (Alt+Space)
│   │   └── feedback.py               # Sound cues & speech synthesis (TTS)
│   │
│   ├── vision/                       # === SYSTEM TWO: MULTIMODAL COMPUTER USE ===
│   │   ├── __init__.py
│   │   ├── capture.py                # Fast cross-platform screen grabber (mss + PIL downscaling)
│   │   ├── grounding.py              # Visual coordinate mapper & UI element detector
│   │   └── vlm_agent.py              # Claude 3.5 Computer Use / Gemini 2.5 Flash agent loop
│   │
│   ├── ui/                           # === FLOATING HUD ===
│   │   ├── __init__.py
│   │   └── overlay.py                # PySide6 transparent, click-through pill HUD
│   │
│   └── main.py                       # CLI & Interactive Assistant runner
│
└── tests/
    ├── test_jev_classification.py    # Test Jev classifications across 20+ realistic user commands
    ├── test_hal_adapters.py          # Verify OS HAL commands, active window detection, app launch
    ├── test_unicode_typing.py        # Verify Hindi/Hinglish/English typing integrity
    └── test_vad_audio.py             # Verify Silero VAD chunking and audio capture
```

---

## 3. Phased Implementation Roadmap

### Phase 1: TypeSafe Jev Classification Engine (Current Step)
- Build `wisprhelp/config.py` loading `JEV_API_KEY` (and `TYPESAFE_API_KEY`).
- Implement `wisprhelp/classification/client.py` and `schemas.py` using TypeSafe's Jev model (`jev-latest`).
- Implement `wisprhelp/classification/classifier.py` with the 5-question parallel schema: `intent`, `target_app`, `complexity`, `is_destructive`, `has_multilingual_content`.
- Create comprehensive test suite `tests/test_jev_classification.py` covering:
  - Complex GUI tasks $\rightarrow$ `gui_workflow_automation`, `complexity: 2`.
  - App launches $\rightarrow$ `direct_app_launch`, `complexity: 0`.
  - Media/volume controls $\rightarrow$ `system_media_control`, `complexity: 0`.
  - Multilingual dictation $\rightarrow$ `text_input_injection`, `has_multilingual_content: True`.
  - Risky actions $\rightarrow$ `is_destructive: True` (triggers voice confirmation).

### Phase 2: Cross-Platform OS Abstraction Layer (HAL) & 3-Priority Execution
- Implement `BaseOSAdapter` interface in `wisprhelp/hal/base.py`.
- Implement `WindowsOSAdapter` in `wisprhelp/hal/windows.py`:
  - Priority 1: `Get-StartApps` + AppID, URI schemes (`whatsapp:`, `ms-windows-store:`), `os.startfile`.
  - Priority 2: Windows UIA via `uiautomation` / `pywinauto` for fast accessibility tree discovery.
  - Unicode clipboard buffer swap for Hindi Devanagari, Hinglish, emojis.
- Implement `MacOSAdapter` in `wisprhelp/hal/macos.py` (AppleScript, `open -a`, AX API, NSPasteboard).
- Implement `LinuxOSAdapter` in `wisprhelp/hal/linux.py` (`xdg-open`, AT-SPI, `xclip`, `xdotool`).
- Implement `get_os_adapter()` in `wisprhelp/hal/factory.py` for host auto-detection.

### Phase 3: Router, Safety Interlocks & Deterministic Fast-Path
- Implement `wisprhelp/core/router.py`:
  - Evaluate Jev confidence and complexity scores.
  - Dispatch `complexity: 0` directly to Priority 1 OS HAL (instant app launch, volume/media control).
  - Dispatch `complexity: 1` text injections directly to Unicode typing engine.
  - Cascade `complexity: 2` through Priority 2 (Accessibility UIA) before falling back to Priority 3 (Vision).
- Implement `wisprhelp/core/safety.py`:
  - Intercept destructive actions with voice confirmation prompt.

### Phase 4: Voice Pipeline (Silero VAD + Whisper Large v3)
- Implement `wisprhelp/voice/vad.py` using Silero VAD for sub-5ms local speech boundary detection.
- Implement `wisprhelp/voice/stt.py` using Whisper Large v3 (via Groq API or local fallback).
- Implement global push-to-talk hotkey hook in `wisprhelp/voice/hotkey.py`.
- Test end-to-end voice capture $\rightarrow$ Silero VAD $\rightarrow$ Whisper STT $\rightarrow$ Jev classification.

### Phase 5: System Two Multimodal Vision Loop & PySide6 Floating HUD
- Implement `wisprhelp/vision/capture.py` for cross-platform screenshot grab with `mss` + `PIL` downscaling.
- Implement `wisprhelp/vision/vlm_agent.py` for Priority 3 visual element grounding and iterative action loops.
- Implement `wisprhelp/ui/overlay.py` with PySide6 click-through floating pill HUD.
- Wire up full runtime in `wisprhelp/core/runtime.py` and `wisprhelp/main.py`.

---

## 4. Verification & Testing Strategy

1. **Automated Unit Tests**:
   ```bash
   python -m pytest tests/test_jev_classification.py -v
   python -m pytest tests/test_hal_adapters.py -v
   python -m pytest tests/test_unicode_typing.py -v
   ```
2. **Fast-Path Benchmarks**:
   - Verify deterministic actions (Priority 1) execute in $< 50\text{ms}$ after classification.
   - Verify accessibility actions (Priority 2) execute in $< 150\text{ms}$ without vision inference.
3. **Multilingual Typing Verification**:
   - Inject English, Hindi Devanagari (`नमस्ते दुनिया`), and Hinglish into active test window; verify zero font corruption.
4. **Safety & Kill Switch Verification**:
   - Verify mouse fling to corner and `Esc` key immediately abort active agent actions.
